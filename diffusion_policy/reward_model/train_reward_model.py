"""
Train a state-space reward model on rollout data using Bradley-Terry preferences.

After training, scores all rollouts + original demos and saves z-score normalized scores.

Usage:
  python reward_model/train_reward_model.py \
    --rollout_data rollouts.npz \
    --demo_hdf5 data/robomimic/datasets/square/mh/low_dim.hdf5 \
    --output_dir reward_model_output
"""

import sys
import os
import pathlib

# Add repo root and reward_model dir to path
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, str(pathlib.Path(__file__).parent))
os.chdir(ROOT_DIR)

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import json
import click
import torch
import numpy as np
import h5py
import wandb
from torch.utils.data import Dataset, DataLoader

from state_reward_model import StateRewardModel, bradley_terry_loss


class PreferencePairDataset(Dataset):
    """Generate preference pairs from rollout metrics."""

    def __init__(self, obs, episode_lengths, metrics, max_seq_len=512, n_pairs=10000, seed=42):
        """
        Args:
            obs: (N, T, D) padded observations
            episode_lengths: (N,) actual lengths
            metrics: (N, K) ground truth metric values per episode
            max_seq_len: truncate sequences to this length
            n_pairs: number of preference pairs to generate
        """
        self.obs = obs
        self.episode_lengths = episode_lengths
        self.metrics = metrics
        self.max_seq_len = max_seq_len

        rng = np.random.RandomState(seed)
        N = len(obs)
        K = metrics.shape[1]

        # Generate random pairs
        idx_a = rng.randint(0, N, size=n_pairs)
        idx_b = rng.randint(0, N, size=n_pairs)
        # Ensure different episodes
        mask = idx_a == idx_b
        idx_b[mask] = (idx_b[mask] + 1) % N

        # Labels: 1.0 if A is preferred (higher metric), 0.0 if B is preferred
        labels = np.zeros((n_pairs, K), dtype=np.float32)
        for k in range(K):
            labels[:, k] = (metrics[idx_a, k] > metrics[idx_b, k]).astype(np.float32)

        self.idx_a = idx_a
        self.idx_b = idx_b
        self.labels = labels

    def __len__(self):
        return len(self.idx_a)

    def __getitem__(self, idx):
        ia, ib = self.idx_a[idx], self.idx_b[idx]
        L = self.max_seq_len

        len_a = min(self.episode_lengths[ia], L)
        len_b = min(self.episode_lengths[ib], L)

        obs_a = np.zeros((L, self.obs.shape[-1]), dtype=np.float32)
        obs_b = np.zeros((L, self.obs.shape[-1]), dtype=np.float32)
        mask_a = np.ones(L, dtype=bool)  # True = padded
        mask_b = np.ones(L, dtype=bool)

        obs_a[:len_a] = self.obs[ia, :len_a]
        obs_b[:len_b] = self.obs[ib, :len_b]
        mask_a[:len_a] = False
        mask_b[:len_b] = False

        return {
            'obs_a': obs_a,
            'obs_b': obs_b,
            'mask_a': mask_a,
            'mask_b': mask_b,
            'labels': self.labels[idx],
        }


def load_demo_obs(demo_hdf5, obs_keys, max_demos=None):
    """Load obs from demo HDF5, return list of (T, D) arrays."""
    episodes = []
    with h5py.File(demo_hdf5, 'r') as f:
        demos = f['data']
        n_demos = len(demos)
        if max_demos is not None:
            n_demos = min(n_demos, max_demos)
        for i in range(n_demos):
            demo = demos[f'demo_{i}']
            obs_parts = [demo['obs'][key][:].astype(np.float32) for key in obs_keys]
            obs = np.concatenate(obs_parts, axis=-1)
            episodes.append(obs)
    return episodes


@click.command()
@click.option('--rollout_data', required=True, help='Path to .npz from collect_rollouts.py')
@click.option('--demo_hdf5', required=True, help='Path to demo HDF5 for scoring original demos')
@click.option('--output_dir', required=True)
@click.option('--obs_keys', default='object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos')
@click.option('--epochs', default=100, type=int)
@click.option('--batch_size', default=64, type=int)
@click.option('--lr', default=1e-4, type=float)
@click.option('--n_pairs', default=20000, type=int)
@click.option('--max_seq_len', default=512, type=int)
@click.option('--max_demos', default=None, type=int, help='Max original demos to include')
@click.option('--device', default='cuda:0')
@click.option('--wandb_project', default='reward_cond_pipeline', help='wandb project name')
def main(rollout_data, demo_hdf5, output_dir, obs_keys, epochs, batch_size, lr,
         n_pairs, max_seq_len, max_demos, device, wandb_project):
    os.makedirs(output_dir, exist_ok=True)
    obs_keys = obs_keys.split(',')
    device = torch.device(device)

    # Init wandb
    wandb.init(
        project=wandb_project,
        name='phase2_reward_model',
        config={
            'rollout_data': rollout_data,
            'demo_hdf5': demo_hdf5,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'n_pairs': n_pairs,
            'max_seq_len': max_seq_len,
        },
    )

    # Load rollout data
    data = np.load(rollout_data)
    obs = data['obs']  # (N, T, D)
    episode_lengths = data['episode_lengths']  # (N,)
    success = data['success']  # (N,)
    speed_reward = data['speed_reward']  # (N,)
    smoothness = data['smoothness']  # (N,)

    # K=3 metrics: success, speed, smoothness
    metrics = np.stack([success, speed_reward, smoothness], axis=-1)  # (N, 3)

    obs_dim = obs.shape[-1]
    num_rewards = 3

    print(f"Loaded {len(obs)} rollouts, obs_dim={obs_dim}")
    print(f"  Success rate: {success.mean():.3f}")
    print(f"  Mean speed:   {speed_reward.mean():.3f}")
    print(f"  Mean smooth:  {smoothness.mean():.3f}")

    # Create dataset and dataloader
    dataset = PreferencePairDataset(
        obs, episode_lengths, metrics,
        max_seq_len=max_seq_len, n_pairs=n_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Create model
    model = StateRewardModel(
        obs_dim=obs_dim,
        num_rewards=num_rewards,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        ffn_dim=512,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Train
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_acc = np.zeros(num_rewards)
        n_batches = 0

        for batch in dataloader:
            obs_a = batch['obs_a'].to(device)
            obs_b = batch['obs_b'].to(device)
            mask_a = batch['mask_a'].to(device)
            mask_b = batch['mask_b'].to(device)
            labels = batch['labels'].to(device)

            rewards_a = model(obs_a, mask_a)
            rewards_b = model(obs_b, mask_b)

            loss, acc = bradley_terry_loss(rewards_a, rewards_b, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.cpu().numpy()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        # Log to wandb every epoch
        wandb.log({
            'reward_model/loss': avg_loss,
            'reward_model/acc_success': avg_acc[0],
            'reward_model/acc_speed': avg_acc[1],
            'reward_model/acc_smoothness': avg_acc[2],
            'reward_model/acc_mean': avg_acc.mean(),
            'reward_model/epoch': epoch + 1,
        }, step=epoch + 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
                  f"acc=[{avg_acc[0]:.3f}, {avg_acc[1]:.3f}, {avg_acc[2]:.3f}]")

    # Save model
    ckpt_path = os.path.join(output_dir, 'reward_model.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved reward model to {ckpt_path}")

    # Score all rollouts
    model.eval()
    rollout_scores = score_episodes(model, obs, episode_lengths, max_seq_len, device)

    # Score original demos
    demo_episodes = load_demo_obs(demo_hdf5, obs_keys, max_demos=max_demos)
    demo_obs_list = []
    demo_lengths = []
    for ep in demo_episodes:
        demo_obs_list.append(ep)
        demo_lengths.append(len(ep))

    # Pad demos
    max_demo_len = max(demo_lengths) if demo_lengths else 0
    demo_obs_padded = np.zeros((len(demo_episodes), max(max_demo_len, 1), obs_dim), dtype=np.float32)
    for i, ep in enumerate(demo_obs_list):
        demo_obs_padded[i, :len(ep)] = ep
    demo_lengths = np.array(demo_lengths, dtype=np.int32)

    demo_scores = score_episodes(model, demo_obs_padded, demo_lengths, max_seq_len, device)

    # Compute z-score normalization stats from rollout scores
    all_scores = np.concatenate([rollout_scores, demo_scores], axis=0)
    score_mean = all_scores.mean(axis=0)  # (K,)
    score_std = all_scores.std(axis=0)  # (K,)
    score_std[score_std < 1e-8] = 1.0  # avoid division by zero

    rollout_z = (rollout_scores - score_mean) / score_std
    demo_z = (demo_scores - score_mean) / score_std

    # Save scores
    scores = {
        'score_mean': score_mean.tolist(),
        'score_std': score_std.tolist(),
        'reward_names': ['success', 'speed', 'smoothness'],
        'rollout_scores_raw': rollout_scores.tolist(),
        'rollout_scores_zscore': rollout_z.tolist(),
        'demo_scores_raw': demo_scores.tolist(),
        'demo_scores_zscore': demo_z.tolist(),
        'n_rollouts': len(rollout_scores),
        'n_demos': len(demo_scores),
    }
    scores_path = os.path.join(output_dir, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=2)

    print(f"\nScoring complete:")
    print(f"  Score mean: {score_mean}")
    print(f"  Score std:  {score_std}")
    print(f"  Rollout z-scores range: [{rollout_z.min(axis=0)}, {rollout_z.max(axis=0)}]")
    print(f"  Demo z-scores range:    [{demo_z.min(axis=0)}, {demo_z.max(axis=0)}]")
    print(f"  Saved scores to {scores_path}")

    # Log scoring summary to wandb
    reward_names = ['success', 'speed', 'smoothness']
    for k, name in enumerate(reward_names):
        wandb.summary[f'scoring/score_mean_{name}'] = float(score_mean[k])
        wandb.summary[f'scoring/score_std_{name}'] = float(score_std[k])
        wandb.summary[f'scoring/rollout_zscore_min_{name}'] = float(rollout_z[:, k].min())
        wandb.summary[f'scoring/rollout_zscore_max_{name}'] = float(rollout_z[:, k].max())
        wandb.summary[f'scoring/demo_zscore_min_{name}'] = float(demo_z[:, k].min())
        wandb.summary[f'scoring/demo_zscore_max_{name}'] = float(demo_z[:, k].max())
        wandb.summary[f'scoring/demo_zscore_mean_{name}'] = float(demo_z[:, k].mean())
    wandb.summary['scoring/n_rollouts'] = len(rollout_scores)
    wandb.summary['scoring/n_demos'] = len(demo_scores)
    wandb.finish()


def score_episodes(model, obs, episode_lengths, max_seq_len, device, batch_size=64):
    """Score episodes with the reward model. Returns (N, K) array."""
    N = len(obs)
    all_scores = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_obs = []
        batch_masks = []

        for i in range(start, end):
            L = min(int(episode_lengths[i]), max_seq_len)
            padded = np.zeros((max_seq_len, obs.shape[-1]), dtype=np.float32)
            mask = np.ones(max_seq_len, dtype=bool)
            padded[:L] = obs[i, :L]
            mask[:L] = False
            batch_obs.append(padded)
            batch_masks.append(mask)

        batch_obs = torch.from_numpy(np.stack(batch_obs)).to(device)
        batch_masks = torch.from_numpy(np.stack(batch_masks)).to(device)

        with torch.no_grad():
            scores = model(batch_obs, batch_masks)
        all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores, axis=0)


if __name__ == '__main__':
    main()
