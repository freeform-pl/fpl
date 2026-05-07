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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
@click.option('--reward_axes', default=None,
              help='Comma-separated reward axes to use. Any combination of: success,speed_reward,smoothness,peg_reward,composite')
def main(rollout_data, demo_hdf5, output_dir, obs_keys, epochs, batch_size, lr,
         n_pairs, max_seq_len, max_demos, device, wandb_project, reward_axes):
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

    # All available axes
    available = {
        'success': ('success', success),
        'speed_reward': ('speed', speed_reward),
        'smoothness': ('smoothness', smoothness),
    }
    if 'peg_reward' in data:
        available['peg_reward'] = ('peg', data['peg_reward'])
    available['composite'] = ('composite', success + speed_reward / 0.5 + smoothness / 0.2)

    # Select axes
    if reward_axes is not None:
        axes = [a.strip() for a in reward_axes.split(',')]
    else:
        # Default: all available non-composite axes
        axes = ['success', 'speed_reward', 'smoothness']
        if 'peg_reward' in data:
            axes.append('peg_reward')

    reward_names = []
    metric_cols = []
    for ax in axes:
        if ax not in available:
            raise ValueError(f"Unknown reward axis '{ax}'. Available: {list(available.keys())}")
        name, values = available[ax]
        reward_names.append(name)
        metric_cols.append(values)
    metrics = np.stack(metric_cols, axis=-1)  # (N, K)

    obs_dim = obs.shape[-1]
    num_rewards = metrics.shape[1]

    print(f"Loaded {len(obs)} rollouts, obs_dim={obs_dim}, num_rewards={num_rewards}")
    print(f"  Success rate: {success.mean():.3f}")
    print(f"  Mean speed:   {speed_reward.mean():.3f}")
    print(f"  Mean smooth:  {smoothness.mean():.3f}")
    if 'peg_reward' in data:
        peg_reward = data['peg_reward']
        print(f"  Peg: left={np.sum(peg_reward > 0)}, right={np.sum(peg_reward < 0)}, none={np.sum(peg_reward == 0)}")
    if 'speed_left' in data and 'speed_right' in data:
        speed_left = data['speed_left']
        speed_right = data['speed_right']
        print(f"  Mean speed_left:  {speed_left[speed_left > 0].mean():.3f} ({np.sum(speed_left > 0)} eps)")
        print(f"  Mean speed_right: {speed_right[speed_right > 0].mean():.3f} ({np.sum(speed_right > 0)} eps)")

    # Log ground truth metric distributions
    fig, axes = plt.subplots(1, num_rewards, figsize=(4 * num_rewards, 3), squeeze=False)
    for k, name in enumerate(reward_names):
        ax = axes[0, k]
        ax.hist(metrics[:, k], bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(f'{name} (ground truth)')
        ax.set_xlabel('value')
        ax.set_ylabel('count')
    fig.suptitle('Ground Truth Metric Distributions')
    fig.tight_layout()
    wandb.log({'reward_model/gt_distributions': wandb.Image(fig)})
    plt.close(fig)

    # Create train/val datasets
    val_ratio = 0.1
    n_val_pairs = max(int(n_pairs * val_ratio), 100)
    n_train_pairs = n_pairs - n_val_pairs

    train_dataset = PreferencePairDataset(
        obs, episode_lengths, metrics,
        max_seq_len=max_seq_len, n_pairs=n_train_pairs, seed=42)
    val_dataset = PreferencePairDataset(
        obs, episode_lengths, metrics,
        max_seq_len=max_seq_len, n_pairs=n_val_pairs, seed=123)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

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

        # Validation
        model.eval()
        val_total_loss = 0.0
        val_total_acc = np.zeros(num_rewards)
        val_n_batches = 0
        with torch.no_grad():
            for batch in val_dataloader:
                obs_a = batch['obs_a'].to(device)
                obs_b = batch['obs_b'].to(device)
                mask_a = batch['mask_a'].to(device)
                mask_b = batch['mask_b'].to(device)
                labels = batch['labels'].to(device)

                rewards_a = model(obs_a, mask_a)
                rewards_b = model(obs_b, mask_b)
                loss_val, acc_val = bradley_terry_loss(rewards_a, rewards_b, labels)

                val_total_loss += loss_val.item()
                val_total_acc += acc_val.cpu().numpy()
                val_n_batches += 1

        val_avg_loss = val_total_loss / val_n_batches
        val_avg_acc = val_total_acc / val_n_batches

        # Log to wandb every epoch
        log_dict = {
            'reward_model/train_loss': avg_loss,
            'reward_model/train_acc_mean': avg_acc.mean(),
            'reward_model/val_loss': val_avg_loss,
            'reward_model/val_acc_mean': val_avg_acc.mean(),
            'reward_model/epoch': epoch + 1,
        }
        for k, name in enumerate(reward_names):
            log_dict[f'reward_model/train_acc_{name}'] = avg_acc[k]
            log_dict[f'reward_model/val_acc_{name}'] = val_avg_acc[k]
        # Log predicted score distributions every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            pred_scores = score_episodes(model, obs, episode_lengths, max_seq_len, device)
            fig, axes = plt.subplots(1, num_rewards, figsize=(4 * num_rewards, 3), squeeze=False)
            for k, name in enumerate(reward_names):
                ax = axes[0, k]
                ax.hist(pred_scores[:, k], bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{name} (predicted)')
                ax.set_xlabel('score')
                ax.set_ylabel('count')
            fig.suptitle(f'Predicted Score Distributions (epoch {epoch+1})')
            fig.tight_layout()
            log_dict['reward_model/pred_distributions'] = wandb.Image(fig)
            plt.close(fig)

        wandb.log(log_dict, step=epoch + 1)

        acc_str = ', '.join(f'{avg_acc[k]:.3f}' for k in range(num_rewards))
        val_acc_str = ', '.join(f'{val_avg_acc[k]:.3f}' for k in range(num_rewards))
        print(f"Epoch {epoch+1}/{epochs}  train_loss={avg_loss:.4f}  train_acc=[{acc_str}]  val_loss={val_avg_loss:.4f}  val_acc=[{val_acc_str}]")

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
        'reward_names': reward_names,
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
    print(f"  Reward dims: {reward_names}")
    print(f"  Score mean: {score_mean}")
    print(f"  Score std:  {score_std}")
    print(f"  Rollout z-scores range: [{rollout_z.min(axis=0)}, {rollout_z.max(axis=0)}]")
    print(f"  Demo z-scores range:    [{demo_z.min(axis=0)}, {demo_z.max(axis=0)}]")
    print(f"  Saved scores to {scores_path}")

    # Log scoring summary to wandb
    for k, name in enumerate(reward_names):
        wandb.summary[f'scoring/score_mean_{name}'] = float(score_mean[k])
        wandb.summary[f'scoring/score_std_{name}'] = float(score_std[k])
        wandb.summary[f'scoring/rollout_zscore_min_{name}'] = float(rollout_z[:, k].min())
        wandb.summary[f'scoring/rollout_zscore_max_{name}'] = float(rollout_z[:, k].max())
        if len(demo_scores) > 0:
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
