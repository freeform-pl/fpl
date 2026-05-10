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

    # Load rollout data (skip if path is "none" — demos-only mode)
    has_rollouts = rollout_data and rollout_data != "none"
    if has_rollouts:
        data = np.load(rollout_data)
        rollout_obs = data['obs']  # (N, T, D)
        rollout_lengths = data['episode_lengths']  # (N,)
        rollout_success = data['success']  # (N,)
        rollout_speed = data['speed_reward']  # (N,)
        rollout_smoothness = data['smoothness']  # (N,)
        rollout_peg = data['peg_reward'] if 'peg_reward' in data else None

        n_rollouts = len(rollout_obs)
        obs_dim = rollout_obs.shape[-1]

        print(f"Loaded {n_rollouts} rollouts, obs_dim={obs_dim}")
        print(f"  Success rate: {rollout_success.mean():.3f}")
        print(f"  Mean speed:   {rollout_speed.mean():.3f}")
        print(f"  Mean smooth:  {rollout_smoothness.mean():.3f}")
        if rollout_peg is not None:
            print(f"  Peg: left={np.sum(rollout_peg < 0)}, right={np.sum(rollout_peg > 0)}, none={np.sum(rollout_peg == 0)}")
    else:
        n_rollouts = 0
        rollout_obs = None
        rollout_lengths = None
        rollout_success = None
        rollout_speed = None
        rollout_smoothness = None
        rollout_peg = None
        obs_dim = None
        print("No rollout data — training on demos only.")

    # Load demo data and compute ground-truth metrics
    demo_episodes = load_demo_obs(demo_hdf5, obs_keys, max_demos=max_demos)
    n_demos = len(demo_episodes)

    # Compute demo metrics from HDF5
    demo_success_list = []
    demo_speed_list = []
    demo_smooth_list = []
    demo_peg_list = []
    demo_lengths_list = []
    with h5py.File(demo_hdf5, 'r') as f:
        demos_group = f['data']
        for i in range(n_demos):
            demo = demos_group[f'demo_{i}']
            actions = demo['actions'][:].astype(np.float32)
            L = len(actions)
            demo_lengths_list.append(len(demo_episodes[i]))

            # Success: demos are expert/scripted, assume success
            demo_success_list.append(1.0)

            # Speed: based on episode length
            demo_speed_list.append(1.0 - 0.9 * (L / 600.0))

            # Smoothness: jerk-based
            if L >= 3:
                vel = np.diff(actions, axis=0)
                acc = np.diff(vel, axis=0)
                jerk = np.diff(acc, axis=0)
                jerk_mag = float(np.mean(np.linalg.norm(jerk, axis=-1)))
                demo_smooth_list.append(float(np.exp(-10.0 * jerk_mag)))
            else:
                demo_smooth_list.append(1.0)

            # Peg reward: from attrs if available, else 0.0
            target_peg = demo.attrs.get('target_peg', None)
            if target_peg is not None:
                demo_peg_list.append(1.0 if target_peg == 'right' else -1.0)
            else:
                demo_peg_list.append(0.0)

    demo_success = np.array(demo_success_list, dtype=np.float32)
    demo_speed = np.array(demo_speed_list, dtype=np.float32)
    demo_smoothness = np.array(demo_smooth_list, dtype=np.float32)
    demo_peg = np.array(demo_peg_list, dtype=np.float32)

    # Infer obs_dim from demos if no rollouts
    if obs_dim is None:
        obs_dim = demo_episodes[0].shape[-1]

    # Pad demo obs to same format as rollouts
    max_demo_len = max(demo_lengths_list) if demo_lengths_list else 0
    max_T = max(rollout_obs.shape[1], max_demo_len) if has_rollouts else max_demo_len
    demo_obs_padded = np.zeros((n_demos, max_T, obs_dim), dtype=np.float32)
    for i, ep in enumerate(demo_episodes):
        demo_obs_padded[i, :len(ep)] = ep
    demo_lengths = np.array(demo_lengths_list, dtype=np.int32)

    # Pad rollout obs if demos are longer
    if has_rollouts and max_T > rollout_obs.shape[1]:
        padded = np.zeros((n_rollouts, max_T, obs_dim), dtype=np.float32)
        padded[:, :rollout_obs.shape[1]] = rollout_obs
        rollout_obs = padded

    print(f"Loaded {n_demos} demos")
    print(f"  Demo mean speed:   {demo_speed.mean():.3f}")
    print(f"  Demo mean smooth:  {demo_smoothness.mean():.3f}")

    # Concatenate rollouts + demos for preference training
    if has_rollouts:
        all_obs = np.concatenate([rollout_obs, demo_obs_padded], axis=0)
        all_lengths = np.concatenate([rollout_lengths, demo_lengths], axis=0)
        all_success = np.concatenate([rollout_success, demo_success], axis=0)
        all_speed = np.concatenate([rollout_speed, demo_speed], axis=0)
        all_smoothness = np.concatenate([rollout_smoothness, demo_smoothness], axis=0)
        all_peg = np.concatenate([rollout_peg, demo_peg], axis=0) if rollout_peg is not None else None
    else:
        all_obs = demo_obs_padded
        all_lengths = demo_lengths
        all_success = demo_success
        all_speed = demo_speed
        all_smoothness = demo_smoothness
        all_peg = demo_peg

    # All available axes (over combined rollouts + demos)
    available = {
        'success': ('success', all_success),
        'speed_reward': ('speed', all_speed),
        'smoothness': ('smoothness', all_smoothness),
    }
    if all_peg is not None:
        available['peg_reward'] = ('peg', all_peg)
    available['composite'] = ('composite', (all_success + all_speed + all_smoothness) / 3)

    # Select axes
    if reward_axes is not None:
        axes = [a.strip() for a in reward_axes.split(',')]
    else:
        # Default: all available non-composite axes
        axes = ['success', 'speed_reward', 'smoothness']
        if all_peg is not None:
            axes.append('peg_reward')

    reward_names = []
    metric_cols = []
    for ax in axes:
        if ax not in available:
            raise ValueError(f"Unknown reward axis '{ax}'. Available: {list(available.keys())}")
        name, values = available[ax]
        reward_names.append(name)
        metric_cols.append(values)
    metrics = np.stack(metric_cols, axis=-1)  # (N_rollouts + N_demos, K)

    num_rewards = metrics.shape[1]
    print(f"Training reward model on {len(all_obs)} episodes ({n_rollouts} rollouts + {n_demos} demos), num_rewards={num_rewards}")

    # Log ground truth metric distributions (rollouts vs demos)
    rollout_metrics = metrics[:n_rollouts]
    demo_metrics = metrics[n_rollouts:]
    fig, axes = plt.subplots(1, num_rewards, figsize=(4 * num_rewards, 3), squeeze=False)
    for k, name in enumerate(reward_names):
        ax = axes[0, k]
        ax.hist(rollout_metrics[:, k], bins=30, alpha=0.6, label='rollouts', edgecolor='black')
        ax.hist(demo_metrics[:, k], bins=30, alpha=0.6, label='demos', edgecolor='black')
        ax.set_title(f'{name} (ground truth)')
        ax.set_xlabel('value')
        ax.set_ylabel('count')
        ax.legend(fontsize=8)
    fig.suptitle('Ground Truth Metric Distributions (Rollouts + Demos)')
    fig.tight_layout()
    wandb.log({'reward_model/gt_distributions': wandb.Image(fig)})
    plt.close(fig)

    # Create train/val datasets (preferences across rollouts AND demos)
    val_ratio = 0.1
    n_val_pairs = max(int(n_pairs * val_ratio), 100)
    n_train_pairs = n_pairs - n_val_pairs

    train_dataset = PreferencePairDataset(
        all_obs, all_lengths, metrics,
        max_seq_len=max_seq_len, n_pairs=n_train_pairs, seed=42)
    val_dataset = PreferencePairDataset(
        all_obs, all_lengths, metrics,
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
            pred_scores = score_episodes(model, all_obs, all_lengths, max_seq_len, device)
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

    # Score rollouts and demos separately
    model.eval()
    if has_rollouts:
        rollout_scores = score_episodes(model, rollout_obs, rollout_lengths, max_seq_len, device)
    else:
        rollout_scores = np.zeros((0, num_rewards), dtype=np.float32)
    demo_scores = score_episodes(model, demo_obs_padded, demo_lengths, max_seq_len, device)

    # Normalize scores to [-1, 1] using min/max
    all_scores = np.concatenate([rollout_scores, demo_scores], axis=0)
    score_min = all_scores.min(axis=0)  # (K,)
    score_max = all_scores.max(axis=0)  # (K,)
    score_range = score_max - score_min
    score_range[score_range < 1e-8] = 1.0  # avoid division by zero

    rollout_z = 2.0 * (rollout_scores - score_min) / score_range - 1.0 if has_rollouts else np.zeros((0, num_rewards), dtype=np.float32)
    demo_z = 2.0 * (demo_scores - score_min) / score_range - 1.0

    # Save scores
    scores = {
        'score_min': score_min.tolist(),
        'score_max': score_max.tolist(),
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
    print(f"  Score min: {score_min}")
    print(f"  Score max: {score_max}")
    if has_rollouts:
        print(f"  Rollout normalized range: [{rollout_z.min(axis=0)}, {rollout_z.max(axis=0)}]")
    print(f"  Demo normalized range:    [{demo_z.min(axis=0)}, {demo_z.max(axis=0)}]")
    print(f"  Saved scores to {scores_path}")

    # Log scoring summary to wandb
    for k, name in enumerate(reward_names):
        wandb.summary[f'scoring/score_min_{name}'] = float(score_min[k])
        wandb.summary[f'scoring/score_max_{name}'] = float(score_max[k])
        if has_rollouts:
            wandb.summary[f'scoring/rollout_norm_min_{name}'] = float(rollout_z[:, k].min())
            wandb.summary[f'scoring/rollout_norm_max_{name}'] = float(rollout_z[:, k].max())
        if len(demo_scores) > 0:
            wandb.summary[f'scoring/demo_norm_min_{name}'] = float(demo_z[:, k].min())
            wandb.summary[f'scoring/demo_norm_max_{name}'] = float(demo_z[:, k].max())
            wandb.summary[f'scoring/demo_norm_mean_{name}'] = float(demo_z[:, k].mean())
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
