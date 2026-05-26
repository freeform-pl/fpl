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
from reward_functions import AXIS_FUNCTIONS, compute_axes


def stride_indices(episode_len: int, max_seq_len: int, stride: int = 1) -> np.ndarray:
    """Return at most `max_seq_len` indices into [0, episode_len), spaced by `stride`.

    stride=1 (default): indices 0,1,...,min(L,max_seq_len)-1 — same prefix-only
    behavior as before; long episodes get truncated at max_seq_len.
    stride>1: indices 0,stride,2*stride,... capped to max_seq_len entries; the
    reward model then sees the whole trajectory at lower temporal resolution.
    """
    L = int(episode_len)
    if L <= 0:
        return np.zeros(0, dtype=np.int64)
    stride = max(int(stride), 1)
    if stride == 1:
        return np.arange(min(L, max_seq_len), dtype=np.int64)
    idx = np.arange(0, L, stride, dtype=np.int64)
    return idx[:max_seq_len]


class PreferencePairDataset(Dataset):
    """Generate preference pairs from rollout metrics."""

    def __init__(self, obs, episode_lengths, metrics, max_seq_len=512, stride=1,
                 n_pairs=None, seed=42):
        """
        Args:
            obs: (N, T, D) padded observations
            episode_lengths: (N,) actual lengths
            metrics: (N, K) ground truth metric values per episode
            max_seq_len: truncate sequences to this length
            stride: 1 = take consecutive prefix (default), >1 = stride-subsample
                    so the whole trajectory is seen at lower temporal resolution.
            n_pairs: number of preference pairs to generate.
                     None (default) = all unique pairs N*(N-1)/2.
                     An integer = randomly sample that many pairs.
        """
        self.obs = obs
        self.episode_lengths = episode_lengths
        self.metrics = metrics
        self.max_seq_len = max_seq_len
        self.stride = int(stride)

        rng = np.random.RandomState(seed)
        N = len(obs)
        K = metrics.shape[1]

        if n_pairs is None:
            # Generate all unique pairs (i, j) with i < j
            from itertools import combinations
            all_pairs = list(combinations(range(N), 2))
            idx_a = np.array([p[0] for p in all_pairs])
            idx_b = np.array([p[1] for p in all_pairs])
            n_pairs = len(all_pairs)
        else:
            # Randomly sample n_pairs pairs
            all_unique = N * (N - 1) // 2
            if n_pairs >= all_unique:
                # If requesting more than all unique pairs, just use all
                from itertools import combinations
                all_pairs = list(combinations(range(N), 2))
                idx_a = np.array([p[0] for p in all_pairs])
                idx_b = np.array([p[1] for p in all_pairs])
                n_pairs = len(all_pairs)
            else:
                idx_a = rng.randint(0, N, size=n_pairs)
                idx_b = rng.randint(0, N, size=n_pairs)
                # Ensure different episodes
                mask = idx_a == idx_b
                idx_b[mask] = (idx_b[mask] + 1) % N

        # Labels: 1.0 if A preferred, 0.0 if B preferred, 0.5 if equal
        labels = np.full((n_pairs, K), 0.5, dtype=np.float32)
        for k in range(K):
            labels[:, k] = np.where(
                metrics[idx_a, k] > metrics[idx_b, k], 1.0,
                np.where(metrics[idx_a, k] < metrics[idx_b, k], 0.0, 0.5)
            )

        self.idx_a = idx_a
        self.idx_b = idx_b
        self.labels = labels

    def __len__(self):
        return len(self.idx_a)

    def __getitem__(self, idx):
        ia, ib = self.idx_a[idx], self.idx_b[idx]
        L = self.max_seq_len

        idx_a_steps = stride_indices(int(self.episode_lengths[ia]), L, self.stride)
        idx_b_steps = stride_indices(int(self.episode_lengths[ib]), L, self.stride)

        obs_a = np.zeros((L, self.obs.shape[-1]), dtype=np.float32)
        obs_b = np.zeros((L, self.obs.shape[-1]), dtype=np.float32)
        mask_a = np.ones(L, dtype=bool)  # True = padded
        mask_b = np.ones(L, dtype=bool)

        obs_a[:len(idx_a_steps)] = self.obs[ia, idx_a_steps]
        obs_b[:len(idx_b_steps)] = self.obs[ib, idx_b_steps]
        mask_a[:len(idx_a_steps)] = False
        mask_b[:len(idx_b_steps)] = False

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
@click.option('--rollout_data', required=True, help='Path(s) to .npz from collect_rollouts.py (comma-separated for multiple files)')
@click.option('--demo_hdf5', required=True, help='Path to demo HDF5 for scoring original demos')
@click.option('--output_dir', required=True)
@click.option('--obs_keys', default='object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos')
@click.option('--epochs', default=100, type=int)
@click.option('--batch_size', default=64, type=int)
@click.option('--lr', default=1e-4, type=float)
@click.option('--n_pairs', default=None, type=int, help='Number of preference pairs. Default: all unique pairs N*(N-1)/2')
@click.option('--max_seq_len', default=512, type=int)
@click.option('--stride', default=1, type=int,
              help='1=prefix only (default, preserves prior behavior); >1 stride-subsamples so the whole trajectory fits in max_seq_len.')
@click.option('--max_demos', default=None, type=int, help='Max original demos to include')
@click.option('--device', default='cuda:0')
@click.option('--wandb_project', default='reward_cond_pipeline', help='wandb project name')
@click.option('--reward_axes', default=None,
              help='Comma-separated reward axes to use. Any combination of: success,speed_reward,smoothness,peg_reward,order_reward,milk_placed,bread_placed,cereal_placed,can_placed,drop_reward,composite(...)')
def main(rollout_data, demo_hdf5, output_dir, obs_keys, epochs, batch_size, lr,
         n_pairs, max_seq_len, stride, max_demos, device, wandb_project, reward_axes):
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
            'n_pairs': n_pairs if n_pairs is not None else 'all',
            'max_seq_len': max_seq_len,
        },
    )

    # Load rollout data (skip if path is "none" — demos-only mode)
    # Supports comma-separated list of npz files
    rollout_paths = [p.strip() for p in rollout_data.split(',') if p.strip() and p.strip() != 'none']
    has_rollouts = len(rollout_paths) > 0
    # --------------------------------------------------------------------- #
    # Load obs/actions/lengths from rollouts + demos. Axis values are computed
    # at the end via reward_functions.AXIS_FUNCTIONS — one source of truth.
    # --------------------------------------------------------------------- #
    if has_rollouts:
        all_obs_chunks, all_action_chunks, all_lengths_chunks = [], [], []
        all_reward_chunks = []   # per-step env rewards (for axis fns that need them)
        for rpath in rollout_paths:
            data = np.load(rpath)
            n = len(data['episode_lengths'])
            all_obs_chunks.append(data['obs'][:n])
            all_lengths_chunks.append(data['episode_lengths'])
            if 'actions' in data:
                all_action_chunks.append(data['actions'][:n])
            else:
                # Some legacy npz files don't store actions — substitute zeros
                # so the smoothness axis just returns 0 for those trajectories.
                all_action_chunks.append(np.zeros((n, data['obs'].shape[1], 1), dtype=np.float32))
            if 'rewards' in data:
                all_reward_chunks.append(data['rewards'][:n])
            else:
                all_reward_chunks.append(None)
            print(f"  Loaded {n} rollouts from {rpath}")

        # Pad obs/actions to a common max T before concatenating.
        max_t = max(a.shape[1] for a in all_obs_chunks)
        obs_dim = all_obs_chunks[0].shape[-1]
        act_dim = all_action_chunks[0].shape[-1] if all_action_chunks[0] is not None else 1
        for i, obs_arr in enumerate(all_obs_chunks):
            if obs_arr.shape[1] < max_t:
                pad = np.zeros((obs_arr.shape[0], max_t - obs_arr.shape[1], obs_dim), dtype=np.float32)
                all_obs_chunks[i] = np.concatenate([obs_arr, pad], axis=1)
        for i, act_arr in enumerate(all_action_chunks):
            if act_arr.shape[1] < max_t:
                pad = np.zeros((act_arr.shape[0], max_t - act_arr.shape[1], act_arr.shape[-1]), dtype=np.float32)
                all_action_chunks[i] = np.concatenate([act_arr, pad], axis=1)

        rollout_obs = np.concatenate(all_obs_chunks, axis=0)
        rollout_actions = np.concatenate(all_action_chunks, axis=0)
        rollout_lengths = np.concatenate(all_lengths_chunks)
        n_rollouts = len(rollout_obs)
        print(f"Total: {n_rollouts} rollouts from {len(rollout_paths)} file(s), obs_dim={obs_dim}")
    else:
        n_rollouts = 0
        rollout_obs = None
        rollout_actions = None
        rollout_lengths = None
        obs_dim = None
        print("No rollout data — training on demos only.")

    # Load demo trajectories from HDF5.
    demo_episodes = load_demo_obs(demo_hdf5, obs_keys, max_demos=max_demos)
    n_demos = len(demo_episodes)
    demo_actions_list = []
    demo_lengths_list = []
    with h5py.File(demo_hdf5, 'r') as f:
        demos_group = f['data']
        for i in range(n_demos):
            demo = demos_group[f'demo_{i}']
            actions = demo['actions'][:].astype(np.float32)
            demo_actions_list.append(actions)
            demo_lengths_list.append(len(demo_episodes[i]))

    # Infer obs_dim from demos if no rollouts
    if obs_dim is None:
        obs_dim = demo_episodes[0].shape[-1]

    # Pad demo obs/actions to same T as rollouts. max_T = max over the
    # longest trajectory in any of: rollout obs, rollout actions, demos.
    # Rollout obs and actions can have different padded T across iteration-
    # rollout files (one file's actions may be longer than another file's
    # obs), so we need to size the combined array to the absolute longest.
    max_demo_len = max(demo_lengths_list) if demo_lengths_list else 0
    if has_rollouts:
        max_T = max(rollout_obs.shape[1], rollout_actions.shape[1], max_demo_len)
    else:
        max_T = max_demo_len
    act_dim_demo = demo_actions_list[0].shape[-1] if demo_actions_list else (
        rollout_actions.shape[-1] if has_rollouts else 1)

    demo_obs_padded = np.zeros((n_demos, max_T, obs_dim), dtype=np.float32)
    demo_actions_padded = np.zeros((n_demos, max_T, act_dim_demo), dtype=np.float32)
    for i, ep in enumerate(demo_episodes):
        demo_obs_padded[i, :len(ep)] = ep
        a = demo_actions_list[i]
        demo_actions_padded[i, :len(a), :a.shape[-1]] = a
    demo_lengths = np.array(demo_lengths_list, dtype=np.int32)

    # Pad rollout obs/actions to max_T if either is shorter. (Either dimension
    # could be the shorter one — obs typically T+1 vs action T, plus per-file
    # padding can leave them inconsistent across files.)
    if has_rollouts and rollout_obs.shape[1] < max_T:
        new_obs = np.zeros((n_rollouts, max_T, obs_dim), dtype=np.float32)
        new_obs[:, :rollout_obs.shape[1]] = rollout_obs
        rollout_obs = new_obs
    if has_rollouts and rollout_actions.shape[1] < max_T:
        new_act = np.zeros((n_rollouts, max_T, rollout_actions.shape[-1]), dtype=np.float32)
        new_act[:, :rollout_actions.shape[1]] = rollout_actions
        rollout_actions = new_act

    print(f"Loaded {n_demos} demos")

    # Concatenate rollouts + demos. Both halves go through the same axis
    # computation below.
    if has_rollouts:
        all_obs = np.concatenate([rollout_obs, demo_obs_padded], axis=0)
        all_actions_arr = np.concatenate([rollout_actions, demo_actions_padded], axis=0) \
            if rollout_actions.shape[-1] == demo_actions_padded.shape[-1] else None
        all_lengths = np.concatenate([rollout_lengths, demo_lengths], axis=0)
    else:
        all_obs = demo_obs_padded
        all_actions_arr = demo_actions_padded
        all_lengths = demo_lengths

    # --------------------------------------------------------------------- #
    # Select axes and compute their values per trajectory by calling the
    # functions in reward_model/reward_functions.py.
    # --------------------------------------------------------------------- #
    import re
    if reward_axes is not None:
        requested_axes = [a.strip() for a in reward_axes.split(',')]
    else:
        # Default: success + speed + smoothness only; the user opts into the
        # task-specific axes via --reward_axes.
        requested_axes = ['success', 'speed_reward', 'smoothness']

    # Expand composite(...) entries — collect the unique base axes we need to
    # compute, then average them back together as composites.
    base_axes_needed = set()
    expanded = []   # list of (kind, payload) — kind in {'plain', 'composite'}
    for ax in requested_axes:
        m = re.match(r'^composite\((.+)\)$', ax)
        if m:
            sub = [s.strip() for s in m.group(1).split('+')]
            base_axes_needed.update(sub)
            expanded.append(('composite', sub))
        else:
            base_axes_needed.add(ax)
            expanded.append(('plain', ax))
    unknown = [a for a in base_axes_needed if a not in AXIS_FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown reward axis(es): {unknown}. Available: {list(AXIS_FUNCTIONS.keys())}")

    # Compute base axis values per trajectory. Trim padded obs/actions down
    # to the actual episode length BEFORE handing them to the reward fns —
    # the fns operate on whole-trajectory data and don't accept a length arg.
    print(f"Computing per-axis rewards for {len(all_obs)} trajectories "
          f"over {len(base_axes_needed)} base axes: {sorted(base_axes_needed)}")
    base_axis_values = {name: np.zeros(len(all_obs), dtype=np.float32) for name in base_axes_needed}
    for i in range(len(all_obs)):
        L_i = int(all_lengths[i])
        obs_i = all_obs[i][:L_i]
        act_i = all_actions_arr[i][:L_i] if all_actions_arr is not None else None
        for name in base_axes_needed:
            base_axis_values[name][i] = AXIS_FUNCTIONS[name](obs_i, actions=act_i)

    # Stack into (N, K) metrics in the requested order (with composites).
    reward_names = []
    metric_cols = []
    for kind, payload in expanded:
        if kind == 'plain':
            reward_names.append(payload)
            metric_cols.append(base_axis_values[payload])
        else:
            sub = payload
            reward_names.append('composite(' + '+'.join(sub) + ')')
            metric_cols.append(sum(base_axis_values[s] for s in sub) / len(sub))
    metrics = np.stack(metric_cols, axis=-1)  # (N_rollouts + N_demos, K)

    # Convenience pulls used downstream by the histogram plot.
    def _compute_or_default(name):
        if name in base_axis_values:
            return base_axis_values[name]
        vals = np.zeros(len(all_obs), dtype=np.float32)
        for i in range(len(all_obs)):
            L_i = int(all_lengths[i])
            obs_i = all_obs[i][:L_i]
            act_i = all_actions_arr[i][:L_i] if all_actions_arr is not None else None
            vals[i] = AXIS_FUNCTIONS[name](obs_i, actions=act_i)
        return vals

    all_success = _compute_or_default('success')
    all_speed = _compute_or_default('speed_reward')
    all_smoothness = _compute_or_default('smoothness')
    print(f"  Success rate: {all_success.mean():.3f}  "
          f"Mean speed: {all_speed.mean():.3f}  Mean smooth: {all_smoothness.mean():.3f}")

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
    N_total = len(all_obs)
    all_unique_pairs = N_total * (N_total - 1) // 2
    effective_n_pairs = n_pairs if n_pairs is not None else all_unique_pairs
    n_val_pairs = max(int(effective_n_pairs * val_ratio), min(100, effective_n_pairs))
    # Ensure we don't allocate all pairs to validation
    n_val_pairs = min(n_val_pairs, int(effective_n_pairs * 0.5))
    n_val_pairs = max(n_val_pairs, 1)
    n_train_pairs = effective_n_pairs - n_val_pairs
    print(f"  Preference pairs: {effective_n_pairs} total ({n_train_pairs} train, {n_val_pairs} val)"
          + (f" [all unique pairs]" if n_pairs is None else f" [specified]"))

    train_dataset = PreferencePairDataset(
        all_obs, all_lengths, metrics,
        max_seq_len=max_seq_len, stride=stride, n_pairs=n_train_pairs, seed=42)
    val_dataset = PreferencePairDataset(
        all_obs, all_lengths, metrics,
        max_seq_len=max_seq_len, stride=stride, n_pairs=n_val_pairs, seed=123)
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
            pred_scores = score_episodes(model, all_obs, all_lengths, max_seq_len, device, stride=stride)
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
        rollout_scores = score_episodes(model, rollout_obs, rollout_lengths, max_seq_len, device, stride=stride)
    else:
        rollout_scores = np.zeros((0, num_rewards), dtype=np.float32)
    demo_scores = score_episodes(model, demo_obs_padded, demo_lengths, max_seq_len, device, stride=stride)

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


def score_episodes(model, obs, episode_lengths, max_seq_len, device, batch_size=64, stride=1):
    """Score episodes with the reward model. Returns (N, K) array.

    Uses the same stride as training so saved scores match what the model saw.
    """
    N = len(obs)
    all_scores = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_obs = []
        batch_masks = []

        for i in range(start, end):
            idx_steps = stride_indices(int(episode_lengths[i]), max_seq_len, stride)
            padded = np.zeros((max_seq_len, obs.shape[-1]), dtype=np.float32)
            mask = np.ones(max_seq_len, dtype=bool)
            padded[:len(idx_steps)] = obs[i, idx_steps]
            mask[:len(idx_steps)] = False
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
