"""
Compute reward scores and z-score normalize for scripted rollout data.

Unlike the learned reward model pipeline, this computes ground-truth metrics
directly from the trajectories:
  - success: always 1.0 for scripted data
  - speed: how quickly the task was completed
  - smoothness: jerk-based smoothness of actions
  - peg: +1 for left peg, -1 for right peg

Usage:
  python scripts/compute_scores_from_scripted.py \
    --rollout_data pipeline_output/rollouts.npz \
    --demo_hdf5 pipeline_output/demos.hdf5 \
    --output_dir pipeline_output/scores
"""

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import click
import numpy as np
import h5py


OBS_KEYS = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']


@click.command()
@click.option('--rollout_data', required=True, help='Path to .npz from collect_initial_scripted_rollouts.py')
@click.option('--demo_hdf5', required=True, help='Path to demos.hdf5 (same data, used for demo_scores in scores.json)')
@click.option('--output_dir', required=True)
@click.option('--reward_axes', default=None,
              help='Comma-separated reward axes. Any combination of: success,speed_reward,smoothness,peg_reward,composite')
def main(rollout_data, demo_hdf5, output_dir, reward_axes):
    os.makedirs(output_dir, exist_ok=True)

    # Load rollout data
    data = np.load(rollout_data)
    success = data['success']
    speed_reward = data['speed_reward']
    smoothness = data['smoothness']
    peg_reward = data['peg_reward']

    available = {
        'success': ('success', success),
        'speed_reward': ('speed', speed_reward),
        'smoothness': ('smoothness', smoothness),
        'peg_reward': ('peg', peg_reward),
        'composite': ('composite', (success + speed_reward + smoothness) / 3),
    }

    if reward_axes is not None:
        axes = [a.strip() for a in reward_axes.split(',')]
    else:
        axes = ['success', 'speed_reward', 'smoothness', 'peg_reward']

    reward_names = []
    metric_cols = []
    for ax in axes:
        if ax not in available:
            raise ValueError(f"Unknown reward axis '{ax}'. Available: {list(available.keys())}")
        name, values = available[ax]
        reward_names.append(name)
        metric_cols.append(values)
    metrics = np.stack(metric_cols, axis=-1)

    print(f"Loaded {len(metrics)} rollouts")
    print(f"  Success rate: {success.mean():.3f}")
    print(f"  Mean speed:   {speed_reward.mean():.3f}")
    print(f"  Mean smooth:  {smoothness.mean():.3f}")
    print(f"  Mean peg:     {peg_reward.mean():.3f}")
    print(f"  Left peg:     {(peg_reward > 0).sum()}, Right peg: {(peg_reward < 0).sum()}")
    if 'speed_left' in data and 'speed_right' in data:
        speed_left = data['speed_left']
        speed_right = data['speed_right']
        print(f"  Mean speed_left:  {speed_left[speed_left > 0].mean():.3f} ({(speed_left > 0).sum()} eps)")
        print(f"  Mean speed_right: {speed_right[speed_right > 0].mean():.3f} ({(speed_right > 0).sum()} eps)")

    # Z-score normalize
    score_mean = metrics.mean(axis=0)
    score_std = metrics.std(axis=0)
    score_std[score_std < 1e-8] = 1.0  # avoid division by zero

    rollout_z = (metrics - score_mean) / score_std

    # For this pipeline, demos and rollouts are the same data.
    # Set demo_scores_zscore to empty so RewardConditionedLowdimDataset skips demo loading.
    scores = {
        'score_mean': score_mean.tolist(),
        'score_std': score_std.tolist(),
        'reward_names': ['success', 'speed', 'smoothness', 'peg'],
        'rollout_scores_raw': metrics.tolist(),
        'rollout_scores_zscore': rollout_z.tolist(),
        'demo_scores_raw': [],
        'demo_scores_zscore': [],
        'n_rollouts': len(metrics),
        'n_demos': 0,
    }

    scores_path = os.path.join(output_dir, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=2)

    print(f"\nScoring complete:")
    print(f"  Score mean: {score_mean}")
    print(f"  Score std:  {score_std}")
    print(f"  Z-scores range: [{rollout_z.min(axis=0)}, {rollout_z.max(axis=0)}]")
    print(f"  Saved scores to {scores_path}")


if __name__ == '__main__':
    main()
