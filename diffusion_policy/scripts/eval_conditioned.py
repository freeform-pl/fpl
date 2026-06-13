"""
Compare original policy vs reward-conditioned policy at different conditioning values.

Evaluates at z_positive, z_zero, z_negative (per-axis), matching the training-time eval.
Logs all metrics including per-peg success/speed/throughput for slow_fast setups.

Usage:
  python scripts/eval_conditioned.py \
    --original_ckpt <path> \
    --ckpt <path> \
    --scores_path <path_to_scores.json> \
    --n_rollouts 50
"""

import sys
import os
import pathlib

# Add repo root to path so diffusion_policy is importable
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import json
import click
import hydra
import torch
import dill
import numpy as np
import wandb
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.reward_conditioned_lowdim_runner import RewardConditionedLowdimRunner
from diffusion_policy.env_runner.robomimic_lowdim_runner import RobomimicLowdimRunner


def load_policy(checkpoint, device):
    """Load policy from checkpoint, return (policy, cfg)."""
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(device)
    policy.eval()
    return policy, cfg


def run_original_policy(policy, cfg, n_rollouts, n_videos, output_dir, device):
    """Run the original (unconditioned) policy."""
    runner_cfg = cfg.task.env_runner
    runner_cfg.n_train = 0
    runner_cfg.n_train_vis = 0
    runner_cfg.n_test = n_rollouts
    runner_cfg.n_test_vis = n_videos

    runner = hydra.utils.instantiate(runner_cfg, output_dir=output_dir)
    log_data = runner.run(policy)
    return log_data


def run_conditioned_policy(policy, cfg, n_rollouts, target_z_array, num_reward_dims,
                           n_videos, output_dir, device):
    """Run conditioned policy with given per-axis z-score reward targets."""
    runner_cfg = cfg.task.env_runner
    runner_kwargs = OmegaConf.to_container(runner_cfg, resolve=True)
    runner_kwargs.pop('_target_', None)
    runner_kwargs.pop('target_rewards', None)
    runner_kwargs.pop('num_reward_dims', None)
    runner_kwargs['output_dir'] = output_dir
    runner_kwargs['n_train'] = 0
    runner_kwargs['n_train_vis'] = 0
    runner_kwargs['n_test'] = n_rollouts
    runner_kwargs['n_test_vis'] = n_videos

    runner = RewardConditionedLowdimRunner(
        target_rewards=target_z_array.tolist(),
        num_reward_dims=num_reward_dims,
        **runner_kwargs)
    log_data = runner.run(policy)
    return log_data


def log_videos_to_wandb(log_data, prefix):
    """Forward any sim_video_* entries in log_data to wandb under eval/<prefix>/."""
    videos = {}
    for key, value in log_data.items():
        if not key.startswith('test/sim_video_'):
            continue
        seed = key[len('test/sim_video_'):]
        videos[f'eval/{prefix}/sim_video_{seed}'] = value
    if videos:
        wandb.log(videos)
        print(f"  Logged {len(videos)} video(s) to wandb under eval/{prefix}/")


# Slow_fast-specific keys, ordered for a per-peg sub-table. Anything not in this
# list is still extracted and printed — this just controls the per-peg layout.
PEG_KEYS = [
    'mean_success_left', 'mean_success_right',
    'mean_speed_left', 'mean_speed_right',
    'mean_throughput_left', 'mean_throughput_right',
    'mean_score_left', 'mean_score_right',
    'mean_first_success_step_left', 'mean_first_success_step_right',
    'left_peg_rate', 'right_peg_rate',
]

# Core metrics shown first in the main table when present.
CORE_KEYS = [
    'mean_success', 'mean_partial_success', 'mean_full_success',
    'mean_strict_success', 'mean_score',
    'mean_speed_reward', 'mean_smoothness',
    'mean_throughput', 'mean_first_success_step',
    'mean_n_placed', 'mean_n_placed_final', 'mean_first_placement_step',
]


def extract_metrics(log_data):
    """Extract every numeric test/* metric from runner log data.

    Generic over slow_fast and pickplace — both put their per-axis values
    (order_reward, bread_placed, *_drop, peg_reward, ...) under test/ in
    log_data. Non-numeric entries (wandb.Video, etc.) are skipped.
    """
    metrics = {}
    for full_key, value in log_data.items():
        if not full_key.startswith('test/'):
            continue
        key = full_key[len('test/'):]
        # Skip per-seed sim_video / sim_max_reward entries and anything
        # that isn't a plain scalar (wandb.Video etc.).
        if key.startswith('sim_video_') or key.startswith('sim_max_reward_'):
            continue
        if isinstance(value, (int, float, np.floating, np.integer)):
            metrics[key] = float(value)
    return metrics


def log_metrics_to_wandb(metrics, prefix):
    """Log all extracted metrics to wandb under a prefix."""
    log_dict = {}
    for key, value in metrics.items():
        log_dict[f'eval/{prefix}_{key}'] = value
    wandb.log(log_dict)


@click.command()
@click.option('--ckpt', required=True, help='Policy checkpoint to evaluate')
@click.option('--scores_path', required=False, default=None, help='Path to scores.json from reward model training')
@click.option('--n_rollouts', default=50, type=int)
@click.option('--num_reward_dims', default=3, type=int)
@click.option('--eval_z_positive', default=None, type=str, help='Per-axis positive z-targets, e.g. "[1.0,1.0,1.0]"')
@click.option('--eval_z_negative', default=None, type=str, help='Per-axis negative z-targets, e.g. "[-1.0,-1.0,-1.0]"')
@click.option('--is_conditioned', is_flag=True, default=False, help='Whether the policy is reward-conditioned (eval at z_pos/z_zero/z_neg)')
@click.option('--n_videos', default=3, type=int, help='Number of rollout videos to record/upload per z-target (0 disables)')
@click.option('--output_dir', default='eval_conditioned_output')
@click.option('--device', default='cuda:0')
@click.option('--wandb_project', default='reward_cond_pipeline', help='wandb project name')
def main(ckpt, scores_path, n_rollouts, num_reward_dims,
         eval_z_positive, eval_z_negative, is_conditioned, n_videos,
         output_dir, device, wandb_project):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    # Determine num_reward_dims from scores.json if available
    if scores_path and os.path.exists(scores_path):
        with open(scores_path, 'r') as f:
            scores_data = json.load(f)
        num_reward_dims = len(scores_data['reward_names'])
        print(f"Reward score stats: min={scores_data['score_min']}, max={scores_data['score_max']}")
        print(f"Reward dims: {num_reward_dims} ({scores_data['reward_names']})")
    else:
        print("No scores file provided or found, skipping reward score stats.")

    # Build per-axis conditioning targets (scores normalized to [-1, 1])
    z_zero = np.zeros(num_reward_dims, dtype=np.float32)
    if eval_z_positive is not None:
        z_positive = np.array(json.loads(eval_z_positive), dtype=np.float32)
    else:
        z_positive = np.full(num_reward_dims, 0.9, dtype=np.float32)
    if eval_z_negative is not None:
        z_negative = np.array(json.loads(eval_z_negative), dtype=np.float32)
    else:
        z_negative = np.full(num_reward_dims, -0.9, dtype=np.float32)

    print(f"z_positive: {z_positive}")
    print(f"z_zero:     {z_zero}")
    print(f"z_negative: {z_negative}")

    # Init wandb
    wandb.init(
        project=wandb_project,
        name='phase5_eval',
        config={
            'ckpt': ckpt,
            'n_rollouts': n_rollouts,
            'num_reward_dims': num_reward_dims,
            'is_conditioned': is_conditioned,
            'z_positive': z_positive.tolist() if is_conditioned else None,
            'z_negative': z_negative.tolist() if is_conditioned else None,
        },
    )

    results = {}
    policy, cfg = load_policy(ckpt, device)

    if is_conditioned:
        for z_label, z_target in [('z_pos', z_positive), ('z_zero', z_zero), ('z_neg', z_negative)]:
            print(f"\n{'=' * 60}")
            print(f"Running CONDITIONED policy @ {z_label}={z_target}...")
            print("=" * 60)
            cond_log = run_conditioned_policy(
                policy, cfg, n_rollouts, z_target, num_reward_dims,
                n_videos, output_dir, device)
            results[z_label] = extract_metrics(cond_log)
            log_metrics_to_wandb(results[z_label], z_label)
            log_videos_to_wandb(cond_log, z_label)
    else:
        print("\n" + "=" * 60)
        print("Running policy (unconditioned)...")
        print("=" * 60)
        log = run_original_policy(policy, cfg, n_rollouts, n_videos, output_dir, device)
        results['policy'] = extract_metrics(log)
        log_metrics_to_wandb(results['policy'], 'policy')
        log_videos_to_wandb(log, 'policy')

    del policy
    torch.cuda.empty_cache()

    # Collect every metric that appeared in any result, ordered: CORE first,
    # then per-axis (alphabetical), then PEG, then anything else. This makes
    # the eval show drop / order / per-object placed / strict_success for
    # pickplace and the slow_fast peg metrics alike.
    all_seen = set()
    for m in results.values():
        all_seen.update(m.keys())

    def order_keys(seen):
        ordered = []
        for k in CORE_KEYS:
            if k in seen:
                ordered.append(k)
                seen.discard(k)
        peg_present = [k for k in PEG_KEYS if k in seen]
        for k in peg_present:
            seen.discard(k)
        axis_like = sorted(k for k in seen
                           if not k.startswith('mean_') and not k.startswith('sim_'))
        for k in axis_like:
            seen.discard(k)
        rest = sorted(seen)
        return ordered, axis_like, peg_present, rest

    core_present, axis_keys, peg_present, rest_keys = order_keys(set(all_seen))

    def print_table(title, keys, width=14, fmt=".3f"):
        if not keys:
            return
        print("\n" + "=" * (15 + (width + 1) * len(keys)))
        print(f"COMPARISON RESULTS — {title}")
        print("=" * (15 + (width + 1) * len(keys)))
        header = f"{'Policy':<15s}"
        for k in keys:
            header += f" {k.replace('mean_', '')[:width]:>{width}s}"
        print(header)
        print("-" * len(header))
        for name, metrics in results.items():
            row = f"{name:<15s}"
            for k in keys:
                val = metrics.get(k, None)
                row += f" {val:>{width}{fmt}}" if val is not None else f" {'n/a':>{width}s}"
            print(row)

    print_table("Core Metrics", core_present)
    print_table("Per-Axis Metrics", axis_keys)
    print_table("Per-Peg Metrics", peg_present)
    print_table("Other Metrics", rest_keys)
    print("=" * 100)

    # Log all results as wandb summary
    for name, metrics in results.items():
        for metric_name, value in metrics.items():
            wandb.summary[f'comparison/{name}/{metric_name}'] = value

    # Log comparison table to wandb — include every metric we saw.
    all_keys = core_present + axis_keys + peg_present + rest_keys
    table_cols = ['policy'] + [k.replace('mean_', '') for k in all_keys]
    table = wandb.Table(columns=table_cols)
    for name, metrics in results.items():
        row = [name] + [metrics.get(k, None) for k in all_keys]
        table.add_data(*row)
    wandb.log({'eval/comparison_table': table})

    # Save results
    results_path = os.path.join(output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    wandb.finish()


if __name__ == '__main__':
    main()
