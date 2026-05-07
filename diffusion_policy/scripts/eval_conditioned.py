"""
Compare original policy vs reward-conditioned policy at different conditioning values.

Runs 4 sets of rollouts:
  1. Original policy (no conditioning)
  2. Conditioned policy @ z=1.5 (high reward)
  3. Conditioned policy @ z=0.0 (average)
  4. Conditioned policy @ z=-1.5 (low reward)

Usage:
  python scripts/eval_conditioned.py \
    --original_ckpt <path> \
    --conditioned_ckpt <path> \
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


def run_original_policy(policy, cfg, n_rollouts, output_dir, device):
    """Run the original (unconditioned) policy."""
    runner_cfg = cfg.task.env_runner
    runner_cfg.n_train = 0
    runner_cfg.n_train_vis = 0
    runner_cfg.n_test = n_rollouts
    runner_cfg.n_test_vis = 0

    runner = hydra.utils.instantiate(runner_cfg, output_dir=output_dir)
    log_data = runner.run(policy)
    return log_data


def run_conditioned_policy(policy, cfg, n_rollouts, target_z, num_reward_dims, output_dir, device):
    """Run conditioned policy with given z-score reward targets."""
    runner_cfg = cfg.task.env_runner
    # We need to create a RewardConditionedLowdimRunner manually
    # Extract runner kwargs from cfg
    runner_kwargs = OmegaConf.to_container(runner_cfg, resolve=True)
    runner_kwargs.pop('_target_', None)
    runner_kwargs['output_dir'] = output_dir
    runner_kwargs['n_train'] = 0
    runner_kwargs['n_train_vis'] = 0
    runner_kwargs['n_test'] = n_rollouts
    runner_kwargs['n_test_vis'] = 0

    target_rewards = [target_z] * num_reward_dims
    runner = RewardConditionedLowdimRunner(
        target_rewards=target_rewards,
        num_reward_dims=num_reward_dims,
        **runner_kwargs)
    log_data = runner.run(policy)
    return log_data


@click.command()
@click.option('--original_ckpt', required=True, help='Original policy checkpoint')
@click.option('--conditioned_ckpt', required=True, help='Reward-conditioned policy checkpoint')
@click.option('--scores_path', required=False, default=None, help='Path to scores.json from reward model training')
@click.option('--n_rollouts', default=50, type=int)
@click.option('--num_reward_dims', default=3, type=int)
@click.option('--output_dir', default='eval_conditioned_output')
@click.option('--device', default='cuda:0')
@click.option('--wandb_project', default='reward_cond_pipeline', help='wandb project name')
def main(original_ckpt, conditioned_ckpt, scores_path, n_rollouts, num_reward_dims, output_dir, device, wandb_project):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    # Init wandb
    wandb.init(
        project=wandb_project,
        name='phase4_eval_comparison',
        config={
            'original_ckpt': original_ckpt,
            'conditioned_ckpt': conditioned_ckpt,
            'n_rollouts': n_rollouts,
            'num_reward_dims': num_reward_dims,
        },
    )

    # Load scores for reference
    if scores_path and os.path.exists(scores_path):
        with open(scores_path, 'r') as f:
            scores_data = json.load(f)
        print(f"Reward score stats: mean={scores_data['score_mean']}, std={scores_data['score_std']}")
    else:
        print("No scores file provided or found, skipping reward score stats.")

    results = {}

    # 1. Original policy
    print("\n" + "=" * 60)
    print("Running ORIGINAL policy...")
    print("=" * 60)
    orig_policy, orig_cfg = load_policy(original_ckpt, device)
    orig_log = run_original_policy(orig_policy, orig_cfg, n_rollouts, output_dir, device)
    results['original'] = extract_metrics(orig_log)
    wandb.log({
        'eval/original_success': results['original']['success'],
        'eval/original_speed': results['original']['speed'],
        'eval/original_smoothness': results['original']['smoothness'],
        'eval/original_score': results['original']['score'],
        'eval/original_throughput': results['original']['throughput'],
    })
    del orig_policy
    torch.cuda.empty_cache()

    # 2-4. Conditioned policy at different z-scores
    # Skip if conditioned checkpoint is the same as original (e.g. demo_only baseline)
    if os.path.abspath(conditioned_ckpt) == os.path.abspath(original_ckpt):
        print("\nConditioned checkpoint is the same as original — skipping conditioned eval.")
    else:
        cond_policy, cond_cfg = load_policy(conditioned_ckpt, device)
        for z_val in [1.5, 0.0, -1.5]:
            label = f"conditioned_z{z_val:+.1f}"
            print(f"\n{'=' * 60}")
            print(f"Running CONDITIONED policy @ z={z_val}...")
            print("=" * 60)
            cond_log = run_conditioned_policy(
                cond_policy, cond_cfg, n_rollouts, z_val, num_reward_dims, output_dir, device)
            results[label] = extract_metrics(cond_log)
            z_label = f"z{z_val:+.1f}".replace('.', '_').replace('+', 'p').replace('-', 'n')
            wandb.log({
                f'eval/{z_label}_success': results[label]['success'],
                f'eval/{z_label}_speed': results[label]['speed'],
                f'eval/{z_label}_smoothness': results[label]['smoothness'],
                f'eval/{z_label}_score': results[label]['score'],
                f'eval/{z_label}_throughput': results[label]['throughput'],
            })

        del cond_policy
        torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Policy':<30s} {'Success':>10s} {'Speed':>10s} {'Smoothness':>12s} {'Score':>10s} {'Throughput':>12s}")
    print("-" * 92)
    for name, metrics in results.items():
        print(f"{name:<30s} {metrics['success']:>10.3f} {metrics['speed']:>10.3f} "
              f"{metrics['smoothness']:>12.3f} {metrics['score']:>10.3f} {metrics['throughput']:>12.4f}")
    print("=" * 92)

    # Log all results as wandb summary
    for name, metrics in results.items():
        for metric_name, value in metrics.items():
            wandb.summary[f'comparison/{name}/{metric_name}'] = value

    # Log comparison table to wandb
    table = wandb.Table(columns=['policy', 'success', 'speed', 'smoothness', 'score', 'throughput'])
    for name, metrics in results.items():
        table.add_data(name, metrics['success'], metrics['speed'],
                       metrics['smoothness'], metrics['score'], metrics['throughput'])
    wandb.log({'eval/comparison_table': table})

    # Save results
    results_path = os.path.join(output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    wandb.finish()


def extract_metrics(log_data):
    """Extract test metrics from runner log data."""
    return {
        'success': log_data.get('test/mean_success', 0.0),
        'speed': log_data.get('test/mean_speed_reward', 0.0),
        'smoothness': log_data.get('test/mean_smoothness', 0.0),
        'score': log_data.get('test/mean_score', 0.0),
        'throughput': log_data.get('test/mean_throughput', 0.0),
    }


if __name__ == '__main__':
    main()
