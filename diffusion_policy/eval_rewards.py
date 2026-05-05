"""
Evaluation script that computes multiple reward metrics for diffusion policy rollouts.

Metrics computed per episode:
  - success:    whether the task was completed (binary)
  - speed:      normalized speed reward (1 = fastest possible, decays with steps used)
  - smoothness: smoothness of the executed action trajectory (penalizes jerk)

Usage:
  python eval_rewards.py --checkpoint <path_to_ckpt> -o <output_dir> [--n_test 50] [--device cuda:0]
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import math
import pathlib
import collections
import json
import click
import hydra
import torch
import dill
import tqdm
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply


def compute_smoothness(actions):
    """
    Compute smoothness of an action trajectory.

    Args:
        actions: np.array of shape (T, action_dim) — sequence of actions executed.

    Returns:
        smoothness: float in [0, 1]. 1 = perfectly smooth, 0 = maximally jerky.
        jerk_mean: mean absolute jerk (for logging).
    """
    if len(actions) < 3:
        return 1.0, 0.0

    actions = np.array(actions)
    # velocity (first difference)
    vel = np.diff(actions, axis=0)
    # acceleration (second difference)
    acc = np.diff(vel, axis=0)
    # jerk (third difference)
    jerk = np.diff(acc, axis=0)

    jerk_magnitude = np.linalg.norm(jerk, axis=-1)  # (T-3,)
    jerk_mean = float(np.mean(jerk_magnitude))

    # Normalize to [0,1] using exponential decay. Scale chosen so that
    # typical robomimic jerk values (~0.01-0.1) map to a useful range.
    smoothness = float(np.exp(-10.0 * jerk_mean))
    return smoothness, jerk_mean


def compute_speed_reward(success, steps_taken, max_steps):
    """
    Speed reward: only awarded if the task succeeded.
    Linearly interpolates between 1 (immediate success) and a small positive
    value (success at the last step).

    Args:
        success: bool
        steps_taken: int — step at which task was completed (or total steps if failed)
        max_steps: int — maximum allowed steps

    Returns:
        speed_reward: float in [0, 1]
    """
    if not success:
        return 0.0
    # Linear interpolation: completing at step 0 gives 1.0, at max_steps gives 0.1
    return 1.0 - 0.9 * (steps_taken / max_steps)


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n_test', default=None, type=int, help='Override number of test rollouts')
@click.option('--n_train', default=0, type=int, help='Number of train rollouts (default 0)')
def main(checkpoint, output_dir, device, n_test, n_train):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # Override runner config if requested
    runner_cfg = cfg.task.env_runner
    if n_test is not None:
        runner_cfg.n_test = n_test
    runner_cfg.n_train = n_train
    runner_cfg.n_train_vis = 0
    runner_cfg.n_test_vis = 0

    # Build env runner (we'll use its setup but run our own rollout loop)
    env_runner = hydra.utils.instantiate(
        runner_cfg,
        output_dir=output_dir)

    # Run rollouts with detailed metric collection
    results = run_eval_rollouts(policy, env_runner, device)

    # Save results
    out_path = os.path.join(output_dir, 'eval_rewards.json')
    json.dump(results, open(out_path, 'w'), indent=2, sort_keys=True)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes:         {results['n_episodes']}")
    print(f"  Success rate:     {results['mean_success']:.3f}")
    print(f"  Mean speed:       {results['mean_speed_reward']:.3f}")
    print(f"  Mean smoothness:  {results['mean_smoothness']:.3f}")
    print(f"  Mean steps:       {results['mean_steps']:.1f} / {results['max_steps']}")
    print(f"  Mean jerk:        {results['mean_jerk']:.6f}")
    print("=" * 60)
    print(f"Results saved to {out_path}")


def run_eval_rollouts(policy, env_runner, device):
    """
    Run evaluation rollouts and collect detailed per-step data
    for computing speed, success, and smoothness rewards.
    """
    env = env_runner.env
    n_envs = len(env_runner.env_fns)
    n_inits = len(env_runner.env_init_fn_dills)
    n_chunks = math.ceil(n_inits / n_envs)

    n_obs_steps = env_runner.n_obs_steps
    n_action_steps = env_runner.n_action_steps
    n_latency_steps = env_runner.n_latency_steps
    max_steps = env_runner.max_steps

    all_episode_metrics = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * n_envs
        end = min(n_inits, start + n_envs)
        this_n_active = end - start
        this_global_slice = slice(start, end)
        this_local_slice = slice(0, this_n_active)

        this_init_fns = env_runner.env_init_fn_dills[this_global_slice]
        n_diff = n_envs - len(this_init_fns)
        if n_diff > 0:
            this_init_fns.extend([env_runner.env_init_fn_dills[0]] * n_diff)

        # init envs
        env.call_each('run_dill_function',
                       args_list=[(x,) for x in this_init_fns])

        # reset
        obs = env.reset()
        past_action = None
        policy.reset()

        # Per-env tracking
        # actions_history[i] stores all single-step actions for env i
        actions_history = [[] for _ in range(n_envs)]
        # Track per-step rewards to detect first success step
        rewards_history = [[] for _ in range(n_envs)]
        env_done = [False] * n_envs

        pbar = tqdm.tqdm(total=max_steps,
                         desc=f"Eval chunk {chunk_idx+1}/{n_chunks}",
                         leave=False, mininterval=1.0)

        done = False
        while not done:
            # create obs dict
            np_obs_dict = {
                'obs': obs[:, :n_obs_steps].astype(np.float32)
            }
            if env_runner.past_action and (past_action is not None):
                np_obs_dict['past_action'] = past_action[
                    :, -(n_obs_steps - 1):].astype(np.float32)

            obs_dict = dict_apply(np_obs_dict,
                lambda x: torch.from_numpy(x).to(device=device))

            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action'][:, n_latency_steps:]
            if not np.all(np.isfinite(action)):
                print(action)
                raise RuntimeError("Nan or Inf action")

            # Record individual actions for smoothness computation
            # action shape: (n_envs, n_action_steps_actual, action_dim)
            for i in range(this_n_active):
                if not env_done[i]:
                    for t in range(action.shape[1]):
                        actions_history[i].append(action[i, t].copy())

            env_action = action
            if env_runner.abs_action:
                env_action = env_runner.undo_transform_action(action)

            obs, reward, done_step, info = env.step(env_action)

            # Collect per-step rewards from the multistep wrapper
            step_rewards = env.call('get_attr', 'reward')
            for i in range(this_n_active):
                if not env_done[i]:
                    # get_attr('reward') returns the full reward list
                    rewards_history[i] = list(step_rewards[i])
                    # Check if this env just finished
                    # We check done from the parallel env

            done = np.all(done_step)
            past_action = action
            pbar.update(action.shape[1])
        pbar.close()

        # Collect per-env rewards for success detection
        all_rewards = env.call('get_attr', 'reward')

        # Compute metrics for each active env in this chunk
        for i in range(this_n_active):
            global_idx = start + i
            seed = env_runner.env_seeds[global_idx]
            prefix = env_runner.env_prefixs[global_idx]

            rewards = np.array(all_rewards[i])
            actions = np.array(actions_history[i])

            # Success: max reward >= 1.0 (robomimic convention)
            success = bool(np.max(rewards) >= 1.0)

            # Steps to success: first step where reward == 1.0
            success_steps = np.where(rewards >= 1.0)[0]
            first_success_step = int(success_steps[0]) if len(success_steps) > 0 else len(rewards)

            # Speed reward
            speed_reward = compute_speed_reward(success, first_success_step, max_steps)

            # Smoothness
            smoothness, jerk_mean = compute_smoothness(actions)

            episode_metrics = {
                'seed': int(seed),
                'prefix': prefix,
                'success': success,
                'total_steps': len(rewards),
                'first_success_step': first_success_step,
                'speed_reward': speed_reward,
                'smoothness': smoothness,
                'jerk_mean': jerk_mean,
                'max_reward': float(np.max(rewards)),
                'mean_reward': float(np.mean(rewards)),
            }
            all_episode_metrics.append(episode_metrics)

    # Aggregate results
    successes = [m['success'] for m in all_episode_metrics]
    speed_rewards = [m['speed_reward'] for m in all_episode_metrics]
    smoothnesses = [m['smoothness'] for m in all_episode_metrics]
    jerks = [m['jerk_mean'] for m in all_episode_metrics]
    steps = [m['total_steps'] for m in all_episode_metrics]

    # Group by prefix (train vs test)
    grouped = collections.defaultdict(list)
    for m in all_episode_metrics:
        grouped[m['prefix']].append(m)

    summary_by_prefix = {}
    for prefix, metrics in grouped.items():
        summary_by_prefix[prefix] = {
            'n_episodes': len(metrics),
            'mean_success': float(np.mean([m['success'] for m in metrics])),
            'mean_speed_reward': float(np.mean([m['speed_reward'] for m in metrics])),
            'mean_smoothness': float(np.mean([m['smoothness'] for m in metrics])),
            'mean_jerk': float(np.mean([m['jerk_mean'] for m in metrics])),
        }

    results = {
        'n_episodes': len(all_episode_metrics),
        'max_steps': max_steps,
        'mean_success': float(np.mean(successes)),
        'mean_speed_reward': float(np.mean(speed_rewards)),
        'mean_smoothness': float(np.mean(smoothnesses)),
        'mean_jerk': float(np.mean(jerks)),
        'mean_steps': float(np.mean(steps)),
        'std_success': float(np.std(successes)),
        'std_speed_reward': float(np.std(speed_rewards)),
        'std_smoothness': float(np.std(smoothnesses)),
        'summary_by_prefix': summary_by_prefix,
        'episodes': all_episode_metrics,
    }

    return results


if __name__ == '__main__':
    main()
