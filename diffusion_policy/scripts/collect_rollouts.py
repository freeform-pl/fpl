"""
Collect rollouts from a trained policy and save trajectories + metrics.

Usage:
  python scripts/collect_rollouts.py --checkpoint <path> --n_rollouts 200 --output_path rollouts.npz
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
import math
import click
import hydra
import torch
import dill
import tqdm
import numpy as np
import wandb

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply


def compute_smoothness(actions):
    if len(actions) < 3:
        return 1.0, 0.0
    actions = np.array(actions)
    vel = np.diff(actions, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    jerk_magnitude = np.linalg.norm(jerk, axis=-1)
    jerk_mean = float(np.mean(jerk_magnitude))
    smoothness = float(np.exp(-10.0 * jerk_mean))
    return smoothness, jerk_mean


def compute_speed_reward(success, steps_taken, max_steps):
    if not success:
        return 0.0
    return 1.0 - 0.9 * (steps_taken / max_steps)


def classify_peg_from_obs(obs):
    """
    Determine peg reward from final obs.
    obs layout: object(14), robot0_eef_pos(3), robot0_eef_quat(4), robot0_gripper_qpos(2)
    object[:3] = nut_pos
    Peg1 (left): [0.23, 0.1, 0.85], Peg2 (right): [0.23, -0.1, 0.85]
    Returns: -1.0 (left), +1.0 (right), 0.0 (neither)
    """
    nut_pos = obs[:3]
    peg1_pos = np.array([0.23, 0.1, 0.85])
    peg2_pos = np.array([0.23, -0.1, 0.85])
    table_z = 0.8
    if (abs(nut_pos[0] - peg1_pos[0]) < 0.03 and
        abs(nut_pos[1] - peg1_pos[1]) < 0.03 and
        nut_pos[2] < table_z + 0.05):
        return -1.0
    if (abs(nut_pos[0] - peg2_pos[0]) < 0.03 and
        abs(nut_pos[1] - peg2_pos[1]) < 0.03 and
        nut_pos[2] < table_z + 0.05):
        return 1.0
    return 0.0


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to policy checkpoint')
@click.option('--n_rollouts', '-n', default=200, type=int, help='Number of rollouts to collect')
@click.option('--output_path', '-o', required=True, help='Output .npz file path')
@click.option('--max_steps', default=None, type=int, help='Override max steps per episode')
@click.option('--device', '-d', default='cuda:0')
@click.option('--wandb_project', default='reward_cond_pipeline', help='wandb project name')
def main(checkpoint, n_rollouts, output_path, max_steps, device, wandb_project):
    # Init wandb
    wandb.init(
        project=wandb_project,
        name='phase1_collect_rollouts',
        config={
            'checkpoint': checkpoint,
            'n_rollouts': n_rollouts,
            'max_steps': max_steps,
        },
    )

    # Load checkpoint
    print("start")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    print("loaded checkpoint")
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)

    output_dir = os.path.dirname(output_path) or '.'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    print("loaded workspace")
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # Setup env runner for rollout collection
    runner_cfg = cfg.task.env_runner
    runner_cfg.n_train = 0
    runner_cfg.n_train_vis = 0
    runner_cfg.n_test = n_rollouts
    runner_cfg.n_test_vis = 0
    if max_steps is not None:
        runner_cfg.max_steps = max_steps

    env_runner = hydra.utils.instantiate(runner_cfg, output_dir=output_dir)
    print("env runner instantiated")
    actual_max_steps = env_runner.max_steps
    n_obs_steps = env_runner.n_obs_steps
    n_action_steps = env_runner.n_action_steps
    n_latency_steps = env_runner.n_latency_steps

    env = env_runner.env
    n_envs = len(env_runner.env_fns)
    n_inits = len(env_runner.env_init_fn_dills)
    n_chunks = math.ceil(n_inits / n_envs)

    # Collect all episodes
    all_obs_episodes = []
    all_action_episodes = []
    all_episode_lengths = []
    all_success = []
    all_speed_reward = []
    all_smoothness = []
    all_peg_reward = []

    print(f"\nCollecting {n_rollouts} rollouts ({n_chunks} chunks, {n_envs} envs, max_steps={actual_max_steps})")
    episode_pbar = tqdm.tqdm(total=n_rollouts, desc="Episodes collected", position=0)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * n_envs
        end = min(n_inits, start + n_envs)
        this_n_active = end - start

        this_init_fns = env_runner.env_init_fn_dills[start:end]
        n_diff = n_envs - len(this_init_fns)
        if n_diff > 0:
            this_init_fns.extend([env_runner.env_init_fn_dills[0]] * n_diff)

        env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

        obs = env.reset()
        past_action = None
        policy.reset()

        # Per-env tracking
        obs_history = [[] for _ in range(n_envs)]
        actions_history = [[] for _ in range(n_envs)]

        # Record initial obs for each env
        for i in range(this_n_active):
            obs_history[i].append(obs[i, 0].copy())  # first obs step

        pbar = tqdm.tqdm(total=actual_max_steps,
                         desc=f"Chunk {chunk_idx+1}/{n_chunks} steps",
                         leave=False, mininterval=1.0, position=1)
        done = False
        while not done:
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
                raise RuntimeError("Nan or Inf action")

            for i in range(this_n_active):
                for t in range(action.shape[1]):
                    actions_history[i].append(action[i, t].copy())

            env_action = action
            if env_runner.abs_action:
                env_action = env_runner.undo_transform_action(action)

            obs, reward, done_step, info = env.step(env_action)

            # Record obs at each step
            for i in range(this_n_active):
                obs_history[i].append(obs[i, 0].copy())

            done = np.all(done_step)
            past_action = action
            pbar.update(action.shape[1])
        pbar.close()

        # Get rewards for success detection
        all_rewards = env.call('get_attr', 'reward')

        for i in range(this_n_active):
            rewards = np.array(all_rewards[i])
            actions = np.array(actions_history[i])
            obs_seq = np.array(obs_history[i])

            success = bool(np.max(rewards) >= 1.0)
            success_steps = np.where(rewards >= 1.0)[0]
            first_success_step = int(success_steps[0]) if len(success_steps) > 0 else len(rewards)
            speed_reward = compute_speed_reward(success, first_success_step, actual_max_steps)
            smoothness, _ = compute_smoothness(actions)

            # Peg reward from final obs
            final_obs = obs_seq[-1]
            peg_reward = classify_peg_from_obs(final_obs)

            all_obs_episodes.append(obs_seq)
            all_action_episodes.append(actions)
            all_episode_lengths.append(len(obs_seq))
            all_success.append(float(success))
            all_speed_reward.append(speed_reward)
            all_smoothness.append(smoothness)
            all_peg_reward.append(peg_reward)

        episode_pbar.update(this_n_active)
        n_collected = len(all_obs_episodes)
        cur_success = np.mean(all_success)
        cur_speed = np.mean(all_speed_reward)
        cur_smooth = np.mean(all_smoothness)
        episode_pbar.set_postfix(success=f"{cur_success:.3f}", total=n_collected)

        # Log running metrics to wandb
        wandb.log({
            'collect/episodes_collected': n_collected,
            'collect/running_success': cur_success,
            'collect/running_speed_reward': cur_speed,
            'collect/running_smoothness': cur_smooth,
        }, step=n_collected)

    episode_pbar.close()

    # Pad to same length and save
    max_obs_len = max(len(ep) for ep in all_obs_episodes)
    max_act_len = max(len(ep) for ep in all_action_episodes)
    obs_dim = all_obs_episodes[0].shape[-1]
    action_dim = all_action_episodes[0].shape[-1]
    n_episodes = len(all_obs_episodes)

    obs_padded = np.zeros((n_episodes, max_obs_len, obs_dim), dtype=np.float32)
    actions_padded = np.zeros((n_episodes, max_act_len, action_dim), dtype=np.float32)

    for i in range(n_episodes):
        obs_padded[i, :len(all_obs_episodes[i])] = all_obs_episodes[i]
        actions_padded[i, :len(all_action_episodes[i])] = all_action_episodes[i]

    np.savez(output_path,
             obs=obs_padded,
             actions=actions_padded,
             episode_lengths=np.array(all_episode_lengths, dtype=np.int32),
             success=np.array(all_success, dtype=np.float32),
             speed_reward=np.array(all_speed_reward, dtype=np.float32),
             smoothness=np.array(all_smoothness, dtype=np.float32),
             peg_reward=np.array(all_peg_reward, dtype=np.float32))

    peg_rewards = np.array(all_peg_reward)
    print(f"\nSaved {n_episodes} rollouts to {output_path}")
    print(f"  Success rate: {np.mean(all_success):.3f}")
    print(f"  Mean speed:   {np.mean(all_speed_reward):.3f}")
    print(f"  Mean smooth:  {np.mean(all_smoothness):.3f}")
    print(f"  Peg: left={np.sum(peg_rewards > 0)}, right={np.sum(peg_rewards < 0)}, none={np.sum(peg_rewards == 0)}")

    # Log final summary to wandb
    wandb.summary['collect/final_success'] = float(np.mean(all_success))
    wandb.summary['collect/final_speed_reward'] = float(np.mean(all_speed_reward))
    wandb.summary['collect/final_smoothness'] = float(np.mean(all_smoothness))
    wandb.summary['collect/left_peg_count'] = int(np.sum(peg_rewards > 0))
    wandb.summary['collect/right_peg_count'] = int(np.sum(peg_rewards < 0))
    wandb.summary['collect/n_episodes'] = n_episodes
    wandb.finish()


if __name__ == '__main__':
    main()
