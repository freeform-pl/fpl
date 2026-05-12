"""
Collect rollouts from a trained policy and save trajectories + metrics.

Usage:
  # Unconditioned (base policy):
  python scripts/collect_rollouts.py --checkpoint <path> --n_rollouts 200 --output_path rollouts.npz

  # Conditioned with multiple targets (one output file per target):
  python scripts/collect_rollouts.py --checkpoint <path> --n_rollouts 50 \
      --output_dir rollouts/ --conditioned --num_reward_dims 1 --discrete_conditioning \
      --conditioning_targets "0.9;0.0;-0.9"
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


def augment_obs_with_conditioning(obs, target_rewards, discrete_conditioning=False, n_cond_bins=21):
    """Append conditioning vector to obs. obs: (B, T, D) -> (B, T, D+C)"""
    B, T, D = obs.shape
    if discrete_conditioning:
        from diffusion_policy.dataset.reward_conditioned_lowdim_dataset import scores_to_onehot
        cond_vec = scores_to_onehot(target_rewards, n_cond_bins)
    else:
        cond_vec = np.array(target_rewards, dtype=np.float32)
    cond_dim = len(cond_vec)
    reward_aug = np.broadcast_to(cond_vec, (B, T, cond_dim)).copy()
    return np.concatenate([obs, reward_aug], axis=-1)


def collect_rollouts_with_conditioning(
    policy, env_runner, target_rewards, n_rollouts,
    conditioned, discrete_conditioning, n_cond_bins, device, actual_max_steps
):
    """Collect n_rollouts episodes with a fixed conditioning value. Returns lists of metrics."""
    n_obs_steps = env_runner.n_obs_steps
    n_action_steps = env_runner.n_action_steps
    n_latency_steps = env_runner.n_latency_steps
    env = env_runner.env
    n_envs = len(env_runner.env_fns)
    n_inits = len(env_runner.env_init_fn_dills)
    n_chunks = math.ceil(n_rollouts / n_envs)

    all_obs_episodes = []
    all_action_episodes = []
    all_episode_lengths = []
    all_success = []
    all_speed_reward = []
    all_smoothness = []
    all_peg_reward = []

    episode_pbar = tqdm.tqdm(total=n_rollouts, desc=f"  cond={target_rewards}", position=0, leave=True)

    for chunk_idx in range(n_chunks):
        start = chunk_idx * n_envs
        end = min(n_rollouts, start + n_envs)
        this_n_active = end - start

        # Cycle through init fns
        this_init_fns = []
        for j in range(start, end):
            this_init_fns.append(env_runner.env_init_fn_dills[j % n_inits])
        n_diff = n_envs - len(this_init_fns)
        if n_diff > 0:
            this_init_fns.extend([env_runner.env_init_fn_dills[0]] * n_diff)

        env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

        obs = env.reset()
        past_action = None
        policy.reset()

        obs_history = [[] for _ in range(n_envs)]
        actions_history = [[] for _ in range(n_envs)]

        for i in range(this_n_active):
            obs_history[i].append(obs[i, 0].copy())

        done = False
        while not done:
            obs_for_policy = obs[:, :n_obs_steps].astype(np.float32)
            if conditioned:
                obs_for_policy = augment_obs_with_conditioning(
                    obs_for_policy, target_rewards,
                    discrete_conditioning=discrete_conditioning, n_cond_bins=n_cond_bins)
            np_obs_dict = {'obs': obs_for_policy}
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

            for i in range(this_n_active):
                obs_history[i].append(obs[i, 0].copy())

            done = np.all(done_step)
            past_action = action

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

    episode_pbar.close()

    return {
        'obs_episodes': all_obs_episodes,
        'action_episodes': all_action_episodes,
        'episode_lengths': all_episode_lengths,
        'success': all_success,
        'speed_reward': all_speed_reward,
        'smoothness': all_smoothness,
        'peg_reward': all_peg_reward,
    }


def save_rollouts_npz(output_path, results, conditioning=None):
    """Pad and save rollout episodes to npz."""
    all_obs_episodes = results['obs_episodes']
    all_action_episodes = results['action_episodes']
    n_episodes = len(all_obs_episodes)

    max_obs_len = max(len(ep) for ep in all_obs_episodes)
    max_act_len = max(len(ep) for ep in all_action_episodes)
    obs_dim = all_obs_episodes[0].shape[-1]
    action_dim = all_action_episodes[0].shape[-1]

    obs_padded = np.zeros((n_episodes, max_obs_len, obs_dim), dtype=np.float32)
    actions_padded = np.zeros((n_episodes, max_act_len, action_dim), dtype=np.float32)

    for i in range(n_episodes):
        obs_padded[i, :len(all_obs_episodes[i])] = all_obs_episodes[i]
        actions_padded[i, :len(all_action_episodes[i])] = all_action_episodes[i]

    save_dict = dict(
        obs=obs_padded,
        actions=actions_padded,
        episode_lengths=np.array(results['episode_lengths'], dtype=np.int32),
        success=np.array(results['success'], dtype=np.float32),
        speed_reward=np.array(results['speed_reward'], dtype=np.float32),
        smoothness=np.array(results['smoothness'], dtype=np.float32),
        peg_reward=np.array(results['peg_reward'], dtype=np.float32),
    )
    if conditioning is not None:
        save_dict['conditioning'] = np.tile(
            np.array(conditioning, dtype=np.float32), (n_episodes, 1))
    np.savez(output_path, **save_dict)


def print_stats(label, results):
    """Print summary stats for a batch of rollouts."""
    n = len(results['success'])
    success = np.array(results['success'])
    speed = np.array(results['speed_reward'])
    smooth = np.array(results['smoothness'])
    peg = np.array(results['peg_reward'])
    print(f"  [{label}] n={n}  success={np.mean(success):.3f}  "
          f"speed={np.mean(speed):.3f}  smooth={np.mean(smooth):.3f}  "
          f"peg: left={np.sum(peg < 0)}, right={np.sum(peg > 0)}, none={np.sum(peg == 0)}")


def parse_conditioning_targets(targets_str, num_reward_dims):
    """Parse semicolon-separated conditioning targets.
    E.g. "0.9;0.0;-0.9" for 1D, "0.9,0.5;-0.9,-0.5" for 2D.
    Returns list of np arrays.
    """
    targets = []
    for part in targets_str.split(';'):
        vals = np.array([float(x) for x in part.strip().split(',')], dtype=np.float64)
        assert len(vals) == num_reward_dims, \
            f"Conditioning target {part} has {len(vals)} dims but num_reward_dims={num_reward_dims}"
        targets.append(vals)
    return targets


@click.command()
@click.option('--checkpoint', '-c', required=True, help='Path to policy checkpoint')
@click.option('--n_rollouts', '-n', default=200, type=int, help='Total number of rollouts (split evenly across conditioning targets if conditioned)')
@click.option('--output_path', '-o', default=None, help='Output .npz file path (for unconditioned collection)')
@click.option('--output_dir', default=None, help='Output directory for conditioned collection (one file per target)')
@click.option('--max_steps', default=None, type=int, help='Override max steps per episode')
@click.option('--device', '-d', default='cuda:0')
@click.option('--wandb_project', default='reward_cond_pipeline', help='wandb project name')
@click.option('--conditioned', is_flag=True, default=False, help='Use conditioned policy (augment obs)')
@click.option('--num_reward_dims', default=3, type=int, help='Number of reward dimensions for conditioning')
@click.option('--discrete_conditioning', is_flag=True, default=False, help='Use one-hot discrete conditioning')
@click.option('--n_cond_bins', default=21, type=int, help='Number of bins for discrete conditioning')
@click.option('--conditioning_targets', default=None, type=str,
              help='Semicolon-separated conditioning targets, e.g. "0.9;0.0;-0.9" for 1D or "0.9,0.5;-0.9,-0.5" for 2D')
def main(checkpoint, n_rollouts, output_path, output_dir, max_steps, device, wandb_project,
         conditioned, num_reward_dims, discrete_conditioning, n_cond_bins, conditioning_targets):

    if conditioned:
        assert conditioning_targets is not None, \
            "Must provide --conditioning_targets when using --conditioned"
        assert output_dir is not None, \
            "Must provide --output_dir when using --conditioned"
        targets = parse_conditioning_targets(conditioning_targets, num_reward_dims)
    else:
        assert output_path is not None, \
            "Must provide --output_path for unconditioned collection"
        targets = [None]

    # Init wandb
    wandb.init(
        project=wandb_project,
        name='collect_rollouts',
        config={
            'checkpoint': checkpoint,
            'n_rollouts': n_rollouts,
            'max_steps': max_steps,
            'conditioned': conditioned,
            'num_reward_dims': num_reward_dims,
            'discrete_conditioning': discrete_conditioning,
            'conditioning_targets': conditioning_targets,
        },
    )

    # Load checkpoint
    print("start")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    print("loaded checkpoint")
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)

    base_dir = output_dir if output_dir else (os.path.dirname(output_path) or '.')
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    print("loaded workspace")
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # Setup env runner — use enough test seeds for the largest batch
    runner_cfg = cfg.task.env_runner
    runner_cfg.n_train = 0
    runner_cfg.n_train_vis = 0
    runner_cfg.n_test = n_rollouts
    runner_cfg.n_test_vis = 0
    if max_steps is not None:
        runner_cfg.max_steps = max_steps

    env_runner = hydra.utils.instantiate(runner_cfg, output_dir=base_dir)
    print("env runner instantiated")
    actual_max_steps = env_runner.max_steps

    if conditioned:
        n_targets = len(targets)
        n_per_target = n_rollouts // n_targets
        remainder = n_rollouts % n_targets
        print(f"\nConditioned collection: num_reward_dims={num_reward_dims}, "
              f"discrete={discrete_conditioning}")
        print(f"Total rollouts: {n_rollouts}, targets: {n_targets}, per target: {n_per_target}"
              + (f" (+1 for first {remainder})" if remainder > 0 else ""))
        print(f"Targets: {[t.tolist() for t in targets]}")
    else:
        n_per_target = n_rollouts
        remainder = 0
        print(f"\nUnconditioned collection: n_rollouts={n_rollouts}")

    all_output_files = []

    for t_idx, target in enumerate(targets):
        # Distribute remainder to first targets
        this_n = n_per_target + (1 if t_idx < remainder else 0)
        if conditioned:
            target_label = ','.join(f'{v:.2f}' for v in target)
            print(f"\n--- Target {t_idx+1}/{len(targets)}: [{target_label}] ({this_n} rollouts) ---")
        else:
            target_label = "unconditioned"

        results = collect_rollouts_with_conditioning(
            policy=policy,
            env_runner=env_runner,
            target_rewards=target if target is not None else np.zeros(num_reward_dims),
            n_rollouts=this_n,
            conditioned=conditioned,
            discrete_conditioning=discrete_conditioning,
            n_cond_bins=n_cond_bins,
            device=device,
            actual_max_steps=actual_max_steps,
        )

        print_stats(target_label, results)

        # Save
        if conditioned:
            fname = f"rollouts_cond_{'_'.join(f'{v:.2f}' for v in target)}.npz"
            fpath = os.path.join(output_dir, fname)
        else:
            fpath = output_path

        save_rollouts_npz(fpath, results, conditioning=target)
        all_output_files.append(fpath)
        print(f"  Saved {len(results['success'])} rollouts -> {fpath}")

        # Log to wandb
        success_rate = float(np.mean(results['success']))
        mean_speed = float(np.mean(results['speed_reward']))
        mean_smooth = float(np.mean(results['smoothness']))
        wandb.log({
            f'collect/{target_label}/success': success_rate,
            f'collect/{target_label}/speed_reward': mean_speed,
            f'collect/{target_label}/smoothness': mean_smooth,
            f'collect/{target_label}/n_episodes': len(results['success']),
        })

    # Print overall summary
    print(f"\n{'='*60}")
    print(f"Collection complete. {len(all_output_files)} file(s) saved:")
    for f in all_output_files:
        print(f"  {f}")

    wandb.summary['collect/n_targets'] = len(targets)
    wandb.summary['collect/n_rollouts_per_target'] = n_rollouts
    wandb.summary['collect/output_files'] = all_output_files
    wandb.finish()


if __name__ == '__main__':
    main()
