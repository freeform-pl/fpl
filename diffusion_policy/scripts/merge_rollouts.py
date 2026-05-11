"""
Merge multiple rollout .npz files into a single accumulated file.

Usage:
  python scripts/merge_rollouts.py -o merged.npz file1.npz file2.npz ...
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import click
import numpy as np


@click.command()
@click.argument('input_files', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('-o', '--output', required=True, help='Output merged .npz path')
def main(input_files, output):
    all_obs = []
    all_actions = []
    all_lengths = []
    all_success = []
    all_speed = []
    all_smoothness = []
    all_peg = []
    all_conditioning = []

    for f in input_files:
        data = np.load(f)
        n = len(data['episode_lengths'])
        all_obs.append(data['obs'][:n])
        all_actions.append(data['actions'][:n])
        all_lengths.append(data['episode_lengths'])
        all_success.append(data['success'])
        all_speed.append(data['speed_reward'])
        all_smoothness.append(data['smoothness'])
        all_peg.append(data['peg_reward'])
        if 'conditioning' in data:
            all_conditioning.append(data['conditioning'])
        print(f"  {f}: {n} episodes")

    # Pad obs and actions to common max length
    max_obs_len = max(a.shape[1] for a in all_obs)
    max_act_len = max(a.shape[1] for a in all_actions)
    obs_dim = all_obs[0].shape[-1]
    act_dim = all_actions[0].shape[-1]

    padded_obs = []
    padded_actions = []
    for obs, acts in zip(all_obs, all_actions):
        n = obs.shape[0]
        if obs.shape[1] < max_obs_len:
            pad = np.zeros((n, max_obs_len - obs.shape[1], obs_dim), dtype=np.float32)
            obs = np.concatenate([obs, pad], axis=1)
        padded_obs.append(obs)
        if acts.shape[1] < max_act_len:
            pad = np.zeros((n, max_act_len - acts.shape[1], act_dim), dtype=np.float32)
            acts = np.concatenate([acts, pad], axis=1)
        padded_actions.append(acts)

    save_dict = dict(
        obs=np.concatenate(padded_obs, axis=0),
        actions=np.concatenate(padded_actions, axis=0),
        episode_lengths=np.concatenate(all_lengths),
        success=np.concatenate(all_success),
        speed_reward=np.concatenate(all_speed),
        smoothness=np.concatenate(all_smoothness),
        peg_reward=np.concatenate(all_peg),
    )
    if all_conditioning:
        save_dict['conditioning'] = np.concatenate(all_conditioning, axis=0)

    total = len(save_dict['episode_lengths'])
    np.savez(output, **save_dict)
    print(f"Merged {total} episodes from {len(input_files)} files -> {output}")


if __name__ == '__main__':
    main()
