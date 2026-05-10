"""
Filter rollout .npz and demo .hdf5 to only keep successful right-peg episodes.
Creates new files in the output directory that can be used directly by demo_success baseline.

Usage:
  python scripts/filter_right_peg_success.py \
    --rollout_npz pipeline_output_slow_fast_rhp/rollouts.npz \
    --demo_hdf5 pipeline_output_slow_fast_rhp/scripted_data/demos.hdf5 \
    --output_dir pipeline_output_slow_fast_success_right_peg
"""

import os
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import h5py
import click


@click.command()
@click.option('--rollout_npz', required=True, help='Path to source rollouts.npz')
@click.option('--demo_hdf5', required=True, help='Path to source demos.hdf5')
@click.option('--output_dir', required=True, help='Output pipeline directory')
def main(rollout_npz, demo_hdf5, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    scripted_dir = os.path.join(output_dir, 'scripted_data')
    os.makedirs(scripted_dir, exist_ok=True)

    # --- Filter rollouts ---
    print(f"Loading rollouts from {rollout_npz}")
    data = np.load(rollout_npz)
    success = data['success']
    peg_reward = data['peg_reward']  # +1 right, -1 left

    speed_reward = data['speed_reward']

    # Print per-peg speed stats
    left_mask = peg_reward < 0.0
    right_mask = peg_reward > 0.0
    success_left = (success >= 1.0) & left_mask
    success_right = (success >= 1.0) & right_mask
    print(f"\n--- Rollout speed stats ---")
    print(f"  Left peg  (n={left_mask.sum()}, success={success_left.sum()}): "
          f"mean_speed={speed_reward[left_mask].mean():.3f}, "
          f"success_only={speed_reward[success_left].mean():.3f}" if success_left.any() else "  Left peg: no successful episodes")
    print(f"  Right peg (n={right_mask.sum()}, success={success_right.sum()}): "
          f"mean_speed={speed_reward[right_mask].mean():.3f}, "
          f"success_only={speed_reward[success_right].mean():.3f}" if success_right.any() else "  Right peg: no successful episodes")

    # Also print episode lengths as a proxy for speed
    episode_lengths = data['episode_lengths']
    if success_left.any():
        print(f"  Left peg  success ep lengths: mean={episode_lengths[success_left].mean():.1f}, "
              f"min={episode_lengths[success_left].min()}, max={episode_lengths[success_left].max()}")
    if success_right.any():
        print(f"  Right peg success ep lengths: mean={episode_lengths[success_right].mean():.1f}, "
              f"min={episode_lengths[success_right].min()}, max={episode_lengths[success_right].max()}")
    print()

    mask = (success >= 1.0) & (peg_reward > 0.0)
    n_total = len(success)
    n_kept = mask.sum()
    print(f"Rollouts: keeping {n_kept}/{n_total} (successful + right peg)")

    # Filter all arrays
    filtered = {}
    for key in data.files:
        filtered[key] = data[key][mask]

    out_npz = os.path.join(output_dir, 'rollouts.npz')
    np.savez(out_npz, **filtered)
    print(f"Saved filtered rollouts to {out_npz}")

    # --- Filter demos ---
    print(f"\nLoading demos from {demo_hdf5}")
    out_hdf5 = os.path.join(scripted_dir, 'demos.hdf5')

    n_demos_total = 0
    n_demos_kept = 0
    demo_lengths_left = []
    demo_lengths_right = []

    with h5py.File(demo_hdf5, 'r') as src, h5py.File(out_hdf5, 'w') as dst:
        dst_data = dst.create_group('data')

        # Copy top-level attrs
        for attr_key in src['data'].attrs:
            dst_data.attrs[attr_key] = src['data'].attrs[attr_key]

        for key in sorted(src['data'].keys(), key=lambda x: int(x.split('_')[1])):
            demo = src['data'][key]
            n_demos_total += 1

            target_peg = demo.attrs.get('target_peg', 'left')
            ep_len = len(demo['actions'])
            if target_peg == 'left':
                demo_lengths_left.append(ep_len)
            else:
                demo_lengths_right.append(ep_len)
            if target_peg != 'right':
                continue

            new_key = f'demo_{n_demos_kept}'
            src.copy(f'data/{key}', dst_data, name=new_key)
            n_demos_kept += 1

        dst_data.attrs['num_demos'] = n_demos_kept

    print(f"\n--- Demo speed stats (episode lengths) ---")
    if demo_lengths_left:
        ll = np.array(demo_lengths_left)
        print(f"  Left peg  (n={len(ll)}): mean={ll.mean():.1f}, min={ll.min()}, max={ll.max()}")
    if demo_lengths_right:
        rl = np.array(demo_lengths_right)
        print(f"  Right peg (n={len(rl)}): mean={rl.mean():.1f}, min={rl.min()}, max={rl.max()}")
    print()

    print(f"Demos: keeping {n_demos_kept}/{n_demos_total} (right peg)")
    print(f"Saved filtered demos to {out_hdf5}")

    print(f"\nDone. Use this output_dir as PIPELINE_DIR for demo_success baseline.")


if __name__ == '__main__':
    main()
