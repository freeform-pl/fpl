"""
Visualize a reward-conditioned policy by running rollouts and saving videos.

Usage:
  python scripts/visualize_conditioned_policy.py \
    --checkpoint <path_to_ckpt> \
    --output_dir vis_output \
    --z_values 0.0,1.5,-1.5
"""

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import click
import hydra
import torch
import dill
import numpy as np
from omegaconf import OmegaConf

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper


OBS_KEYS = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']


def create_env():
    dataset_path = "data/robomimic/datasets/square/mh/low_dim.hdf5"
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta["env_kwargs"]["controller_configs"]['control_delta'] = False

    ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': OBS_KEYS})
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )

    fps = 10
    env = MultiStepWrapper(
        VideoRecordingWrapper(
            RobomimicLowdimWrapper(
                env=robomimic_env,
                obs_keys=OBS_KEYS,
                init_state=None,
                render_hw=(256, 256),
                render_camera_name='agentview',
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=fps,
                codec='h264',
                input_pix_fmt='rgb24',
                crf=22,
                thread_type='FRAME',
                thread_count=1,
            ),
            file_path=None,
            steps_per_render=2,
        ),
        n_obs_steps=3,
        n_action_steps=1,
        max_episode_steps=400,
    )
    return env


def augment_obs(obs, target_rewards):
    """Append reward z-scores to obs. obs: (n_obs_steps, D) -> (n_obs_steps, D+K)"""
    T, D = obs.shape
    reward_aug = np.broadcast_to(target_rewards, (T, len(target_rewards))).copy()
    return np.concatenate([obs, reward_aug], axis=-1)


@click.command()
@click.option('--checkpoint', required=True, help='Path to conditioned policy checkpoint')
@click.option('--output_dir', default='vis_output', help='Where to save videos')
@click.option('--z_values', default='0.0,1.5,-1.5', help='Comma-separated z-values to test (broadcast to all dims)')
@click.option('--z_peg_values', default=None, help='If set, test specific peg z-values (overrides dim 3). Comma-separated.')
@click.option('--n_episodes', default=3, type=int, help='Episodes per z-value')
@click.option('--device', default='cuda:0')
def main(checkpoint, output_dir, z_values, z_peg_values, n_episodes, device):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load policy
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill, map_location=device)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    num_reward_dims = cfg.get('num_reward_dims', 3)
    n_obs_steps = cfg.n_obs_steps

    print(f"Loaded policy: obs_dim={cfg.obs_dim}, action_dim={cfg.action_dim}, num_reward_dims={num_reward_dims}")

    env = create_env()

    # Parse z-values
    z_vals = [float(x) for x in z_values.split(',')]

    # Build list of (label, target_rewards) to test
    test_configs = []
    for z in z_vals:
        rewards = np.array([z] * num_reward_dims, dtype=np.float32)
        test_configs.append((f"z{z:+.1f}_all", rewards))

    # Optionally test specific peg z-values
    if z_peg_values:
        for pz in [float(x) for x in z_peg_values.split(',')]:
            rewards = np.array([0.0] * num_reward_dims, dtype=np.float32)
            rewards[3] = pz  # peg dim is index 3
            test_configs.append((f"z_peg{pz:+.1f}", rewards))

    for label, target_rewards in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {label}, target_rewards={target_rewards}")
        print(f"{'='*60}")

        for ep in range(n_episodes):
            # Set up video
            video_path = pathlib.Path(output_dir) / f"{label}_ep{ep}.mp4"
            env.env.file_path = str(video_path)

            env.seed(ep * 1000 + 42)
            obs = env.reset()
            policy.reset()

            total_reward = 0
            for step in range(400):
                # obs shape from MultiStepWrapper: (n_obs_steps_padded, obs_dim)
                obs_for_policy = augment_obs(obs[:n_obs_steps], target_rewards)

                np_obs_dict = {
                    'obs': obs_for_policy[np.newaxis].astype(np.float32)  # (1, T, D+K)
                }
                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                action = action_dict['action'].detach().cpu().numpy()  # (1, n_action_steps, action_dim)
                action = action[:, 0:]  # all action steps

                if not np.all(np.isfinite(action)):
                    print(f"  WARNING: NaN/Inf action at step {step}")
                    break

                obs, reward, done, info = env.step(action)
                total_reward += reward

                if done:
                    break

            # Get final nut position
            nut_pos = obs[-1, :3]
            print(f"  Episode {ep}: reward={total_reward:.0f}, steps={step+1}, "
                  f"nut_final_pos=[{nut_pos[0]:.3f}, {nut_pos[1]:.3f}, {nut_pos[2]:.3f}], "
                  f"video={video_path}")

            env.env.video_recoder.stop()

    print(f"\nDone! Videos saved to {output_dir}/")


if __name__ == '__main__':
    main()
