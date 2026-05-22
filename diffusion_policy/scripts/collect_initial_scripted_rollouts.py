"""
Collect scripted rollouts for the square nut assembly task (state-space / lowdim).

Collects data with:
- Random peg target (left or right) per episode
- Varying noise levels for trajectory smoothness

Usage:
python scripts/collect_initial_scripted_rollouts.py -o data/multimodal_square -n 200
"""

import sys
import os
import pathlib

# Add repo root to path so diffusion_policy is importable
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import click
import torch
import json
import numpy as np
import copy
import random
import h5py

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.common.replay_buffer import ReplayBuffer

from scipy.spatial.transform import Rotation as R
import gym


OBS_KEYS = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']


class MultimodalSquareLowdimWrapper(gym.Env):
    """Wrapper that randomizes nut position on reset and tracks reward progress."""

    def __init__(self, env):
        self.env = env
        self.seed_state_map = dict()
        self._seed = None
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def get_observation(self):
        return self.env.get_observation()

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        nut_pos = np.random.uniform([-0.2, -0.2], [0, 0.2], size=(2))

        reset_state = np.array([
            0., -0.02921895, 0.17810908, 0.02728627, -2.63967499,
            -0.01431297, 2.9502351, 0.77126893, 0.020833, -0.020833,
            -0.11083517, 0.11445349, 0.89, -0.98421243, 0.,
            0., 0.17699121, 10., 10., 10.,
            1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ])

        self.rew = 0
        reset_state[10:12] = nut_pos
        self.last_reset_state = reset_state.copy()
        # env chain: self.env = RobomimicLowdimWrapper, .env = EnvRobosuite
        self.env.env.reset_to({"states": reset_state})

        # return obs
        obs = self.get_observation()
        return obs

    def _get_robosuite_env(self):
        """Get the underlying robosuite env (has .sim, .obj_body_id, etc.)"""
        return self.env.env.env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        robosuite_env = self._get_robosuite_env()
        nut_pos = robosuite_env.sim.data.body_xpos[robosuite_env.obj_body_id['SquareNut']]

        # Success = nut placed on either peg (single insertion)
        on_peg = robosuite_env.on_peg(nut_pos, 0) or robosuite_env.on_peg(nut_pos, 1)
        if on_peg:
            self.rew = 1

        reward = 1 if self.rew == 1 else 0

        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)


class ScriptedPolicy:

    def __init__(self, env=None):
        self.env = env
        self.rotation_transformer = RotationTransformer('euler_angles', 'axis_angle', from_convention='XYZ')
        self.reset()

    def predict_action(self, obs):
        if self.start is None:
            self.start = obs[-1][14:14+6]

        action = np.zeros((1, 7))
        action[0][:3] = self.start[:3]
        action[0][2] -= self.step_num * .001
        action[0][4] = 3.14
        action[0][5] = 1.5
        action[0][3:6] = self.rotation_transformer.forward(action[:, 3:6].reshape((1, 3)))
        self.step_num += 1
        return action

    def reset(self):
        self.step_num = 0
        self.start = None


class SmoothScriptedPolicy(ScriptedPolicy):

    def __init__(self, env=None):
        super().__init__(env)
        self.last_action = None

    def _get_robosuite_env(self):
        """Get robosuite env from policy's env reference.
        Chain: MultiStepWrapper → VideoRecording → MultimodalSquare → LowdimWrapper → EnvRobosuite → robosuite
        """
        return self.env.env.env.env.env.env

    def _get_env_robosuite(self):
        """Get EnvRobosuite (has get_observation() returning dict)."""
        return self.env.env.env.env.env

    def generate_trajectory(self, mode='middle'):
        robosuite_env = self._get_robosuite_env()
        self.nut_pos = robosuite_env.sim.data.body_xpos[robosuite_env.obj_body_id['SquareNut']]
        self.nut_ori = robosuite_env.sim.data.body_xmat[robosuite_env.obj_body_id['SquareNut']].reshape(3, 3)
        self.nut_ori = R.from_matrix(self.nut_ori).as_euler('xyz')
        self.nut_rot = self.nut_ori[2]

        print("nut pose", self.nut_pos, self.nut_ori)

        peg_pos = np.array(robosuite_env.sim.data.body_xpos[robosuite_env.peg1_body_id])

        above_peg_pos = peg_pos.copy()
        above_peg_pos[2] += .25
        above_peg_pos[1] -= 0.032

        if self.nut_rot > np.pi / 2:
            pick_rot = self.nut_rot - np.pi
        elif self.nut_rot < -np.pi / 2:
            pick_rot = np.pi + self.nut_rot
        else:
            pick_rot = self.nut_rot

        grasp_pos = self.nut_pos.copy()
        grasp_pos[2] -= 0.062

        if np.abs(self.nut_rot) < 4.:
            grasp_pos[1] -= 0.029 * np.cos(pick_rot)
            grasp_pos[0] += 0.029 * np.sin(pick_rot)

        above_nut_pos = grasp_pos.copy()
        above_nut_pos[2] += .2

        self.trajectory = [
            {"t": 30, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, -1])])},
            {"t": 60, "action": np.concatenate([grasp_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, -1.])])},
            {"t": 90, "action": np.concatenate([grasp_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, 1.])])},
            {"t": 120, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, 1.])])},
            {"t": 140, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2, 1.])])},
            {"t": 180, "action": np.concatenate([above_peg_pos, np.array([0., 3.14159, 3.14159/2, 1.])])},
            {"t": 185, "action": np.concatenate([above_peg_pos, np.array([0., 3.14159, 3.14159/2, 1.])])},
            {"t": 190, "action": np.concatenate([above_peg_pos, np.array([0., 3.14159, 3.14159/2, -1.])])},
            {"t": 500, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2, -1.])])},
        ]

    def reset(self):
        super().reset()
        self.mode = random.choice(['left', 'right'])
        print('mode:', self.mode)
        self.generate_trajectory(self.mode)
        cur_xyz = self._get_env_robosuite().get_observation()['robot0_eef_pos']
        self.last_t = 0
        self.last_action = np.concatenate([cur_xyz, np.array([0., 3.14159, 3.14159/2, 0.])])

    def replan(self):
        super().reset()
        self.generate_trajectory(self.mode)
        cur_xyz = self._get_env_robosuite().get_observation()['robot0_eef_pos']
        self.last_t = 0
        self.last_action = np.concatenate([cur_xyz, np.array([0., 3.14159, 3.14159/2, 0.])])

    def predict_action(self, obs):
        if self.step_num == self.trajectory[0]["t"]:
            self.last_action = self.trajectory[0]['action']
            self.last_t = self.trajectory[0]['t']
            self.trajectory.pop(0)

        action = self.last_action + (self.trajectory[0]['action'] - self.last_action) * (self.step_num - self.last_t) / (self.trajectory[0]["t"] - self.last_t)
        action = action.reshape((1, 7))
        action[0][3:6] = self.rotation_transformer.forward(action[:, 3:6].reshape((1, 3)))
        self.step_num += 1
        return action


class SquareSideScriptedPolicy(SmoothScriptedPolicy):

    def __init__(self, env=None, target_peg='random', noise_level=0.0, speed_factor=1.0):
        """
        :param target_peg: 'left', 'right', or 'random' (randomly chosen each reset)
        :param noise_level: Controls vmax for bezier curves (0.0 = perfectly smooth, higher = noisier)
        :param speed_factor: Scales trajectory timesteps. >1 = slower, <1 = faster.
        """
        self._target_peg_setting = target_peg
        self.noise_level = noise_level
        self.speed_factor = speed_factor
        self.target_peg = 'left'  # will be set in reset()
        super().__init__(env)

    def generate_curved_trajectory(self, start, end, mode='middle', vmax=0.0, num_points=30):
        """Generates a curved trajectory between two points using a quadratic bezier curve."""
        control_point = (start + end) / 2
        offset = np.random.uniform(low=-vmax, high=vmax, size=3)
        offset[2] = 0.0
        control_point += offset

        t_values = np.linspace(0, 1, num=num_points)
        trajectory = [
            (1 - t) ** 2 * start + 2 * (1 - t) * t * control_point + t ** 2 * end
            for t in t_values
        ]
        return np.array(trajectory)

    def generate_trajectory(self, mode='middle'):
        robosuite_env = self._get_robosuite_env()
        self.nut_pos = robosuite_env.sim.data.body_xpos[robosuite_env.obj_body_id['SquareNut']]
        self.nut_ori = robosuite_env.sim.data.body_xmat[robosuite_env.obj_body_id['SquareNut']].reshape(3, 3)
        self.nut_ori = R.from_matrix(self.nut_ori).as_euler('xyz')
        self.nut_rot = self.nut_ori[2]

        print("nut pose", self.nut_pos, self.nut_ori)

        peg_pos1 = np.array(robosuite_env.sim.data.body_xpos[robosuite_env.peg1_body_id])
        peg_pos2 = np.array(robosuite_env.sim.data.body_xpos[robosuite_env.peg2_body_id])

        # Choose target peg
        if self.target_peg == 'left':
            peg_pos = peg_pos1
        else:
            peg_pos = peg_pos2

        if self.nut_rot > np.pi / 2:
            pick_rot = self.nut_rot - np.pi
        elif self.nut_rot < -np.pi / 2:
            pick_rot = np.pi + self.nut_rot
        else:
            pick_rot = self.nut_rot

        # Grasp position (offset into nut)
        grasp_pos = self.nut_pos.copy()
        grasp_pos[2] -= 0.062
        grasp_pos[1] += 0.07 * np.sin(self.nut_rot)
        grasp_pos[0] += 0.07 * np.cos(self.nut_rot)

        above_nut_pos = grasp_pos.copy()
        above_nut_pos[2] += .2

        # Peg positions
        above_peg_pos = peg_pos.copy()
        above_peg_pos[2] += .25
        above_peg_pos[1] += 0.07

        # Insert position (drop nut onto peg)
        insert_pos = peg_pos.copy()
        insert_pos[2] += 0.05
        insert_pos[1] += 0.07

        # Generate curved trajectories — noise_level controls bezier deviation
        v = self.noise_level
        # Phase 1: Move above nut -> descend to grasp
        curved_descend = self.generate_curved_trajectory(above_nut_pos, grasp_pos, mode, v, 30)
        # Phase 2: Lift nut up
        curved_lift = self.generate_curved_trajectory(grasp_pos, above_nut_pos, mode, v, 30)
        # Phase 3: Move to above peg
        curved_transit = self.generate_curved_trajectory(above_nut_pos, above_peg_pos, mode, v, 30)
        # Phase 4: Insert down onto peg (no noise — precision required)
        curved_insert = self.generate_curved_trajectory(above_peg_pos, insert_pos, mode, 0.0, 25)

        angle = 0  # aligned rotation for drop
        sf = self.speed_factor  # >1 = slower, <1 = faster

        # Build trajectory using cursor so phases never overlap
        t = int(40 * sf)
        self.trajectory = [
            {"t": t, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159 / 2 - pick_rot, -1])])}
        ]

        # Descend to grasp (gripper open)
        t += int(10 * sf)
        for i, point in enumerate(curved_descend):
            self.trajectory.append({"t": t + i, "action": np.concatenate([point, np.array([0., 3.14159, 3.14159 / 2 - pick_rot, -1])])})
        t += len(curved_descend) - 1

        # Close gripper
        t += int(5 * sf)
        self.trajectory.append({"t": t, "action": np.concatenate([grasp_pos, np.array([0., 3.14159, 3.14159 / 2 - pick_rot, 1.])])})

        # Lift up (gripper closed)
        t += int(15 * sf)
        for i, point in enumerate(curved_lift):
            self.trajectory.append({"t": t + i, "action": np.concatenate([point, np.array([0., 3.14159, 3.14159 / 2 - pick_rot, 1.])])})
        t += len(curved_lift) - 1

        # Rotate to aligned angle
        t += int(20 * sf)
        self.trajectory.append({"t": t, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, angle, 1.])])})

        # Transit to above peg
        t += 1
        for i, point in enumerate(curved_transit):
            self.trajectory.append({"t": t + i, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        t += len(curved_transit) - 1

        # Insert down onto peg
        t += 1
        for i, point in enumerate(curved_insert):
            self.trajectory.append({"t": t + i, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        t += len(curved_insert) - 1

        # Release gripper
        t += 1
        self.trajectory.append({"t": t, "action": np.concatenate([insert_pos, np.array([0., 3.14159, angle, -1.])])})

        # Hold (end)
        self.trajectory.append({"t": 600, "action": np.concatenate([insert_pos, np.array([0., 3.14159, angle, -1.])])})

    def reset(self):
        # Choose target peg
        if self._target_peg_setting == 'random':
            self.target_peg = random.choice(['left', 'right'])
        else:
            self.target_peg = self._target_peg_setting
        super().reset()

    def replan(self):
        # Keep same target_peg but replan trajectory
        super().replan()


def create_robomimic_env():
    dataset_path = "data/robomimic/datasets/square/mh/low_dim.hdf5"

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta["env_kwargs"]["controller_configs"]['control_delta'] = False

    print(env_meta)

    ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': OBS_KEYS})
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )

    fps = 10
    crf = 22
    robosuite_fps = 20
    steps_per_render = max(robosuite_fps // fps, 1)

    env = MultiStepWrapper(
        VideoRecordingWrapper(
            MultimodalSquareLowdimWrapper(
                RobomimicLowdimWrapper(
                    env=robomimic_env,
                    obs_keys=OBS_KEYS,
                    init_state=None,
                ),
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=fps,
                codec='h264',
                input_pix_fmt='rgb24',
                crf=crf,
                thread_type='FRAME',
                thread_count=1
            ),
            file_path=None,
            steps_per_render=steps_per_render
        ),
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=600
    )

    return env, env_meta


def create_policy(env, target_peg='random', noise_level=0.0, speed_factor=1.0):
    return SquareSideScriptedPolicy(env, target_peg=target_peg, noise_level=noise_level, speed_factor=speed_factor)


@click.command()
@click.option('-o', '--output_dir', default='data/multimodal_square')
@click.option('-n', '--num_episodes', type=int, default=200)
@click.option('--seed', type=int, default=0)
@click.option('--noise_min', type=float, default=0.0, help='Min noise level (vmax) for bezier curves')
@click.option('--noise_max', type=float, default=0.12, help='Max noise level (vmax) for bezier curves')
@click.option('--speed_factor_left', type=float, default=1.0, help='Speed factor for left peg (>1=slower, <1=faster)')
@click.option('--speed_factor_right', type=float, default=1.0, help='Speed factor for right peg (>1=slower, <1=faster)')
@click.option('--speed_factor_range_left', type=(float, float), default=None, help='Sample left peg speed uniformly from (min, max)')
@click.option('--speed_factor_range_right', type=(float, float), default=None, help='Sample right peg speed uniformly from (min, max)')
@click.option('--target_peg', type=click.Choice(['random', 'left', 'right']), default='random', help='Which peg to target (default: random)')
def main(output_dir, num_episodes, seed, noise_min, noise_max, speed_factor_left, speed_factor_right, speed_factor_range_left, speed_factor_range_right, target_peg):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Seed set to: {seed}")
    print(f"Collecting {num_episodes} episodes with target_peg={target_peg}, noise in [{noise_min}, {noise_max}]")
    if speed_factor_range_left is not None:
        print(f"Speed factor left: uniform [{speed_factor_range_left[0]}, {speed_factor_range_left[1]}]")
    else:
        print(f"Speed factor left: {speed_factor_left}")
    if speed_factor_range_right is not None:
        print(f"Speed factor right: uniform [{speed_factor_range_right[0]}, {speed_factor_range_right[1]}]")
    else:
        print(f"Speed factor right: {speed_factor_right}")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    env, env_cfg = create_robomimic_env()

    dataset_file = h5py.File(pathlib.Path(output_dir).joinpath("demos.hdf5"), 'w')
    dataset_data_group = dataset_file.create_group('data')
    dataset_data_group.attrs['env_args'] = json.dumps(env_cfg.copy())

    total_rollouts = 0
    successful_rollouts = 0

    # Noise levels: uniform distribution between noise_min and noise_max
    # Low noise = very smooth, high noise = non-smooth trajectories
    noise_levels = np.linspace(noise_min, noise_max, num_episodes)
    np.random.shuffle(noise_levels)

    for ep_num in range(num_episodes):
        success = False
        noise = noise_levels[ep_num]
        # Create policy with random peg; speed_factor set based on chosen peg
        policy = create_policy(env, target_peg=target_peg, noise_level=noise, speed_factor=1.0)
        # Set speed factor (range or fixed per peg)
        if policy.target_peg == 'left':
            if speed_factor_range_left is not None:
                policy.speed_factor = np.random.uniform(speed_factor_range_left[0], speed_factor_range_left[1])
            else:
                policy.speed_factor = speed_factor_left
        else:
            if speed_factor_range_right is not None:
                policy.speed_factor = np.random.uniform(speed_factor_range_right[0], speed_factor_range_right[1])
            else:
                policy.speed_factor = speed_factor_right

        print(f'Episode {ep_num}: target_peg={policy.target_peg}, noise_level={noise:.4f}, speed_factor={policy.speed_factor:.2f}')

        attempt = 0
        max_attempts = 20
        while not success and attempt < max_attempts:
            obs_list = []
            action_list = []

            # start video — save for first 10 episodes AND first 5 failed attempts for debugging
            assert isinstance(env.env, VideoRecordingWrapper)
            env.env.video_recoder.stop()

            if ep_num < 10 or attempt < 5:
                filename = pathlib.Path(output_dir).joinpath(f"vids/ep{ep_num}_attempt{attempt}.mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                env.env.file_path = str(filename)
            else:
                env.env.file_path = None

            # reset env
            env.env.env.env.init_state = None
            env.seed(np.random.randint(0, 10000000))

            obs = env.reset()
            # On retry, keep the same speed_factor (already set above)
            policy.replan()

            rews = []
            max_rew_stage = 0
            for step in range(600):
                action = policy.predict_action(obs)

                # obs is (n_obs_steps, obs_dim) from MultiStepWrapper; store the latest
                obs_list.append(obs[-1].copy())
                action_list.append(action[0].copy())

                obs, reward, done, info = env.step(action)
                rews.append(reward)
                # Track reward stage progress
                max_rew_stage = max(max_rew_stage, env.env.env.rew)

                if reward >= 1.0:
                    success = True
                    print("success")
                    break

            if not success:
                print(f"  attempt {attempt} FAILED: nut_on_peg={max_rew_stage>0}, steps={len(rews)}, peg={policy.target_peg}")

            if success:
                successful_rollouts += 1
            total_rollouts += 1
            attempt += 1

        if not success:
            print(f"  WARNING: Episode {ep_num} failed after {max_attempts} attempts, using last attempt")

        # Compute per-axis scores for this episode
        action_array_ep = np.stack(action_list, axis=0)
        ep_steps = len(action_list)
        ep_smoothness = compute_smoothness(action_array_ep)
        # Gate by success — failed demos contribute smoothness=0 so the reward
        # model can't reward "smoothly never finishing".
        if not success:
            ep_smoothness = 0.0
        ep_speed = compute_speed_reward(ep_steps)
        ep_peg = -1.0 if policy.target_peg == 'left' else 1.0
        print(f"Episode {ep_num} complete: steps={ep_steps}, peg={policy.target_peg}({ep_peg:+.0f}), "
              f"speed={ep_speed:.3f}, smoothness={ep_smoothness:.3f}, noise={noise:.4f}")

        # Save to HDF5 — lowdim obs stored per-key for compatibility with existing dataset loaders
        ep_group = dataset_data_group.create_group(f'demo_{ep_num}')
        obs_group = ep_group.create_group('obs')

        obs_array = np.stack(obs_list, axis=0)  # (T, obs_dim)
        # Split back into individual obs keys for compatibility
        idx = 0
        for key in OBS_KEYS:
            raw_obs = env.env.env.env.env.get_observation()  # just to get shapes
            dim = raw_obs[key].shape[0]
            obs_group.create_dataset(key, data=obs_array[:, idx:idx+dim])
            idx += dim

        action_array = np.stack(action_list, axis=0)
        ep_group.create_dataset('actions', data=action_array)

        # Save initial sim state — needed by env runner to reset to this episode's starting config
        # env chain: MultiStepWrapper → VideoRecording → MultimodalSquare → LowdimWrapper → ...
        init_state = env.env.env.last_reset_state  # from MultimodalSquareLowdimWrapper
        # states shape: (T, 45) — runner only uses states[0], so broadcast init_state to all timesteps
        states = np.broadcast_to(init_state, (len(obs_list), len(init_state))).copy()
        ep_group.create_dataset('states', data=states)

        ep_group.attrs['target_peg'] = policy.target_peg
        ep_group.attrs['noise_level'] = noise
        ep_group.attrs['speed_factor'] = policy.speed_factor

    dataset_data_group.attrs['data collection'] = f"{successful_rollouts} of {total_rollouts} total rollouts successful"
    dataset_file.close()

    # Also save as .npz for compatibility with reward model pipeline
    print("Saving .npz format...")
    save_npz_from_hdf5(output_dir)

    print(f"{successful_rollouts} of {total_rollouts} total rollouts successful")


def compute_smoothness(actions):
    """Compute smoothness of an action trajectory. 1 = perfectly smooth, 0 = maximally jerky."""
    if len(actions) < 3:
        return 1.0
    actions = np.array(actions)
    vel = np.diff(actions, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    jerk_magnitude = np.linalg.norm(jerk, axis=-1)
    jerk_mean = float(np.mean(jerk_magnitude))
    return float(np.exp(-10.0 * jerk_mean))


def compute_speed_reward(steps_taken, max_steps=600):
    """Speed reward: linear interpolation, 1.0 at step 0, 0.1 at max_steps."""
    return 1.0 - 0.9 * (steps_taken / max_steps)


def save_npz_from_hdf5(output_dir):
    """Convert HDF5 demos to .npz format with computed metrics."""
    hdf5_path = pathlib.Path(output_dir) / "demos.hdf5"
    npz_path = pathlib.Path(output_dir) / "rollouts.npz"

    # Convert 7D axis_angle actions to 10D rot6d to match policy action space
    rot_transformer = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')

    with h5py.File(hdf5_path, 'r') as f:
        demos = f['data']
        n_episodes = len([k for k in demos.keys() if k.startswith('demo_')])

        all_obs = []
        all_actions = []
        all_lengths = []
        all_success = []
        all_speed = []
        all_smoothness = []
        all_peg_reward = []
        all_speed_left = []
        all_speed_right = []

        max_len = 0
        for i in range(n_episodes):
            demo = demos[f'demo_{i}']
            obs_parts = [demo['obs'][key][:].astype(np.float32) for key in OBS_KEYS]
            obs = np.concatenate(obs_parts, axis=-1)
            actions = demo['actions'][:].astype(np.float32)

            # Convert 7D (pos3 + axis_angle3 + gripper1) to 10D (pos3 + rot6d + gripper1)
            if actions.shape[-1] == 7:
                pos = actions[:, :3]
                rot = actions[:, 3:6]
                gripper = actions[:, 6:]
                rot6d = rot_transformer.forward(rot)
                actions = np.concatenate([pos, rot6d, gripper], axis=-1).astype(np.float32)
            L = min(len(obs), len(actions))
            all_obs.append(obs[:L])
            all_actions.append(actions[:L])
            all_lengths.append(L)
            max_len = max(max_len, L)

            # Metrics
            all_success.append(1.0)  # all scripted episodes succeed
            speed = compute_speed_reward(L)
            all_speed.append(speed)
            all_smoothness.append(compute_smoothness(actions[:L]))

            # Peg reward: +1 for right peg, -1 for left peg
            target_peg = demo.attrs.get('target_peg', 'left')
            all_peg_reward.append(1.0 if target_peg == 'right' else -1.0)

            # Per-peg speed: speed for matching peg, 0 for the other
            all_speed_left.append(speed if target_peg == 'left' else 0.0)
            all_speed_right.append(speed if target_peg == 'right' else 0.0)

        # Pad to same length
        obs_dim = all_obs[0].shape[-1]
        act_dim = all_actions[0].shape[-1]
        obs_padded = np.zeros((n_episodes, max_len, obs_dim), dtype=np.float32)
        act_padded = np.zeros((n_episodes, max_len, act_dim), dtype=np.float32)
        for i in range(n_episodes):
            L = all_lengths[i]
            obs_padded[i, :L] = all_obs[i]
            act_padded[i, :L] = all_actions[i]

        np.savez(npz_path,
            obs=obs_padded,
            actions=act_padded,
            episode_lengths=np.array(all_lengths, dtype=np.int32),
            success=np.array(all_success, dtype=np.float32),
            speed_reward=np.array(all_speed, dtype=np.float32),
            smoothness=np.array(all_smoothness, dtype=np.float32),
            peg_reward=np.array(all_peg_reward, dtype=np.float32),
            speed_left=np.array(all_speed_left, dtype=np.float32),
            speed_right=np.array(all_speed_right, dtype=np.float32),
        )
        print(f"Saved {n_episodes} episodes to {npz_path}")
        print(f"  obs shape: {obs_padded.shape}, action shape: {act_padded.shape}")
        print(f"  Mean speed: {np.mean(all_speed):.3f}, Mean smoothness: {np.mean(all_smoothness):.3f}")
        print(f"  Left peg: {sum(1 for p in all_peg_reward if p > 0)}, Right peg: {sum(1 for p in all_peg_reward if p < 0)}")
        print(f"  Mean speed_left: {np.mean([s for s in all_speed_left if s > 0]):.3f}, "
              f"Mean speed_right: {np.mean([s for s in all_speed_right if s > 0]):.3f}")


if __name__ == '__main__':
    main()
