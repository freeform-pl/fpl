"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

import numpy as np

import collections

import copy

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

from scipy.spatial.transform import Rotation as R

from diffusion_policy.env_runner.robomimic_image_runner import create_env as robomimic_image_create_env
from diffusion_policy.common.replay_buffer import ReplayBuffer
import random

import h5py

class MultimodalSquareWrapper(RobomimicImageWrapper):
    def __init__(self, 
        env: RobomimicImageWrapper,
        ):

        self.env = env
        # self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        
      
        self.action_space =self.env.action_space
        self.observation_space = self.env.observation_space

    def get_observation(self):
        return self.env.get_observation()

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        # state = self.env.get_state()['states'] 

        nut_pos = np.random.uniform([-0.2, -0.2], [0, 0.2], size=(2))

        reset_state = np.array([ 0.        , -0.02921895,  0.17810908,  0.02728627, -2.63967499, \
                -0.01431297,  2.9502351 ,  0.77126893,  0.020833  , -0.020833  , \
                -0.11083517,  0.11445349,  0.89      ,  -0.98421243,  0.        , # <- first trhee for some value \ 
                    0.        ,  0.17699121, 10.        , 10.        , 10.        , \
                    1.        ,  0.        ,  0.        ,  0.        ,  0.        , \
                    0.        ,  0.        ,  0.        ,  0.        ,  0.        , \
                    0.        ,  0.        ,  0.        ,  0.        ,  0.        , \
                    0.        ,  0.        ,  0.        ,  0.        ,  0.        , \
                    0.        ,  0.        ,  0.        ,  0.        ,  0.        ])  

        self.rew = 0
        reset_state[10:12] = nut_pos
        self.env.env.reset_to({"states": reset_state})
        nut_pos = self.env.env.env.sim.data.body_xpos[self.env.env.env.obj_body_id['SquareNut']]
        
        peg_pos1 = np.array(self.env.env.env.sim.data.body_xpos[self.env.env.env.peg1_body_id])
        
        peg_pos2 = np.array(self.env.env.env.sim.data.body_xpos[self.env.env.env.peg2_body_id])


        if np.linalg.norm(nut_pos - peg_pos1) < np.linalg.norm(nut_pos - peg_pos2):
            self.target_peg_id = 1
        else:
            self.target_peg_id = 0

        # return obs
        obs = self.get_observation()
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # TODO change reward
        peg_id = 0
        peg_pos = self.env.env.env.sim.data.body_xpos[self.env.env.env.obj_body_id['SquareNut']]
        on_peg = self.env.env.env.on_peg(peg_pos, peg_id)

        on_saddle_drop = np.linalg.norm(obs["robot0_eef_pos"] - np.array([0,0,0.89])) < 0.03
        on_saddle = np.linalg.norm(obs["robot0_eef_pos"] - np.array([0,0,0.89+0.2])) < 0.09

        if self.rew == 0 and on_peg:
            self.rew += 1
        elif self.rew == 1 and on_saddle:
            self.rew += 1
        elif self.rew == 2 and on_peg:
            self.rew += 1
        elif self.rew == 3 and on_saddle_drop:
            self.rew += 1

        reward = 1 if self.rew == 4 else 0

        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)
            
class ScriptedPolicy:

    def __init__(self, env=None):
        self.env = env
        self.rotation_transformer = RotationTransformer('euler_angles', 'axis_angle', from_convention='XYZ')
        self.reset()

    def predict_action(self, obs_dict):

        #pose = self.env.env.env.env.get_observation()

        if self.start is None:
            self.start = obs_dict[-1][14:14+6]

        action = np.zeros((1,7))

        action[0][:3] = self.start[:3]
        action[0][2] -= self.step_num*.001

        action[0][4] = 3.14
        action[0][5] = 1.5

        action[0][3:6] = self.rotation_transformer.forward(action[:,3:6].reshape((1,3)))

        self.step_num += 1

        return action

    def reset(self):
        self.step_num = 0
        self.start = None

def get_body_euler_angles_from_matrix(sim, body_name, axes='xyz', degrees=False):

    # Get the body ID
    body_id = sim.model.body_name2id(body_name)
    
    # Retrieve and reshape the rotation matrix
    orientation_mat = sim.data.body_xmat[body_id].reshape(3, 3)
    
    # Create a Rotation object
    rotation = R.from_matrix(orientation_mat)
    
    # Convert to Euler angles
    euler_angles = rotation.as_euler(axes, degrees=degrees)
    
    return euler_angles

class TestPosScriptedPolicy(ScriptedPolicy):

    def __init__(self, env=None):
        super().__init__(env)

    def reset(self):
        super().reset()
        self.nut_pos = self.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.obj_body_id['SquareNut']]
        self.nut_ori = self.env.env.env.env.env.sim.data.body_xmat[self.env.env.env.env.env.obj_body_id['SquareNut']].reshape(3,3)
        self.nut_ori = R.from_matrix(self.nut_ori).as_euler('xyz')

        self.nut_rot = self.nut_ori[2]

        print("nut pose", self.nut_pos, self.nut_ori)

    def predict_action(self, obs_dict):
        action = np.zeros((1,7))

        action[0][:3] = self.nut_pos
        action[0][2] += 0.08

        action[0][4] = 3.14159
        if self.nut_rot > 0 and self.nut_rot < 1.:
            action[0][5] = 3.14159/2 - self.nut_rot

        action[0][3:6] = self.rotation_transformer.forward(action[:,3:6].reshape((1,3)))

        return action
    
class SmoothScriptedPolicy(ScriptedPolicy):

    def __init__(self, env=None):
        super().__init__(env)

        self.last_action = None

    def generate_trajectory(self):
        self.nut_pos = self.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.obj_body_id['SquareNut']]
        self.nut_ori = self.env.env.env.env.env.sim.data.body_xmat[self.env.env.env.env.env.obj_body_id['SquareNut']].reshape(3,3)
        self.nut_ori = R.from_matrix(self.nut_ori).as_euler('xyz')

        self.nut_rot = self.nut_ori[2]

        print("nut pose", self.nut_pos, self.nut_ori)

        peg_pos = np.array(self.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.peg1_body_id])

        above_peg_pos = peg_pos.copy()
        above_peg_pos[2] += .25
        above_peg_pos[1] -= 0.032

        if self.nut_rot > np.pi/2:
            pick_rot = self.nut_rot - np.pi
            drop_rot = 0.
        elif self.nut_rot < -np.pi/2:
            pick_rot = np.pi + self.nut_rot
        else:
            pick_rot = self.nut_rot

        grasp_pos = self.nut_pos.copy()
        grasp_pos[2] -= 0.062

        if np.abs(self.nut_rot) < 4.:
            grasp_pos[1] -= 0.029*np.cos(pick_rot)
            grasp_pos[0] += 0.029*np.sin(pick_rot)

        above_nut_pos = grasp_pos.copy()
        above_nut_pos[2] += .2

        print(pick_rot)

        self.trajectory = [{"t":30, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, -1])])},
        {"t":60, "action": np.concatenate([grasp_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, -1.])])},
        {"t":90, "action": np.concatenate([grasp_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, 1.])])},
        {"t":120, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2 - pick_rot, 1.])])},
        {"t":140, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2, 1.])])},
        {"t":180, "action": np.concatenate([above_peg_pos, np.array([0., 3.14159, 3.14159/2, 1.])])},
        {"t":185, "action": np.concatenate([above_peg_pos, np.array([0., 3.14159, 3.14159/2, 1.])])},
        {"t":190, "action": np.concatenate([above_peg_pos, np.array([0., 3.14159, 3.14159/2, -1.])])},
        {"t":500, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159/2, -1.])])}]

    def reset(self):
        super().reset()
        self.mode = random.choice(['left', 'right'])  # Randomly select a curvature mode
        # self.mode = random.choice(['left', 'middle', 'right'])  # Randomly select a curvature mode
        # self.mode = 'middle'

        print('mode:', self.mode)
        self.generate_trajectory(self.mode)
        # self.trajectory_bkp = copy.deepcopy(self.trajectory)
        cur_xyz = self.env.env.env.env.env.get_observation()['robot0_eef_pos']

        self.last_t = 0
        self.last_action = np.concatenate([cur_xyz, np.array([0., 3.14159, 3.14159/2, 0.])])

    def replan(self):
        super().reset()
        self.generate_trajectory(self.mode)
        # self.trajectory = copy.deepcopy(self.trajectory_bkp)
        # print('self.trajectory_bkp', len(self.trajectory_bkp))
        cur_xyz = self.env.env.env.env.env.get_observation()['robot0_eef_pos']
        self.last_t = 0
        self.last_action = np.concatenate([cur_xyz, np.array([0., 3.14159, 3.14159/2, 0.])])

    def predict_action(self, obs_dict):

        if self.step_num == self.trajectory[0]["t"]:
            self.last_action = self.trajectory[0]['action']
            self.last_t = self.trajectory[0]['t']
            self.trajectory.pop(0)
        
        action = self.last_action + (self.trajectory[0]['action'] - self.last_action)*(self.step_num - self.last_t)/(self.trajectory[0]["t"] - self.last_t)
        action = action.reshape((1,7))

        action[0][3:6] = self.rotation_transformer.forward(action[:,3:6].reshape((1,3)))

        self.step_num += 1
        return action

class SquareSideScriptedPolicy(SmoothScriptedPolicy):

    def generate_curved_trajectory(self, start, end, mode='middle', vmax=0.0, num_points=30):
        """
        Generates a curved trajectory between two points.
        :param start: Starting position (x, y, z).
        :param end: Ending position (x, y, z).
        :param mode: 'left', 'right', or 'middle' for curvature direction.
        :return: List of trajectory waypoints.
        """
        control_point = (start + end) / 2
        offset = np.random.uniform(low=-vmax, high=vmax, size=3)
        offset[2] = 0.0
        control_point += offset
        # offset = np.array([0.02, 0.02, 0.0])  # Example offset for curvature

        # Generate intermediate points (e.g., quadratic bezier curve)
        t_values = np.linspace(0, 1, num=num_points)  # Adjust number of points as needed
        trajectory = [
            (1 - t) ** 2 * start + 2 * (1 - t) * t * control_point + t ** 2 * end
            for t in t_values
        ]
        return np.array(trajectory)

    def generate_trajectory(self, mode='middle'):
        self.nut_pos = self.env.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.env.obj_body_id['SquareNut']]
        self.nut_ori = self.env.env.env.env.env.env.sim.data.body_xmat[self.env.env.env.env.env.env.obj_body_id['SquareNut']].reshape(3, 3)
        self.nut_ori = R.from_matrix(self.nut_ori).as_euler('xyz')

        self.nut_rot = self.nut_ori[2]

        print("nut pose", self.nut_pos, self.nut_ori)

        peg_pos1 = np.array(self.env.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.env.peg1_body_id])
        
        peg_pos2 = np.array(self.env.env.env.env.env.env.sim.data.body_xpos[self.env.env.env.env.env.env.peg2_body_id])

        grasp_pos = self.nut_pos.copy()

        peg_pos = peg_pos1

        above_peg_pos = peg_pos.copy()
        above_peg_pos[2] += .25
        above_peg_pos[1] += 0.07

        top_peg_pos = peg_pos.copy()
        top_peg_pos[2] += 0.0
        top_peg_pos[1] += 0.07

        if self.nut_rot > np.pi / 2:
            pick_rot = self.nut_rot - np.pi
        elif self.nut_rot < -np.pi / 2:
            pick_rot = np.pi + self.nut_rot
        else:
            pick_rot = self.nut_rot

        # Compute the initial grasp position
        grasp_pos = self.nut_pos.copy()
        grasp_pos[2] -= 0.062

        saddle_pos = np.array([0,0,0])

        saddle_pos[2] = 0.89 + 0.2# stay in sadle for 30 seconds

        saddle_pos_down = np.array([0,0,0.89])

        grasp_pos[1] += 0.07 * np.sin(self.nut_rot)
        grasp_pos[0] += 0.07 * np.cos(self.nut_rot)

        above_nut_pos = grasp_pos.copy()
        above_nut_pos[2] += .2

        # Create trajectory with curvature applied to specific segments
        self.trajectory = [{"t": 30, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159 / 2 - pick_rot, -1])])}]
        self.trajectory = [{"t": 40, "action": np.concatenate([above_nut_pos, np.array([0., 3.14159, 3.14159 / 2 - pick_rot, -1])])}]

        # Generate curved trajectories for selected segments
        curved_traj_1 = self.generate_curved_trajectory(above_nut_pos, grasp_pos, mode, 0.0, 30)
        curved_traj_2 = self.generate_curved_trajectory(grasp_pos, above_nut_pos, mode, 0.0, 30)
        curved_traj_3 = self.generate_curved_trajectory(above_nut_pos, above_peg_pos, mode, 0.0, 30)
        curved_traj_4 = self.generate_curved_trajectory(above_peg_pos, top_peg_pos, mode, 0.0, 25)
        curved_traj_5 = self.generate_curved_trajectory(top_peg_pos, above_peg_pos, mode, 0.0, 25)
        curved_traj_6 = self.generate_curved_trajectory(above_peg_pos, saddle_pos, mode, 0.0, 30)
        curved_traj_7 = self.generate_curved_trajectory(saddle_pos, above_peg_pos, mode, 0.0, 30)
        curved_traj_8 = self.generate_curved_trajectory(above_peg_pos, top_peg_pos, mode, 0.0, 25)
        curved_traj_9 = self.generate_curved_trajectory(top_peg_pos, above_peg_pos, mode, 0.0, 25)
        curved_traj_10 = self.generate_curved_trajectory(above_peg_pos, saddle_pos, mode, 0.0, 30)
        curved_traj_11 = self.generate_curved_trajectory(saddle_pos, saddle_pos_down, mode, 0.0, 25)

        angle = 0
        # Segment 1: Above Nut -> Grasp Position
        for t, point in enumerate(curved_traj_1, start=50):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, 3.14159 / 2 - self.nut_rot, -1])])})
        # Grasp Position -> Grasp Position with Force
        self.trajectory.append({"t": 85, "action": np.concatenate([grasp_pos, np.array([0., 3.14159, 3.14159 / 2 - self.nut_rot, 1.])])})
        # Segment 2: Grasp Position -> Above Nut
        for t, point in enumerate(curved_traj_2, start=100):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, 3.14159 / 2 - self.nut_rot, 1.])])})
        t = t+30
        self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        # Above Nut (Aligned) -> Above Peg
        for t, point in enumerate(curved_traj_3, start=t+1):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        # down peg 
        for t, point in enumerate(curved_traj_4, start=t+1):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        # up peg 
        for t, point in enumerate(curved_traj_5, start=t+1):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        # go to saddle
        for t, point in enumerate(curved_traj_6, start=t+1):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
            
        t+=20
        self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        
        #  back to nut
        for t, point in enumerate(curved_traj_7, start=t+1):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        # go down
        for t, point in enumerate(curved_traj_8, start=t+1):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        # go up
        for t, point in enumerate(curved_traj_9, start=t+1):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        # go to saddle
        for t, point in enumerate(curved_traj_10, start=t+1):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        t+=20
        self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        
        # go down 
        for t, point in enumerate(curved_traj_11, start=t+1):
            self.trajectory.append({"t": t, "action": np.concatenate([point, np.array([0., 3.14159, angle, 1.])])})
        # End segment: Above Peg (Static Points)
        self.trajectory += [
            {"t": t+1, "action": np.concatenate([saddle_pos_down, np.array([0., 3.14159, angle, -1.])])},
            {"t": 600, "action": np.concatenate([saddle_pos_down, np.array([0., 3.14159, angle, -1.])])}
        ]

def create_robomimic_env():
    dev = True
    cam_h = 84 #if not dev else 640
    cam_w = 84 #if not dev else 640

    env_meta = {'env_name': 'NutAssemblySquare', 'type': 1, 'env_kwargs': {'has_renderer': True, 'has_offscreen_renderer': False, 'ignore_done': True, 'use_object_obs': True, 'use_camera_obs': True, 'control_freq': 20, 'controller_configs': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': False, 'interpolation': None, 'ramp_ratio': 0.2}, 'robots': ['Panda'], 'camera_depths': False, 'camera_heights': 84, 'camera_widths': 84, 'reward_shaping': False}}
    shape_meta = {
        "action": {"shape":[10]},
        "obs":{
            "agentview_image": {"shape":[3, 84, 84], "type": "rgb"},
            "robot0_eef_pos": {"shape":[3]},
            "robot0_eef_quat": {"shape":[4]},
            "robot0_eye_in_hand_image": {"shape":[3, 84, 84], "type": "rgb"},
            "robot0_gripper_qpos":{"shape":[2]}
        }

    }
    import robomimic.utils.file_utils as FileUtils  

    env_meta = FileUtils.get_env_metadata_from_dataset("data/robomimic/datasets/square/mh/image_abs.hdf5")

    env_meta["env_kwargs"]["controller_configs"]['control_delta'] =  False

    print(env_meta)
    env = robomimic_image_create_env(env_meta, shape_meta=shape_meta)

    fps = 10
    crf = 22
    robosuite_fps = 20
    steps_per_render = max(robosuite_fps // fps, 1)
    env_n_obs_steps = 1
    env_n_action_steps = 1
    max_steps = 600


    # hard reset doesn't influence lowdim env
    # robomimic_env.env.hard_reset = False
    env = MultiStepWrapper(
            VideoRecordingWrapper(
                MultimodalSquareWrapper(
                    RobomimicImageWrapper(
                        env=env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key="agentview_image"
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
            n_obs_steps=env_n_obs_steps,
            n_action_steps=env_n_action_steps,
            max_episode_steps=max_steps
        )
    
    return env, env_meta

def create_policy(env, mode='both'):
    return SquareSideScriptedPolicy(env), "side"

@click.command()
# @click.option('-o', '--output_dir', default='data/uniform')
@click.option('-o', '--output_dir', default='data/longhistsquare100')
@click.option('-d', '--device', default='cuda:0')
@click.option('-n', '--num_episodes', type=int, default=100)
@click.option('-s', '--num_side', type=int, default=5)
@click.option('--seed', type=int, default=0)
def main(output_dir, device, num_episodes, num_side, seed):
    # Set up seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Seed set to: {seed}")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    env, env_cfg = create_robomimic_env()

    zarr_path = str(pathlib.Path(output_dir).joinpath('replay_buffer.zarr').absolute())
    replay_buffer = ReplayBuffer.create_from_path(
        zarr_path=zarr_path, mode='a')

    dataset_file = h5py.File(pathlib.Path(output_dir).joinpath("demos.hdf5"), 'w')
    dataset_data_group = dataset_file.create_group('data')

    dataset_data_group.attrs['env_args'] = json.dumps(env_cfg.copy())

    total_rollouts = 0
    successful_rollouts = 0

    modes = ['side']*num_side + ['corner']*(num_episodes - num_side)
    random.shuffle(modes)

    for ep_num in range(num_episodes):

        success = False
        policy, policy_type = create_policy(env, mode='side')

        print('policy_type:', policy_type)

        while not success:
            #NOTE Dataloader in this repo only needs obs and actions to train (see diffusion_policy/dataset/robomimic_replay_lowdim_dataset.py)
            trajectory = {"obs": [],
            "actions": [],}

            #start video
            assert isinstance(env.env, VideoRecordingWrapper)
            env.env.video_recoder.stop()

            if ep_num < 5:
                filename = pathlib.Path(output_dir).joinpath(
                f"vids/episode{ep_num}.mp4")
                filename.parent.mkdir(parents=False, exist_ok=True)
                filename = str(filename)
                env.env.file_path = filename
            else:
                env.env.file_path = None

            #reset env with seed
            assert isinstance(env.env.env, RobomimicImageWrapper)
            env.env.env.init_state = None
            env.seed(np.random.randint(0, 10000000))

            obs = env.reset()
            policy.replan()

            rews = []
            for step in range(600):
                obs_dict = obs

                action = policy.predict_action(obs)

                trajectory['actions'].append(action[0].copy())

                for key in obs_dict.keys():
                    obs_dict[key] = obs_dict[key][0]

                camera_keys = ["agentview_image", "robot0_eye_in_hand_image"]
                for key in camera_keys:
                    obs_dict[key] =  (obs_dict[key]*255).astype(np.uint8).transpose((1,2,0))
                trajectory['obs'].append(obs_dict)


                #TODO: why do two methods of getting raw_obs disagree slightly?
                #raw_obs = env.env.env.env.get_observation()
                obs, reward, done, info = env.step(action)
                rews.append(reward)

                if reward >= 1.0:
                    success = True
                    print("success")
                    break

            if success:
                successful_rollouts += 1

            total_rollouts += 1

        print(f"Episode {ep_num} complete", max(rews), policy_type)

        ep_group = dataset_data_group.create_group(f'demo_{ep_num}')
        obs_group = ep_group.create_group('obs')
        for obs_kwrd in trajectory['obs'][0].keys():
            obs_kwrd = str(obs_kwrd)
            obs_array = np.stack([od[obs_kwrd] for od in trajectory['obs']], axis=0)
            obs_group.create_dataset(obs_kwrd, data=obs_array)
        action_array = np.stack([a for a in trajectory['actions']], axis=0)
        ep_group.create_dataset('actions', data=action_array)
        ep_group.attrs['scripted_policy_type'] = policy_type

        # TODO the images need to be saved in format
    dataset_data_group.attrs['data collection'] = f"{successful_rollouts} of {total_rollouts} total rollouts successful"
    dataset_file.close()
    print(f"{successful_rollouts} of {total_rollouts} total rollouts successful")

if __name__ == '__main__':
    main()