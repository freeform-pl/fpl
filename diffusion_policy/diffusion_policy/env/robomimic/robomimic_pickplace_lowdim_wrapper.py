"""
Lowdim wrapper for the 4-object PickPlace task (milk, bread, cereal, can).

On reset, lets the PickPlace env's own placement_initializer randomize the four
object positions in bin1 (matching how scripted data collection rolls). Tracks
per-object placement and exposes:
  - reward = number of objects placed in their correct bin (0..4)
  - info['objects_in_bins'] = per-object 0/1 array
"""

from typing import List, Optional
import numpy as np
import gym
from gym.spaces import Box
from robomimic.envs.env_robosuite import EnvRobosuite


OBJ_NAMES = ['Milk', 'Bread', 'Cereal', 'Can']
# Right-first canonical placement order in object-id space. Used to choose
# which subset is "active" when n_active_objects < 4 (e.g. pickplace_2 keeps
# the first two: Bread + Can).
CANONICAL_RIGHT_FIRST = [1, 3, 0, 2]

# Canonical object orientations (wxyz quaternions) used by quadrant placement.
# These are axis-aligned poses so the scripted policy can grip with a fixed
# gripper yaw — without this the env's random z-rotation occasionally lines
# the wide side of objects (esp. Cereal, ~8 cm wide vs the Panda's 8.4 cm
# max opening) with the gripper closing direction and grasps fail.
IDENTITY_QUAT = np.array([1., 0., 0., 0.])
# 90° z-rotation: cos(pi/4) + sin(pi/4) k
ROT_Z_90_QUAT = np.array([np.cos(np.pi / 4), 0., 0., np.sin(np.pi / 4)])
DEFAULT_OBJ_QUATS = {
    'Milk':   IDENTITY_QUAT,
    'Bread':  IDENTITY_QUAT,
    'Cereal': IDENTITY_QUAT,
    'Can':    IDENTITY_QUAT,
}


class RobomimicPickPlaceLowdimWrapper(gym.Env):
    """Wrapper for PickPlace 4-object task that mirrors the two-peg wrapper API."""

    def __init__(self,
            env: EnvRobosuite,
            obs_keys: List[str] = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos'],
            init_state: Optional[np.ndarray] = None,
            render_hw=(256, 256),
            render_camera_name='agentview',
            quadrant_placement: bool = True,
            quadrant_noise: float = 0.03,
            settle_steps: int = 40,
            n_active_objects: int = 4,
        ):
        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        # Fixed-quadrant placement: each of the 4 objects starts at the center
        # of its own quadrant in bin1 with at most ±quadrant_noise xy jitter.
        # Replaces the previous rejection-sampling scheme, which sometimes left
        # objects close enough to bump each other during grasping.
        self.quadrant_placement = bool(quadrant_placement)
        self.quadrant_noise = float(quadrant_noise)
        self.settle_steps = int(settle_steps)
        # Subset of objects "active" for the task. The rest are cleared out of
        # the scene at reset so they don't clutter the workspace or obstruct
        # grasping. Active ids are the first n in the right-first canonical
        # order: 4 → [Bread,Can,Milk,Cereal], 2 → [Bread,Can].
        self.n_active_objects = int(n_active_objects)
        self.active_object_ids = sorted(CANONICAL_RIGHT_FIRST[:self.n_active_objects])
        self.active_object_names = [OBJ_NAMES[i] for i in self.active_object_ids]
        self.seed_state_map = dict()
        self._seed = None

        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low, high=high, shape=low.shape, dtype=low.dtype
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low, high=high, shape=low.shape, dtype=low.dtype
        )

    def get_observation(self):
        raw_obs = self.env.get_observation()
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def _min_pairwise_distance(self):
        rs = self.env.env  # robosuite PickPlace env
        positions = [
            np.array(rs.sim.data.body_xpos[rs.obj_body_id[obj.name]][:2])
            for obj in rs.objects
        ]
        min_d = np.inf
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                if d < min_d:
                    min_d = d
        return min_d

    def _place_in_quadrants(self):
        """Set each object's xy to the center of its own bin1 quadrant plus
        ±quadrant_noise jitter, keep the random orientation that the env's
        placement_initializer chose, and let physics settle for a few steps."""
        # Start from the env's normal reset so the visual goal objects, joints,
        # qvel, etc. are all initialized in a consistent state.
        self.env.reset()

        rs = self.env.env  # robosuite PickPlace env
        bin1_x, bin1_y, bin1_z = rs.bin1_pos
        size_x, size_y, _ = rs.bin_size
        # Mirror the bin2 quadrant convention used by target_bin_placements:
        # object id 0 = (x-low, y-low), 1 = (x-high, y-low), 2 = (x-low, y-high), 3 = (x-high, y-high).
        offsets = [
            (-size_x / 4.0, -size_y / 4.0),
            ( size_x / 4.0, -size_y / 4.0),
            (-size_x / 4.0,  size_y / 4.0),
            ( size_x / 4.0,  size_y / 4.0),
        ]
        inactive_names = []
        for i, obj in enumerate(rs.objects):
            if i not in self.active_object_ids:
                inactive_names.append(obj.name)
                continue
            dx, dy = offsets[i]
            nx = np.random.uniform(-self.quadrant_noise, self.quadrant_noise)
            ny = np.random.uniform(-self.quadrant_noise, self.quadrant_noise)
            joint_name = obj.joints[0]
            qpos = rs.sim.data.get_joint_qpos(joint_name).copy()
            # qpos layout for a free joint: [x, y, z, qw, qx, qy, qz].
            # Drop z must keep the object's bottom above the bin floor; using
            # the body center directly would clip tall objects (e.g. Cereal's
            # 10 cm half-height) into the floor and leave them in a bad pose.
            bottom_offset_z = float(obj.bottom_offset[2])  # negative
            drop_z = bin1_z + abs(bottom_offset_z) + 0.01
            qpos[0] = bin1_x + dx + nx
            qpos[1] = bin1_y + dy + ny
            qpos[2] = drop_z
            qpos[3:7] = DEFAULT_OBJ_QUATS.get(obj.name, IDENTITY_QUAT)
            rs.sim.data.set_joint_qpos(joint_name, qpos)
            # Zero out velocities so they don't carry over from the prior step.
            rs.sim.data.set_joint_qvel(joint_name, np.zeros(6))

        # Move objects we don't want in the scene to a far-away parking spot
        # (robosuite's clear_objects helper). This keeps the obs object slots
        # populated but pushes the bodies off-table so they can't interfere.
        if inactive_names:
            rs.clear_objects(inactive_names)
        # Settle under gravity so objects come to rest on the bin floor.
        rs.sim.forward()
        for _ in range(self.settle_steps):
            rs.sim.step()

    def _do_random_reset(self):
        if self.quadrant_placement:
            self._place_in_quadrants()
        else:
            self.env.reset()

    def reset(self):
        if self.init_state is not None:
            self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                np.random.seed(seed)
                self._do_random_reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            self._do_random_reset()

        obs = self.get_observation()
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)

        # robosuite's _check_success gates "in_bin" on the gripper also being
        # far from the object — that suppresses placements while the gripper is
        # still hovering over them post-release. Compute a physics-only check
        # so the scripted collector / env_runner can see persistent placements.
        robosuite_env = self.env.env
        robosuite_env._check_success()
        physical_in_bins = np.zeros(len(robosuite_env.objects), dtype=np.float32)
        for i, obj in enumerate(robosuite_env.objects):
            obj_pos = robosuite_env.sim.data.body_xpos[robosuite_env.obj_body_id[obj.name]]
            physical_in_bins[i] = float(not robosuite_env.not_in_bin(obj_pos, i))

        n_in_bins = float(np.sum(physical_in_bins))
        info['objects_in_bins'] = physical_in_bins.copy()
        info['objects_in_bins_strict'] = robosuite_env.objects_in_bins.copy()
        reward = n_in_bins

        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, height=h, width=w,
                               camera_name=self.render_camera_name)
