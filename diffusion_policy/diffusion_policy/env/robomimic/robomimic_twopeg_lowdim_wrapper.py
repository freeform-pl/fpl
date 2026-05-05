"""
Lowdim wrapper for the two-peg square nut assembly task.

Randomizes nut position on reset (matching scripted data collection distribution)
and checks success on either peg.
"""

from typing import List, Optional
import numpy as np
import gym
from gym.spaces import Box
from robomimic.envs.env_robosuite import EnvRobosuite


# Base reset state template — robot in home position, nut xy overwritten per episode
_BASE_RESET_STATE = np.array([
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


class RobomimicTwoPegLowdimWrapper(gym.Env):
    """
    Wrapper for the square nut assembly task with two pegs.

    On reset, randomizes the nut's xy position in [-0.2, -0.2] to [0, 0.2],
    matching the distribution used by collect_initial_scripted_rollouts.py.
    Checks success on either peg (left or right).
    """

    def __init__(self,
            env: EnvRobosuite,
            obs_keys: List[str] = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos'],
            init_state: Optional[np.ndarray] = None,
            render_hw=(256, 256),
            render_camera_name='agentview'
        ):
        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = dict()
        self._seed = None

        # setup spaces
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

    def _make_init_state(self, rng=None):
        """Generate a reset state with randomized nut position."""
        if rng is None:
            rng = np.random
        state = _BASE_RESET_STATE.copy()
        nut_pos = rng.uniform([-0.2, -0.2], [0, 0.2], size=(2,))
        state[10:12] = nut_pos
        return state

    def reset(self):
        if self.init_state is not None:
            self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                rng = np.random.RandomState(seed)
                state = self._make_init_state(rng)
                self.env.reset_to({'states': state})
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            state = self._make_init_state()
            self.env.reset_to({'states': state})

        obs = self.get_observation()
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)

        # Check success on either peg
        robosuite_env = self.env.env
        nut_pos = robosuite_env.sim.data.body_xpos[robosuite_env.obj_body_id['SquareNut']]
        on_peg = robosuite_env.on_peg(nut_pos, 0) or robosuite_env.on_peg(nut_pos, 1)
        if on_peg:
            reward = 1.0

        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, height=h, width=w,
                               camera_name=self.render_camera_name)
