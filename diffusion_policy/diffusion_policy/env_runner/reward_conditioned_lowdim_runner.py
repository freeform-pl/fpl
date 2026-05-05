"""
Reward-conditioned lowdim runner.

Same as RobomimicLowdimRunner but appends target reward z-scores to obs
before passing to the policy.
"""

import os
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.robomimic_lowdim_runner import RobomimicLowdimRunner


class RewardConditionedLowdimRunner(RobomimicLowdimRunner):
    """
    Extends RobomimicLowdimRunner to append target reward values to observations.
    """

    def __init__(self, target_rewards, num_reward_dims=3, **kwargs):
        """
        Args:
            target_rewards: list of K floats — z-score reward values to condition on
            num_reward_dims: number of reward dimensions
            **kwargs: passed to RobomimicLowdimRunner
        """
        super().__init__(**kwargs)
        self.target_rewards = np.array(target_rewards, dtype=np.float32)
        self.num_reward_dims = num_reward_dims
        assert len(self.target_rewards) == num_reward_dims

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_actions = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()

            chunk_actions = [[] for _ in range(n_envs)]

            pbar = tqdm.tqdm(total=self.max_steps,
                desc=f"Eval conditioned chunk {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                # Augment obs with reward conditioning
                # obs shape: (n_envs, n_obs_steps_total, obs_dim)
                obs_for_policy = self._augment_obs(obs[:, :self.n_obs_steps])

                np_obs_dict = {
                    'obs': obs_for_policy.astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :, -(self.n_obs_steps - 1):].astype(np.float32)

                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'][:, self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                for i in range(this_n_active_envs):
                    for t in range(action.shape[1]):
                        chunk_actions[i].append(action[i, t].copy())

                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            for i in range(this_n_active_envs):
                all_actions[start + i] = np.array(chunk_actions[i])

        # Compute metrics
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        prefix_success = collections.defaultdict(list)
        prefix_speed = collections.defaultdict(list)
        prefix_smoothness = collections.defaultdict(list)

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            rewards = np.array(all_rewards[i])
            actions = all_actions[i]

            max_reward = np.max(rewards)
            max_rewards[prefix].append(max_reward)

            success = float(max_reward >= 1.0)
            prefix_success[prefix].append(success)

            success_steps = np.where(rewards >= 1.0)[0]
            first_success_step = int(success_steps[0]) if len(success_steps) > 0 else len(rewards)
            speed_reward = 0.0
            if success:
                speed_reward = 1.0 - 0.9 * (first_success_step / self.max_steps)
            prefix_speed[prefix].append(speed_reward)

            smoothness = 1.0
            if len(actions) >= 3:
                jerk = np.diff(actions, n=3, axis=0)
                jerk_mag = np.mean(np.linalg.norm(jerk, axis=-1))
                smoothness = float(np.exp(-10.0 * jerk_mag))
            prefix_smoothness[prefix].append(smoothness)

        for prefix, value in max_rewards.items():
            log_data[prefix + 'mean_score'] = np.mean(value)
            log_data[prefix + 'mean_success'] = np.mean(prefix_success[prefix])
            log_data[prefix + 'mean_speed_reward'] = np.mean(prefix_speed[prefix])
            log_data[prefix + 'mean_smoothness'] = np.mean(prefix_smoothness[prefix])

        return log_data

    def _augment_obs(self, obs):
        """Append target reward values to obs. obs: (B, T, D) -> (B, T, D+K)"""
        B, T, D = obs.shape
        reward_aug = np.broadcast_to(
            self.target_rewards, (B, T, self.num_reward_dims)).copy()
        return np.concatenate([obs, reward_aug], axis=-1)
