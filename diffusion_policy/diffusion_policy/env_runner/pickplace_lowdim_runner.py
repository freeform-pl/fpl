"""
Lowdim runner for the 4-object PickPlace task.

Mirrors TwoPegLowdimRunner, but swaps in the PickPlace wrapper and ignores the
peg-specific metrics from the base class (they evaluate to 'none' on PickPlace
observations and the resulting fields stay empty).
"""

import os
import collections
import math
import numpy as np
import pathlib
import h5py
import dill
import tqdm
import torch
import wandb
import wandb.sdk.data_types.video as wv

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.robomimic_lowdim_runner import RobomimicLowdimRunner, create_env
from diffusion_policy.env.robomimic.robomimic_pickplace_lowdim_wrapper import RobomimicPickPlaceLowdimWrapper
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class PickPlaceLowdimRunner(RobomimicLowdimRunner):
    """Runner for the PickPlace 4-object benchmark."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        import robomimic.utils.file_utils as FileUtils
        dataset_path = os.path.expanduser(kwargs['dataset_path'])
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        # Scripted data is collected with absolute actions.
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

        obs_keys = kwargs.get('obs_keys', ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])
        render_hw = kwargs.get('render_hw', (256, 256))
        render_camera_name = kwargs.get('render_camera_name', 'agentview')
        fps = kwargs.get('fps', 10)
        crf = kwargs.get('crf', 22)
        max_steps = kwargs.get('max_steps', 1200)
        n_obs_steps = kwargs.get('n_obs_steps', 2)
        n_action_steps = kwargs.get('n_action_steps', 8)
        n_latency_steps = kwargs.get('n_latency_steps', 0)
        quadrant_placement = kwargs.get('quadrant_placement', True)
        quadrant_noise = kwargs.get('quadrant_noise', 0.03)
        settle_steps = kwargs.get('settle_steps', 40)
        n_active_objects = kwargs.get('n_active_objects', 4)
        self.n_active_objects = int(n_active_objects)

        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        def env_fn():
            robomimic_env = create_env(env_meta=env_meta, obs_keys=obs_keys)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicPickPlaceLowdimWrapper(
                        env=robomimic_env,
                        obs_keys=obs_keys,
                        init_state=None,
                        render_hw=render_hw,
                        render_camera_name=render_camera_name,
                        quadrant_placement=quadrant_placement,
                        quadrant_noise=quadrant_noise,
                        settle_steps=settle_steps,
                        n_active_objects=n_active_objects,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps, codec='h264', input_pix_fmt='rgb24',
                        crf=crf, thread_type='FRAME', thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        n_envs = len(self.env_fns)
        self.env_fns = [env_fn] * n_envs

        output_dir = self.output_dir
        new_init_fn_dills = []
        for seed, prefix in zip(self.env_seeds, self.env_prefixs):
            if prefix == 'train/':
                train_idx = seed
                enable_render = train_idx < kwargs.get('n_train_vis', 3)
                with h5py.File(dataset_path, 'r') as f:
                    init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    assert isinstance(env.env.env, RobomimicPickPlaceLowdimWrapper)
                    env.env.env.init_state = init_state

                new_init_fn_dills.append(dill.dumps(init_fn))
            else:
                enable_render = sum(1 for s, p in zip(self.env_seeds, self.env_prefixs)
                                    if p == 'test/' and s < seed) < kwargs.get('n_test_vis', 6)

                def init_fn(env, seed=seed, enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    assert isinstance(env.env.env, RobomimicPickPlaceLowdimWrapper)
                    env.env.env.init_state = None
                    env.seed(seed)

                new_init_fn_dills.append(dill.dumps(init_fn))

        self.env_init_fn_dills = new_init_fn_dills
        self.env = AsyncVectorEnv(self.env_fns)

    def run(self, policy: BaseLowdimPolicy):
        """
        Overrides RobomimicLowdimRunner.run() to use PickPlace-appropriate
        metrics — reward is in [0, 4] (count of objects placed). We expose:
          - mean_n_placed: average #objects placed (0..4)
          - mean_partial_success: any object placed
          - mean_full_success: all 4 placed
          - mean_speed_reward / mean_smoothness / mean_score
        """
        device = policy.device
        dtype = policy.dtype
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

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            past_action = None
            policy.reset()

            chunk_actions = [[] for _ in range(n_envs)]
            chunk_done = [False] * n_envs

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps,
                desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                np_obs_dict = {'obs': obs[:, :self.n_obs_steps].astype(np.float32)}
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[:, -(self.n_obs_steps - 1):].astype(np.float32)

                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'][:, self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                for i in range(this_n_active_envs):
                    if not chunk_done[i]:
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

            this_local_slice = slice(0, this_n_active_envs)
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            for i in range(this_n_active_envs):
                all_actions[start + i] = np.array(chunk_actions[i])

        log_data = dict()
        prefix_n_placed = collections.defaultdict(list)
        prefix_partial_success = collections.defaultdict(list)
        prefix_full_success = collections.defaultdict(list)
        prefix_speed = collections.defaultdict(list)
        prefix_smoothness = collections.defaultdict(list)
        prefix_first_placement_step = collections.defaultdict(list)

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            rewards = np.array(all_rewards[i])
            actions = all_actions[i]

            max_reward = float(np.max(rewards))
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            n_placed = max_reward  # already in [0, n_active_objects]
            partial = float(n_placed >= 1.0)
            full = float(n_placed >= self.n_active_objects - 1e-6)
            prefix_n_placed[prefix].append(n_placed)
            prefix_partial_success[prefix].append(partial)
            prefix_full_success[prefix].append(full)

            placement_steps = np.where(rewards >= 1.0)[0]
            first_step = int(placement_steps[0]) if len(placement_steps) > 0 else len(rewards)
            speed_reward = 0.0
            if partial:
                speed_reward = 1.0 - 0.9 * (first_step / self.max_steps)
                prefix_first_placement_step[prefix].append(first_step)
            prefix_speed[prefix].append(speed_reward)

            smoothness = 1.0
            if len(actions) >= 4:
                jerk = np.diff(actions, n=3, axis=0)
                jerk_mag = float(np.mean(np.linalg.norm(jerk, axis=-1)))
                smoothness = float(np.exp(-10.0 * jerk_mag))
            # Gate by success — failed rollouts get smoothness=0 so the metric
            # only rewards trajectories that actually placed something.
            if not partial:
                smoothness = 0.0
            prefix_smoothness[prefix].append(smoothness)

            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[prefix + f'sim_video_{seed}'] = wandb.Video(video_path)

        for prefix in prefix_n_placed.keys():
            mean_n = np.mean(prefix_n_placed[prefix])
            mean_partial = np.mean(prefix_partial_success[prefix])
            mean_full = np.mean(prefix_full_success[prefix])
            mean_speed = np.mean(prefix_speed[prefix])
            mean_smooth = np.mean(prefix_smoothness[prefix])
            log_data[prefix + 'mean_n_placed'] = mean_n
            log_data[prefix + 'mean_success'] = mean_partial   # for compatibility (>=1 object placed)
            log_data[prefix + 'mean_partial_success'] = mean_partial
            log_data[prefix + 'mean_full_success'] = mean_full
            log_data[prefix + 'mean_speed_reward'] = mean_speed
            log_data[prefix + 'mean_smoothness'] = mean_smooth
            placed_frac = mean_n / max(self.n_active_objects, 1)
            log_data[prefix + 'mean_score'] = (placed_frac + mean_speed + mean_smooth) / 3
            if prefix_first_placement_step[prefix]:
                log_data[prefix + 'mean_first_placement_step'] = np.mean(prefix_first_placement_step[prefix])

        return log_data
