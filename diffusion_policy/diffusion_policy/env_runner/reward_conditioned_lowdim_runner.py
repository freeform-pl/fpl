"""
Reward-conditioned lowdim runner for the two-peg square nut assembly task.

Same as RobomimicLowdimRunner but:
- Uses RobomimicTwoPegLowdimWrapper for consistent nut randomization
- Appends target reward z-scores to obs before passing to the policy
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
import wandb
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.robomimic_lowdim_runner import RobomimicLowdimRunner, create_env
from diffusion_policy.env.robomimic.robomimic_twopeg_lowdim_wrapper import RobomimicTwoPegLowdimWrapper


def classify_peg_from_obs(obs):
    """
    Determine which peg the nut is on from the final observation.
    obs layout: object(14), robot0_eef_pos(3), robot0_eef_quat(4), robot0_gripper_qpos(2) = 23 dims
    object[:3] = nut_pos

    Peg positions (from robosuite NutAssemblySquare):
      peg1 (left):  [0.23,  0.1, 0.85]
      peg2 (right): [0.23, -0.1, 0.85]

    Returns: 'left', 'right', or 'none'
    """
    nut_pos = obs[:3]
    peg1_pos = np.array([0.23, 0.1, 0.85])
    peg2_pos = np.array([0.23, -0.1, 0.85])

    # Same threshold as robosuite on_peg: xy within 0.03, z below table + 0.05
    table_z = 0.8
    if (abs(nut_pos[0] - peg1_pos[0]) < 0.03 and
        abs(nut_pos[1] - peg1_pos[1]) < 0.03 and
        nut_pos[2] < table_z + 0.05):
        return 'left'

    if (abs(nut_pos[0] - peg2_pos[0]) < 0.03 and
        abs(nut_pos[1] - peg2_pos[1]) < 0.03 and
        nut_pos[2] < table_z + 0.05):
        return 'right'

    return 'none'


class RewardConditionedLowdimRunner(RobomimicLowdimRunner):
    """
    Extends RobomimicLowdimRunner to:
    - Use RobomimicTwoPegLowdimWrapper for consistent nut randomization
    - Append target reward values to observations
    """

    def __init__(self, num_reward_dims=3, target_rewards=None, use_twopeg_wrapper=False, **kwargs):
        """
        Args:
            num_reward_dims: number of reward dimensions
            target_rewards: optional list of K floats — overridden by workspace at eval
            use_twopeg_wrapper: if True, replace envs with TwoPeg wrapper (for scripted two-peg tasks)
            **kwargs: passed to RobomimicLowdimRunner
        """
        super().__init__(**kwargs)
        self.num_reward_dims = num_reward_dims
        if target_rewards is not None:
            self.target_rewards = np.array(target_rewards, dtype=np.float32)
        else:
            self.target_rewards = np.zeros(num_reward_dims, dtype=np.float32)

        if not use_twopeg_wrapper:
            # Standard task: keep parent's envs, only add obs augmentation
            return

        # Replace env_fns with two-peg wrapper version
        # Re-read env_meta from dataset_path (same as parent)
        import robomimic.utils.file_utils as FileUtils
        dataset_path = os.path.expanduser(kwargs['dataset_path'])
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        # Scripted data is always collected with absolute actions (control_delta=False)
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

        obs_keys = kwargs.get('obs_keys', ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])
        render_hw = kwargs.get('render_hw', (256, 256))
        render_camera_name = kwargs.get('render_camera_name', 'agentview')
        fps = kwargs.get('fps', 10)
        crf = kwargs.get('crf', 22)
        max_steps = kwargs.get('max_steps', 400)
        n_obs_steps = kwargs.get('n_obs_steps', 2)
        n_action_steps = kwargs.get('n_action_steps', 8)
        n_latency_steps = kwargs.get('n_latency_steps', 0)

        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        def env_fn():
            robomimic_env = create_env(env_meta=env_meta, obs_keys=obs_keys)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicTwoPegLowdimWrapper(
                        env=robomimic_env,
                        obs_keys=obs_keys,
                        init_state=None,
                        render_hw=render_hw,
                        render_camera_name=render_camera_name
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

        # Rebuild init functions with correct assert for TwoPeg wrapper
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
                    assert isinstance(env.env.env, RobomimicTwoPegLowdimWrapper)
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
                    assert isinstance(env.env.env, RobomimicTwoPegLowdimWrapper)
                    env.env.env.init_state = None
                    env.seed(seed)

                new_init_fn_dills.append(dill.dumps(init_fn))

        self.env_init_fn_dills = new_init_fn_dills
        self.env = AsyncVectorEnv(self.env_fns)

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_actions = [None] * n_inits
        all_final_obs = [None] * n_inits

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
            # Store final obs for peg classification (obs shape: n_envs, n_obs_steps, obs_dim)
            for i in range(this_n_active_envs):
                all_final_obs[start + i] = obs[i, -1, :].copy()

        # Compute metrics
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        prefix_success = collections.defaultdict(list)
        prefix_speed = collections.defaultdict(list)
        prefix_smoothness = collections.defaultdict(list)
        prefix_left_peg = collections.defaultdict(list)
        prefix_right_peg = collections.defaultdict(list)
        prefix_speed_left = collections.defaultdict(list)
        prefix_speed_right = collections.defaultdict(list)
        prefix_throughput = collections.defaultdict(list)
        prefix_success_left = collections.defaultdict(list)
        prefix_success_right = collections.defaultdict(list)
        prefix_score_left = collections.defaultdict(list)
        prefix_score_right = collections.defaultdict(list)
        prefix_throughput_left = collections.defaultdict(list)
        prefix_throughput_right = collections.defaultdict(list)

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

            throughput = max_reward / (first_success_step + 1)
            prefix_throughput[prefix].append(throughput)

            smoothness = 1.0
            if len(actions) >= 3:
                jerk = np.diff(actions, n=3, axis=0)
                jerk_mag = np.mean(np.linalg.norm(jerk, axis=-1))
                smoothness = float(np.exp(-10.0 * jerk_mag))
            prefix_smoothness[prefix].append(smoothness)

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{seed}'] = sim_video

            # Per-peg tracking from final obs
            score = (success + speed_reward + smoothness) / 3
            final_obs = all_final_obs[i]
            if final_obs is not None:
                peg_status = classify_peg_from_obs(final_obs)
                prefix_left_peg[prefix].append(1.0 if peg_status == 'left' else 0.0)
                prefix_right_peg[prefix].append(1.0 if peg_status == 'right' else 0.0)
                # success_left/right = successfully placed on that peg (over all trials)
                prefix_success_left[prefix].append(1.0 if (success and peg_status == 'left') else 0.0)
                prefix_success_right[prefix].append(1.0 if (success and peg_status == 'right') else 0.0)
                if peg_status == 'left':
                    prefix_speed_left[prefix].append(speed_reward)
                    prefix_score_left[prefix].append(score)
                    prefix_throughput_left[prefix].append(throughput)
                elif peg_status == 'right':
                    prefix_speed_right[prefix].append(speed_reward)
                    prefix_score_right[prefix].append(score)
                    prefix_throughput_right[prefix].append(throughput)

        for prefix in prefix_success.keys():
            mean_success = np.mean(prefix_success[prefix])
            mean_speed = np.mean(prefix_speed[prefix])
            mean_smoothness = np.mean(prefix_smoothness[prefix])
            log_data[prefix + 'mean_score'] = (mean_success + mean_speed + mean_smoothness) / 3
            log_data[prefix + 'mean_success'] = mean_success
            log_data[prefix + 'mean_speed_reward'] = mean_speed
            log_data[prefix + 'mean_smoothness'] = mean_smoothness
            log_data[prefix + 'mean_throughput'] = np.mean(prefix_throughput[prefix])
            if prefix in prefix_left_peg:
                log_data[prefix + 'left_peg_rate'] = np.mean(prefix_left_peg[prefix])
                log_data[prefix + 'right_peg_rate'] = np.mean(prefix_right_peg[prefix])
            if prefix_speed_left[prefix]:
                log_data[prefix + 'mean_speed_left'] = np.mean(prefix_speed_left[prefix])
            if prefix_speed_right[prefix]:
                log_data[prefix + 'mean_speed_right'] = np.mean(prefix_speed_right[prefix])
            if prefix_success_left[prefix]:
                log_data[prefix + 'mean_success_left'] = np.mean(prefix_success_left[prefix])
            if prefix_success_right[prefix]:
                log_data[prefix + 'mean_success_right'] = np.mean(prefix_success_right[prefix])
            if prefix_score_left[prefix]:
                log_data[prefix + 'mean_score_left'] = np.mean(prefix_score_left[prefix])
            if prefix_score_right[prefix]:
                log_data[prefix + 'mean_score_right'] = np.mean(prefix_score_right[prefix])
            if prefix_throughput_left[prefix]:
                log_data[prefix + 'mean_throughput_left'] = np.mean(prefix_throughput_left[prefix])
            if prefix_throughput_right[prefix]:
                log_data[prefix + 'mean_throughput_right'] = np.mean(prefix_throughput_right[prefix])

        return log_data

    def _augment_obs(self, obs):
        """Append target reward values to obs. obs: (B, T, D) -> (B, T, D+K)"""
        B, T, D = obs.shape
        reward_aug = np.broadcast_to(
            self.target_rewards, (B, T, self.num_reward_dims)).copy()
        return np.concatenate([obs, reward_aug], axis=-1)
