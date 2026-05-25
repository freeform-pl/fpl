"""
Lowdim runner for the two-peg square nut assembly task.

Same as RobomimicLowdimRunner but uses RobomimicTwoPegLowdimWrapper
for consistent nut randomization and two-peg success checking.
"""

import os
import numpy as np
import pathlib
import h5py
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.env_runner.robomimic_lowdim_runner import RobomimicLowdimRunner, create_env
from diffusion_policy.env.robomimic.robomimic_twopeg_lowdim_wrapper import RobomimicTwoPegLowdimWrapper


class TwoPegLowdimRunner(RobomimicLowdimRunner):
    """
    Same as RobomimicLowdimRunner but uses RobomimicTwoPegLowdimWrapper
    so that nut randomization and success checking match the scripted data collection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Rebuild env_fns with TwoPeg wrapper
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

    def _extra_eval_log(self, all_obs_seqs, all_actions, all_rewards):
        """Per-axis reward logging for the slow_fast / twopeg task. Computes
        each axis (success / speed_reward / smoothness / peg_reward /
        peg_reward_raw) on every rollout's obs sequence and logs the mean per
        prefix. Lets every twopeg-derived baseline (base policy, AWR,
        demo_success, demo_only) be compared on the same axes as RHP."""
        from reward_model.reward_functions import (
            compute_pickplace_eval_log, get_slow_fast_logging_axes)
        n_inits = len(self.env_init_fn_dills)
        prefixes = [self.env_prefixs[i] for i in range(n_inits)]
        return compute_pickplace_eval_log(
            obs_seqs=all_obs_seqs,
            action_seqs=all_actions,
            prefixes=prefixes,
            n_active_objects=4,  # unused for slow_fast (no strict_success)
            axis_names=get_slow_fast_logging_axes(),
        )
