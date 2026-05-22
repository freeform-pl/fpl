"""
Reward-conditioned lowdim runner for the two-peg square nut assembly task.

Same as RobomimicLowdimRunner but:
- Uses RobomimicTwoPegLowdimWrapper for consistent nut randomization
- Appends target reward z-scores to obs before passing to the policy
"""

import os
import re
import sys
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
from diffusion_policy.env_runner.robomimic_lowdim_runner import RobomimicLowdimRunner, create_env, classify_peg_from_obs
from diffusion_policy.env.robomimic.robomimic_twopeg_lowdim_wrapper import RobomimicTwoPegLowdimWrapper

# Per-axis reward-function registry shared with the reward model.
_REWARD_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'reward_model')
if _REWARD_MODEL_DIR not in sys.path:
    sys.path.insert(0, _REWARD_MODEL_DIR)
from reward_functions import AXIS_FUNCTIONS as REWARD_AXIS_FUNCTIONS
from diffusion_policy.env.robomimic.robomimic_pickplace_lowdim_wrapper import RobomimicPickPlaceLowdimWrapper




class RewardConditionedLowdimRunner(RobomimicLowdimRunner):
    """
    Extends RobomimicLowdimRunner to:
    - Use RobomimicTwoPegLowdimWrapper for consistent nut randomization
    - Append target reward values to observations
    """

    @staticmethod
    def _parse_reward_axes(reward_axes):
        """Parse `reward_axes` into a flat list of base axis names recognised
        by reward_functions.AXIS_FUNCTIONS.

        Accepts:
          - None / empty → []
          - str: comma-separated, supports composite(a+b+c)
          - any iterable (list, tuple, omegaconf.ListConfig, etc.) of strings
        Composite(a+b+c) → [a, b, c]. Unknown axis names are silently dropped.
        """
        if reward_axes is None:
            return []
        if isinstance(reward_axes, str):
            entries = [a.strip() for a in reward_axes.split(',') if a.strip()]
        else:
            # Anything iterable that isn't a string (list, tuple, ListConfig).
            try:
                entries = [str(a).strip() for a in reward_axes]
            except TypeError:
                entries = []
        names = []
        for ax in entries:
            m = re.match(r'^composite\((.+)\)$', ax)
            if m:
                names.extend(s.strip() for s in m.group(1).split('+'))
            else:
                names.append(ax)
        seen, out = set(), []
        for n in names:
            if n in seen or n not in REWARD_AXIS_FUNCTIONS:
                continue
            seen.add(n)
            out.append(n)
        return out

    def __init__(self, num_reward_dims=3, target_rewards=None, use_twopeg_wrapper=False,
                 use_pickplace_wrapper=False,
                 discrete_conditioning=False, n_cond_bins=21,
                 reward_axes=None, **kwargs):
        """
        Args:
            num_reward_dims: number of reward dimensions
            target_rewards: optional list of K floats — overridden by workspace at eval
            use_twopeg_wrapper: if True, replace envs with TwoPeg wrapper (for scripted two-peg tasks)
            use_pickplace_wrapper: if True, replace envs with the PickPlace 4-obj
                wrapper. The wrapper-specific kwargs (quadrant_placement,
                quadrant_noise, settle_steps) are popped out of `kwargs` here
                so the parent runner doesn't see them.
            discrete_conditioning: if True, convert target_rewards to one-hot encoding
            n_cond_bins: number of bins for discrete conditioning (default 21 for [-1, 1] in 0.1 steps)
            **kwargs: passed to RobomimicLowdimRunner
        """
        # Strip PickPlace wrapper kwargs before passing kwargs upstream — the
        # parent RobomimicLowdimRunner doesn't accept them and would TypeError.
        pp_kwargs = {
            'quadrant_placement': kwargs.pop('quadrant_placement', True),
            'quadrant_noise': kwargs.pop('quadrant_noise', 0.03),
            'settle_steps': kwargs.pop('settle_steps', 40),
            'n_active_objects': kwargs.pop('n_active_objects', 4),
        }
        super().__init__(**kwargs)
        self.num_reward_dims = num_reward_dims
        self.discrete_conditioning = discrete_conditioning
        self.n_cond_bins = n_cond_bins
        self.use_pickplace_wrapper = use_pickplace_wrapper
        # Stored for metric thresholds (full_success, mean_score denominator).
        self.n_active_objects = int(pp_kwargs.get('n_active_objects', 4))
        # Per-axis logging — when non-empty, eval computes each named reward-
        # axis value on every rollout and logs `prefix + axis_name` to wandb.
        # The workspace overrides this directly from `scores.json` after
        # construction (so no Hydra wiring is needed for the common case).
        self.reward_axis_names = self._parse_reward_axes(reward_axes)
        if target_rewards is not None:
            tr = np.array(target_rewards, dtype=np.float32)
            # Truncate or pad to match num_reward_dims
            if len(tr) != num_reward_dims:
                self.target_rewards = np.zeros(num_reward_dims, dtype=np.float32)
                self.target_rewards[:min(len(tr), num_reward_dims)] = tr[:num_reward_dims]
            else:
                self.target_rewards = tr
        else:
            self.target_rewards = np.zeros(num_reward_dims, dtype=np.float32)

        if use_pickplace_wrapper:
            self._install_pickplace_wrapper(kwargs, pp_kwargs)
            return

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

                # Extract nut xy from the demo state and build a fresh init state
                # so reset_to works properly with the renderer-enabled env
                nut_xy = init_state[10:12].copy()

                def init_fn(env, nut_xy=nut_xy, enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    assert isinstance(env.env.env, RobomimicTwoPegLowdimWrapper)
                    fresh_state = env.env.env._make_init_state()
                    fresh_state[10:12] = nut_xy
                    env.env.env.init_state = fresh_state

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

    def _install_pickplace_wrapper(self, kwargs, pp_kwargs):
        """Replace env_fns + init_fns with the PickPlace 4-obj wrapper."""
        import robomimic.utils.file_utils as FileUtils
        dataset_path = os.path.expanduser(kwargs['dataset_path'])
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
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
                        quadrant_placement=pp_kwargs['quadrant_placement'],
                        quadrant_noise=pp_kwargs['quadrant_noise'],
                        settle_steps=pp_kwargs['settle_steps'],
                        n_active_objects=pp_kwargs['n_active_objects'],
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
        device = policy.device
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_actions = [None] * n_inits
        all_final_obs = [None] * n_inits
        # Per-rollout obs sequences (at multistep boundaries — last obs of each
        # n_action_steps chunk). Used to compute per-axis reward values for
        # wandb logging, e.g. test/order_reward, test/bread_placed, ...
        all_obs_seqs = [None] * n_inits

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
            chunk_obs_seq = [[obs[i, -1].copy()] for i in range(n_envs)]

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
                # Record one obs per multistep boundary for per-axis logging.
                for i in range(this_n_active_envs):
                    chunk_obs_seq[i].append(obs[i, -1].copy())
                done = np.all(done)
                past_action = action
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            for i in range(this_n_active_envs):
                all_actions[start + i] = np.array(chunk_actions[i])
                all_obs_seqs[start + i] = np.stack(chunk_obs_seq[i], axis=0)
            # Store final obs for peg classification (obs shape: n_envs, n_obs_steps, obs_dim)
            for i in range(this_n_active_envs):
                all_final_obs[start + i] = obs[i, -1, :].copy()

        # PickPlace branch: reward is the count of objects placed (0..4) at each
        # step, not a 0/1 success signal — so the peg-style metrics below would
        # all be nonsense. Compute task-appropriate ones and return early.
        if getattr(self, 'use_pickplace_wrapper', False):
            log_data = dict()
            prefix_max_placed = collections.defaultdict(list)
            prefix_final_placed = collections.defaultdict(list)
            prefix_partial = collections.defaultdict(list)   # >= 1 object
            prefix_full = collections.defaultdict(list)      # all 4 objects
            prefix_speed = collections.defaultdict(list)
            prefix_smoothness = collections.defaultdict(list)
            prefix_first_placement_step = collections.defaultdict(list)

            for i in range(n_inits):
                seed = self.env_seeds[i]
                prefix = self.env_prefixs[i]
                rewards = np.array(all_rewards[i])
                actions = all_actions[i]

                max_r = float(np.max(rewards)) if len(rewards) else 0.0
                final_r = float(rewards[-1]) if len(rewards) else 0.0
                prefix_max_placed[prefix].append(max_r)
                prefix_final_placed[prefix].append(final_r)
                prefix_partial[prefix].append(float(max_r >= 1.0))
                # Full success = all *active* objects placed, not all 4. Uses
                # self.n_active_objects (set from pp_kwargs) so pickplace_2
                # correctly logs full_success=1 when both Bread+Can land.
                prefix_full[prefix].append(float(max_r >= self.n_active_objects - 1e-6))

                placement_steps = np.where(rewards >= 1.0)[0]
                first_step = int(placement_steps[0]) if len(placement_steps) > 0 else len(rewards)
                speed_reward = 0.0
                if max_r >= 1.0:
                    speed_reward = 1.0 - 0.9 * (first_step / self.max_steps)
                    prefix_first_placement_step[prefix].append(first_step)
                prefix_speed[prefix].append(speed_reward)

                smoothness = 1.0
                if len(actions) >= 4:
                    jerk = np.diff(actions, n=3, axis=0)
                    jerk_mag = float(np.mean(np.linalg.norm(jerk, axis=-1)))
                    smoothness = float(np.exp(-10.0 * jerk_mag))
                # Gate by success: only smooth completed trajectories get credit.
                if max_r < 1.0:
                    smoothness = 0.0
                prefix_smoothness[prefix].append(smoothness)

                log_data[prefix + f'sim_max_reward_{seed}'] = max_r
                video_path = all_video_paths[i]
                if video_path is not None:
                    log_data[prefix + f'sim_video_{seed}'] = wandb.Video(video_path)

            # Per-axis reward values for each rollout — logged as prefix + axis
            # name (e.g. test/order_reward, test/bread_placed). Same axis
            # functions as the reward model uses, so the policy's eval values
            # are directly comparable to what the reward model is scoring.
            # Also computes a strict success criterion: order_reward = +1 AND
            # every per-object _drop axis > 0 (i.e. careful drop on every
            # active object). Generalises to pickplace_2 / pickplace_4 via the
            # configured reward axes.
            prefix_strict_success = collections.defaultdict(list)
            strict_drop_axes = [ax for ax in self.reward_axis_names
                                if ax.endswith('_drop') and ax != 'drop_reward']
            strict_axes_available = ('order_reward' in self.reward_axis_names
                                     and len(strict_drop_axes) > 0)
            if self.reward_axis_names:
                axis_accum = {ax: collections.defaultdict(list) for ax in self.reward_axis_names}
                for i in range(n_inits):
                    prefix = self.env_prefixs[i]
                    obs_seq = all_obs_seqs[i]
                    if obs_seq is None or len(obs_seq) == 0:
                        continue
                    act_seq = all_actions[i]
                    rollout_vals = {}
                    for ax in self.reward_axis_names:
                        fn = REWARD_AXIS_FUNCTIONS.get(ax)
                        if fn is None:
                            continue
                        try:
                            v = float(fn(obs_seq, actions=act_seq))
                        except Exception:
                            v = 0.0
                        axis_accum[ax][prefix].append(v)
                        rollout_vals[ax] = v
                    if strict_axes_available:
                        order_ok = rollout_vals.get('order_reward', 0.0) >= 1.0 - 1e-6
                        drops_ok = all(rollout_vals.get(ax, 0.0) > 0
                                       for ax in strict_drop_axes)
                        prefix_strict_success[prefix].append(
                            float(order_ok and drops_ok))
                for ax, by_prefix in axis_accum.items():
                    for prefix, vals in by_prefix.items():
                        if vals:
                            log_data[prefix + ax] = float(np.mean(vals))
            for prefix, vals in prefix_strict_success.items():
                if vals:
                    log_data[prefix + 'mean_strict_success'] = float(np.mean(vals))

            for prefix in prefix_max_placed.keys():
                mean_n = float(np.mean(prefix_max_placed[prefix]))
                mean_final = float(np.mean(prefix_final_placed[prefix]))
                mean_partial = float(np.mean(prefix_partial[prefix]))
                mean_full = float(np.mean(prefix_full[prefix]))
                mean_speed = float(np.mean(prefix_speed[prefix]))
                mean_smooth = float(np.mean(prefix_smoothness[prefix]))
                log_data[prefix + 'mean_n_placed'] = mean_n
                log_data[prefix + 'mean_n_placed_final'] = mean_final
                log_data[prefix + 'mean_success'] = mean_partial   # kept for ckpt-naming compatibility
                log_data[prefix + 'mean_partial_success'] = mean_partial
                log_data[prefix + 'mean_full_success'] = mean_full
                log_data[prefix + 'mean_speed_reward'] = mean_speed
                log_data[prefix + 'mean_smoothness'] = mean_smooth
                # Score = avg of (placed-fraction, speed, smoothness), each in [0, 1].
                # Divides mean_n by n_active_objects so the score is comparable
                # across variants (pickplace_2: /2, pickplace_4: /4).
                placed_frac = mean_n / max(self.n_active_objects, 1)
                log_data[prefix + 'mean_score'] = (placed_frac + mean_speed + mean_smooth) / 3
                if prefix_first_placement_step[prefix]:
                    log_data[prefix + 'mean_first_placement_step'] = float(np.mean(prefix_first_placement_step[prefix]))
            return log_data

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
        prefix_first_success_step = collections.defaultdict(list)
        prefix_first_success_step_left = collections.defaultdict(list)
        prefix_first_success_step_right = collections.defaultdict(list)

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
            if success:
                prefix_first_success_step[prefix].append(first_success_step)

            throughput = success / ((first_success_step + 1) / self.max_steps) if success else 0.0
            prefix_throughput[prefix].append(throughput)

            smoothness = 1.0
            if len(actions) >= 3:
                jerk = np.diff(actions, n=3, axis=0)
                jerk_mag = np.mean(np.linalg.norm(jerk, axis=-1))
                smoothness = float(np.exp(-10.0 * jerk_mag))
            # Gate by success — failed rollouts get smoothness=0.
            if not success:
                smoothness = 0.0
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
                    if success:
                        prefix_first_success_step_left[prefix].append(first_success_step)
                elif peg_status == 'right':
                    prefix_speed_right[prefix].append(speed_reward)
                    prefix_score_right[prefix].append(score)
                    prefix_throughput_right[prefix].append(throughput)
                    if success:
                        prefix_first_success_step_right[prefix].append(first_success_step)

        for prefix in prefix_success.keys():
            mean_success = np.mean(prefix_success[prefix])
            mean_speed = np.mean(prefix_speed[prefix])
            mean_smoothness = np.mean(prefix_smoothness[prefix])
            log_data[prefix + 'mean_score'] = (mean_success + mean_speed + mean_smoothness) / 3
            log_data[prefix + 'mean_success'] = mean_success
            log_data[prefix + 'mean_speed_reward'] = mean_speed
            log_data[prefix + 'mean_smoothness'] = mean_smoothness
            log_data[prefix + 'mean_throughput'] = np.mean(prefix_throughput[prefix])
            if prefix_first_success_step[prefix]:
                log_data[prefix + 'mean_first_success_step'] = np.mean(prefix_first_success_step[prefix])
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
            if prefix_first_success_step_left[prefix]:
                log_data[prefix + 'mean_first_success_step_left'] = np.mean(prefix_first_success_step_left[prefix])
            if prefix_first_success_step_right[prefix]:
                log_data[prefix + 'mean_first_success_step_right'] = np.mean(prefix_first_success_step_right[prefix])

        return log_data

    def _augment_obs(self, obs):
        """Append target reward conditioning to obs. obs: (B, T, D) -> (B, T, D+C)"""
        B, T, D = obs.shape
        if self.discrete_conditioning:
            from diffusion_policy.dataset.reward_conditioned_lowdim_dataset import scores_to_onehot
            cond_vec = scores_to_onehot(self.target_rewards, self.n_cond_bins)
            cond_dim = len(cond_vec)
        else:
            cond_vec = self.target_rewards
            cond_dim = self.num_reward_dims
        reward_aug = np.broadcast_to(cond_vec, (B, T, cond_dim)).copy()
        return np.concatenate([obs, reward_aug], axis=-1)
