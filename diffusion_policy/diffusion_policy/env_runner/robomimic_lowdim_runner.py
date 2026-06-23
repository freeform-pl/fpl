import os
import json
import wandb
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
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


def classify_peg_from_obs(obs):
    """
    Determine which peg the nut is on from the final observation.
    obs[:3] = nut_pos. Peg1 (left): [0.23, 0.1, 0.85], Peg2 (right): [0.23, -0.1, 0.85].
    Returns: 'left', 'right', or 'none'
    """
    nut_pos = obs[:3]
    table_z = 0.8
    if (abs(nut_pos[0] - 0.23) < 0.03 and abs(nut_pos[1] - 0.1) < 0.03 and nut_pos[2] < table_z + 0.05):
        return 'left'
    if (abs(nut_pos[0] - 0.23) < 0.03 and abs(nut_pos[1] - (-0.1)) < 0.03 and nut_pos[2] < table_z + 0.05):
        return 'right'
    return 'none'
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env


class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicLowdimWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
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

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        env = AsyncVectorEnv(env_fns)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_actions = [None] * n_inits
        all_final_obs = [None] * n_inits
        # Per-rollout obs sequence (one obs per multistep boundary) — used
        # by subclasses for post-eval per-axis reward logging. Memory cost is
        # small (T_max * obs_dim per rollout), so we always accumulate.
        all_obs_seqs = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            # track per-env actions for smoothness computation
            chunk_actions = [[] for _ in range(n_envs)]
            chunk_obs_seq = [[obs[i, -1].copy()] for i in range(n_envs)]
            chunk_done = [False] * n_envs

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[:,:self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # record per-step actions for smoothness
                for i in range(this_n_active_envs):
                    if not chunk_done[i]:
                        for t in range(action.shape[1]):
                            chunk_actions[i].append(action[i, t].copy())

                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                # Record one obs per multistep boundary for post-eval per-axis logging.
                for i in range(this_n_active_envs):
                    chunk_obs_seq[i].append(obs[i, -1].copy())
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            for i in range(this_n_active_envs):
                all_actions[start + i] = np.array(chunk_actions[i])
                all_final_obs[start + i] = obs[i, -1, :].copy()
                all_obs_seqs[start + i] = np.stack(chunk_obs_seq[i], axis=0)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        # per-prefix accumulators for reward metrics
        prefix_success = collections.defaultdict(list)
        prefix_speed = collections.defaultdict(list)
        prefix_smoothness = collections.defaultdict(list)
        prefix_throughput = collections.defaultdict(list)
        prefix_left_peg = collections.defaultdict(list)
        prefix_right_peg = collections.defaultdict(list)
        prefix_success_left = collections.defaultdict(list)
        prefix_success_right = collections.defaultdict(list)
        prefix_speed_left = collections.defaultdict(list)
        prefix_speed_right = collections.defaultdict(list)
        prefix_throughput_left = collections.defaultdict(list)
        prefix_throughput_right = collections.defaultdict(list)
        prefix_score_left = collections.defaultdict(list)
        prefix_score_right = collections.defaultdict(list)
        prefix_first_success_step = collections.defaultdict(list)
        prefix_first_success_step_left = collections.defaultdict(list)
        prefix_first_success_step_right = collections.defaultdict(list)

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            rewards = np.array(all_rewards[i])
            actions = all_actions[i]

            max_reward = np.max(rewards)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # --- success ---
            success = float(max_reward >= 1.0)
            prefix_success[prefix].append(success)

            # --- speed reward ---
            success_steps = np.where(rewards >= 1.0)[0]
            first_success_step = int(success_steps[0]) if len(success_steps) > 0 else len(rewards)
            speed_reward = 0.0
            if success:
                speed_reward = 1.0 - 0.9 * (first_success_step / self.max_steps)
            prefix_speed[prefix].append(speed_reward)
            if success:
                prefix_first_success_step[prefix].append(first_success_step)

            # --- throughput: success * 300 / time_to_first_success ---
            throughput = success * 300.0 / first_success_step if (success and first_success_step > 0) else 0.0
            prefix_throughput[prefix].append(throughput)

            # --- smoothness --- 0 on failure so the metric only credits
            # smooth trajectories that actually completed the task.
            smoothness = 1.0
            if len(actions) >= 3:
                jerk = np.diff(actions, n=3, axis=0)
                jerk_mag = np.mean(np.linalg.norm(jerk, axis=-1))
                smoothness = float(np.exp(-10.0 * jerk_mag))
            if not success:
                smoothness = 0.0
            prefix_smoothness[prefix].append(smoothness)

            # --- per-peg metrics ---
            final_obs = all_final_obs[i]
            if final_obs is not None:
                peg_status = classify_peg_from_obs(final_obs)
                prefix_left_peg[prefix].append(1.0 if peg_status == 'left' else 0.0)
                prefix_right_peg[prefix].append(1.0 if peg_status == 'right' else 0.0)

                prefix_success_left[prefix].append(1.0 if (success and peg_status == 'left') else 0.0)
                prefix_success_right[prefix].append(1.0 if (success and peg_status == 'right') else 0.0)
                if peg_status == 'left':
                    prefix_speed_left[prefix].append(speed_reward)
                    prefix_throughput_left[prefix].append(throughput)
                    score = (success + speed_reward + smoothness) / 3
                    prefix_score_left[prefix].append(score)
                    if success:
                        prefix_first_success_step_left[prefix].append(first_success_step)
                elif peg_status == 'right':
                    prefix_speed_right[prefix].append(speed_reward)
                    prefix_throughput_right[prefix].append(throughput)
                    score = (success + speed_reward + smoothness) / 3
                    prefix_score_right[prefix].append(score)
                    if success:
                        prefix_first_success_step_right[prefix].append(first_success_step)

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix in prefix_success.keys():
            mean_success = np.mean(prefix_success[prefix])
            mean_speed = np.mean(prefix_speed[prefix])
            mean_smoothness = np.mean(prefix_smoothness[prefix])
            log_data[prefix+'mean_score'] = (mean_success + mean_speed + mean_smoothness) / 3
            log_data[prefix+'mean_success'] = mean_success
            log_data[prefix+'mean_speed_reward'] = mean_speed
            log_data[prefix+'mean_smoothness'] = mean_smoothness
            log_data[prefix+'mean_throughput'] = np.mean(prefix_throughput[prefix])
            if prefix_first_success_step[prefix]:
                log_data[prefix+'mean_first_success_step'] = np.mean(prefix_first_success_step[prefix])
            if prefix_left_peg[prefix]:
                log_data[prefix+'left_peg_rate'] = np.mean(prefix_left_peg[prefix])
                log_data[prefix+'right_peg_rate'] = np.mean(prefix_right_peg[prefix])
            if prefix_success_left[prefix]:
                log_data[prefix+'mean_success_left'] = np.mean(prefix_success_left[prefix])
            if prefix_success_right[prefix]:
                log_data[prefix+'mean_success_right'] = np.mean(prefix_success_right[prefix])
            if prefix_speed_left[prefix]:
                log_data[prefix+'mean_speed_left'] = np.mean(prefix_speed_left[prefix])
            if prefix_speed_right[prefix]:
                log_data[prefix+'mean_speed_right'] = np.mean(prefix_speed_right[prefix])
            if prefix_throughput_left[prefix]:
                log_data[prefix+'mean_throughput_left'] = np.mean(prefix_throughput_left[prefix])
            if prefix_throughput_right[prefix]:
                log_data[prefix+'mean_throughput_right'] = np.mean(prefix_throughput_right[prefix])
            if prefix_score_left[prefix]:
                log_data[prefix+'mean_score_left'] = np.mean(prefix_score_left[prefix])
            if prefix_score_right[prefix]:
                log_data[prefix+'mean_score_right'] = np.mean(prefix_score_right[prefix])
            if prefix_first_success_step_left[prefix]:
                log_data[prefix+'mean_first_success_step_left'] = np.mean(prefix_first_success_step_left[prefix])
            if prefix_first_success_step_right[prefix]:
                log_data[prefix+'mean_first_success_step_right'] = np.mean(prefix_first_success_step_right[prefix])

        # Dump per-rollout first-success step lists. Each call to run() (one
        # per eval condition) appends a JSON line to per_rollout_steps.jsonl
        # in self.output_dir so downstream analysis can recover individual
        # rollout values (we otherwise only persist the means).
        try:
            tr = getattr(self, 'target_rewards', None)
            if hasattr(tr, 'tolist'):
                tr_serializable = [float(x) for x in tr.tolist()]
            elif tr is None:
                tr_serializable = None
            else:
                tr_serializable = [float(x) for x in list(tr)]

            prefixes_out = {}
            for prefix in prefix_success.keys():
                prefixes_out[prefix] = {
                    'first_success_step_left':  [int(x) for x in prefix_first_success_step_left[prefix]],
                    'first_success_step_right': [int(x) for x in prefix_first_success_step_right[prefix]],
                    'first_success_step':       [int(x) for x in prefix_first_success_step[prefix]],
                }

            out_dir = getattr(self, 'output_dir', None)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, 'per_rollout_steps.jsonl'), 'a') as _f:
                    _f.write(json.dumps({
                        'target_rewards': tr_serializable,
                        'prefixes': prefixes_out,
                    }) + '\n')
                print(f"[runner] per-rollout steps dumped under {out_dir}/per_rollout_steps.jsonl "
                      f"(target_rewards={tr_serializable})")
        except Exception as _e:
            print(f"[runner] failed to dump per-rollout steps: {_e}")

        # Subclass hook: per-axis post-eval logging. Default returns nothing.
        extra = self._extra_eval_log(all_obs_seqs, all_actions, all_rewards)
        if extra:
            log_data.update(extra)
        return log_data

    def _extra_eval_log(self, all_obs_seqs, all_actions, all_rewards):
        """Hook for subclasses to add per-axis reward logging on top of the
        default peg-style metrics. Override to return {prefix + key: mean_value}.
        Default: no extra logging.
        """
        return {}

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
