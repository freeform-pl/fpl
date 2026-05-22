"""
Reward-conditioned lowdim dataset.

Loads rollout data (.npz) + original demo HDF5, augments obs with z-score reward values.
The obs becomes (T, obs_dim + num_reward_dims). Reward dims use identity normalization
so z-score values pass through unchanged.
"""

from typing import Dict, List
import os
import torch
import numpy as np
import json
import h5py
from tqdm import tqdm
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
    array_to_stats
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )


def score_to_onehot(score, n_bins=21):
    """Convert a score in [-1, 1] to a one-hot vector of length n_bins.
    Bins are centered at -1.0, -0.9, ..., 0.9, 1.0."""
    bucket = int(round((score + 1.0) * (n_bins - 1) / 2.0))
    bucket = max(0, min(n_bins - 1, bucket))
    onehot = np.zeros(n_bins, dtype=np.float32)
    onehot[bucket] = 1.0
    return onehot


def scores_to_onehot(scores, n_bins=21):
    """Convert (K,) scores to (K * n_bins,) concatenated one-hot vectors."""
    parts = [score_to_onehot(s, n_bins) for s in scores]
    return np.concatenate(parts)


class RewardConditionedLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            rollout_data_path: str,
            scores_path: str,
            demo_hdf5_path: str,
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            obs_keys: List[str] = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos'],
            num_reward_dims: int = None,  # inferred from scores.json if not set
            discrete_conditioning: bool = False,
            n_cond_bins: int = 21,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: int = None,
            n_active_objects: int = 4,  # PickPlace: zero out inactive object slots in obs
            **kwargs,  # absorb extra keys from base task config (dataset_path, abs_action, etc.)
        ):
        self.n_active_objects = int(n_active_objects)
        self.obs_keys = list(obs_keys)
        obs_keys = list(obs_keys)

        # Load scores (z-score normalized)
        with open(scores_path, 'r') as f:
            scores_data = json.load(f)

        rollout_z = np.array(scores_data['rollout_scores_zscore'], dtype=np.float32)  # (N_rollout, K)
        demo_z = np.array(scores_data['demo_scores_zscore'], dtype=np.float32)  # (N_demo, K)

        # Discretize scores into 0.1 buckets: -1.0, -0.9, ..., 0.9, 1.0
        if len(rollout_z) > 0:
            rollout_z = np.round(rollout_z * 10) / 10
            rollout_z = np.clip(rollout_z, -1.0, 1.0)
        if len(demo_z) > 0:
            demo_z = np.round(demo_z * 10) / 10
            demo_z = np.clip(demo_z, -1.0, 1.0)

        # Infer num_reward_dims from scores file
        if num_reward_dims is None:
            num_reward_dims = rollout_z.shape[1]
        conditioning_dims = num_reward_dims * n_cond_bins if discrete_conditioning else num_reward_dims
        print(f"Reward dims: {num_reward_dims} ({scores_data.get('reward_names', [])}), "
              f"discrete={discrete_conditioning}, conditioning_dims={conditioning_dims}")

        replay_buffer = ReplayBuffer.create_empty_numpy()

        # Load rollout data (skip if rollout_data_path is "none" — demos-only mode)
        # Supports comma-separated list of npz files
        rollout_paths = [p.strip() for p in rollout_data_path.split(',')
                         if p.strip() and p.strip() != 'none'] if rollout_data_path else []
        has_rollouts = len(rollout_paths) > 0
        rollout_action_dim = None
        if has_rollouts:
            # Load and concatenate all rollout files
            all_rollout_obs, all_rollout_actions, all_rollout_lengths = [], [], []
            total_rollout_eps = 0
            for rpath in rollout_paths:
                data = np.load(rpath)
                n = len(data['episode_lengths'])
                all_rollout_obs.append((data['obs'][:n], data['episode_lengths']))
                all_rollout_actions.append(data['actions'][:n])
                all_rollout_lengths.append(data['episode_lengths'])
                total_rollout_eps += n
                print(f"  Loaded {n} rollouts from {rpath}")
            if total_rollout_eps != len(rollout_z):
                raise ValueError(
                    f"Rollout count mismatch between rollouts.npz files "
                    f"({total_rollout_eps} total episodes across {len(rollout_paths)} "
                    f"file(s)) and scores.json rollout_scores_zscore "
                    f"({len(rollout_z)}). Re-run Phase 3 (reward model training) "
                    f"on the current rollout set so scores.json is fresh."
                )

            # Process each file's episodes with the corresponding z-scores
            rollout_offset = 0
            for file_idx, rpath in enumerate(rollout_paths):
                r_obs = all_rollout_obs[file_idx][0]
                r_lengths = all_rollout_lengths[file_idx]
                r_actions = all_rollout_actions[file_idx]
                n_eps = len(r_lengths)
                rollout_action_dim = r_actions.shape[-1]

                for i in tqdm(range(n_eps), desc=f"Loading rollouts from {os.path.basename(rpath)}"):
                    L_obs = int(r_lengths[i])
                    L_act = min(L_obs - 1, r_actions.shape[1])
                    if L_act <= 0:
                        continue

                    obs_i = r_obs[i, :L_obs].astype(np.float32)
                    act_i = r_actions[i, :L_act].astype(np.float32)

                    # Augment obs with reward conditioning (same for all timesteps)
                    reward_vals = rollout_z[rollout_offset + i]  # (K,)
                    if discrete_conditioning:
                        cond_vec = scores_to_onehot(reward_vals, n_cond_bins)
                    else:
                        cond_vec = reward_vals
                    reward_aug = np.broadcast_to(cond_vec, (L_obs, conditioning_dims)).copy()
                    obs_aug = np.concatenate([obs_i, reward_aug], axis=-1)

                    L = min(len(obs_aug), L_act)
                    episode = {
                        'obs': obs_aug[:L],
                        'action': act_i[:L],
                    }
                    replay_buffer.add_episode(episode)
                rollout_offset += n_eps
            print(f"Total: {rollout_offset} rollout episodes from {len(rollout_paths)} file(s)")
        else:
            print("No rollout data — loading demos only.")

        # Load original demo data
        # Always convert 7D axis_angle demos to 10D rot6d (policy action space)
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep='rotation_6d')

        with h5py.File(demo_hdf5_path, 'r') as f:
            demos = f['data']
            n_hdf5_demos = len([k for k in demos.keys() if k.startswith('demo_')])
            n_score_demos = len(demo_z)
            if n_hdf5_demos != n_score_demos:
                raise ValueError(
                    f"Demo count mismatch between demos.hdf5 ({n_hdf5_demos}) "
                    f"and scores.json ({n_score_demos}). The reward model was "
                    f"trained on a different demo set than is being loaded here. "
                    f"Re-run Phase 3 (reward model training) on the current "
                    f"demos.hdf5 ({demo_hdf5_path}) so scores.json is fresh, "
                    f"or point scores_path={scores_path} to a matching file."
                )
            n_demos = n_hdf5_demos
            for i in tqdm(range(n_demos), desc="Loading demo episodes"):
                demo = demos[f'demo_{i}']
                obs_parts = [demo['obs'][key][:].astype(np.float32) for key in obs_keys]
                obs_i = np.concatenate(obs_parts, axis=-1)
                act_i = demo['actions'][:].astype(np.float32)

                # Convert 7D axis_angle -> 10D rot6d
                if act_i.shape[-1] == 7:
                    pos = act_i[:, :3]
                    rot = act_i[:, 3:6]
                    gripper = act_i[:, 6:]
                    rot6d = rotation_transformer.forward(rot)
                    act_i = np.concatenate([pos, rot6d, gripper], axis=-1).astype(np.float32)

                L = min(len(obs_i), len(act_i))
                obs_i = obs_i[:L]
                act_i = act_i[:L]

                # Augment obs with demo reward conditioning
                reward_vals = demo_z[i]
                if discrete_conditioning:
                    cond_vec = scores_to_onehot(reward_vals, n_cond_bins)
                else:
                    cond_vec = reward_vals
                reward_aug = np.broadcast_to(cond_vec, (L, conditioning_dims)).copy()
                obs_aug = np.concatenate([obs_i, reward_aug], axis=-1)

                episode = {
                    'obs': obs_aug,
                    'action': act_i,
                }
                replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.num_reward_dims = num_reward_dims
        self.conditioning_dims = conditioning_dims

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Action: per-dim range normalizer. Actions are 10D
        # [pos(3) + rot6d(6) + gripper(1)]. The position dims are in
        # world-frame meters (~0.1-0.3) so an identity normalizer made the
        # MSE loss heavily favor position errors over rot6d/gripper errors.
        # Per-dim range scaling balances the loss across action components.
        action_stat = array_to_stats(self.replay_buffer['action'])
        normalizer['action'] = get_range_normalizer_from_stat(action_stat)

        # Obs: normalize the base obs dims with per-dim range scaling, and
        # leave the appended conditioning dims with an identity transform
        # (they're already z-scores). The previous implementation used a
        # SINGLE global max-abs scalar for the whole base obs vector — which
        # let off-table parked objects at xy≈10 dominate `max_abs` and squash
        # the active-object position dims to ~0.01, creating a large
        # train-test gap. Per-dim min/max scaling restores per-dim signal.
        all_obs = self.replay_buffer['obs']  # (total_steps, D+C)
        D = all_obs.shape[-1] - self.conditioning_dims

        obs_base = all_obs[:, :D]
        base_stat = array_to_stats(obs_base)
        base_normalizer = get_range_normalizer_from_stat(base_stat)
        base_sd = base_normalizer.state_dict()
        base_scale = base_sd['params_dict.scale'].cpu().numpy().copy()
        base_offset = base_sd['params_dict.offset'].cpu().numpy().copy()

        # Explicitly mask the four high-scale active-object quat dims that
        # otherwise amplify tiny physics noise enormously (per training-set
        # range → scale: Can.q.z=0.0012→1693, eef_q.w=0.014→144,
        # Can.q.y=0.022→91, Bread.q.z=0.040→50). These all have near-zero
        # signal across demos so killing them is a safe denoise.
        # Indices match the obs layout for PickPlace (4 obj × 14 + eef +
        # gripper). Only applied if obs has the PickPlace 65-dim base.
        HIGH_SCALE_MASK_DIMS = {
            48,   # Can.q.z
            59,   # eef_q.w
            47,   # Can.q.y
            20,   # Bread.q.z
        }
        if D >= 56 and len(self.obs_keys) > 0 and self.obs_keys[0] == 'object':
            for d in HIGH_SCALE_MASK_DIMS:
                if 0 <= d < D:
                    base_scale[d] = 0.0
                    base_offset[d] = 0.0
            print(f"[normalizer] explicit-mask {sorted(HIGH_SCALE_MASK_DIMS)} "
                  f"→ scale=0, offset=0 (Can.q.z, eef_q.w, Can.q.y, Bread.q.z)")

        # PickPlace specific: when fewer than 4 objects are active in the
        # scene, the inactive object_ids are determined by the right-first
        # canonical order. Their 14-dim slots in `object` are populated by
        # `clear_objects` (parked off-table) and drift over time — which
        # makes the normalized obs differ between train and eval. Force
        # those slots to (scale=0, offset=0) so they always normalize to 0
        # regardless of input. Inferred from obs schema: PickPlace `object`
        # is 56 dims = 4 × 14, and the obs starts with the 'object' key.
        # Only apply when obs_keys begins with 'object' AND base dims ≥ 56.
        if (D >= 56 and self.n_active_objects < 4
                and len(self.obs_keys) > 0 and self.obs_keys[0] == 'object'):
            # Right-first canonical order: Bread(1), Can(3), Milk(0), Cereal(2)
            canonical_order = [1, 3, 0, 2]
            active_ids = set(canonical_order[:self.n_active_objects])
            inactive_ids = [i for i in range(4) if i not in active_ids]
            for obj_id in inactive_ids:
                base_scale[obj_id * 14: obj_id * 14 + 14] = 0.0
                base_offset[obj_id * 14: obj_id * 14 + 14] = 0.0
            print(f"[normalizer] n_active_objects={self.n_active_objects} → "
                  f"zeroing obs slots for inactive object ids {inactive_ids} "
                  f"(dims {[(i*14, i*14+14) for i in inactive_ids]})")

        # Reward dims: identity (z-scores pass through).
        reward_scale = np.ones(self.conditioning_dims, dtype=np.float32)
        reward_offset = np.zeros(self.conditioning_dims, dtype=np.float32)

        full_stat = array_to_stats(all_obs)
        full_scale = np.concatenate([base_scale, reward_scale]).astype(np.float32)
        full_offset = np.concatenate([base_offset, reward_offset]).astype(np.float32)

        normalizer['obs'] = SingleFieldLinearNormalizer.create_manual(
            scale=full_scale,
            offset=full_offset,
            input_stats_dict=full_stat
        )

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
