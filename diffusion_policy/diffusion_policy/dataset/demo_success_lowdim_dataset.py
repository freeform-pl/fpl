"""
Dataset that combines original demos + successful rollouts (no reward model).
Filters rollouts to only include episodes where success == 1.
"""

from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
    pickplace_masked_range_scale_offset,
    array_to_stats,
)


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )


class DemoSuccessLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            rollout_data_path: str,
            demo_hdf5_path: str,
            horizon: int = 1,
            pad_before: int = 0,
            pad_after: int = 0,
            obs_keys: List[str] = [
                'object',
                'robot0_eef_pos',
                'robot0_eef_quat',
                'robot0_gripper_qpos'],
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: int = None,
            n_active_objects: int = 4,  # PickPlace: zero out inactive object slots in obs
            **kwargs,
        ):
        self.n_active_objects = int(n_active_objects)
        self.obs_keys = list(obs_keys)
        obs_keys = list(obs_keys)
        replay_buffer = ReplayBuffer.create_empty_numpy()

        n_rollouts_total = 0
        n_rollouts_success = 0

        # Load rollout data — only successful episodes. Skip entirely when
        # rollout_data_path is unset / "none" (demos-only mode).
        has_rollouts = bool(rollout_data_path) and rollout_data_path.strip() != 'none'
        rollout_action_dim = None
        if has_rollouts:
            data = np.load(rollout_data_path)
            rollout_obs = data['obs']
            rollout_actions = data['actions']
            rollout_lengths = data['episode_lengths']
            success = data['success']
            rollout_action_dim = rollout_actions.shape[-1]

            for i in tqdm(range(len(rollout_obs)), desc="Loading successful rollouts"):
                n_rollouts_total += 1
                if success[i] < 1.0:
                    continue
                n_rollouts_success += 1

                L_obs = int(rollout_lengths[i])
                L_act = min(L_obs - 1, rollout_actions.shape[1])
                if L_act <= 0:
                    continue

                obs_i = rollout_obs[i, :L_obs].astype(np.float32)
                act_i = rollout_actions[i, :L_act].astype(np.float32)
                L = min(len(obs_i), L_act)

                replay_buffer.add_episode({
                    'obs': obs_i[:L],
                    'action': act_i[:L],
                })

            print(f"Rollouts: {n_rollouts_success}/{n_rollouts_total} successful")
        else:
            print("Rollouts: SKIPPED (rollout_data_path='none' — demos-only mode)")

        # Load original demo data
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep='rotation_6d')

        n_demos = 0
        with h5py.File(demo_hdf5_path, 'r') as f:
            demos = f['data']
            for i in tqdm(range(len(demos)), desc="Loading demo episodes"):
                demo = demos[f'demo_{i}']
                obs_parts = [demo['obs'][key][:].astype(np.float32) for key in obs_keys]
                obs_i = np.concatenate(obs_parts, axis=-1)
                act_i = demo['actions'][:].astype(np.float32)

                # Convert actions if dim mismatch (7D axis_angle -> 10D rot6d)
                if act_i.shape[-1] != rollout_action_dim and act_i.shape[-1] == 7:
                    pos = act_i[:, :3]
                    rot = act_i[:, 3:6]
                    gripper = act_i[:, 6:]
                    rot6d = rotation_transformer.forward(rot)
                    act_i = np.concatenate([pos, rot6d, gripper], axis=-1).astype(np.float32)

                L = min(len(obs_i), len(act_i))
                replay_buffer.add_episode({
                    'obs': obs_i[:L],
                    'action': act_i[:L],
                })
                n_demos += 1

        print(f"Total episodes: {replay_buffer.n_episodes} ({n_demos} demos + {n_rollouts_success} successful rollouts)")

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

        # Action: per-dim range normalizer (matches reward_conditioned setup).
        action_stat = array_to_stats(self.replay_buffer['action'])
        normalizer['action'] = get_range_normalizer_from_stat(action_stat)

        # Obs: per-dim range scaling with PickPlace masks applied via the
        # shared helper (zeros 4 high-amplifying quat dims and the inactive
        # object slots when n_active_objects < 4).
        all_obs = self.replay_buffer['obs']
        obs_stat = array_to_stats(all_obs)
        obs_starts_with_object = (
            len(self.obs_keys) > 0 and self.obs_keys[0] == 'object')
        scale, offset = pickplace_masked_range_scale_offset(
            obs_stat,
            n_active_objects=self.n_active_objects,
            obs_starts_with_object=obs_starts_with_object,
        )
        normalizer['obs'] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict=obs_stat,
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
