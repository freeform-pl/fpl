"""
Preference dataset for reward learning.

Each sample is a pair of trajectories (A, B) with K binary preference labels.
Trajectories are represented as strided sequences of (third-person, wrist) frame pairs.
"""

import json
import os
from pathlib import Path
from typing import Optional, Union

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tasks import TASKS


def _strided_indices(total: int, stride: int, seq_len: int, offset: int) -> list:
    """Return up to seq_len frame indices starting at offset with the given stride."""
    return list(range(offset, total, stride))[:seq_len]


def _resize_frames(frames: np.ndarray, img_size: tuple) -> np.ndarray:
    """Resize (T, H, W, 3) array to (T, img_size[1], img_size[0], 3)."""
    return np.stack([cv2.resize(f, img_size, interpolation=cv2.INTER_AREA) for f in frames])


def _load_from_hdf5(hdf5_path: str, stride: int, seq_len: int, offset: int, img_size: tuple, action_chunk_size: int = 0) -> tuple:
    with h5py.File(hdf5_path, "r") as f:
        demo_key = next(iter(f["data"].keys()))
        obs = f[f"data/{demo_key}/obs"]
        total = obs["agent_view"].shape[0]
        indices = _strided_indices(total, stride, seq_len, offset)
        tp = obs["agent_view"][indices]  # (T, H, W, 3) uint8
        wr = obs["wrist"][indices]

        proprio = obs["JOINT_POS"][indices].astype(np.float32)  # (T, 7)

        action_chunks = None
        if action_chunk_size > 0:
            acts_jp = f[f"data/{demo_key}/actions/joint_position"][:]  # (total, 7)
            acts_gr = f[f"data/{demo_key}/actions/gripper_position"][:]  # (total,) or (total, 1)
            if acts_gr.ndim == 1:
                acts_gr = acts_gr[:, np.newaxis]
            acts = np.concatenate([acts_jp, acts_gr], axis=-1).astype(np.float32)  # (total, 8)
            action_dim = acts.shape[-1]

            chunks = []
            for idx in indices:
                end = idx + action_chunk_size
                if end <= total:
                    chunk = acts[idx:end]
                else:
                    chunk = acts[idx:]
                    pad_len = action_chunk_size - len(chunk)
                    chunk = np.concatenate([chunk, np.zeros((pad_len, action_dim), dtype=chunk.dtype)], axis=0)
                chunks.append(chunk)
            action_chunks = np.stack(chunks)  # (n_real, action_chunk_size, action_dim)

    n_real = len(indices)
    if n_real < seq_len:
        pad = seq_len - n_real
        tp = np.concatenate([tp, np.zeros((pad, *tp.shape[1:]), dtype=tp.dtype)], axis=0)
        wr = np.concatenate([wr, np.zeros((pad, *wr.shape[1:]), dtype=wr.dtype)], axis=0)
        proprio = np.concatenate([proprio, np.zeros((pad, proprio.shape[1]), dtype=proprio.dtype)], axis=0)
        if action_chunks is not None:
            action_chunks = np.concatenate([
                action_chunks,
                np.zeros((pad, action_chunk_size, action_dim), dtype=action_chunks.dtype),
            ], axis=0)

    tp = _resize_frames(tp, img_size)
    wr = _resize_frames(wr, img_size)
    return tp, wr, proprio, n_real, action_chunks


def load_trajectory(
    hdf5_path: str,
    stride: int,
    seq_len: int,
    img_size: tuple = (128, 128),
    offset: int = 0,
    action_chunk_size: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Load a trajectory as (third_person, wrist) image sequences.

    Frames are sampled at [offset, offset+stride, offset+2*stride, ...].
    Pass offset in [0, stride-1] to cover all temporal shifts during training.

    Returns:
        third_person:   (T, 3, H, W)  uint8
        wrist:          (T, 3, H, W)  uint8
        padding_mask:   (T,) bool
        action_chunks:  (T, action_chunk_size, action_dim) float32  [only if action_chunk_size > 0]
    """
    tp, wr, proprio, n_real, action_chunks = _load_from_hdf5(hdf5_path, stride, seq_len, offset, img_size, action_chunk_size)

    tp_t = torch.from_numpy(tp).permute(0, 3, 1, 2)
    wr_t = torch.from_numpy(wr).permute(0, 3, 1, 2)
    padding_mask = torch.zeros(seq_len, dtype=torch.bool)
    padding_mask[n_real:] = True
    result = {
        "third_person": tp_t,
        "wrist": wr_t,
        "proprio": torch.from_numpy(proprio),  # (T, 7)
        "padding_mask": padding_mask,
    }
    if action_chunks is not None:
        result["action_chunks"] = torch.from_numpy(action_chunks)
    return result


def parse_preference_labels(preferences: dict, preference_keys: list) -> torch.Tensor:
    """
    Convert preference dict to a float tensor of shape (K,).

    Encoding:
        1.0 = A is preferred
        0.0 = B is preferred
        0.5 = Equal
    """
    labels = []
    for key in preference_keys:
        val = preferences.get(key, "Equal")
        if val == "A":
            labels.append(1.0)
        elif val == "B":
            labels.append(0.0)
        else:
            labels.append(0.5)
    return torch.tensor(labels, dtype=torch.float32)


class PreferenceDataset(Dataset):
    """
    Dataset of pairwise trajectory comparisons.

    Each item:
        traj_a:  dict with keys 'third_person', 'wrist', each (T, 3, H, W)
        traj_b:  dict with keys 'third_person', 'wrist', each (T, 3, H, W)
        labels:  (K,) float tensor  — 1=A preferred, 0=B preferred, 0.5=Equal
        session: str timestamp (for identification / visualization)
    """

    def __init__(
        self,
        preference_dirs: list[str],
        preference_keys: list,
        stride: int = 4,
        seq_len: int = 28,
        img_size: tuple = (128, 128),
        training: bool = True,
        preload: bool = False,
        action_chunk_size: int = 0,
        preload_offsets: int = 5,
    ):
        self.stride = stride
        self.seq_len = seq_len
        self.img_size = img_size
        self.training = training
        self.preload = preload
        self.preference_keys = preference_keys
        self.action_chunk_size = action_chunk_size
        self.preload_offsets = preload_offsets

        self.samples = []
        for d in preference_dirs:
            pref_file = os.path.join(d, "preference.json")
            hdf5_a = os.path.join(d, "rollout_A.hdf5")
            hdf5_b = os.path.join(d, "rollout_B.hdf5")

            if not (os.path.exists(pref_file) and os.path.exists(hdf5_a) and os.path.exists(hdf5_b)):
                continue

            try:
                for path in (hdf5_a, hdf5_b):
                    with h5py.File(path, "r") as f:
                        demo_key = next(iter(f["data"].keys()))
                        _ = f[f"data/{demo_key}/obs/agent_view"].shape
                        _ = f[f"data/{demo_key}/obs/JOINT_POS"].shape
                        if action_chunk_size > 0:
                            _ = f[f"data/{demo_key}/actions/joint_position"].shape
                            _ = f[f"data/{demo_key}/actions/gripper_position"].shape
            except (OSError, KeyError):
                print(f"Skipping corrupted HDF5 in {d}")
                continue

            with open(pref_file) as f:
                meta = json.load(f)

            labels = parse_preference_labels(meta["preferences"], preference_keys)
            session = meta.get("session_timestamp", os.path.basename(d))


            succeeded_a = meta["rollout_A"].get("succeeded", None)
            succeeded_b = meta["rollout_B"].get("succeeded", None)

            self.samples.append({
                "hdf5_a": hdf5_a,
                "hdf5_b": hdf5_b,
                "labels": labels,
                "session": session,
                "instruction": meta.get("instruction", ""),
                "raw_preferences": meta["preferences"],
                "succeeded_a": torch.tensor(1 if succeeded_a is True else (0 if succeeded_a is False else -1), dtype=torch.int8),
                "succeeded_b": torch.tensor(1 if succeeded_b is True else (0 if succeeded_b is False else -1), dtype=torch.int8),
            })

        if preload:
            n_off = preload_offsets if training else 1
            offsets = [int(i * stride / n_off) for i in range(n_off)]
            print(f"Preloading {len(self.samples)} trajectory pairs × {n_off} offset(s) {offsets}...")
            for s in self.samples:
                s["trajs_a"] = [load_trajectory(s["hdf5_a"], stride, seq_len, img_size, o, action_chunk_size) for o in offsets]
                s["trajs_b"] = [load_trajectory(s["hdf5_b"], stride, seq_len, img_size, o, action_chunk_size) for o in offsets]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.preload:
            i = int(np.random.randint(0, len(s["trajs_a"]))) if self.training else 0
            traj_a = s["trajs_a"][i]
            traj_b = s["trajs_b"][i]
        else:
            offset = int(np.random.randint(0, self.stride)) if self.training else 0
            traj_a = load_trajectory(s["hdf5_a"], self.stride, self.seq_len, self.img_size, offset, self.action_chunk_size)
            traj_b = load_trajectory(s["hdf5_b"], self.stride, self.seq_len, self.img_size, offset, self.action_chunk_size)

        return {
            "traj_a": traj_a,
            "traj_b": traj_b,
            "labels": s["labels"],
            "session": s["session"],
            "succeeded_a": s["succeeded_a"],
            "succeeded_b": s["succeeded_b"],
        }


def print_dataset_stats(train_ds: "PreferenceDataset", val_ds: "PreferenceDataset"):
    """Print episode length statistics for train and val splits."""
    def collect_lengths(ds):
        lengths = []
        for s in ds.samples:
            for path in (s["hdf5_a"], s["hdf5_b"]):
                try:
                    with h5py.File(path, "r") as f:
                        demo_key = next(iter(f["data"].keys()))
                        lengths.append(f[f"data/{demo_key}/obs/agent_view"].shape[0])
                except OSError:
                    pass
        return lengths

    stride = train_ds.stride
    seq_len = train_ds.seq_len
    max_covered = stride * seq_len
    print(f"Stride={stride}, seq_len={seq_len} → max frames covered per episode: {max_covered}")

    for split, ds in (("Train", train_ds), ("Val", val_ds)):
        lengths = collect_lengths(ds)
        if not lengths:
            print(f"{split}: no episodes found")
            continue
        arr = np.array(lengths)
        print(
            f"{split} episode lengths ({len(arr)} rollouts): "
            f"min={arr.min()}  max={arr.max()}  "
            f"mean={arr.mean():.1f}  median={np.median(arr):.1f}  std={arr.std():.1f}"
        )


def make_datasets(
    task: str,
    preferences_dir: Union[str, list] = "preferences",
    val_fraction: float = 0.2,
    stride: int = 4,
    seq_len: int = 28,
    img_size: tuple = (128, 128),
    seed: int = 0,
    preload: bool = True,
    action_chunk_size: int = 0,
    preload_offsets: int = 5,
) -> tuple[PreferenceDataset, PreferenceDataset]:
    """
    Randomly assign val_fraction of preference sessions to validation and the
    rest to training. Each session's trajectories are kept whole — no timestep
    from a validation trajectory is ever seen during training.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Available: {list(TASKS.keys())}")
    preference_keys = TASKS[task]

    roots = [preferences_dir] if isinstance(preferences_dir, str) else preferences_dir
    dirs = sorted(
        os.path.join(root, d)
        for root in roots
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(dirs))
    n_val = max(1, int(len(dirs) * val_fraction))

    val_dirs   = [dirs[i] for i in perm[:n_val]]
    train_dirs = [dirs[i] for i in perm[n_val:]]

    print(f"Task: {task} | Keys: {preference_keys}")
    print(f"Train: {len(train_dirs)} sessions, Val: {len(val_dirs)} sessions")

    train_ds = PreferenceDataset(train_dirs, preference_keys=preference_keys, stride=stride, seq_len=seq_len, img_size=img_size, training=True,  preload=preload, action_chunk_size=action_chunk_size, preload_offsets=preload_offsets)
    val_ds   = PreferenceDataset(val_dirs,   preference_keys=preference_keys, stride=stride, seq_len=seq_len, img_size=img_size, training=False, preload=preload, action_chunk_size=action_chunk_size, preload_offsets=preload_offsets)
    return train_ds, val_ds


def load_anchors(
    anchors_file: str,
    preference_keys: list,
    stride: int,
    seq_len: int,
    img_size: tuple = (128, 128),
    action_chunk_size: int = 0,
) -> list:
    """
    Load anchor trajectories from a JSON file.

    Each entry in the returned list is:
        {"traj": dict, "dim": int, "target": float}

    where target=1.0 means "good" and target=0.0 means "bad".
    Only keys present in preference_keys are loaded.
    """
    with open(anchors_file) as f:
        data = json.load(f)

    entries = []
    for key_name, splits in data.items():
        if key_name not in preference_keys:
            print(f"[anchors] Skipping unknown key '{key_name}'")
            continue
        dim = preference_keys.index(key_name)
        for label, paths in splits.items():
            target = 1.0 if label == "good" else 0.0
            for path in paths:
                try:
                    traj = load_trajectory(path, stride, seq_len, img_size, offset=0,
                                           action_chunk_size=action_chunk_size)
                    entries.append({"traj": traj, "dim": dim, "target": target})
                except Exception as e:
                    print(f"[anchors] Skipping {path}: {e}")

    print(f"Loaded {len(entries)} anchor trajectories from {anchors_file}")
    return entries
