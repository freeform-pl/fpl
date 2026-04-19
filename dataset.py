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


def _load_from_hdf5(hdf5_path: str, stride: int, seq_len: int, offset: int, img_size: tuple) -> tuple:
    with h5py.File(hdf5_path, "r") as f:
        demo_key = next(iter(f["data"].keys()))
        obs = f[f"data/{demo_key}/obs"]
        total = obs["agent_view"].shape[0]
        indices = _strided_indices(total, stride, seq_len, offset)
        tp = obs["agent_view"][indices]  # (T, H, W, 3) uint8
        wr = obs["wrist"][indices]

    n_real = len(indices)
    if n_real < seq_len:
        pad = seq_len - n_real
        tp = np.concatenate([tp, np.zeros((pad, *tp.shape[1:]), dtype=tp.dtype)], axis=0)
        wr = np.concatenate([wr, np.zeros((pad, *wr.shape[1:]), dtype=wr.dtype)], axis=0)

    tp = _resize_frames(tp, img_size)
    wr = _resize_frames(wr, img_size)
    return tp, wr, n_real


# def _load_from_video(video_path: str, stride: int, seq_len: int, img_size: tuple, offset: int) -> tuple:
#     cap = cv2.VideoCapture(video_path)
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()

#     indices = _strided_indices(total, stride, seq_len, offset)
#     target_set = set(indices)

#     cap = cv2.VideoCapture(video_path)
#     tp_frames, wr_frames = [], []
#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_idx in target_set:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             half_w = frame.shape[1] // 2
#             tp = cv2.resize(frame[:, :half_w], img_size, interpolation=cv2.INTER_AREA)
#             wr = cv2.resize(frame[:, half_w:], img_size, interpolation=cv2.INTER_AREA)
#             tp_frames.append(tp)
#             wr_frames.append(wr)
#         frame_idx += 1
#     cap.release()

#     while len(tp_frames) < seq_len:
#         tp_frames.append(tp_frames[-1])
#         wr_frames.append(wr_frames[-1])

#     return np.stack(tp_frames), np.stack(wr_frames)


def load_trajectory(
    hdf5_path: str,
    stride: int,
    seq_len: int,
    img_size: tuple = (128, 128),
    offset: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Load a trajectory as (third_person, wrist) image sequences.

    Frames are sampled at [offset, offset+stride, offset+2*stride, ...].
    Pass offset in [0, stride-1] to cover all temporal shifts during training.

    Loads from HDF5 if observation/third_person is present (run preprocess.py
    to populate), otherwise falls back to the alongside .mp4 video.

    Returns:
        third_person: (T, 3, H, W)  uint8  — normalization done at batch time
        wrist:        (T, 3, H, W)  uint8
    """
    tp, wr, n_real = _load_from_hdf5(hdf5_path, stride, seq_len, offset, img_size)

    # (T, H, W, 3) -> (T, 3, H, W)
    tp_t = torch.from_numpy(tp).permute(0, 3, 1, 2)
    wr_t = torch.from_numpy(wr).permute(0, 3, 1, 2)
    padding_mask = torch.zeros(seq_len, dtype=torch.bool)
    padding_mask[n_real:] = True
    return {"third_person": tp_t, "wrist": wr_t, "padding_mask": padding_mask}


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
    ):
        self.stride = stride
        self.seq_len = seq_len
        self.img_size = img_size
        self.training = training
        self.preload = preload
        self.preference_keys = preference_keys

        self.samples = []
        for d in preference_dirs:
            pref_file = os.path.join(d, "preference.json")
            hdf5_a = os.path.join(d, "rollout_A.hdf5")
            hdf5_b = os.path.join(d, "rollout_B.hdf5")

            if not (os.path.exists(pref_file) and os.path.exists(hdf5_a) and os.path.exists(hdf5_b)):
                continue

            try:
                with h5py.File(hdf5_a, "r") as _fa, h5py.File(hdf5_b, "r") as _fb:
                    pass
            except OSError:
                print(f"Skipping corrupted HDF5 in {d}")
                continue

            with open(pref_file) as f:
                meta = json.load(f)

            labels = parse_preference_labels(meta["preferences"], preference_keys)
            session = meta.get("session_timestamp", os.path.basename(d))

            self.samples.append({
                "hdf5_a": hdf5_a,
                "hdf5_b": hdf5_b,
                "labels": labels,
                "session": session,
                "instruction": meta.get("instruction", ""),
                "raw_preferences": meta["preferences"],
            })

        if preload:
            if training:
                print("Warning: preload=True with training=True fixes the stride offset at load time (no per-epoch augmentation).")
            print(f"Preloading {len(self.samples)} trajectory pairs...")
            for s in self.samples:
                offset = int(np.random.randint(0, stride)) if training else 0
                s["traj_a"] = load_trajectory(s["hdf5_a"], stride, seq_len, img_size, offset)
                s["traj_b"] = load_trajectory(s["hdf5_b"], stride, seq_len, img_size, offset)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.preload:
            traj_a = s["traj_a"]
            traj_b = s["traj_b"]
        else:
            offset = int(np.random.randint(0, self.stride)) if self.training else 0
            traj_a = load_trajectory(s["hdf5_a"], self.stride, self.seq_len, self.img_size, offset)
            traj_b = load_trajectory(s["hdf5_b"], self.stride, self.seq_len, self.img_size, offset)

        return {
            "traj_a": traj_a,
            "traj_b": traj_b,
            "labels": s["labels"],
            "session": s["session"],
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

    train_ds = PreferenceDataset(train_dirs, preference_keys=preference_keys, stride=stride, seq_len=seq_len, img_size=img_size, training=True,  preload=preload)
    val_ds   = PreferenceDataset(val_dirs,   preference_keys=preference_keys, stride=stride, seq_len=seq_len, img_size=img_size, training=False, preload=preload)
    return train_ds, val_ds
