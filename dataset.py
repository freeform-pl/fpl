"""
Preference dataset for reward learning.

Each sample is a pair of trajectories (A, B) with K binary preference labels.
Trajectories are represented as strided sequences of (third-person, wrist) frame pairs.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tasks import TASKS


def _resolve_iris_path(path: str) -> str:
    """Translate /iris/u/... -> /hai/scratch/marcelto/data/... when the iris path
    is unavailable (haic doesn't mount /iris)."""
    if not isinstance(path, str) or not path.startswith("/iris/u/"):
        return path
    if os.path.exists(path):
        return path
    return "/hai/scratch/marcelto/data/" + path[len("/iris/u/"):]


def _strided_indices(total: int, stride: int, seq_len: int, offset: int) -> list:
    """Return up to seq_len frame indices starting at offset with the given stride."""
    return list(range(offset, total, stride))[:seq_len]


def _resize_frames(frames: np.ndarray, img_size: tuple) -> np.ndarray:
    """Resize (T, H, W, 3) array to (T, img_size[1], img_size[0], 3)."""
    return np.stack([cv2.resize(f, img_size, interpolation=cv2.INTER_AREA) for f in frames])


def _load_raw_hdf5(hdf5_path: str, action_chunk_size: int = 0) -> dict:
    """Load all raw data from an HDF5 file once. Returns a dict with numpy arrays."""
    with h5py.File(hdf5_path, "r") as f:
        demo_key = next(iter(f["data"].keys()))
        obs = f[f"data/{demo_key}/obs"]
        raw = {
            "agent_view": obs["agent_view"][:],   # (total, H, W, 3) uint8
            "wrist": obs["wrist"][:],              # (total, H, W, 3) uint8
            "proprio": obs["JOINT_POS"][:].astype(np.float32),  # (total, 7)
        }
        if action_chunk_size > 0:
            acts_jp = f[f"data/{demo_key}/actions/joint_position"][:]  # (total, 7)
            acts_gr = f[f"data/{demo_key}/actions/gripper_position"][:]  # (total,) or (total, 1)
            if acts_gr.ndim == 1:
                acts_gr = acts_gr[:, np.newaxis]
            raw["actions"] = np.concatenate([acts_jp, acts_gr], axis=-1).astype(np.float32)  # (total, 8)
    return raw


def _extract_trajectory(raw: dict, stride: int, seq_len: int, offset: int, img_size: tuple, action_chunk_size: int = 0) -> tuple:
    """Extract a strided trajectory from pre-loaded raw data."""
    total = raw["agent_view"].shape[0]
    indices = _strided_indices(total, stride, seq_len, offset)
    tp = raw["agent_view"][indices]
    wr = raw["wrist"][indices]
    proprio = raw["proprio"][indices]

    action_chunks = None
    if action_chunk_size > 0:
        acts = raw["actions"]
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
        action_chunks = np.stack(chunks)

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


def _load_from_hdf5(hdf5_path: str, stride: int, seq_len: int, offset: int, img_size: tuple, action_chunk_size: int = 0) -> tuple:
    raw = _load_raw_hdf5(hdf5_path, action_chunk_size)
    return _extract_trajectory(raw, stride, seq_len, offset, img_size, action_chunk_size)


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


def _trajectories_from_raw(
    raw: dict,
    stride: int,
    seq_len: int,
    img_size: tuple,
    offsets: list[int],
    action_chunk_size: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Extract trajectory dicts for all offsets from pre-loaded raw data."""
    results = []
    for o in offsets:
        tp, wr, proprio, n_real, action_chunks = _extract_trajectory(
            raw, stride, seq_len, o, img_size, action_chunk_size
        )
        tp_t = torch.from_numpy(tp).permute(0, 3, 1, 2)
        wr_t = torch.from_numpy(wr).permute(0, 3, 1, 2)
        padding_mask = torch.zeros(seq_len, dtype=torch.bool)
        padding_mask[n_real:] = True
        result = {
            "third_person": tp_t,
            "wrist": wr_t,
            "proprio": torch.from_numpy(proprio),
            "padding_mask": padding_mask,
        }
        if action_chunks is not None:
            result["action_chunks"] = torch.from_numpy(action_chunks)
        results.append(result)
    return results


def load_trajectories_all_offsets(
    hdf5_path: str,
    stride: int,
    seq_len: int,
    img_size: tuple,
    offsets: list[int],
    action_chunk_size: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Load HDF5 once and return a trajectory dict for each offset."""
    raw = _load_raw_hdf5(hdf5_path, action_chunk_size)
    return _trajectories_from_raw(raw, stride, seq_len, img_size, offsets, action_chunk_size)


def auto_detect_preference_keys(preference_dirs: list, cross_dirs: list) -> list:
    """Scan all preference JSON files and return the sorted union of keys observed.

    Used when --task auto is passed: caller doesn't know the schema ahead of time
    and wants the dataset's own labels to define the axes.
    """
    import glob as _glob
    keys = set()
    for pdir in preference_dirs or []:
        if not os.path.isdir(pdir):
            continue
        for session in os.listdir(pdir):
            pf = os.path.join(pdir, session, "preference.json")
            if not os.path.exists(pf):
                continue
            try:
                with open(pf) as f:
                    keys.update(json.load(f).get("preferences", {}).keys())
            except Exception:
                pass
    for cdir in cross_dirs or []:
        if not os.path.isdir(cdir):
            continue
        for cf in _glob.glob(os.path.join(cdir, "preference_*.json")):
            try:
                with open(cf) as f:
                    keys.update(json.load(f).get("preferences", {}).keys())
            except Exception:
                pass
    return sorted(keys)


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
        only_large: bool = False,
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
        t_start = time.time()
        n_skipped_large = 0

        # When preloading, we load raw HDF5 data during validation (single read per file)
        # and extract trajectories at the end, so each file is opened exactly once.
        raw_cache = {} if preload else None
        if preload:
            n_off = preload_offsets if training else 1
            offsets = [int(i * stride / n_off) for i in range(n_off)]

        for di, d in enumerate(preference_dirs):
            pref_file = os.path.join(d, "preference.json")
            hdf5_a = os.path.join(d, "rollout_A.hdf5")
            hdf5_b = os.path.join(d, "rollout_B.hdf5")

            if only_large:
                hdf5_a = os.path.join(d, "rollout_A_large.hdf5")
                hdf5_b = os.path.join(d, "rollout_B_large.hdf5")

            if not (os.path.exists(pref_file) and os.path.exists(hdf5_a) and os.path.exists(hdf5_b)):
                if only_large:
                    n_skipped_large += 1
                continue

            try:
                if preload:
                    # Load raw data (validates implicitly — will raise on corrupt files)
                    for path in (hdf5_a, hdf5_b):
                        if path not in raw_cache:
                            t_file = time.time()
                            raw_cache[path] = _load_raw_hdf5(path, action_chunk_size)
                            print(f"    Loaded {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} "
                                  f"in {time.time() - t_file:.2f}s", flush=True)
                else:
                    # Validate only — just check that required keys exist
                    for path in (hdf5_a, hdf5_b):
                        with h5py.File(path, "r") as f:
                            demo_key = next(iter(f["data"].keys()))
                            _ = f[f"data/{demo_key}/obs/agent_view"].shape
                            _ = f[f"data/{demo_key}/obs/JOINT_POS"].shape
                            if action_chunk_size > 0:
                                _ = f[f"data/{demo_key}/actions/joint_position"].shape
                                _ = f[f"data/{demo_key}/actions/gripper_position"].shape
            except (OSError, KeyError):
                print(f"Skipping corrupted HDF5 in {d}", flush=True)
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

            log_every = 1 if preload else 20
            if (di + 1) % log_every == 0 or di + 1 == len(preference_dirs):
                print(f"  {'Loaded' if preload else 'Validated'} {di + 1}/{len(preference_dirs)} dirs "
                      f"({len(self.samples)} valid, {len(raw_cache) if raw_cache is not None else 0} cached files) "
                      f"[{time.time() - t_start:.1f}s]", flush=True)

        print(f"{'Loading' if preload else 'Validation'} complete: {len(self.samples)} valid pairs "
              f"from {len(preference_dirs)} dirs in {time.time() - t_start:.1f}s", flush=True)
        if only_large and n_skipped_large > 0:
            print(f"  (--only_large: skipped {n_skipped_large} dirs missing *_large.hdf5)", flush=True)

        self.preload_time_s = 0.0
        if preload:
            print(f"Extracting {len(self.samples)} trajectory pairs × {n_off} offset(s) {offsets} "
                  f"from {len(raw_cache)} cached files...", flush=True)
            t_extract = time.time()
            for si, s in enumerate(self.samples):
                t0 = time.time()
                s["trajs_a"] = _trajectories_from_raw(raw_cache[s["hdf5_a"]], stride, seq_len, img_size, offsets, action_chunk_size)
                s["trajs_b"] = _trajectories_from_raw(raw_cache[s["hdf5_b"]], stride, seq_len, img_size, offsets, action_chunk_size)
                if (si + 1) % 10 == 0 or si + 1 == len(self.samples):
                    print(f"  Extracted {si + 1}/{len(self.samples)} pairs "
                          f"[{time.time() - t_extract:.1f}s total, last={time.time() - t0:.2f}s]", flush=True)
            del raw_cache
            self.preload_time_s = time.time() - t_start
            print(f"Total preload time: {self.preload_time_s:.1f}s "
                  f"({self.preload_time_s / max(len(self.samples), 1):.2f}s/pair)", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.preload and "trajs_a" in s:
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
    only_large: bool = False,
    preference_keys: Optional[list] = None,
) -> tuple[PreferenceDataset, PreferenceDataset]:
    """
    Randomly assign val_fraction of preference sessions to validation and the
    rest to training. Each session's trajectories are kept whole — no timestep
    from a validation trajectory is ever seen during training.

    If preference_keys is provided it overrides the TASKS lookup (used for
    --task auto, where keys come from the data itself).
    """
    if preference_keys is None:
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

    train_ds = PreferenceDataset(train_dirs, preference_keys=preference_keys, stride=stride, seq_len=seq_len, img_size=img_size, training=True,  preload=preload, action_chunk_size=action_chunk_size, preload_offsets=preload_offsets, only_large=only_large)
    val_ds   = PreferenceDataset(val_dirs,   preference_keys=preference_keys, stride=stride, seq_len=seq_len, img_size=img_size, training=False, preload=preload, action_chunk_size=action_chunk_size, preload_offsets=preload_offsets, only_large=only_large)
    return train_ds, val_ds


def load_cross_preferences(
    cross_dir: str,
    preference_dirs: list,
    preference_keys: list,
    stride: int,
    seq_len: int,
    img_size: tuple = (128, 128),
    action_chunk_size: int = 0,
) -> list:
    """
    Load cross-preference samples from a directory of preference_X.json files.

    Each JSON has rollout_A_timestamp / rollout_B_timestamp that match the
    rollout_A.timestamp / rollout_B.timestamp fields inside regular preference.json
    sessions.  The corresponding HDF5 files are looked up from preference_dirs.

    Returns a list of samples in the same dict format as PreferenceDataset.samples.
    """
    import glob as _glob

    # Build timestamp → (hdf5_path, succeeded) across all regular preference sessions.
    ts_map = {}  # str timestamp → (abs hdf5 path, succeeded bool|None)
    for pref_dir in preference_dirs:
        if not os.path.isdir(pref_dir):
            continue
        for session_name in os.listdir(pref_dir):
            pref_file = os.path.join(pref_dir, session_name, "preference.json")
            if not os.path.exists(pref_file):
                continue
            try:
                with open(pref_file) as f:
                    meta = json.load(f)
            except Exception:
                continue
            session_path = os.path.join(pref_dir, session_name)
            for rollout_key, hdf5_name in [("rollout_A", "rollout_A.hdf5"),
                                            ("rollout_B", "rollout_B.hdf5")]:
                info = meta.get(rollout_key, {})
                ts = info.get("timestamp")
                if ts is None:
                    continue
                hdf5_path = os.path.join(session_path, hdf5_name)
                if os.path.exists(hdf5_path):
                    ts_map[ts] = (hdf5_path, info.get("succeeded", None))

    if not ts_map:
        print("[cross_preferences] Warning: no rollout timestamps found in preference_dirs "
              "(cross-preferences using rollout_A_id/rollout_B_id will still work)")

    cross_files = sorted(_glob.glob(os.path.join(cross_dir, "preference_*.json")))
    if not cross_files:
        print(f"[cross_preferences] No preference_*.json files found in {cross_dir}")
        return []

    samples = []
    n_skip = 0
    for cross_file in cross_files:
        try:
            with open(cross_file) as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[cross_preferences] Skipping {os.path.basename(cross_file)}: {e}")
            n_skip += 1
            continue

        # Support two formats:
        #   1) rollout_A_id / rollout_B_id  — direct path to HDF5 (or dir containing .hdf5)
        #   2) rollout_A_timestamp / rollout_B_timestamp — looked up via ts_map
        id_a = meta.get("rollout_A_id")
        id_b = meta.get("rollout_B_id")

        if id_a is not None and id_b is not None:
            # Direct-path mode: resolve to .hdf5 file
            hdf5_a = id_a if id_a.endswith(".hdf5") else id_a + ".hdf5"
            hdf5_b = id_b if id_b.endswith(".hdf5") else id_b + ".hdf5"
            hdf5_a = _resolve_iris_path(hdf5_a)
            hdf5_b = _resolve_iris_path(hdf5_b)
            if not os.path.exists(hdf5_a):
                print(f"[cross_preferences] Skipping {os.path.basename(cross_file)}: "
                      f"rollout_A_id path not found: {hdf5_a}")
                n_skip += 1
                continue
            if not os.path.exists(hdf5_b):
                print(f"[cross_preferences] Skipping {os.path.basename(cross_file)}: "
                      f"rollout_B_id path not found: {hdf5_b}")
                n_skip += 1
                continue
            succeeded_a = None
            succeeded_b = None
        else:
            # Timestamp-lookup mode
            ts_a = meta.get("rollout_A_timestamp")
            ts_b = meta.get("rollout_B_timestamp")

            if ts_a not in ts_map:
                print(f"[cross_preferences] Skipping {os.path.basename(cross_file)}: "
                      f"rollout_A_timestamp '{ts_a}' not found")
                n_skip += 1
                continue
            if ts_b not in ts_map:
                print(f"[cross_preferences] Skipping {os.path.basename(cross_file)}: "
                      f"rollout_B_timestamp '{ts_b}' not found")
                n_skip += 1
                continue

            hdf5_a, succeeded_a = ts_map[ts_a]
            hdf5_b, succeeded_b = ts_map[ts_b]

        # Validate that both HDF5 files can be opened.
        try:
            for path in (hdf5_a, hdf5_b):
                with h5py.File(path, "r") as f:
                    demo_key = next(iter(f["data"].keys()))
                    _ = f[f"data/{demo_key}/obs/agent_view"].shape
        except (OSError, KeyError) as e:
            print(f"[cross_preferences] Skipping {os.path.basename(cross_file)}: "
                  f"corrupted HDF5 — {e}")
            n_skip += 1
            continue

        labels = parse_preference_labels(meta["preferences"], preference_keys)
        session = meta.get("session_timestamp", os.path.basename(cross_file))

        samples.append({
            "hdf5_a": hdf5_a,
            "hdf5_b": hdf5_b,
            "labels": labels,
            "session": session,
            "instruction": meta.get("instruction", ""),
            "raw_preferences": meta["preferences"],
            "succeeded_a": torch.tensor(
                1 if succeeded_a is True else (0 if succeeded_a is False else -1),
                dtype=torch.int8,
            ),
            "succeeded_b": torch.tensor(
                1 if succeeded_b is True else (0 if succeeded_b is False else -1),
                dtype=torch.int8,
            ),
        })

    unique_paths = set()
    for s in samples:
        unique_paths.add(s["hdf5_a"])
        unique_paths.add(s["hdf5_b"])
    print(f"[cross_preferences] Loaded {len(samples)} pairs from {cross_dir} "
          f"({len(unique_paths)} unique trajectories), "
          f"skipped {n_skip} out of {len(cross_files)} files")
    return samples


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


# ---------------------------------------------------------------------------
# Open-axis wrapper (for qwen_open)
# ---------------------------------------------------------------------------

class DummyPreferenceDataset(Dataset):
    """Tiny synthetic dataset for fast DDP/OOM debugging.

    Returns the same schema as PreferenceDataset (third_person, wrist,
    padding_mask, proprio, labels, succeeded_a/b, session) but generates
    random uint8 image tensors on the fly. No HDF5, no preloading.

    Exposes a `.samples` list so OpenPreferenceDataset can wrap this.
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        img_size: tuple,
        num_preferences: int,
        proprio_dim: int = 8,
        seed: int = 0,
    ):
        self.n = n_samples
        self.T = seq_len
        H = img_size[0] if isinstance(img_size, (tuple, list)) else int(img_size)
        self.H = H
        self.K = num_preferences
        self.proprio_dim = proprio_dim
        self.seed = seed
        # OpenPreferenceDataset and downstream stat code read `.samples` —
        # we only need `labels` present for the open wrapper to enumerate axes.
        # Pre-seed each sample's label so axis prompts get a usable signal.
        g = torch.Generator().manual_seed(seed)
        self.samples = []
        for i in range(n_samples):
            labels = torch.zeros(num_preferences, dtype=torch.float32)
            # Alternate 0/1 across samples and axes so cross-entropy isn't degenerate.
            for k in range(num_preferences):
                labels[k] = float((i + k) % 2)
            self.samples.append({"labels": labels})
        # Stride/seq_len attrs so print_dataset_stats-style code can read them.
        self.stride = 1
        self.seq_len = seq_len
        self.preload_time_s = 0.0

    def __len__(self):
        return self.n

    def _make_traj(self):
        T, H = self.T, self.H
        return {
            "third_person": torch.randint(0, 256, (T, 3, H, H), dtype=torch.uint8),
            "wrist": torch.randint(0, 256, (T, 3, H, H), dtype=torch.uint8),
            "padding_mask": torch.zeros(T, dtype=torch.bool),
            "proprio": torch.zeros(T, self.proprio_dim, dtype=torch.float32),
        }

    def __getitem__(self, idx):
        labels = self.samples[idx]["labels"].clone()
        return {
            "traj_a": self._make_traj(),
            "traj_b": self._make_traj(),
            "labels": labels,
            "succeeded_a": torch.tensor(int(labels[0].item() == 1.0), dtype=torch.int8),
            "succeeded_b": torch.tensor(int(labels[0].item() == 0.0), dtype=torch.int8),
            "session": "dummy",
        }


class OpenPreferenceDataset(Dataset):
    """Wraps a PreferenceDataset, exploding K-axis samples into per-axis samples.

    Each multi-axis sample (with K preference labels) becomes up to K single-axis
    samples, each containing the axis name and a scalar preference label.
    When `skip_equal=True`, axes with label 0.5 (equal/unlabeled) are skipped,
    which is required when training with equal_weight == 0 to avoid all-equal
    per-rank batches that produce a loss with no grad_fn.
    """

    def __init__(self, base_dataset, preference_keys: list[str], skip_equal: bool = False):
        self.base = base_dataset
        self.preference_keys = preference_keys
        self.skip_equal = skip_equal
        self.items = []  # (base_idx, axis_idx, axis_name)
        for i, sample in enumerate(base_dataset.samples):
            labels = sample["labels"]
            for k, key in enumerate(preference_keys):
                if skip_equal and float(labels[k]) == 0.5:
                    continue
                self.items.append((i, k, key))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        base_idx, axis_idx, axis_name = self.items[idx]
        sample = self.base[base_idx]
        return {
            "traj_a": sample["traj_a"],
            "traj_b": sample["traj_b"],
            "labels": sample["labels"][axis_idx:axis_idx + 1],  # (1,)
            "axis_label": axis_name,
            "succeeded_a": sample.get("succeeded_a", torch.tensor(0, dtype=torch.int8)),
            "succeeded_b": sample.get("succeeded_b", torch.tensor(0, dtype=torch.int8)),
            "session": sample.get("session", ""),
        }
