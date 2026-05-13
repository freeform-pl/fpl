"""
Generate episode_metadata.json for existing LeRobot datasets that were created
by convert_custom_droid_to_lerobot.py before metadata tracking was added.

Replays the same score JSON iteration order to reconstruct the episode mapping.

Usage:
  python generate_episode_metadata.py \
    --scores-dir /path/to/infer_output/... \
    --repo-name marcelto/fold_pants_multi_standardized_1dp \
    --task-prompt "fold the shorts" \
    --score-type standardized \
    --decimal-places 1
"""

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
from tqdm import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME


def find_score_jsons(scores_dir: Path) -> list[Path]:
    entries = []
    for f in sorted(scores_dir.glob("*_score*.json")):
        entries.append(f)
    for subdir in sorted(scores_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for f in sorted(subdir.glob("*_score*.json")):
            entries.append(f)
    return entries


def build_prompt(task_prompt: str, scores: dict, decimal_places: int) -> str:
    parts = [f"{k}: {v:.{decimal_places}f}" for k, v in scores.items()]
    return task_prompt + ", " + ", ".join(parts)


@dataclass
class Args:
    scores_dir: str
    repo_name: str
    task_prompt: str
    score_type: str = "standardized"
    decimal_places: int = 1


def main(args: Args):
    scores_dir = Path(args.scores_dir)
    dataset_path = HF_LEROBOT_HOME / args.repo_name

    score_files = find_score_jsons(scores_dir)
    print(f"Found {len(score_files)} score files")

    episode_metadata = []
    ep_idx = 0

    n_missing_hdf5 = 0
    n_missing_scores = 0
    n_hdf5_error = 0

    for score_file in tqdm(score_files, desc="Replaying iteration"):
        with open(score_file) as f:
            score_data = json.load(f)

        source_hdf5 = Path(score_data["source_hdf5"])
        if not source_hdf5.exists():
            n_missing_hdf5 += 1
            if n_missing_hdf5 <= 3:
                print(f"  [skip] HDF5 not found: {source_hdf5}")
            continue

        scores = score_data.get(args.score_type)
        if scores is None:
            n_missing_scores += 1
            if n_missing_scores <= 3:
                print(f"  [skip] no '{args.score_type}' in {score_file.name}, keys: {list(score_data.keys())}")
            continue

        task_language = build_prompt(args.task_prompt, scores, args.decimal_places)

        try:
            with h5py.File(source_hdf5, "r") as f:
                demo_keys = list(f["data"].keys())
                if ep_idx == 0 and len(episode_metadata) == 0:
                    print(f"  [debug] first file: {source_hdf5}")
                    print(f"  [debug] top-level keys: {list(f.keys())}")
                    print(f"  [debug] data keys: {demo_keys}")
        except OSError as e:
            n_hdf5_error += 1
            if n_hdf5_error <= 3:
                print(f"  [skip] HDF5 error: {source_hdf5}: {e}")
            continue

        if not demo_keys and ep_idx == 0:
            print(f"  [debug] no demo keys in {source_hdf5}")

        for demo_key in demo_keys:
            try:
                with h5py.File(source_hdf5, "r") as f:
                    demo = f[f"data/{demo_key}"]
                    if isinstance(demo["actions"], h5py.Dataset):
                        T = demo["actions"].shape[0]
                    else:
                        T = demo["actions"]["joint_velocity"].shape[0]
            except Exception as e:
                print(f"  [skip] {source_hdf5.name}/{demo_key}: {e}")
                continue

            episode_metadata.append({
                "episode_index": ep_idx,
                "source_hdf5": str(source_hdf5),
                "demo_key": demo_key,
                "score_file": str(score_file),
                "task_language": task_language,
                "num_frames": T,
            })
            ep_idx += 1

    print(f"\n  Missing HDF5: {n_missing_hdf5}, Missing score type: {n_missing_scores}, HDF5 errors: {n_hdf5_error}")
    out_path = dataset_path / "episode_metadata.json"
    with open(out_path, "w") as f:
        json.dump(episode_metadata, f, indent=2)

    print(f"\nSaved {len(episode_metadata)} episodes to {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
