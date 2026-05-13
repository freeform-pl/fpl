"""
Create a modified LeRobot dataset from an existing one by filtering and/or re-labeling.

Uses episode_metadata.json (saved by convert_custom_droid_to_lerobot.py) to map
episodes back to their source trajectories, then filters (e.g. success-only)
and/or re-labels with different scores.

The source dataset is read directly from parquet + image files to avoid
LeRobot API compatibility issues.

Usage:
  # Keep only successful rollouts (same scores)
  python create_new_lerobot.py \
    --args.source-repo-name marcelto/fold_pants_multi_standardized_1dp \
    --args.repo-name marcelto/fold_pants_success_only2 \
    --args.task-prompt "fold the shorts" \
    --args.success-only

  # Re-label with different score type (using a different scores dir)
  python create_new_lerobot.py \
    --args.source-repo-name marcelto/fold_pants_multi_standardized_1dp \
    --args.repo-name marcelto/fold_pants_multi_normalized \
    --args.task-prompt "fold the shorts" \
    --args.scores-dir /path/to/NEW_infer_output/... \
    --args.score-type normalized \
    --args.decimal-places 2
"""

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def build_prompt(task_prompt: str, scores: dict, decimal_places: int) -> str:
    parts = [f"{k}: {v:.{decimal_places}f}" for k, v in scores.items()]
    return task_prompt + ", " + ", ".join(parts)


def check_succeeded(source_hdf5: str) -> bool | None:
    """Check if a rollout succeeded via preference.json in its parent dir."""
    source = Path(source_hdf5)
    pref_json = source.parent / "preference.json"
    if not pref_json.exists():
        return None

    with open(pref_json) as f:
        pref_data = json.load(f)

    if source.stem == "rollout_A":
        rollout_key = "rollout_A"
    elif source.stem == "rollout_B":
        rollout_key = "rollout_B"
    else:
        return None

    return pref_data.get(rollout_key, {}).get("succeeded", None)


@dataclass
class Args:
    source_repo_name: str
    """Source LeRobot dataset repo ID to read from (must have episode_metadata.json)"""

    repo_name: str
    """New LeRobot dataset repo ID to create"""

    task_prompt: str
    """Base task description (e.g. 'fold the shorts')"""

    scores_dir: str = ""
    """Optional: directory with new score JSONs for re-labeling. If empty, uses existing task_language from metadata."""

    score_type: str = "standardized"
    """Which score dict to read: 'raw', 'normalized', 'standardized', 'buckets', 'buckets_quantile'"""

    decimal_places: int = 1
    """Number of decimal places for score values in the prompt"""

    success_only: bool = False
    """Only include rollouts where preference.json has succeeded=True. Demos are always included."""

    include_demos: bool = True
    """Include demo trajectories (flat demos_*.hdf5 with no preference.json)"""

    push_to_hub: bool = False
    """Push dataset to HuggingFace Hub after creation"""


def main(args: Args):
    source_dataset_path = HF_LEROBOT_HOME / args.source_repo_name
    final_output_path = HF_LEROBOT_HOME / args.repo_name

    # ── Step 1: Load episode metadata ──────────────────────────────────
    metadata_path = source_dataset_path / "episode_metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found. Run generate_episode_metadata.py first.")
        return

    with open(metadata_path) as f:
        episode_metadata = json.load(f)
    print(f"Loaded metadata for {len(episode_metadata)} episodes from {args.source_repo_name}")

    # ── Step 2: Build new score lookup if re-labeling ──────────────────
    new_scores = {}  # source_hdf5 -> score_data
    if args.scores_dir:
        scores_dir = Path(args.scores_dir)
        print(f"\n=== Loading new scores from {scores_dir} ===")
        for f_path in sorted(scores_dir.glob("*_score*.json")):
            with open(f_path) as f:
                data = json.load(f)
            new_scores[data["source_hdf5"]] = data
        for subdir in sorted(scores_dir.iterdir()):
            if not subdir.is_dir():
                continue
            for f_path in sorted(subdir.glob("*_score*.json")):
                with open(f_path) as f:
                    data = json.load(f)
                new_scores[data["source_hdf5"]] = data
        print(f"  Loaded {len(new_scores)} score files")

    # ── Step 3: Filter episodes ────────────────────────────────────────
    episodes_to_keep = []
    n_success = 0
    n_failed = 0
    n_demos = 0

    for ep in episode_metadata:
        source_hdf5 = ep["source_hdf5"]

        # Filter by success
        if args.success_only:
            succeeded = check_succeeded(source_hdf5)
            if succeeded is None:
                if args.include_demos:
                    episodes_to_keep.append(ep)
                    n_demos += 1
                continue
            if succeeded:
                episodes_to_keep.append(ep)
                n_success += 1
            else:
                n_failed += 1
        else:
            episodes_to_keep.append(ep)

    if args.success_only:
        print(f"\n=== Filtering for success_only ===")
        print(f"  Succeeded: {n_success}, Failed (excluded): {n_failed}, Demos: {n_demos}")
    print(f"  Keeping {len(episodes_to_keep)}/{len(episode_metadata)} episodes")

    if not episodes_to_keep:
        print("No episodes to keep.")
        return

    # ── Step 4: Determine task language for each episode ────────────────
    for ep in episodes_to_keep:
        if args.scores_dir and ep["source_hdf5"] in new_scores:
            scores = new_scores[ep["source_hdf5"]].get(args.score_type)
            if scores:
                ep["new_task_language"] = build_prompt(args.task_prompt, scores, args.decimal_places)
            else:
                ep["new_task_language"] = ep["task_language"]
        elif args.task_prompt:
            if args.success_only:
                # Success-only filtering: use plain task prompt, no score conditioning
                ep["new_task_language"] = args.task_prompt
            else:
                # Re-use existing scores from the original score file if available
                score_file = ep.get("score_file")
                if score_file and Path(score_file).exists():
                    with open(score_file) as f:
                        score_data = json.load(f)
                    scores = score_data.get(args.score_type)
                    if scores:
                        ep["new_task_language"] = build_prompt(args.task_prompt, scores, args.decimal_places)
                    else:
                        ep["new_task_language"] = ep["task_language"]
                else:
                    ep["new_task_language"] = ep["task_language"]
        else:
            ep["new_task_language"] = ep["task_language"]

    # ── Step 5: Set up local scratch for output ────────────────────────
    scr_dir = Path("/scr/marcelto")
    if scr_dir.exists():
        local_tmp = scr_dir / "lerobot_convert"
        local_tmp.mkdir(exist_ok=True)
    else:
        local_tmp = Path(tempfile.mkdtemp(prefix="lerobot_new_"))

    local_lerobot_home = local_tmp / "lerobot_output"
    local_lerobot_home.mkdir(exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = str(local_lerobot_home)
    import lerobot.common.datasets.lerobot_dataset as lds
    lds.HF_LEROBOT_HOME = local_lerobot_home
    local_output_path = local_lerobot_home / args.repo_name

    if local_output_path.exists():
        shutil.rmtree(local_output_path)

    # ── Step 6: Load source dataset and copy filtered episodes ─────────
    print(f"\n=== Loading source dataset: {args.source_repo_name} ===")
    # Load from the NFS path (original HF_LEROBOT_HOME)
    source_ds = LeRobotDataset(args.source_repo_name, root=source_dataset_path)
    print(f"  {source_ds.num_episodes} episodes, {source_ds.num_frames} frames")

    print(f"\n=== Creating new dataset: {args.repo_name} ===")
    new_ds = LeRobotDataset.create(
        repo_id=args.repo_name,
        robot_type="panda",
        fps=source_ds.fps,
        features={k: v for k, v in source_ds.features.items()
                  if k not in ("timestamp", "frame_index", "episode_index", "index", "task_index")},
        image_writer_threads=8,
        image_writer_processes=0,
    )

    t_start = time.time()
    total_frames = 0
    new_episode_metadata = []

    for ep in tqdm(episodes_to_keep, desc="Copying episodes"):
        ep_idx = ep["episode_index"]
        task_language = ep["new_task_language"]

        if ep_idx >= source_ds.num_episodes:
            print(f"  [skip] episode {ep_idx} >= source episodes {source_ds.num_episodes}")
            continue

        from_idx = source_ds.episode_data_index["from"][ep_idx].item()
        to_idx = source_ds.episode_data_index["to"][ep_idx].item()

        for frame_idx in range(from_idx, to_idx):
            item = source_ds[frame_idx]

            frame = {}
            skip_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            for key in new_ds.features:
                if key in skip_keys:
                    continue
                if key in item:
                    val = item[key]
                    if isinstance(val, torch.Tensor):
                        # Images: CHW -> HWC for add_frame
                        if val.ndim == 3 and val.shape[0] in (1, 3):
                            val = val.permute(1, 2, 0).numpy()
                        else:
                            val = val.numpy()
                        # Scalars: ensure shape matches feature spec
                        feat = new_ds.features[key]
                        if isinstance(feat, dict) and "shape" in feat:
                            expected = tuple(feat["shape"])
                            if val.shape != expected:
                                val = val.reshape(expected)
                    frame[key] = val
            frame["task"] = task_language

            new_ds.add_frame(frame)
            total_frames += 1

        new_ds.save_episode()

        new_episode_metadata.append({
            "episode_index": len(new_episode_metadata),
            "source_hdf5": ep["source_hdf5"],
            "demo_key": ep["demo_key"],
            "score_file": ep.get("score_file", ""),
            "task_language": task_language,
            "num_frames": to_idx - from_idx,
        })

        print(f"  ep {ep_idx} -> {task_language[:80]}...")

    t_total = time.time() - t_start
    print(f"\n--- Done in {t_total:.1f}s ---")
    print(f"  {len(new_episode_metadata)} episodes, {total_frames} frames")

    # Save metadata for the new dataset
    metadata_out = local_output_path / "episode_metadata.json"
    with open(metadata_out, "w") as f:
        json.dump(new_episode_metadata, f, indent=2)
    print(f"  Saved episode_metadata.json")

    # ── Step 7: Copy dataset back to NFS ───────────────────────────────
    print(f"\n=== Copying dataset back to NFS ===")
    print(f"  {local_output_path} -> {final_output_path}")
    t0 = time.time()
    if final_output_path.exists():
        shutil.rmtree(final_output_path)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_output_path, final_output_path)
    print(f"  Copied in {time.time() - t0:.1f}s")

    if not scr_dir.exists():
        shutil.rmtree(local_tmp, ignore_errors=True)
    print(f"\nDone! -> {final_output_path}")

    if args.push_to_hub:
        lds.HF_LEROBOT_HOME = HF_LEROBOT_HOME
        os.environ["HF_LEROBOT_HOME"] = str(HF_LEROBOT_HOME)
        dataset = LeRobotDataset(repo_id=args.repo_name)
        dataset.push_to_hub(
            tags=["droid", "panda", "preference"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
