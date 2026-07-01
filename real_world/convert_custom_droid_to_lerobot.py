"""
Convert scored trajectories to LeRobot format for pi0.5-droid fine-tuning.

Reads score JSONs produced by infer.py, pairs them with source HDF5 trajectories,
and builds a LeRobot dataset with reward-conditioned language prompts.

Usage:
  python convert_custom_droid_to_lerobot.py \
    --scores_dir /path/to/infer_output/... \
    --repo_name marcelto/setup_table_standardized_1dp \
    --task_prompt "set up the table" \
    --score_type standardized \
    --decimal_places 1

Input:
  - Each score JSON must contain a "source_hdf5" path and a score dict under the
    key given by --score_type (e.g. "standardized"). If the JSONs were produced
    on a different machine, pass --hdf5_root to remap the paths while preserving
    their relative subpath.
  - Each HDF5 must contain, per demo under "data/<demo>":
      obs/JOINT_POS, obs/GRIPPER, obs/agent_view, obs/wrist
      and either "actions" + "actions_abs", or
      "actions/joint_velocity" + "actions/gripper_position".
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from tqdm import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

IMAGE_SIZE = (224, 224)  # (width, height)


# def resize_image(img_array: np.ndarray) -> np.ndarray:
#     img = Image.fromarray(img_array)
#     return np.array(img.resize(IMAGE_SIZE, resample=Image.BICUBIC))


def find_score_jsons(scores_dir: Path) -> list[Path]:
    """Find all *_score*.json files in scores_dir (flat and in subdirs)."""
    entries = []
    # Flat score files (for demos)
    for f in sorted(scores_dir.glob("*_score*.json")):
        entries.append(f)
    # Score files inside subdirs (for preference rollouts)
    for subdir in sorted(scores_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for f in sorted(subdir.glob("*_score*.json")):
            entries.append(f)
    return entries


def build_prompt(task_prompt: str, scores: dict, decimal_places: int) -> str:
    parts = [f"{k}: {v:.{decimal_places}f}" for k, v in scores.items()]
    return task_prompt + ", " + ", ".join(parts)


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    added_batch = False
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        # Convert to channels-first for torch operations
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
            added_batch = True
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
            added_batch = True

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # F.interpolate requires float input
    orig_dtype = images.dtype
    if orig_dtype == torch.uint8:
        images = images.float()

    # Resize
    resized_images = F.interpolate(
        images, size=(resized_height, resized_width), mode=mode, align_corners=False if mode == "bilinear" else None
    )

    # Handle dtype-specific clipping
    if orig_dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif orig_dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if orig_dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]
    if added_batch:
        padded_images = padded_images.squeeze(0)  # Remove batch dimension if it was added

    return padded_images

@dataclass
class Args:
    scores_dir: str
    """Directory containing score JSONs from infer.py"""

    repo_name: str
    """LeRobot repo ID (e.g. marcelto/setup_table_standardized_1dp). Dataset is saved to $HF_LEROBOT_HOME/<repo_name>"""

    task_prompt: str
    """Base task description (e.g. 'set up the table'). Scores are appended as 'key: value' pairs."""

    score_type: str = "standardized"
    """Which score dict to read from the JSON: 'raw', 'normalized', 'standardized', 'buckets', 'buckets_quantile'"""

    decimal_places: int = 1
    """Number of decimal places for score values in the prompt"""

    scratch_dir: str | None = None
    """Local scratch dir for fast I/O. Defaults to $SCRATCH env var, else a system tempdir."""

    hdf5_root: str | None = None
    """If set, remap each score JSON's `source_hdf5` path under this directory,
    preserving the relative subpath. Use when the JSONs contain absolute paths
    from another machine."""

    fps: int = 15
    """Frames per second recorded in the LeRobot dataset metadata"""

    robot_type: str = "panda"
    """Robot type recorded in the LeRobot dataset metadata"""

    push_to_hub: bool = False
    """Push dataset to HuggingFace Hub after conversion"""

    append: bool = False
    """Append to existing dataset instead of recreating"""

    debug: bool = False
    """Save debug_resize.png showing original vs resized images for the first episode"""

    wandb_project: str = "lerobot_convert"
    """Wandb project name for logging conversion metrics"""

    wandb_entity: str | None = None
    """Wandb entity (team/user). If None, uses default."""

    no_wandb: bool = False
    """Disable wandb logging"""

    wandb_step_window: int = 100
    """Number of add_frame calls to average over before logging a step-level entry to wandb"""


def main(args: Args):
    import os
    import tempfile

    scores_dir = Path(args.scores_dir)
    final_output_path = HF_LEROBOT_HOME / args.repo_name

    # Find all score JSONs
    score_files = find_score_jsons(scores_dir)
    print(f"Found {len(score_files)} score files in {scores_dir}")
    if not score_files:
        print("Nothing to convert.")
        return

    # Wandb init
    wandb_run = None
    if not args.no_wandb:
        # Fall back to offline mode when no credentials are configured so a
        # fresh machine can run the conversion without an interactive login.
        wandb_mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=wandb_mode,
            name=args.repo_name.replace("/", "_"),
            config={
                "scores_dir": str(scores_dir),
                "repo_name": args.repo_name,
                "task_prompt": args.task_prompt,
                "score_type": args.score_type,
                "decimal_places": args.decimal_places,
                "image_size": IMAGE_SIZE,
                "num_score_files": len(score_files),
            },
        )

    # Step 1: Set up local scratch for output
    scratch_root = args.scratch_dir or os.environ.get("SCRATCH")
    use_scratch = bool(scratch_root) and Path(scratch_root).exists()
    if use_scratch:
        local_tmp = Path(scratch_root) / "lerobot_convert"
        local_tmp.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Using local scratch: {local_tmp} ===")
    else:
        local_tmp = Path(tempfile.mkdtemp(prefix="lerobot_convert_"))
        print(f"\n=== Using system tempdir: {local_tmp} ===")

    # Per-process staging dir for HDF5 copies — avoids basename collisions
    # between concurrent conversion jobs sharing the same local_tmp/scr.
    hdf5_stage = Path(tempfile.mkdtemp(prefix="hdf5_stage_", dir=local_tmp))
    print(f"  HDF5 staging: {hdf5_stage}")

    # Point HF_LEROBOT_HOME to local disk for output
    local_lerobot_home = local_tmp / "lerobot_output"
    local_lerobot_home.mkdir(exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = str(local_lerobot_home)
    import lerobot.common.datasets.lerobot_dataset as lds
    lds.HF_LEROBOT_HOME = local_lerobot_home
    local_output_path = local_lerobot_home / args.repo_name

    print(f"  Output will be written to {local_output_path}")

    # Step 2: Set up dataset
    if local_output_path.exists() and not args.append:
        shutil.rmtree(local_output_path)

    if args.append and local_output_path.exists():
        dataset = LeRobotDataset(repo_id=args.repo_name)
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_name,
            robot_type=args.robot_type,
            fps=args.fps,
            features={
                "exterior_image_1_left": {
                    "dtype": "image",
                    "shape": (IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image_left": {
                    "dtype": "image",
                    "shape": (IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
                    "names": ["height", "width", "channel"],
                },
                "joint_position": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["joint_position"],
                },
                "gripper_position": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["gripper_position"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=8,
            image_writer_processes=0,
        )

    # Step 3: Convert
    total_episodes = 0
    skipped = 0
    t_total_start = time.time()
    t_hdf5_read = 0.0
    t_resize = 0.0
    t_add_frame = 0.0
    t_save_episode = 0.0
    t_copy = 0.0
    total_frames = 0
    episode_metadata = []  # per-episode info

    # Rolling window for per-step wandb logging
    step_window = max(1, args.wandb_step_window)
    window_t_sum = 0.0
    window_count = 0
    window_idx = 0

    for score_file in tqdm(score_files, desc="Score files"):
        with open(score_file) as f:
            score_data = json.load(f)

        source_hdf5_str = score_data["source_hdf5"]
        source_hdf5 = Path(source_hdf5_str)
        # Remap paths written on another machine under a local root, preserving
        # the relative subpath (avoids basename collisions across subdirs).
        if args.hdf5_root is not None:
            rel = source_hdf5.relative_to(source_hdf5.anchor)
            source_hdf5 = Path(args.hdf5_root) / rel
        if not source_hdf5.exists():
            print(f"  [skip] source HDF5 not found: {source_hdf5_str}")
            skipped += 1
            continue

        scores = score_data.get(args.score_type)
        if scores is None:
            print(f"  [skip] no '{args.score_type}' scores in {score_file.name}")
            skipped += 1
            continue

        task_language = build_prompt(args.task_prompt, scores, args.decimal_places)

        # Capture source file size before copy
        source_size_bytes = source_hdf5.stat().st_size

        # Copy HDF5 to local disk for fast reads
        t0 = time.time()
        local_hdf5 = hdf5_stage / source_hdf5.name
        shutil.copy2(source_hdf5, local_hdf5)
        t_copy_file = time.time() - t0
        t_copy += t_copy_file

        try:
            with h5py.File(local_hdf5, "r") as f:
                demo_keys = sorted(f["data"].keys())
        except OSError as e:
            print(f"  [skip] corrupted file {source_hdf5_str}: {e}")
            local_hdf5.unlink(missing_ok=True)
            skipped += 1
            continue

        for demo_key in demo_keys:
            try:
                t0 = time.time()
                with h5py.File(local_hdf5, "r") as f:
                    demo = f[f"data/{demo_key}"]
                    obs = demo["obs"]

                    if isinstance(demo["actions"], h5py.Dataset):
                        actions_vel = demo["actions"][:]
                        actions_abs = demo["actions_abs"][:]
                        gripper_action = actions_abs[:, -1:]
                    else:
                        actions_vel = demo["actions"]["joint_velocity"][:]
                        gripper_action = demo["actions"]["gripper_position"][:]

                    joint_pos = obs["JOINT_POS"][:]
                    gripper_obs = obs["GRIPPER"][:]
                    agent_view = obs["agent_view"][:]
                    wrist_view = obs["wrist"][:]
                    T = actions_vel.shape[0]
                t_read_demo = time.time() - t0
                t_hdf5_read += t_read_demo

                # Capture pre-resize shapes
                agent_shape_before = tuple(agent_view.shape)
                wrist_shape_before = tuple(wrist_view.shape)

                t0 = time.time()
                agent_tensor = torch.from_numpy(agent_view)
                wrist_tensor = torch.from_numpy(wrist_view)
                resized_agent = resize_with_pad_torch(agent_tensor, IMAGE_SIZE[1], IMAGE_SIZE[0])
                resized_wrist = resize_with_pad_torch(wrist_tensor, IMAGE_SIZE[1], IMAGE_SIZE[0])
                resized_agent_np = resized_agent.numpy()
                resized_wrist_np = resized_wrist.numpy()
                t_resize_demo = time.time() - t0

                # Capture post-resize shapes
                agent_shape_after = tuple(resized_agent_np.shape)
                wrist_shape_after = tuple(resized_wrist_np.shape)

                if args.debug:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                    axes[0, 0].imshow(agent_view[0])
                    axes[0, 0].set_title("Original agent")
                    axes[0, 1].imshow(resized_agent_np[0])
                    axes[0, 1].set_title(f"Resized agent {resized_agent_np[0].shape}")
                    axes[1, 0].imshow(wrist_view[0])
                    axes[1, 0].set_title("Original wrist")
                    axes[1, 1].imshow(resized_wrist_np[0])
                    axes[1, 1].set_title(f"Resized wrist {resized_wrist_np[0].shape}")
                    plt.tight_layout()
                    plt.savefig("debug_resize.png", dpi=150)
                    plt.close()
                    print(f"  [debug] saved debug_resize.png (orig: {agent_view[0].shape} -> resized: {resized_agent_np[0].shape})")
                    args.debug = False
                t_resize += t_resize_demo

                t_add_frame_demo = 0.0
                for t in range(T):
                    action = np.concatenate([
                        actions_vel[t].astype(np.float32),
                        gripper_action[t].astype(np.float32),
                    ])

                    t0 = time.time()
                    dataset.add_frame({
                        "exterior_image_1_left": resized_agent_np[t],
                        "wrist_image_left": resized_wrist_np[t],
                        "joint_position": joint_pos[t].astype(np.float32),
                        "gripper_position": gripper_obs[t].astype(np.float32),
                        "actions": action,
                        "task": task_language,
                    })
                    dt = time.time() - t0
                    t_add_frame += dt
                    t_add_frame_demo += dt
                    total_frames += 1

                    if wandb_run is not None:
                        window_t_sum += dt
                        window_count += 1
                        if window_count >= step_window:
                            wandb.log({
                                "step/add_frame_avg_s": window_t_sum / window_count,
                                "step/add_frame_avg_ms": (window_t_sum / window_count) * 1000,
                                "step/window_index": window_idx,
                                "step/window_size": window_count,
                                "step/global_frame": total_frames,
                                "step/episode_index": total_episodes,
                            })
                            window_t_sum = 0.0
                            window_count = 0
                            window_idx += 1

                t0 = time.time()
                dataset.save_episode()
                t_save_episode_demo = time.time() - t0
                t_save_episode += t_save_episode_demo

                if wandb_run is not None:
                    wandb.log({
                        "episode_index": total_episodes,
                        "source_file_size_mb": source_size_bytes / (1024 * 1024),
                        "source_file_size_bytes": source_size_bytes,
                        "num_frames": T,
                        "time/hdf5_copy_s": t_copy_file,
                        "time/hdf5_read_s": t_read_demo,
                        "time/resize_s": t_resize_demo,
                        "time/add_frame_s": t_add_frame_demo,
                        "time/add_frame_per_frame_ms": (t_add_frame_demo / T * 1000) if T > 0 else 0,
                        "time/save_episode_s": t_save_episode_demo,
                        "shape/agent_before_T": agent_shape_before[0],
                        "shape/agent_before_H": agent_shape_before[1],
                        "shape/agent_before_W": agent_shape_before[2],
                        "shape/agent_after_H": agent_shape_after[1],
                        "shape/agent_after_W": agent_shape_after[2],
                        "shape/wrist_before_H": wrist_shape_before[1],
                        "shape/wrist_before_W": wrist_shape_before[2],
                        "shape/wrist_after_H": wrist_shape_after[1],
                        "shape/wrist_after_W": wrist_shape_after[2],
                        "shape/agent_before_str": str(agent_shape_before),
                        "shape/agent_after_str": str(agent_shape_after),
                        "shape/wrist_before_str": str(wrist_shape_before),
                        "shape/wrist_after_str": str(wrist_shape_after),
                    })

                episode_metadata.append({
                    "episode_index": total_episodes,
                    "source_hdf5": source_hdf5_str,
                    "demo_key": demo_key,
                    "score_file": str(score_file),
                    "task_language": task_language,
                    "num_frames": T,
                })
                total_episodes += 1
                print(f"  {Path(source_hdf5_str).name}/{demo_key} -> {task_language[:80]}...")

            except Exception as e:
                import traceback
                print(f"  [skip] error {Path(source_hdf5_str).name}/{demo_key}: {e}")
                traceback.print_exc()
                try:
                    dataset.episode_buffer = {"size": 0}
                except Exception:
                    pass
                skipped += 1
                continue

        local_hdf5.unlink(missing_ok=True)

    t_total = time.time() - t_total_start
    print(f"\n--- Timing breakdown ---")
    print(f"  Total:        {t_total:.1f}s")
    print(f"  HDF5 copy:    {t_copy:.1f}s")
    print(f"  HDF5 read:    {t_hdf5_read:.1f}s")
    print(f"  Image resize: {t_resize:.1f}s")
    print(f"  add_frame:    {t_add_frame:.1f}s ({total_frames} frames)")
    print(f"  save_episode: {t_save_episode:.1f}s ({total_episodes} episodes)")

    if wandb_run is not None and window_count > 0:
        wandb.log({
            "step/add_frame_avg_s": window_t_sum / window_count,
            "step/add_frame_avg_ms": (window_t_sum / window_count) * 1000,
            "step/window_index": window_idx,
            "step/window_size": window_count,
            "step/global_frame": total_frames,
            "step/episode_index": total_episodes,
        })

    if wandb_run is not None:
        wandb.summary["total/total_s"] = t_total
        wandb.summary["total/hdf5_copy_s"] = t_copy
        wandb.summary["total/hdf5_read_s"] = t_hdf5_read
        wandb.summary["total/resize_s"] = t_resize
        wandb.summary["total/add_frame_s"] = t_add_frame
        wandb.summary["total/save_episode_s"] = t_save_episode
        wandb.summary["total/total_frames"] = total_frames
        wandb.summary["total/total_episodes"] = total_episodes
        wandb.summary["total/skipped"] = skipped

    # Save episode metadata
    metadata_path = local_output_path / "episode_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(episode_metadata, f, indent=2)
    print(f"  Saved episode metadata ({len(episode_metadata)} episodes) to episode_metadata.json")

    # Step 4: Copy dataset back to NFS
    print(f"\n=== Copying dataset back to NFS ===")
    print(f"  {local_output_path} -> {final_output_path}")
    t0 = time.time()
    if final_output_path.exists():
        shutil.rmtree(final_output_path)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_output_path, final_output_path)
    print(f"  Copied in {time.time() - t0:.1f}s")

    # Always remove the per-job HDF5 staging dir; only blow away local_tmp
    # itself when it's the tempdir fallback (we keep scratch around for reuse).
    shutil.rmtree(hdf5_stage, ignore_errors=True)
    if not use_scratch:
        shutil.rmtree(local_tmp, ignore_errors=True)
    print(f"\nDone! Converted {total_episodes} episodes, skipped {skipped} -> {final_output_path}")

    if args.push_to_hub:
        # Re-point HF_LEROBOT_HOME back to NFS for push
        lds.HF_LEROBOT_HOME = HF_LEROBOT_HOME
        os.environ["HF_LEROBOT_HOME"] = str(HF_LEROBOT_HOME)
        dataset = LeRobotDataset(repo_id=args.repo_name)
        dataset.push_to_hub(
            tags=["droid", "panda", "preference"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    tyro.cli(main)
