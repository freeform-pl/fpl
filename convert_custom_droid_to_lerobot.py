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

    # Pad
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

    push_to_hub: bool = False
    """Push dataset to HuggingFace Hub after conversion"""

    append: bool = False
    """Append to existing dataset instead of recreating"""

    debug: bool = False
    """Save debug_resize.png showing original vs resized images for the first episode"""


def main(args: Args):
    scores_dir = Path(args.scores_dir)
    output_path = HF_LEROBOT_HOME / args.repo_name

    # Find all score JSONs
    score_files = find_score_jsons(scores_dir)
    print(f"Found {len(score_files)} score files in {scores_dir}")
    if not score_files:
        print("Nothing to convert.")
        return

    # Set up dataset
    if output_path.exists() and not args.append:
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    if args.append and output_path.exists():
        dataset = LeRobotDataset(repo_id=args.repo_name)
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_name,
            robot_type="panda",
            fps=15,
            features={
                "exterior_image_1_left": {
                    "dtype": "image",
                    "shape": (IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
                    "names": ["height", "width", "channel"],
                },
                "exterior_image_2_left": {
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

    total_episodes = 0
    skipped = 0

    t_total_start = time.time()


    padding = np.zeros((*IMAGE_SIZE,3))

    for score_file in tqdm(score_files, desc="Score files"):
        t_resize = 0.0
        t_hdf5_read = 0.0
        t_add_frame = 0.0
        t_save_episode = 0.0
        t_json_read = 0.0
        total_frames = 0
        # Load score JSON
        t0 = time.time()
        with open(score_file) as f:
            score_data = json.load(f)
        t_json_read += time.time() - t0

        source_hdf5 = Path(score_data["source_hdf5"])
        if not source_hdf5.exists():
            print(f"  [skip] source HDF5 not found: {source_hdf5}")
            skipped += 1
            continue

        scores = score_data.get(args.score_type)
        if scores is None:
            print(f"  [skip] no '{args.score_type}' scores in {score_file.name}")
            skipped += 1
            continue

        task_language = build_prompt(args.task_prompt, scores, args.decimal_places)

        # Copy HDF5 to local disk to avoid slow NFS reads
        import tempfile
        t_copy = time.time()
        local_hdf5 = Path(tempfile.gettempdir()) / source_hdf5.name
        shutil.copy2(source_hdf5, local_hdf5)
        print(f"  copied to local in {time.time() - t_copy:.1f}s")

        # Read trajectory from HDF5
        try:
            with h5py.File(local_hdf5, "r") as f:
                demo_keys = list(f["data"].keys())
        except OSError as e:
            print(f"  [skip] corrupted file {source_hdf5}: {e}")
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
                t_hdf5_read += time.time() - t0
                print(f"  hdf5 read {time.time() - t0:.1f}s")
                t0 = time.time()
                agent_tensor = torch.from_numpy(agent_view)  # (T, H, W, 3)
                wrist_tensor = torch.from_numpy(wrist_view)
                resized_agent = resize_with_pad_torch(agent_tensor, IMAGE_SIZE[1], IMAGE_SIZE[0])  # (T, 224, 224, 3)
                resized_wrist = resize_with_pad_torch(wrist_tensor, IMAGE_SIZE[1], IMAGE_SIZE[0])
                resized_agent_np = resized_agent.numpy()
                resized_wrist_np = resized_wrist.numpy()
                print("resized image shape", resized_wrist_np.shape)
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
                    args.debug = False  # only save once
                t_resize += time.time() - t0

                for t in range(T):
                    action = np.concatenate([
                        actions_vel[t].astype(np.float32),
                        gripper_action[t].astype(np.float32),
                    ])

                    t0 = time.time()
                    dataset.add_frame({
                        "exterior_image_1_left": resized_agent_np[t],
                        "exterior_image_2_left": resized_agent_np[t],
                        "wrist_image_left": resized_wrist_np[t],
                        "joint_position": joint_pos[t].astype(np.float32),
                        "gripper_position": gripper_obs[t].astype(np.float32),
                        "actions": action,
                        "task": task_language,
                    })
                    t_add_frame += time.time() - t0
                    total_frames += 1

                print("resize", t_resize)
                print("add frame", t_add_frame)

                t0 = time.time()
                dataset.save_episode()
                t_save_episode += time.time() - t0
                print("save episode", time.time() - t0)
                total_episodes += 1
                print(f"  {source_hdf5.name}/{demo_key} -> {task_language[:80]}...")

            except Exception as e:
                import traceback
                print(f"  [skip] error {source_hdf5.name}/{demo_key}: {e}")
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
    print(f"  JSON read:    {t_json_read:.1f}s")
    print(f"  HDF5 read:    {t_hdf5_read:.1f}s")
    print(f"  Image resize: {t_resize:.1f}s ({total_frames * 3} images)")
    print(f"  add_frame:    {t_add_frame:.1f}s ({total_frames} frames)")
    print(f"  save_episode: {t_save_episode:.1f}s ({total_episodes} episodes)")
    print(f"  Other:        {t_total - t_json_read - t_hdf5_read - t_resize - t_add_frame - t_save_episode:.1f}s")

    print(f"\nDone! Converted {total_episodes} episodes, skipped {skipped} -> {output_path}")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["droid", "panda", "preference"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
