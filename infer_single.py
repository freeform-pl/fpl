"""
Single-trajectory inference + paper-style visualization.

Loads a qwen_open_cum or qwen_discounted checkpoint, picks one trajectory
(either an explicit --hdf5 path, or a random rollout sampled from
--preferences_dir), runs per-frame inference, and writes a figure with:

  - Top: filmstrip of third-person frames (one per timestep)
  - Below: K rows of per-axis predicted reward, with the cumulative sum
           reported in each row's title.

Usage:
  python infer_single.py --ckpt path/to/model.pt --hdf5 rollout.hdf5
  python infer_single.py --ckpt path/to/model.pt --preferences_dir preferences/
"""

from __future__ import annotations

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from dataset import load_trajectory
from qwen_model import QwenRewardModel
from tasks import TASKS


SUPPORTED_MODELS = ("qwen_open_cum", "qwen_discounted")


def _build_model(saved_args: dict, preference_keys: list[str], device: torch.device):
    model_type = saved_args.get("model", "transformer")
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"infer_single only supports {SUPPORTED_MODELS}; checkpoint is '{model_type}'"
        )
    is_open_cum = model_type == "qwen_open_cum"
    is_discounted = model_type == "qwen_discounted"
    model = QwenRewardModel(
        num_preferences=1 if is_open_cum else len(preference_keys),
        model_name=saved_args.get("qwen_model_name", "Qwen/Qwen3-VL-4B-Instruct"),
        use_lora=False,
        lora_r=saved_args.get("lora_r", 64),
        lora_alpha=saved_args.get("lora_alpha", 16),
        reward_sigmoid=saved_args.get("reward_sigmoid", False),
        gradient_checkpointing=False,
        discounted=is_discounted,
        open_cum=is_open_cum,
    ).to(device)
    return model, model_type


def _find_rollouts(preferences_dir: list[str]) -> list[str]:
    """Return every rollout_*.hdf5 (or standalone .hdf5) under preferences_dir."""
    found = []
    for root in preferences_dir:
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full):
                for f in sorted(os.listdir(full)):
                    if f.endswith(".hdf5") and not f.endswith("_large.hdf5"):
                        found.append(os.path.join(full, f))
            elif entry.endswith(".hdf5") and not entry.endswith("_large.hdf5"):
                found.append(full)
    return found


def _per_frame_rewards(
    model: QwenRewardModel,
    traj: dict,
    model_type: str,
    preference_keys: list[str],
    device: torch.device,
) -> np.ndarray:
    """Run inference and return per-frame rewards as a (T, K) numpy array.

    For qwen_open_cum we loop over axes (single head, axis goes in the prompt).
    For qwen_discounted the K heads produce all axes in one pass.
    """
    tp = traj["third_person"].unsqueeze(0).to(device)
    wr = traj["wrist"].unsqueeze(0).to(device)
    pm = traj["padding_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "qwen_open_cum":
            cols = []
            for axis in preference_keys:
                pf = model.forward_per_frame(tp, wr, pm, axis_labels=[axis])  # (1, T, 1)
                cols.append(pf[0, :, 0].cpu().numpy())
            per_frame = np.stack(cols, axis=-1)  # (T, K)
        else:  # qwen_discounted
            pf = model.forward_per_frame(tp, wr, pm)  # (1, T, K)
            per_frame = pf[0].cpu().numpy()

    return per_frame


def _plot(
    frames: np.ndarray,
    rewards: np.ndarray,
    preference_keys: list[str],
    title: str,
    out_path: str,
) -> None:
    """Filmstrip on top, K reward rows below. Sized for paper figures."""
    T, K = rewards.shape
    timesteps = np.arange(T)
    cumulative = np.nansum(rewards, axis=0)  # (K,)

    cell_in = max(0.6, min(1.0, 16.0 / max(T, 1)))
    fig_w = max(10.0, cell_in * T + 1.5)
    fig_h = 2.2 + 1.6 * K

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        K + 1, T, figure=fig,
        height_ratios=[1.6] + [1.0] * K,
        hspace=0.45, wspace=0.04,
        left=0.06, right=0.98, top=0.92, bottom=0.06,
    )

    for t in range(T):
        ax = fig.add_subplot(gs[0, t])
        ax.imshow(frames[t])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#888888")
        ax.set_xlabel(str(t), fontsize=7, labelpad=1)

    colors = plt.get_cmap("tab10").colors
    for k, key in enumerate(preference_keys):
        ax = fig.add_subplot(gs[k + 1, :])
        values = rewards[:, k]
        color = colors[k % len(colors)]
        ax.plot(timesteps, values, marker="o", markersize=4,
                linewidth=1.6, color=color)
        ax.fill_between(timesteps, 0, values, alpha=0.12, color=color)
        ax.axhline(0, color="#444444", linewidth=0.6, linestyle="--")
        ax.set_ylabel(key, fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_xticks(timesteps)
        ax.tick_params(axis="both", labelsize=7)
        if k < K - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("timestep (frame pair)", fontsize=9)
        ax.set_title(f"cumulative = {cumulative[k]:+.3f}",
                     fontsize=9, loc="right", pad=2)

    fig.suptitle(title, fontsize=12, y=0.985)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--hdf5", default=None,
                        help="Specific rollout HDF5 to score; if omitted, sample one randomly.")
    parser.add_argument("--preferences_dir", default="preferences",
                        help="Comma-separated dirs; only used when --hdf5 not given.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: <ckpt_dir>/single_<traj>.png)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling --hdf5 (None = system random).")
    parser.add_argument("--task", default=None,
                        help="Override the task stored in the checkpoint.")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})

    for key in ("stride", "seq_len", "img_size"):
        if getattr(args, key) is None:
            setattr(args, key, saved_args.get(key))
            if getattr(args, key) is None:
                parser.error(f"--{key} not in checkpoint; pass it explicitly")

    task = args.task if args.task is not None else saved_args.get("task", "cube_in_three_bowls")
    if task not in TASKS:
        parser.error(f"--task '{task}' not in TASKS (known: {sorted(TASKS)})")
    preference_keys = TASKS[task]

    model, model_type = _build_model(saved_args, preference_keys, device)
    model.load_checkpoint_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {model_type} checkpoint with axes: {preference_keys}")

    if args.hdf5 is not None:
        hdf5_path = args.hdf5
    else:
        roots = [d.strip() for d in args.preferences_dir.split(",")]
        candidates = _find_rollouts(roots)
        if not candidates:
            parser.error(f"No rollout HDF5 files found under {roots}")
        if args.seed is not None:
            random.seed(args.seed)
        hdf5_path = random.choice(candidates)
        print(f"Sampled trajectory: {hdf5_path}")

    traj = load_trajectory(
        hdf5_path, args.stride, args.seq_len, (args.img_size, args.img_size), offset=0,
    )
    n_real = int((~traj["padding_mask"]).sum())
    print(f"Trajectory length (after stride={args.stride}): {n_real} frame pairs")

    per_frame = _per_frame_rewards(model, traj, model_type, preference_keys, device)
    per_frame = per_frame[:n_real]
    frames = traj["third_person"][:n_real].permute(0, 2, 3, 1).numpy()
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)

    if args.out is None:
        traj_name = os.path.splitext(os.path.basename(hdf5_path))[0]
        parent_name = os.path.basename(os.path.dirname(hdf5_path))
        ckpt_dir = os.path.dirname(args.ckpt) or "."
        args.out = os.path.join(ckpt_dir, f"single_{parent_name}_{traj_name}.png")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    title = f"{model_type} — {os.path.basename(os.path.dirname(hdf5_path))}/" \
            f"{os.path.basename(hdf5_path)}"
    _plot(frames, per_frame, preference_keys, title, args.out)


if __name__ == "__main__":
    main()
