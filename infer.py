"""
Run reward model inference on all preference-folder trajectories.

For each preference folder the script writes two files:
    reward_model_{ckptname}_rollout_A_score.json
    reward_model_{ckptname}_rollout_B_score.json

Each file contains:
    raw              — raw reward scores (unbounded)
    normalized       — min-max scaled to [0, 1] across all scored trajectories
    standardized     — z-score normalized (mean=0, std=1) across all scored trajectories
    buckets          — equal-width bucket labels [1-5] over [min, max]
    buckets_quantile — equal-frequency bucket labels [1-5] (same # points per bucket)

Usage:
    python infer.py --ckpt exp/2026-04-12_19-00-11/checkpoints/step000200.pt
    python infer.py --ckpt exp/2026-04-12_19-00-11/checkpoints/step000200.pt \
                    --preferences_dir preferences
"""

import argparse
import json
import os
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import load_trajectory
from model import RewardModel, DiscountedRewardModel
from flow_model import RewardModel as FlowRewardModel
from qwen_model import QwenRewardModel
from tasks import TASKS
from analyze import RewardData, plot_scatter_matrix, plot_dim_histograms


N_BUCKETS = 5
BUCKET_EDGES = [i / N_BUCKETS for i in range(N_BUCKETS + 1)]  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BAR_WIDTH = 30



def to_bucket(score: float, lo: float, hi: float) -> int:
    """Map a score to an equal-width bucket label in [1, N_BUCKETS] over [lo, hi]."""
    if hi == lo:
        return 1
    frac = (score - lo) / (hi - lo)
    return min(int(frac * N_BUCKETS) + 1, N_BUCKETS)


def to_quantile_bucket(score: float, edges: list[float]) -> int:
    """Map a score to an equal-frequency bucket label in [1, 5] given quantile edges."""
    for i in range(N_BUCKETS - 1):
        if score < edges[i + 1]:
            return i + 1
    return N_BUCKETS


def score_trajectory(model, hdf5_path: str, args, device: torch.device) -> dict:
    """Return raw scores only; buckets are added after quantile edges are computed."""
    traj = load_trajectory(hdf5_path, args.stride, args.seq_len, (args.img_size, args.img_size), offset=0)

    with torch.no_grad():
        if isinstance(model, FlowRewardModel):
            obs = {
                "third_person": traj["third_person"].unsqueeze(0).to(device),
                "wrist": traj["wrist"].unsqueeze(0).to(device),
                "padding_mask": traj["padding_mask"].unsqueeze(0).to(device),
                "proprio": traj["proprio"].unsqueeze(0).to(device),
            }
            rewards = model(obs)  # (1, K)
        else:
            rewards = model(
                traj["third_person"].unsqueeze(0).to(device),
                traj["wrist"].unsqueeze(0).to(device),
                traj["padding_mask"].unsqueeze(0).to(device),
            )  # (1, K)

    return {k: float(rewards[0, i]) for i, k in enumerate(args.preference_keys)}


def compute_quantile_edges(all_scores: dict[str, list[float]]) -> dict[str, list[float]]:
    """Compute per-key quantile bucket edges that give equal-frequency bins."""
    percentiles = np.linspace(0, 100, N_BUCKETS + 1)  # [0, 20, 40, 60, 80, 100]
    return {
        key: [float(v) for v in np.percentile(vals, percentiles)]
        for key, vals in all_scores.items()
    }


def _load_frames(hdf5_path: str, stride: int, seq_len: int, cell_size: int) -> np.ndarray:
    """Return agent_view frames as (T, cell_size, cell_size, 3) uint8."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        demo_key = next(iter(f["data"].keys()))
        obs = f[f"data/{demo_key}/obs"]
        total = obs["agent_view"].shape[0]
        indices = list(range(0, total, stride))[:seq_len]
        raw = obs["agent_view"][indices]
    return np.stack([cv2.resize(fr, (cell_size, cell_size)) for fr in raw])


def create_ranking_video(
    entries_sorted: list,
    key: str,
    vis_dir: str,
    args,
    n_grid: int = 100,
    cell_size: int = 96,
) -> str:
    """
    Animated grid of up to n_grid trajectories ordered worst→best.
    Each cell shows agent_view frames playing simultaneously.
    Score is overlaid on each cell.
    Returns the output path.
    """
    n = min(n_grid, len(entries_sorted))
    sel_indices = np.round(np.linspace(0, len(entries_sorted) - 1, n)).astype(int)
    selected = [entries_sorted[i] for i in sel_indices]

    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))

    # Load frames; skip silently on error
    loaded = []
    for entry in selected:
        try:
            frames = _load_frames(entry["hdf5"], args.stride, args.seq_len, cell_size)
            loaded.append((entry, frames))
        except Exception:
            pass

    if not loaded:
        return None

    T = max(f.shape[0] for _, f in loaded)
    grid_h = n_rows * cell_size
    grid_w = n_cols * cell_size

    key_safe = key.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(vis_dir, f"{key_safe}_ranking.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 10.0, (grid_w, grid_h))

    for t in range(T):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for idx, (entry, frames) in enumerate(loaded):
            row, col = divmod(idx, n_cols)
            y0, x0 = row * cell_size, col * cell_size
            if t >= frames.shape[0]:
                cell = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)  # black padding
            else:
                cell = frames[t].copy()
            # Score labels (raw / normalized / standardized) stacked at bottom
            lines = [
                f"r:{entry['raw']:.2f}",
                f"n:{entry['normalized']:.2f}",
                f"z:{entry['standardized']:.2f}",
            ]
            for li, line in enumerate(lines):
                pos = (2, cell_size - 5 - (len(lines) - 1 - li) * 11)
                cv2.putText(cell, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(cell, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
            grid[y0:y0 + cell_size, x0:x0 + cell_size] = cell
        writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    writer.release()
    return out_path


def render_trajectory_video(hdf5_path: str, out_path: str, stride: int, seq_len: int, img_size: int) -> None:
    """Write agent_view + wrist frames side-by-side as an mp4."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        demo_key = next(iter(f["data"].keys()))
        obs = f[f"data/{demo_key}/obs"]
        total = obs["agent_view"].shape[0]
        indices = list(range(0, total, stride))[:seq_len]
        tp = obs["agent_view"][indices]
        wr = obs["wrist"][indices]

    hw = img_size
    tp_r = np.stack([cv2.resize(fr, (hw, hw)) for fr in tp])
    wr_r = np.stack([cv2.resize(fr, (hw, hw)) for fr in wr])
    frames = np.concatenate([tp_r, wr_r], axis=2)  # (T, H, 2W, 3)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 10.0, (hw * 2, hw))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def visualize_distributions(
    entries_by_key: dict,
    quantile_edges: dict,
    all_scores: dict,
    vis_dir: str,
    preference_keys: list,
    args,
    use_wandb: bool = False,
    n_ranking: int = 100,
    ranking_cell_size: int = 96,
) -> None:
    """
    For each preference key:
      - Plot raw reward histogram (data-adaptive bins, no assumed range)
      - Plot quantile-bucket bar chart
      - Render one representative video per bucket (middle entry by score)
    """
    os.makedirs(vis_dir, exist_ok=True)
    wandb_log = {}

    for key in preference_keys:
        entries = entries_by_key.get(key, [])
        if not entries:
            continue

        entries_sorted = sorted(entries, key=lambda e: e["raw"])
        scores = [e["raw"] for e in entries_sorted]
        arr = np.array(scores)
        qe = quantile_edges[key]
        key_safe = key.replace(" ", "_").replace("/", "_")

        n_bins = min(40, max(10, len(arr) // 3))
        norm_arr  = np.array([e["normalized"]   for e in entries_sorted])
        std_arr   = np.array([e["standardized"] for e in entries_sorted])

        # ── Combined histogram: raw / normalized / standardized ────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle(f"{key}  (n={len(arr)})", fontsize=11)

        axes[0].hist(arr,      bins=n_bins, color="steelblue", edgecolor="white", linewidth=0.4)
        axes[0].set_title("raw reward")
        axes[0].set_xlabel("score")
        axes[0].set_ylabel("count")

        axes[1].hist(norm_arr, bins=n_bins, color="mediumseagreen", edgecolor="white", linewidth=0.4)
        axes[1].set_title("normalized  [0, 1]")
        axes[1].set_xlabel("score")

        axes[2].hist(std_arr,  bins=n_bins, color="coral", edgecolor="white", linewidth=0.4)
        axes[2].set_title("standardized  (z-score)")
        axes[2].set_xlabel("score")

        raw_hist_path = os.path.join(vis_dir, f"{key_safe}_histograms.png")
        fig.savefig(raw_hist_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        # ── Quantile-bucket bar chart ──────────────────────────────────────
        bucket_counts = [sum(1 for e in entries if e["q_bucket"] == b) for b in range(1, N_BUCKETS + 1)]
        bucket_labels = [f"B{b}\n[{qe[b-1]:.2f},{qe[b]:.2f})" for b in range(1, N_BUCKETS + 1)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, N_BUCKETS + 1), bucket_counts, tick_label=bucket_labels,
               color="coral", edgecolor="white", linewidth=0.4)
        ax.set_title(f"{key}  —  quantile bucket distribution")
        ax.set_xlabel("bucket")
        ax.set_ylabel("count")
        bucket_hist_path = os.path.join(vis_dir, f"{key_safe}_bucket_distribution.png")
        fig.savefig(bucket_hist_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        if use_wandb:
            import wandb
            wandb_log[f"{key_safe}/histograms"] = wandb.Image(raw_hist_path)
            wandb_log[f"{key_safe}/bucket_distribution"] = wandb.Image(bucket_hist_path)

        # ── Ranking grid video ─────────────────────────────────────────────
        ranking_path = create_ranking_video(
            entries_sorted, key, vis_dir, args,
            n_grid=n_ranking, cell_size=ranking_cell_size,
        )
        if ranking_path and use_wandb:
            import wandb
            wandb_log[f"{key_safe}/ranking"] = wandb.Video(ranking_path, format="mp4")

        # ── One video per bucket ───────────────────────────────────────────
        for b in range(1, N_BUCKETS + 1):
            bucket_entries = sorted(
                [e for e in entries_sorted if e["q_bucket"] == b],
                key=lambda e: e["raw"],
            )
            if not bucket_entries:
                continue
            chosen = bucket_entries[len(bucket_entries) // 2]  # median entry
            vid_path = os.path.join(vis_dir, f"{key_safe}_bucket{b}_score{chosen['raw']:.3f}.mp4")
            try:
                render_trajectory_video(chosen["hdf5"], vid_path, args.stride, args.seq_len, args.img_size)
                if use_wandb:
                    import wandb
                    wandb_log[f"{key_safe}/bucket{b}_video"] = wandb.Video(vid_path, format="mp4")
            except Exception as exc:
                print(f"  [warn] video failed for {key} bucket {b}: {exc}")

    if use_wandb:
        import wandb
        wandb.log(wandb_log)


def plot_per_frame_rewards(
    model: "DiscountedRewardModel",
    hdf5_paths: list[str],
    preference_keys: list[str],
    args,
    device: torch.device,
    out_dir: str,
    n_trajectories: int = 10,
) -> None:
    """Plot per-frame reward predictions for each axis across a trajectory.

    Generates one figure per trajectory with:
      - Top row: filmstrip of third-person camera frames (aligned with x-axis)
      - Below: K subplots (one per preference axis) showing per-frame reward values
    """
    from matplotlib.gridspec import GridSpec

    os.makedirs(out_dir, exist_ok=True)
    selected = hdf5_paths[:n_trajectories]
    K = len(preference_keys)

    for idx, hdf5_path in enumerate(selected):
        traj = load_trajectory(hdf5_path, args.stride, args.seq_len, (args.img_size, args.img_size), offset=0)

        with torch.no_grad():
            frame_rewards = model.forward_per_frame(
                traj["third_person"].unsqueeze(0).to(device),
                traj["wrist"].unsqueeze(0).to(device),
                traj["padding_mask"].unsqueeze(0).to(device),
            )  # (1, T, K)

        fr = frame_rewards[0].cpu().numpy()  # (T, K)
        padding = traj["padding_mask"].numpy()  # (T,)
        n_real = int((~padding).sum())
        fr = fr[:n_real]
        timesteps = np.arange(n_real)

        # Extract frames as (T, H, W, 3) uint8 for display
        frames = traj["third_person"][:n_real].permute(0, 2, 3, 1).numpy()  # (T, H, W, 3)

        traj_name = os.path.splitext(os.path.basename(hdf5_path))[0]
        parent_name = os.path.basename(os.path.dirname(hdf5_path))

        # Layout: filmstrip row (height 2) + K reward rows (height 2 each)
        fig_h = 2 + 2.5 * K
        fig = plt.figure(figsize=(max(12, n_real * 0.8), fig_h))
        gs = GridSpec(K + 1, n_real, figure=fig,
                      height_ratios=[1.5] + [1] * K,
                      hspace=0.35, wspace=0.05)

        # --- Filmstrip row: one image per frame ---
        for t in range(n_real):
            ax_img = fig.add_subplot(gs[0, t])
            ax_img.imshow(frames[t])
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_xlabel(str(t), fontsize=7)

        # --- Reward rows: one plot per preference axis spanning all columns ---
        for k in range(K):
            ax = fig.add_subplot(gs[k + 1, :])
            values = fr[:, k]
            ax.plot(timesteps, values, marker="o", markersize=4, linewidth=1.2)
            ax.set_ylabel(preference_keys[k], fontsize=9)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, n_real - 0.5)
            ax.set_xticks(timesteps)
            if k < K - 1:
                ax.set_xticklabels([])

        fig.suptitle(f"Per-frame rewards — {parent_name}/{traj_name}", fontsize=13, y=0.99)

        out_path = os.path.join(out_dir, f"per_frame_{idx:02d}_{parent_name}_{traj_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Per-frame plot saved → {out_path}")


def print_stats(all_scores: dict[str, list[float]], quantile_edges: dict[str, list[float]]) -> None:
    """Print per-key min/max/percentiles and ASCII histograms for both bucket types."""
    print("\n" + "=" * 70)
    print("SCORE STATISTICS (A and B rollouts combined)")
    print("=" * 70)

    for key, vals in all_scores.items():
        arr = np.array(vals)
        total = len(arr)
        p = np.percentile(arr, [10, 25, 50, 75, 90])

        print(f"\n  {key}  (n={total})")
        print(f"    min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")
        print(f"    p10={p[0]:.3f}  p25={p[1]:.3f}  p50={p[2]:.3f}  p75={p[3]:.3f}  p90={p[4]:.3f}")

        # Equal-width histogram (data-adaptive range)
        ew_edges = np.linspace(arr.min(), arr.max(), N_BUCKETS + 1)
        print(f"\n    Equal-width buckets [{arr.min():.3f}-{arr.max():.3f} → {N_BUCKETS} bins]:")
        counts_ew, _ = np.histogram(arr, bins=ew_edges)
        max_count = max(counts_ew) if max(counts_ew) > 0 else 1
        for i, count in enumerate(counts_ew):
            lo, hi = ew_edges[i], ew_edges[i + 1]
            bar = "█" * int(count / max_count * BAR_WIDTH)
            pct = 100.0 * count / total
            print(f"      [{lo:.2f}-{hi:.2f})  {bar:<{BAR_WIDTH}}  {count:3d}  ({pct:5.1f}%)")

        # Equal-frequency histogram
        qe = quantile_edges[key]
        print(f"\n    Equal-frequency buckets (quantile edges: {', '.join(f'{e:.3f}' for e in qe)}):")
        counts_qf, _ = np.histogram(arr, bins=qe)
        max_count = max(counts_qf) if max(counts_qf) > 0 else 1
        for i, count in enumerate(counts_qf):
            lo, hi = qe[i], qe[i + 1]
            bar = "█" * int(count / max_count * BAR_WIDTH)
            pct = 100.0 * count / total
            print(f"      [{lo:.3f}-{hi:.3f})  {bar:<{BAR_WIDTH}}  {count:3d}  ({pct:5.1f}%)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--preferences_dir", type=str, default="preferences")
    # These are read from the checkpoint by default; override if needed
    parser.add_argument("--stride",    type=int, default=None)
    parser.add_argument("--seq_len",   type=int, default=None)
    parser.add_argument("--img_size",  type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to write score JSON files (default: alongside each preference folder). "
                             "When set, mirrors the preference folder structure under this directory.")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Directory to save histograms and videos (default: <ckpt_dir>/vis_<ckpt_name>)")
    parser.add_argument("--wandb", action="store_true", help="Log histograms and videos to wandb")
    parser.add_argument("--wandb_project", type=str, default="reward_learning")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--n_ranking", type=int, default=100,
                        help="Number of trajectories to show in the ranking grid video")
    parser.add_argument("--ranking_cell_size", type=int, default=96,
                        help="Pixel size of each cell in the ranking grid")
    args = parser.parse_args()

    args.preferences_dir = [d.strip() for d in args.preferences_dir.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})

    for key in ("stride", "seq_len", "img_size", "embed_dim"):
        if getattr(args, key) is None:
            setattr(args, key, saved_args.get(key))
            if getattr(args, key) is None:
                parser.error(f"--{key} not found in checkpoint; pass it explicitly")

    print(f"Args: {args}")

    task = saved_args.get("task", "cube_in_three_bowls")
    args.preference_keys = TASKS[task]

    model_type = saved_args.get("model", "transformer")
    if model_type == "flow":
        model = FlowRewardModel(
            num_preferences=len(args.preference_keys),
            embed_dim=args.embed_dim,
            num_heads=saved_args.get("num_heads", 8),
            num_layers=saved_args.get("num_layers", 4),
            ffn_dim=saved_args.get("ffn_dim", 512),
            dropout=saved_args.get("dropout", 0.1),
            backbone=saved_args.get("backbone", "resnet18"),
            n_sample_steps=saved_args.get("n_sample_steps", 10),
            n_samples=saved_args.get("n_samples", 10),
            ptp=saved_args.get("ptp", False),
            action_chunk_size=saved_args.get("action_chunk_size", 16),
        ).to(device)
    elif model_type == "discounted":
        model = DiscountedRewardModel(
            num_preferences=len(args.preference_keys),
            embed_dim=args.embed_dim,
            gamma=saved_args.get("gamma", 0.99),
        ).to(device)
    elif model_type in ("qwen", "qwen_lora"):
        model = QwenRewardModel(
            num_preferences=len(args.preference_keys),
            model_name=saved_args.get("qwen_model_name", "Qwen/Qwen3-VL-4B-Instruct"),
            use_lora=(model_type == "qwen_lora"),
            lora_r=saved_args.get("lora_r", 64),
            lora_alpha=saved_args.get("lora_alpha", 16),
            reward_sigmoid=saved_args.get("reward_sigmoid", False),
            gradient_checkpointing=False,  # not needed at inference
        ).to(device)
    else:
        model = RewardModel(
            num_preferences=len(args.preference_keys),
            embed_dim=args.embed_dim,
            num_heads=saved_args.get("num_heads", 8),
            num_layers=saved_args.get("num_layers", 4),
            ffn_dim=saved_args.get("ffn_dim", 512),
            dropout=saved_args.get("dropout", 0.1),
            backbone=saved_args.get("backbone", "resnet18"),
        ).to(device)
    if isinstance(model, QwenRewardModel):
        model.load_checkpoint_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    model_folder = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
    prefix = f"reward_model_{model_folder}_{ckpt_name}"

    import h5py

    pref_dirs = sorted(
        os.path.join(root, d)
        for root in args.preferences_dir
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )

    # Collect standalone HDF5 files (not inside subdirs, e.g. demos_62.hdf5)
    standalone_hdf5 = sorted(
        os.path.join(root, f)
        for root in args.preferences_dir
        for f in os.listdir(root)
        if f.endswith(".hdf5") and os.path.isfile(os.path.join(root, f))
    )

    # --- Pass 1: score all trajectories, collect raw scores ---
    all_scores = defaultdict(list)
    results = []       # [(pref_dir, hdf5_a, raw_a, hdf5_b, raw_b)]
    solo_results = []  # [(hdf5_path, raw_scores)]

    for pref_dir in pref_dirs:
        hdf5_a = os.path.join(pref_dir, "rollout_A.hdf5")
        hdf5_b = os.path.join(pref_dir, "rollout_B.hdf5")

        if not (os.path.exists(hdf5_a) and os.path.exists(hdf5_b)):
            print(f"  [skip] {pref_dir} — missing hdf5 files")
            continue

        # Validate HDF5 structure (same check as dataset.py)
        try:
            for path in (hdf5_a, hdf5_b):
                with h5py.File(path, "r") as f:
                    demo_key = next(iter(f["data"].keys()))
                    _ = f[f"data/{demo_key}/obs/agent_view"].shape
                    _ = f[f"data/{demo_key}/obs/JOINT_POS"].shape
        except (OSError, KeyError):
            print(f"  [skip] {pref_dir} — incompatible HDF5 format")
            continue

        try:
            raw_a = score_trajectory(model, hdf5_a, args, device)
            raw_b = score_trajectory(model, hdf5_b, args, device)
        except (KeyError, OSError) as e:
            print(f"  [skip] {pref_dir} — {e}")
            continue
        results.append((pref_dir, hdf5_a, raw_a, hdf5_b, raw_b))

        for k, v in raw_a.items():
            all_scores[k].append(v)
        for k, v in raw_b.items():
            all_scores[k].append(v)

        session = os.path.basename(pref_dir)
        a_str = ", ".join(f"{k}: {v:.3f}" for k, v in raw_a.items())
        b_str = ", ".join(f"{k}: {v:.3f}" for k, v in raw_b.items())
        print(f"[{session}]")
        print(f"  A: {a_str}")
        print(f"  B: {b_str}")

    # Score standalone HDF5 files
    for hdf5_path in standalone_hdf5:
        try:
            with h5py.File(hdf5_path, "r") as f:
                demo_key = next(iter(f["data"].keys()))
                _ = f[f"data/{demo_key}/obs/agent_view"].shape
                _ = f[f"data/{demo_key}/obs/JOINT_POS"].shape
        except (OSError, KeyError):
            print(f"  [skip] {hdf5_path} — incompatible HDF5 format")
            continue
        try:
            raw = score_trajectory(model, hdf5_path, args, device)
        except (KeyError, OSError) as e:
            print(f"  [skip] {hdf5_path} — {e}")
            continue
        solo_results.append((hdf5_path, raw))
        for k, v in raw.items():
            all_scores[k].append(v)
        name = os.path.splitext(os.path.basename(hdf5_path))[0]
        s_str = ", ".join(f"{k}: {v:.3f}" for k, v in raw.items())
        print(f"[{name}]  {s_str}")

    if not results and not solo_results:
        print("No valid trajectories found.")
        return

    # --- Compute quantile edges and per-key stats from full distribution ---
    quantile_edges = compute_quantile_edges(all_scores)
    score_stats = {
        k: {
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
        }
        for k, vals in all_scores.items()
    }

    # --- Pass 2: write JSON files with all label types ---
    def _make_score_dict(raw: dict, source_hdf5: str) -> dict:
        return {
            "source_hdf5": source_hdf5,
            "raw": raw,
            "normalized":    {k: float((v - score_stats[k]["min"]) / max(score_stats[k]["max"] - score_stats[k]["min"], 1e-8))
                               for k, v in raw.items()},
            "standardized":  {k: float((v - score_stats[k]["mean"]) / max(score_stats[k]["std"], 1e-8))
                               for k, v in raw.items()},
            "buckets":         {k: to_bucket(v, score_stats[k]["min"], score_stats[k]["max"]) for k, v in raw.items()},
            "buckets_quantile": {k: to_quantile_bucket(v, quantile_edges[k]) for k, v in raw.items()},
        }

    for pref_dir, hdf5_a, raw_a, hdf5_b, raw_b in results:
        if args.output_dir:
            json_dir = os.path.join(args.output_dir, os.path.basename(pref_dir))
            os.makedirs(json_dir, exist_ok=True)
        else:
            json_dir = pref_dir
        for raw, hdf5, suffix in [(raw_a, hdf5_a, "A"), (raw_b, hdf5_b, "B")]:
            out = _make_score_dict(raw, hdf5)
            out_path = os.path.join(json_dir, f"{prefix}_rollout_{suffix}_score.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

    # Write standalone trajectory score JSONs
    if solo_results:
        solo_dir = args.output_dir or os.path.dirname(args.ckpt)
        os.makedirs(solo_dir, exist_ok=True)
        for i, (hdf5_path, raw) in enumerate(solo_results):
            out = _make_score_dict(raw, hdf5_path)
            name = os.path.splitext(os.path.basename(hdf5_path))[0]
            out_path = os.path.join(solo_dir, f"{prefix}_score_{name}.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

    n_paired = len(results) * 2
    n_solo = len(solo_results)
    print(f"\nDone. Scored {n_paired} paired + {n_solo} standalone trajectories.")

    # --- Build per-key entry list for visualization ---
    entries_by_key = defaultdict(list)
    all_scored = []  # flat list of (hdf5_path, raw_dict, session_name, rollout_label)
    for pref_dir, hdf5_a, raw_a, hdf5_b, raw_b in results:
        session = os.path.basename(pref_dir)
        for raw, hdf5, rlabel in [(raw_a, hdf5_a, "A"), (raw_b, hdf5_b, "B")]:
            all_scored.append((hdf5, raw, session, rlabel))
    for hdf5_path, raw in solo_results:
        name = os.path.splitext(os.path.basename(hdf5_path))[0]
        all_scored.append((hdf5_path, raw, name, "solo"))

    for hdf5, raw, _, _ in all_scored:
        for key, score in raw.items():
            st = score_stats[key]
            entries_by_key[key].append({
                "hdf5": hdf5,
                "raw": score,
                "normalized":   (score - st["min"]) / max(st["max"] - st["min"], 1e-8),
                "standardized": (score - st["mean"]) / max(st["std"], 1e-8),
                "q_bucket": to_quantile_bucket(score, quantile_edges[key]),
            })

    vis_dir = args.vis_dir or os.path.join(os.path.dirname(args.ckpt), f"vis_{ckpt_name}")

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or ckpt_name, config=vars(args))

    visualize_distributions(
        entries_by_key, quantile_edges, all_scores, vis_dir,
        args.preference_keys, args, use_wandb=args.wandb,
        n_ranking=args.n_ranking, ranking_cell_size=args.ranking_cell_size,
    )
    print(f"Visualizations saved → {vis_dir}")

    # --- Per-frame reward plots for discounted models ---
    if model_type == "discounted":
        print("\nGenerating per-frame reward plots...")
        all_hdf5 = [hdf5 for hdf5, _, _, _ in all_scored]
        plot_per_frame_rewards(
            model, all_hdf5, args.preference_keys, args, device,
            out_dir=os.path.join(vis_dir, "per_frame_rewards"),
            n_trajectories=10,
        )

    # --- Stats ---
    print_stats(all_scores, quantile_edges)

    stats_out = {}
    for key, vals in all_scores.items():
        arr = np.array(vals)
        ew_edges_j = np.linspace(arr.min(), arr.max(), N_BUCKETS + 1)
        counts_ew, _ = np.histogram(arr, bins=ew_edges_j)
        qe = quantile_edges[key]
        counts_qf, _ = np.histogram(arr, bins=qe)
        stats_out[key] = {
            "n": len(vals),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "histogram_equal_width": {
                f"{ew_edges_j[i]:.3f}-{ew_edges_j[i+1]:.3f}": int(counts_ew[i])
                for i in range(N_BUCKETS)
            },
            "quantile_edges": qe,
            "histogram_equal_freq": {
                f"{qe[i]:.3f}-{qe[i+1]:.3f}": int(counts_qf[i])
                for i in range(N_BUCKETS)
            },
        }

    if args.output_dir:
        stats_dir = args.output_dir
    elif len(args.preferences_dir) == 1:
        stats_dir = args.preferences_dir[0]
    else:
        stats_dir = os.path.dirname(args.ckpt)
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f"{prefix}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"Stats saved → {stats_path}")

    # --- Save npz for downstream analysis ---
    K = len(args.preference_keys)

    hdf5_paths, sessions, rollouts = [], [], []
    scores_rows, norm_rows, std_rows, qbucket_rows = [], [], [], []

    for hdf5, raw, session, rlabel in all_scored:
        hdf5_paths.append(hdf5)
        sessions.append(session)
        rollouts.append(rlabel)
        st_row, no_row, zs_row, qb_row = [], [], [], []
        for key in args.preference_keys:
            st = score_stats[key]
            v = raw[key]
            st_row.append(v)
            no_row.append((v - st["min"]) / max(st["max"] - st["min"], 1e-8))
            zs_row.append((v - st["mean"]) / max(st["std"], 1e-8))
            qb_row.append(to_quantile_bucket(v, quantile_edges[key]))
        scores_rows.append(st_row)
        norm_rows.append(no_row)
        std_rows.append(zs_row)
        qbucket_rows.append(qb_row)

    qedges_matrix = np.array([quantile_edges[k] for k in args.preference_keys])  # (K, N_BUCKETS+1)

    npz_path = os.path.join(vis_dir, f"{prefix}_data.npz")
    np.savez(
        npz_path,
        preference_keys=np.array(args.preference_keys),
        hdf5_paths=np.array(hdf5_paths),
        sessions=np.array(sessions),
        rollouts=np.array(rollouts),
        scores=np.array(scores_rows, dtype=np.float32),          # (N, K) raw
        normalized=np.array(norm_rows, dtype=np.float32),        # (N, K)
        standardized=np.array(std_rows, dtype=np.float32),       # (N, K)
        q_buckets=np.array(qbucket_rows, dtype=np.int8),         # (N, K)
        stat_min=np.array([score_stats[k]["min"]  for k in args.preference_keys], dtype=np.float32),
        stat_max=np.array([score_stats[k]["max"]  for k in args.preference_keys], dtype=np.float32),
        stat_mean=np.array([score_stats[k]["mean"] for k in args.preference_keys], dtype=np.float32),
        stat_std=np.array([score_stats[k]["std"]  for k in args.preference_keys], dtype=np.float32),
        quantile_edges=qedges_matrix.astype(np.float32),
    )
    print(f"Data saved → {npz_path}")

    # --- Analysis plots ---
    analysis_dir = args.output_dir or vis_dir
    os.makedirs(analysis_dir, exist_ok=True)
    rd = RewardData(npz_path)
    plot_scatter_matrix(rd, os.path.join(analysis_dir, f"{prefix}_scatter_matrix.png"))
    plot_dim_histograms(rd, os.path.join(analysis_dir, f"{prefix}_dim_histograms.png"))

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
