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
    python infer.py --ckpt <path to checkpoint>
    python infer.py --ckpt <path to checkpoint> \
                    --preferences_dir preferences
"""

import argparse
import json
import math
import os
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import load_trajectory, load_trajectories_all_offsets
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


def _qwen_per_frame(model, traj, keys, device) -> np.ndarray:
    """Per-step rewards (n_real, K) for one loaded trajectory (open_cum/discounted)."""
    tp = traj["third_person"].unsqueeze(0).to(device)
    wr = traj["wrist"].unsqueeze(0).to(device)
    pm = traj["padding_mask"].unsqueeze(0).to(device)
    n_real = int((~traj["padding_mask"]).sum())
    with torch.no_grad():
        if model.open_cum:
            cols = [model.forward_per_frame(tp, wr, pm, axis_labels=[k])[0, :, 0]
                    for k in keys]
            pf = torch.stack(cols, dim=-1)            # (T, K)
        else:
            pf = model.forward_per_frame(tp, wr, pm)[0]  # (T, K)
    return pf[:n_real].float().cpu().numpy()


def score_trajectory(model, hdf5_path: str, args, device: torch.device):
    """Return ``(raw_cumulative, per_frame)`` for one trajectory.

    ``raw_cumulative`` is ``{axis: float}`` (buckets are added later, once the
    quantile edges are known). ``per_frame`` is an ``(n_real, K)`` float array
    of per-step rewards, or ``None`` for models whose trajectory score is not a
    sum of per-step scores.

    For the qwen ``open_cum`` / ``discounted`` variants the trajectory score is
    by construction the sum of per-step rewards (see ``_forward_open_cum`` /
    ``_forward_discounted``), so we compute the per-step rewards once and sum
    them — the cumulative score matches the direct forward at no extra cost.

    With ``args.dense`` (and stride > 1) the qwen variants are scored at every
    temporal offset 0..stride-1 and interleaved into a full-resolution per-step
    array indexed by true frame index — stride× more forward passes, but the
    curve gets stride× more points. (Cumulative magnitudes grow accordingly, so
    raw sums are only comparable within a dense run.)
    """
    keys = args.preference_keys
    is_qwen_summable = isinstance(model, QwenRewardModel) and (
        getattr(model, "open_cum", False) or getattr(model, "discounted", False)
    )
    dense = getattr(args, "dense", False) and (args.stride or 1) > 1

    if is_qwen_summable and dense:
        offsets = list(range(args.stride))
        trajs = load_trajectories_all_offsets(
            hdf5_path, args.stride, args.seq_len,
            (args.img_size, args.img_size), offsets=offsets,
        )
        dense_vals = {}  # true_frame_index -> (K,) per-step rewards
        for o, tr in zip(offsets, trajs):
            pf_o = _qwen_per_frame(model, tr, keys, device)  # (n_real_o, K)
            for j in range(pf_o.shape[0]):
                dense_vals[o + j * args.stride] = pf_o[j]
        if dense_vals:
            length = max(dense_vals) + 1
            pf = np.full((length, len(keys)), np.nan, dtype=np.float32)
            for idx, vec in dense_vals.items():
                pf[idx] = vec
        else:
            pf = np.zeros((0, len(keys)), dtype=np.float32)
        raw = {k: float(np.nansum(pf[:, i])) for i, k in enumerate(keys)}
        return raw, pf

    traj = load_trajectory(hdf5_path, args.stride, args.seq_len, (args.img_size, args.img_size), offset=0)

    tp = traj["third_person"].unsqueeze(0).to(device)
    wr = traj["wrist"].unsqueeze(0).to(device)
    pm = traj["padding_mask"].unsqueeze(0).to(device)
    n_real = int((~traj["padding_mask"]).sum())

    with torch.no_grad():
        if isinstance(model, FlowRewardModel):
            obs = {
                "third_person": tp,
                "wrist": wr,
                "padding_mask": pm,
                "proprio": traj["proprio"].unsqueeze(0).to(device),
            }
            rewards = model(obs)  # (1, K)
            return {k: float(rewards[0, i]) for i, k in enumerate(keys)}, None

        # Per-step-summable qwen variants: derive the cumulative score from the
        # per-step rewards so the curve and the score are guaranteed consistent.
        if is_qwen_summable:
            pf = _qwen_per_frame(model, traj, keys, device)
            raw = {k: float(np.nansum(pf[:, i])) for i, k in enumerate(keys)}
            return raw, pf

        if getattr(args, "is_open_qwen", False):
            # Open Qwen (non-cumulative): single reward head, run once per axis
            # with the axis name in the prompt. No per-step decomposition.
            result = {}
            for k in keys:
                rewards = model(tp, wr, pm, axis_labels=[k])  # (1, 1)
                result[k] = float(rewards[0, 0])
            return result, None

        rewards = model(
            tp, wr, pm,
        )  # (1, K)
        raw = {k: float(rewards[0, i]) for i, k in enumerate(keys)}

        per_frame = None
        if isinstance(model, DiscountedRewardModel):
            try:
                pf = model.forward_per_frame(tp, wr, pm)[0][:n_real]
                per_frame = pf.float().cpu().numpy()
            except Exception:
                per_frame = None
        return raw, per_frame


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

        # Combined histogram: raw / normalized / standardized
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

        # Quantile-bucket bar chart
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

        # Ranking grid video
        ranking_path = create_ranking_video(
            entries_sorted, key, vis_dir, args,
            n_grid=n_ranking, cell_size=ranking_cell_size,
        )
        if ranking_path and use_wandb:
            import wandb
            wandb_log[f"{key_safe}/ranking"] = wandb.Video(ranking_path, format="mp4")

        # One video per bucket
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

        # Filmstrip row: one image per frame
        for t in range(n_real):
            ax_img = fig.add_subplot(gs[0, t])
            ax_img.imshow(frames[t])
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_xlabel(str(t), fontsize=7)

        # Reward rows: one plot per preference axis spanning all columns
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


def plot_success_separation(
    results: list,
    solo_results: list,
    score_stats: dict,
    preference_keys: list[str],
    output_dir: str,
    prefix: str,
) -> None:
    """Generate box-plot and histogram comparing demos vs successful/failed rollouts (standardized)."""
    dims = preference_keys

    # Categorize rollouts by success using preference.json
    rollouts_success, rollouts_fail = [], []
    for pref_dir, hdf5_a, raw_a, hdf5_b, raw_b in results:
        pref_file = os.path.join(pref_dir, "preference.json")
        if not os.path.exists(pref_file):
            continue
        with open(pref_file) as f:
            pref_data = json.load(f)
        for raw, suffix in [(raw_a, "A"), (raw_b, "B")]:
            succeeded = pref_data.get(f"rollout_{suffix}", {}).get("succeeded", None)
            if succeeded is None:
                continue
            std_scores = {k: (v - score_stats[k]["mean"]) / max(score_stats[k]["std"], 1e-8)
                          for k, v in raw.items()}
            (rollouts_success if succeeded else rollouts_fail).append(std_scores)

    # Demos are standalone trajectories
    demo_std = []
    for hdf5_path, raw in solo_results:
        std_scores = {k: (v - score_stats[k]["mean"]) / max(score_stats[k]["std"], 1e-8)
                      for k, v in raw.items()}
        demo_std.append(std_scores)

    n_demos = len(demo_std)
    n_success = len(rollouts_success)
    n_fail = len(rollouts_fail)
    if n_success + n_fail == 0:
        print("  [skip] No rollouts with success labels found — skipping success separation plots.")
        return

    def collect(data_list):
        return {d: [r[d] for r in data_list] for d in dims}

    demo_scores = collect(demo_std) if demo_std else {d: [] for d in dims}
    success_scores = collect(rollouts_success)
    fail_scores = collect(rollouts_fail)

    colors = ["#4dabf7", "#51cf66", "#ff6b6b"]

    # Plot 1: per-dimension box plots (demos vs success vs fail)
    n_cols = min(5, len(dims))
    n_rows = int(np.ceil(len(dims) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    categories = []
    cat_labels = []
    cat_colors = []
    if demo_std:
        categories.append(demo_scores)
        cat_labels.append("Demos")
        cat_colors.append(colors[0])
    categories.append(success_scores)
    cat_labels.append("Rollout\nSuccess")
    cat_colors.append(colors[1])
    categories.append(fail_scores)
    cat_labels.append("Rollout\nFail")
    cat_colors.append(colors[2])

    for i, dim in enumerate(dims):
        ax = axes[i]
        data = [cat[dim] for cat in categories]
        bp = ax.boxplot(data, tick_labels=cat_labels, patch_artist=True, widths=0.6)
        for j, box in enumerate(bp["boxes"]):
            box.set_facecolor(cat_colors[j])
            box.set_alpha(0.7)
        ax.set_title(dim, fontsize=10, fontweight="bold")
        ax.set_ylabel("Standardized Score")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3)
    for i in range(len(dims), len(axes)):
        axes[i].set_visible(False)

    title_parts = []
    if demo_std:
        title_parts.append(f"Demos: {n_demos}")
    title_parts += [f"Rollout Success: {n_success}", f"Rollout Fail: {n_fail}"]
    plt.suptitle(
        f"Demos vs Rollouts (Standardized Scores)\n{', '.join(title_parts)}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, f"{prefix}_demos_vs_rollouts_boxplot.png")
    plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: overall histogram
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fail_overall = [np.mean([r[d] for d in dims]) for r in rollouts_fail]
    success_overall = [np.mean([r[d] for d in dims]) for r in rollouts_success]
    demo_overall = [np.mean([r[d] for d in dims]) for r in demo_std] if demo_std else []

    ax2.hist(fail_overall, bins=30, alpha=0.6, color="#ff6b6b",
             label=f"Rollout Fail (n={n_fail})", edgecolor="black")
    ax2.hist(success_overall, bins=30, alpha=0.6, color="#51cf66",
             label=f"Rollout Success (n={n_success})", edgecolor="black")
    if demo_overall:
        ax2.hist(demo_overall, bins=30, alpha=0.6, color="#4dabf7",
                 label=f"Demos (n={n_demos})", edgecolor="black")
    ax2.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Mean (z=0)")
    ax2.set_xlabel("Mean Standardized Score (across all dimensions)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Overall Score Distribution (Standardized)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(output_dir, f"{prefix}_demos_vs_rollouts_histogram.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: per-axis histograms
    n_cols_h = min(5, len(dims))
    n_rows_h = int(np.ceil(len(dims) / n_cols_h))
    fig3, axes3 = plt.subplots(n_rows_h, n_cols_h, figsize=(4.8 * n_cols_h, 4 * n_rows_h))
    axes3 = np.atleast_1d(axes3).flatten()

    for i, dim in enumerate(dims):
        ax = axes3[i]
        all_vals = fail_scores[dim] + success_scores[dim] + demo_scores[dim]
        if not all_vals:
            continue
        lo, hi = min(all_vals), max(all_vals)
        bins = np.linspace(lo - 0.1, hi + 0.1, 25)
        ax.hist(fail_scores[dim], bins=bins, alpha=0.6, color="#ff6b6b",
                label="Rollout Fail", edgecolor="black", linewidth=0.3)
        ax.hist(success_scores[dim], bins=bins, alpha=0.6, color="#51cf66",
                label="Rollout Success", edgecolor="black", linewidth=0.3)
        if demo_scores[dim]:
            ax.hist(demo_scores[dim], bins=bins, alpha=0.6, color="#4dabf7",
                    label="Demos", edgecolor="black", linewidth=0.3)
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.4)
        ax.set_title(dim, fontsize=10, fontweight="bold")
        ax.set_xlabel("Standardized Score")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for i in range(len(dims), len(axes3)):
        axes3[i].set_visible(False)

    plt.suptitle(
        f"Per-Axis Score Distribution (Standardized)\n{', '.join(title_parts)}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    per_axis_path = os.path.join(output_dir, f"{prefix}_demos_vs_rollouts_per_axis.png")
    plt.savefig(per_axis_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print(f"Success-separation plots saved → {boxplot_path}")
    print(f"                                → {hist_path}")
    print(f"                                → {per_axis_path}")


def create_overall_ranking_video(
    rd: "RewardData",
    output_dir: str,
    prefix: str,
    args,
    n_sample: int = 40,
    cell_size: int = 128,
):
    """
    For each preference axis, write a grid video of n_sample random rollouts
    ordered worst→best by that axis's standardized score.

    Each cell shows agent_view frames with the rank, axis z-score, and session
    name overlaid. The same random subset is used across axes so videos are
    comparable. Returns a list of output paths.
    """
    N = rd.N
    if N == 0:
        return []

    # Pick one random subset and reuse it across axes for comparability
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(N, size=min(n_sample, N), replace=False)

    # Pre-load frames once per trajectory in the sample
    frames_by_idx = {}
    for idx in sample_idx:
        hdf5 = str(rd.hdf5_paths[idx])
        try:
            frames_by_idx[idx] = _load_frames(hdf5, args.stride, args.seq_len, cell_size)
        except Exception:
            pass

    if not frames_by_idx:
        return []

    n = len(sample_idx)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    grid_h = n_rows * cell_size
    grid_w = n_cols * cell_size

    out_paths = []
    for ki, key in enumerate(rd.keys):
        # Sort the sample by this axis's standardized score (worst first)
        axis_std = rd.standardized[sample_idx, ki]
        order = np.argsort(axis_std)
        ordered_idx = sample_idx[order]
        ordered_z = axis_std[order]

        loaded = [
            (pos, idx, frames_by_idx[idx], float(z))
            for pos, (idx, z) in enumerate(zip(ordered_idx, ordered_z))
            if idx in frames_by_idx
        ]
        if not loaded:
            continue

        T = max(f.shape[0] for _, _, f, _ in loaded)
        key_safe = key.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(output_dir, f"{prefix}_overall_ranking_{key_safe}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 10.0, (grid_w, grid_h))

        for t in range(T):
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            for pos, idx, frames, z in loaded:
                row, col = divmod(pos, n_cols)
                y0, x0 = row * cell_size, col * cell_size
                if t >= frames.shape[0]:
                    cell = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                else:
                    cell = frames[t].copy()

                # Rank + axis z-score at top
                rank_line = f"#{pos+1}  z={z:.2f}"
                cv2.putText(cell, rank_line, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(cell, rank_line, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

                # Axis name just below top label
                axis_line = key[:18]
                cv2.putText(cell, axis_line, (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(cell, axis_line, (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 255, 200), 1, cv2.LINE_AA)

                # Session + rollout label at bottom
                session = str(rd.sessions[idx])
                rollout = str(rd.rollouts[idx])
                info_line = f"{session[-8:]}/{rollout}" if len(session) > 8 else f"{session}/{rollout}"
                cv2.putText(cell, info_line, (2, cell_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(cell, info_line, (2, cell_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

                grid[y0:y0 + cell_size, x0:x0 + cell_size] = cell
            writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        writer.release()
        print(f"Overall ranking video saved → {out_path}")
        out_paths.append(out_path)

    return out_paths


def plot_session_ranking(
    rd: "RewardData",
    output_dir: str,
    prefix: str,
) -> None:
    """Bar chart of sessions ranked by mean standardized score, with per-dimension breakdown."""
    sessions = np.unique(rd.sessions)
    if len(sessions) < 2:
        print("  [skip] Not enough sessions for ranking plot.")
        return

    # Compute mean standardized score per session (overall and per-dim)
    rows = []
    for s in sessions:
        idx = rd.for_session(s)
        overall = float(rd.standardized[idx].mean())
        per_dim = {k: float(rd.standardized[idx, i].mean()) for i, k in enumerate(rd.keys)}
        rows.append({"session": s, "overall": overall, **per_dim})
    rows.sort(key=lambda r: r["overall"])

    session_labels = [r["session"] for r in rows]
    overall_scores = [r["overall"] for r in rows]
    K = rd.K

    # Plot 1: Overall ranking bar chart
    fig_height = max(6, len(sessions) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    colors = ["#51cf66" if s >= 0 else "#ff6b6b" for s in overall_scores]
    y_pos = np.arange(len(sessions))
    ax.barh(y_pos, overall_scores, color=colors, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(session_labels, fontsize=7)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean Standardized Score (across all dimensions)", fontsize=10)
    ax.set_title(f"Session Ranking by Overall Reward Score\n({len(sessions)} sessions)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    overall_path = os.path.join(output_dir, f"{prefix}_session_ranking.png")
    plt.savefig(overall_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Per-dimension ranking heatmap
    dim_matrix = np.array([[r[k] for k in rd.keys] for r in rows])  # (n_sessions, K)
    fig, ax = plt.subplots(figsize=(max(8, K * 1.2), fig_height))
    vmax = max(abs(dim_matrix.min()), abs(dim_matrix.max()))
    im = ax.imshow(dim_matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(sessions)))
    ax.set_yticklabels(session_labels, fontsize=7)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(rd.keys, fontsize=8, rotation=45, ha="right")
    ax.set_title("Session × Dimension Mean Standardized Scores", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Standardized Score", shrink=0.8)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, f"{prefix}_session_ranking_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Session ranking plots saved → {overall_path}")
    print(f"                            → {heatmap_path}")


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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (flow model sampling + video sampling)")
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
    parser.add_argument("--task", type=str, default=None,
                        help="If set, override the task stored in the checkpoint and run "
                             "inference using this task's preference axes instead.")
    parser.add_argument("--dense", action="store_true",
                        help="Score qwen open_cum/discounted trajectories at every temporal "
                             "offset 0..stride-1 and interleave into full-resolution per-step "
                             "rewards (stride× more forward passes; denser plot points).")
    args = parser.parse_args()

    args.preferences_dir = [d.strip() for d in args.preferences_dir.split(",")]

    torch.manual_seed(args.seed)

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
    if args.dense and (args.stride or 1) > 1:
        print(f"[dense] scoring at all {args.stride} temporal offsets — "
              f"~{args.stride}x more forward passes per trajectory.")

    saved_task = saved_args.get("task", "cube_in_three_bowls")
    if args.task is not None:
        if args.task not in TASKS:
            parser.error(f"--task '{args.task}' not in TASKS (known: {sorted(TASKS)})")
        if args.task != saved_task:
            print(f"[task override] checkpoint trained on '{saved_task}', "
                  f"running inference on '{args.task}'")
        task = args.task
    else:
        task = saved_task
    args.preference_keys = [k.lower() for k in TASKS[task]]

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
    elif model_type in ("qwen", "qwen_lora", "qwen_open",
                         "qwen_discounted", "qwen_open_discounted",
                         "qwen_open_cum"):
        is_open = model_type in ("qwen_open", "qwen_open_discounted", "qwen_open_cum")
        is_discounted = model_type in ("qwen_discounted", "qwen_open_discounted")
        is_open_cum = model_type == "qwen_open_cum"
        model = QwenRewardModel(
            num_preferences=1 if is_open else len(args.preference_keys),
            model_name=saved_args.get("qwen_model_name", "Qwen/Qwen3-VL-4B-Instruct"),
            use_lora=(model_type == "qwen_lora"),
            lora_r=saved_args.get("lora_r", 64),
            lora_alpha=saved_args.get("lora_alpha", 16),
            reward_sigmoid=saved_args.get("reward_sigmoid", False),
            gradient_checkpointing=False,  # not needed at inference
            discounted=is_discounted,
            open_cum=is_open_cum,
        ).to(device)
        args.is_open_qwen = is_open
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
    if not hasattr(args, "is_open_qwen"):
        args.is_open_qwen = False
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

    # Collect standalone HDF5 files (not inside subdirs, e.g. demos_62.hdf5).
    standalone_hdf5 = sorted(
        os.path.join(root, f)
        for root in args.preferences_dir
        for f in os.listdir(root)
        if f.endswith(".hdf5")
        and os.path.isfile(os.path.join(root, f))
    )

    # Pass 1: score all trajectories, collect raw scores
    all_scores = defaultdict(list)
    results = []       # [(pref_dir, hdf5_a, raw_a, hdf5_b, raw_b)]
    solo_results = []  # [(hdf5_path, raw_scores)]
    per_frame_by_path = {}  # hdf5_path -> (n_real, K) per-step rewards (or None)

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
            raw_a, pf_a = score_trajectory(model, hdf5_a, args, device)
            raw_b, pf_b = score_trajectory(model, hdf5_b, args, device)
        except (KeyError, OSError) as e:
            print(f"  [skip] {pref_dir} — {e}")
            continue
        results.append((pref_dir, hdf5_a, raw_a, hdf5_b, raw_b))
        per_frame_by_path[hdf5_a] = pf_a
        per_frame_by_path[hdf5_b] = pf_b

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
        except (OSError, KeyError, StopIteration):
            print(f"  [skip] {hdf5_path} — incompatible HDF5 format")
            continue
        try:
            raw, pf = score_trajectory(model, hdf5_path, args, device)
        except (KeyError, OSError) as e:
            print(f"  [skip] {hdf5_path} — {e}")
            continue
        solo_results.append((hdf5_path, raw))
        per_frame_by_path[hdf5_path] = pf
        for k, v in raw.items():
            all_scores[k].append(v)
        name = os.path.splitext(os.path.basename(hdf5_path))[0]
        s_str = ", ".join(f"{k}: {v:.3f}" for k, v in raw.items())
        print(f"[{name}]  {s_str}")

    if not results and not solo_results:
        print("No valid trajectories found.")
        return

    # Compute quantile edges and per-key stats from full distribution
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

    # Per-timestep population stats: for each axis and timestep index,
    # mean/std of the per-step reward across every trajectory that reaches
    # that timestep. Used by plot_single's --standardize_per_timestep. 
    per_step_vals = defaultdict(lambda: defaultdict(list))  # axis -> t -> [values]
    for pf in per_frame_by_path.values():
        if pf is None:
            continue
        for t in range(pf.shape[0]):
            for i, key in enumerate(args.preference_keys):
                v = pf[t, i]
                if np.isfinite(v):
                    per_step_vals[key][t].append(float(v))
    ts_stats = {}  # axis -> list over t of {"mean", "std"}
    for key in args.preference_keys:
        per_t = per_step_vals.get(key, {})
        t_max = (max(per_t) + 1) if per_t else 0
        ts_stats[key] = [
            {
                "mean": float(np.mean(per_t[t])) if per_t.get(t) else 0.0,
                "std":  float(np.std(per_t[t]))  if per_t.get(t) else 0.0,
            }
            for t in range(t_max)
        ]

    # Pass 2: write JSON files with all label types
    dense_mode = bool(args.dense and (args.stride or 1) > 1)

    def _clean(x) -> float | None:
        # JSON has no NaN/Inf; emit null so plot_single reads valid JSON.
        x = float(x)
        return x if math.isfinite(x) else None

    def _make_score_dict(raw: dict, source_hdf5: str) -> dict:
        pf = per_frame_by_path.get(source_hdf5)  # (n_real, K) or None
        per_frame = None
        if pf is not None:
            per_frame = {
                k: [_clean(pf[t, i]) for t in range(pf.shape[0])]
                for i, k in enumerate(args.preference_keys)
            }
        return {
            "source_hdf5": source_hdf5,
            "raw": raw,
            "normalized":    {k: float((v - score_stats[k]["min"]) / max(score_stats[k]["max"] - score_stats[k]["min"], 1e-8))
                               for k, v in raw.items()},
            "standardized":  {k: float((v - score_stats[k]["mean"]) / max(score_stats[k]["std"], 1e-8))
                               for k, v in raw.items()},
            "buckets":         {k: to_bucket(v, score_stats[k]["min"], score_stats[k]["max"]) for k, v in raw.items()},
            "buckets_quantile": {k: to_quantile_bucket(v, quantile_edges[k]) for k, v in raw.items()},
            # Everything plot_single needs to render this trajectory standalone:
            "per_frame": per_frame,
            "norm_stats": {
                k: {
                    "mean": score_stats[k]["mean"],
                    "std":  score_stats[k]["std"],
                    "min":  score_stats[k]["min"],
                    "max":  score_stats[k]["max"],
                }
                for k in raw
            },
            # Per-timestep population mean/std, aligned 1:1 with per_frame above.
            "norm_stats_per_timestep": (
                {k: ts_stats[k][: pf.shape[0]] for k in args.preference_keys}
                if pf is not None else None
            ),
            "meta": {
                "stride": args.stride,
                "seq_len": args.seq_len,
                "img_size": args.img_size,
                "preference_keys": list(args.preference_keys),
                "model_type": model_type,
                "ckpt": args.ckpt,
                "n_frames": int(pf.shape[0]) if pf is not None else None,
                "dense": bool(dense_mode),
                # How plot_single should load frames so they align 1:1 with
                # per_frame: dense per_frame is indexed by true frame (stride 1).
                "frame_stride": 1 if dense_mode else args.stride,
                "frame_seq_len": int(pf.shape[0]) if (dense_mode and pf is not None) else args.seq_len,
            },
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

    # Build per-key entry list for visualization
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

    # Per-frame reward plots for discounted models
    if model_type == "discounted":
        print("\nGenerating per-frame reward plots...")
        all_hdf5 = [hdf5 for hdf5, _, _, _ in all_scored]
        plot_per_frame_rewards(
            model, all_hdf5, args.preference_keys, args, device,
            out_dir=os.path.join(vis_dir, "per_frame_rewards"),
            n_trajectories=10,
        )

    # Stats
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
            "std": float(arr.std()),
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

    # Save npz for downstream analysis
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

    # Analysis plots
    analysis_dir = args.output_dir or vis_dir
    os.makedirs(analysis_dir, exist_ok=True)
    rd = RewardData(npz_path)
    plot_scatter_matrix(rd, os.path.join(analysis_dir, f"{prefix}_scatter_matrix.png"))
    plot_dim_histograms(rd, os.path.join(analysis_dir, f"{prefix}_dim_histograms.png"))

    # Session ranking plots and video
    plot_session_ranking(rd, analysis_dir, prefix)
    create_overall_ranking_video(rd, analysis_dir, prefix, args)

    # Success separation plots
    plot_success_separation(
        results, solo_results, score_stats,
        args.preference_keys, analysis_dir, prefix,
    )

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
