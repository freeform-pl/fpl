"""
Plot a single trajectory from precomputed infer.py outputs.

infer.py scores every trajectory and writes, per trajectory, a `*_score*.json`
that already contains:
  - per_frame:    {axis: [reward_t0, reward_t1, ...]}  (per-step rewards)
  - raw:          {axis: cumulative reward}
  - standardized: {axis: z-score using the population mean/std}
  - norm_stats:   {axis: {mean, std, min, max}}  (computed over all trajectories)
  - meta:         {stride, seq_len, img_size, preference_keys, model_type, ckpt}

This script renders one such trajectory — a filmstrip + per-axis reward curves
(PNG) and an animated MP4 — WITHOUT running the model. It only reads the JSON
and loads the raw frames from `source_hdf5`.

Usage:
  python plot_single.py --score_json .../reward_model_..._score_demos_1.json
  python plot_single.py --score_json <json> --out fig.png --no-video --fps 2
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec

from dataset import load_trajectory


def _load_frames(hdf5_path: str, meta: dict, n_real: int) -> np.ndarray:
    """Load third-person frames aligned 1:1 with the per-step scores.

    Uses frame_stride/frame_seq_len when present (dense runs index per_frame by
    true frame, i.e. stride 1); falls back to stride/seq_len for older JSONs.
    """
    stride = meta.get("frame_stride", meta["stride"])
    seq_len = max(meta.get("frame_seq_len", meta["seq_len"]), n_real)
    traj = load_trajectory(
        hdf5_path, stride, seq_len,
        (meta["img_size"], meta["img_size"]), offset=0,
    )
    frames = traj["third_person"][:n_real].permute(0, 2, 3, 1).numpy()
    if frames.dtype != np.uint8:
        frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    return frames


def _row_title(key: str, raw: dict, std: dict, mode: str) -> str:
    """Per-axis annotation depending on the standardization mode.

    - "raw":      Σraw (curve sum) and the whole-trajectory z-score.
    - "traj":     curve already sums to the z-score, so label it that way.
    - "timestep": per-timestep z; the curve sum isn't a z-score, so show Σraw.
    """
    parts = []
    if key in raw and raw[key] is not None:
        parts.append(f"Σraw = {raw[key]:+.3f}")
    if mode != "timestep" and key in std and std[key] is not None:
        label = "Σ = z = " if mode == "traj" else "z = "
        parts.append(f"{label}{std[key]:+.2f}")
    return "   ".join(parts)


def _standardize_per_timestep(rewards: np.ndarray, keys: list[str], ts_stats: dict) -> np.ndarray:
    """Standardize each per-step value against the population at the SAME timestep.

    ``ts_stats[axis]`` is a list (indexed by timestep) of {"mean", "std"} over
    all trajectories that reached that timestep. Maps r_t -> (r_t - μ_t) / σ_t,
    so each value reads as "stds above/below the population's per-step reward at
    this timestep." NaNs (padded frames) are preserved.
    """
    out = np.full_like(rewards, np.nan)
    T = rewards.shape[0]
    for i, key in enumerate(keys):
        stats = ts_stats.get(key, [])
        for t in range(T):
            v = rewards[t, i]
            if not np.isfinite(v):
                continue
            if t < len(stats):
                mean = float(stats[t].get("mean", 0.0))
                std = float(stats[t].get("std", 0.0))
            else:
                mean, std = 0.0, 1.0
            # Zero spread (e.g. only one trajectory reached this timestep) → 0.
            out[t, i] = (v - mean) / std if std > 1e-8 else 0.0
    return out


def _standardize_per_frame(rewards: np.ndarray, keys: list[str], norm_stats: dict) -> np.ndarray:
    """Rescale per-step rewards so each axis's curve sums to its z-score.

    For axis k with population (cumulative-score) mean μ_k and std σ_k, and
    n real frames, we map r_t -> (r_t - μ_k / n) / σ_k. Then Σ_t equals
    (Σ_t r_t - μ_k) / σ_k = the standardized cumulative z-score. NaNs (padded
    frames) are preserved.
    """
    out = np.full_like(rewards, np.nan)
    for i, key in enumerate(keys):
        col = rewards[:, i]
        finite = np.isfinite(col)
        n = int(finite.sum())
        st = norm_stats.get(key, {})
        mean = float(st.get("mean", 0.0))
        std = max(float(st.get("std", 1.0)), 1e-8)
        offset = (mean / n) if n > 0 else 0.0
        out[finite, i] = (col[finite] - offset) / std
    return out


def _plot(frames, rewards, keys, raw, std, title, out_path, mode="raw", max_film=40):
    """Filmstrip on top, K reward rows below. Sized for paper figures.

    The filmstrip shows up to ``max_film`` evenly-spaced frames (so dense
    trajectories stay legible), while the reward curves keep full resolution.
    """
    T, K = rewards.shape
    timesteps = np.arange(T)

    # Subsample the filmstrip columns for long (e.g. dense) trajectories.
    n_film = min(T, max_film)
    film_idx = np.unique(np.linspace(0, T - 1, n_film).round().astype(int)) if T else np.array([], int)
    n_film = len(film_idx)

    cell_in = max(0.6, min(1.0, 16.0 / max(n_film, 1)))
    fig_w = max(10.0, cell_in * n_film + 1.5)
    fig_h = 2.2 + 1.6 * K

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        K + 1, n_film, figure=fig,
        height_ratios=[1.6] + [1.0] * K,
        hspace=0.45, wspace=0.04,
        left=0.06, right=0.98, top=0.92, bottom=0.06,
    )

    for c, t in enumerate(film_idx):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(frames[t])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#888888")
        ax.set_xlabel(str(t), fontsize=7, labelpad=1)

    # Cap the number of x-ticks so dense curves stay readable.
    n_ticks = min(T, 20)
    xticks = np.unique(np.linspace(0, T - 1, n_ticks).round().astype(int)) if T else []

    colors = plt.get_cmap("tab10").colors
    for k, key in enumerate(keys):
        ax = fig.add_subplot(gs[k + 1, :])
        values = rewards[:, k]
        color = colors[k % len(colors)]
        marker = "o" if T <= 60 else None
        ax.plot(timesteps, values, marker=marker, markersize=4,
                linewidth=1.6, color=color)
        ax.fill_between(timesteps, 0, values, alpha=0.12, color=color)
        ax.axhline(0, color="#444444", linewidth=0.6, linestyle="--")
        ax.set_ylabel(key, fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_xticks(xticks)
        ax.tick_params(axis="both", labelsize=7)
        if k < K - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("timestep (frame pair)", fontsize=9)
        ax.set_title(_row_title(key, raw, std, mode), fontsize=9, loc="right", pad=2)

    suffix = {"traj": "  [standardized per step]",
              "timestep": "  [standardized per timestep]"}.get(mode, "")
    fig.suptitle(title + suffix, fontsize=12, y=0.985)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {out_path}")


def _animate(frames, rewards, keys, raw, std, title, out_path, fps=4, mode="raw"):
    """MP4: current frame on top, one plot of all metrics below.

    The reward curves draw in progressively as the video advances, with a dot
    marking each axis's value at the current timestep.
    """
    T, K = rewards.shape
    timesteps = np.arange(T)

    finite = rewards[np.isfinite(rewards)]
    ymin = float(finite.min()) if finite.size else -1.0
    ymax = float(finite.max()) if finite.size else 1.0
    pad = 0.05 * max(ymax - ymin, 1e-3)
    ymin, ymax = ymin - pad, ymax + pad

    fig = plt.figure(figsize=(7.5, 8.5))
    gs = GridSpec(
        2, 1, figure=fig, height_ratios=[3, 2], hspace=0.18,
        left=0.10, right=0.97, top=0.92, bottom=0.08,
    )

    img_ax = fig.add_subplot(gs[0])
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    im = img_ax.imshow(frames[0])
    frame_label = img_ax.set_title("t = 0", fontsize=10)

    line_ax = fig.add_subplot(gs[1])
    line_ax.set_xlim(-0.5, T - 0.5)
    line_ax.set_ylim(ymin, ymax)
    line_ax.axhline(0, color="#444444", linewidth=0.6, linestyle="--")
    line_ax.set_xlabel("timestep (frame pair)", fontsize=9)
    line_ax.set_ylabel(
        {"traj": "standardized reward (per step, Σ = z)",
         "timestep": "per-timestep standardized reward (z)"}.get(
            mode, "predicted reward (per step)"), fontsize=9)
    line_ax.grid(True, alpha=0.25, linewidth=0.5)

    colors = plt.get_cmap("tab10").colors
    lines, dots = [], []
    for k, key in enumerate(keys):
        color = colors[k % len(colors)]
        label = key
        if key in std and std[key] is not None:
            label = f"{key} (z={std[key]:+.2f})"
        (ln,) = line_ax.plot([], [], linewidth=1.6, color=color, label=label)
        (dot,) = line_ax.plot([], [], marker="o", markersize=6, color=color)
        lines.append(ln)
        dots.append(dot)
    line_ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9)
    fig.suptitle(title, fontsize=11, y=0.98)

    def update(t):
        im.set_data(frames[t])
        frame_label.set_text(f"t = {t}")
        for k in range(K):
            lines[k].set_data(timesteps[: t + 1], rewards[: t + 1, k])
            dots[k].set_data([timesteps[t]], [rewards[t, k]])
        return [im, frame_label, *lines, *dots]

    anim = FuncAnimation(fig, update, frames=T, interval=1000.0 / fps, blit=False)
    anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=150)
    plt.close(fig)
    print(f"Saved video → {out_path}")


def _resolve_axes(requested: list[str], keys: list[str]) -> list[str]:
    """Map user-supplied axis strings to canonical preference keys.

    Each string matches by exact name, else by unique case-insensitive
    substring. Raises ValueError on no-match or ambiguous matches. The result
    preserves the order given and drops duplicates.
    """
    selected = []
    for raw_q in requested:
        q = raw_q.strip()
        if not q:
            continue
        if q in keys:
            selected.append(q)
            continue
        matches = [k for k in keys if q.lower() in k.lower()]
        if len(matches) == 1:
            selected.append(matches[0])
        elif not matches:
            raise ValueError(f"--axes: '{q}' matches no axis. Available: {keys}")
        else:
            raise ValueError(f"--axes: '{q}' is ambiguous, matches {matches}. Be more specific.")
    out, seen = [], set()
    for k in selected:
        if k not in seen:
            seen.add(k)
            out.append(k)
    if not out:
        raise ValueError("--axes resolved to an empty selection.")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_json", required=True,
                        help="A *_score*.json written by infer.py (must contain per_frame).")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: single_<traj>.png next to the JSON).")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True,
                        help="Also write an animated MP4 next to the PNG.")
    parser.add_argument("--fps", type=int, default=4,
                        help="Frames per second for the --video output.")
    parser.add_argument("--standardize", action="store_true",
                        help="Rescale each axis's per-step rewards with the whole-trajectory "
                             "population norm_stats so the curve sums to the standardized z-score.")
    parser.add_argument("--standardize_per_timestep", action="store_true",
                        help="Standardize each per-step value against the population at the "
                             "SAME timestep (uses norm_stats_per_timestep). Mutually exclusive "
                             "with --standardize.")
    parser.add_argument("--axes", default=None,
                        help="Comma-separated subset of axes to plot (matches "
                             "preference_keys by exact name or unique case-insensitive "
                             "substring). Default: all axes.")
    parser.add_argument("--dense", action="store_true",
                        help="Require the score JSON to be dense (full-resolution per-step "
                             "rewards). Densification is done by infer.py --dense, not here; "
                             "this flag just asserts the data is dense and errors if not.")
    args = parser.parse_args()

    with open(args.score_json) as f:
        data = json.load(f)

    per_frame = data.get("per_frame")
    if per_frame is None:
        parser.error(
            f"{args.score_json} has no 'per_frame' data — the checkpoint's model "
            "type does not expose per-step rewards (only qwen open_cum / discounted "
            "and DiscountedRewardModel do)."
        )

    meta = data["meta"]
    if args.dense and not meta.get("dense"):
        parser.error(
            f"{args.score_json} is not dense. Re-run infer.py with --dense to produce "
            "full-resolution per-step rewards, or drop --dense here."
        )
    all_keys = meta["preference_keys"]
    raw = data.get("raw", {})
    std = data.get("standardized", {})

    T = len(per_frame[all_keys[0]])
    rewards = np.full((T, len(all_keys)), np.nan, dtype=np.float32)
    for i, key in enumerate(all_keys):
        rewards[:, i] = [np.nan if v is None else v for v in per_frame[key]]

    if args.standardize and args.standardize_per_timestep:
        parser.error("pass at most one of --standardize / --standardize_per_timestep.")

    mode = "raw"
    if args.standardize:
        norm_stats = data.get("norm_stats")
        if not norm_stats:
            parser.error(f"{args.score_json} has no 'norm_stats'; cannot --standardize.")
        rewards = _standardize_per_frame(rewards, all_keys, norm_stats)
        mode = "traj"
    elif args.standardize_per_timestep:
        ts_stats = data.get("norm_stats_per_timestep")
        if not ts_stats:
            parser.error(f"{args.score_json} has no 'norm_stats_per_timestep'; "
                         "re-run the updated infer.py to produce it.")
        rewards = _standardize_per_timestep(rewards, all_keys, ts_stats)
        mode = "timestep"

    # Optionally restrict to a subset of axes.
    if args.axes:
        try:
            keys = _resolve_axes(args.axes.split(","), all_keys)
        except ValueError as e:
            parser.error(str(e))
        cols = [all_keys.index(k) for k in keys]
        rewards = rewards[:, cols]
        print(f"Plotting axes: {keys}")
    else:
        keys = all_keys

    hdf5_path = data["source_hdf5"]
    frames = _load_frames(hdf5_path, meta, T)

    model_type = meta.get("model_type", "")
    title = f"{model_type} — {os.path.basename(os.path.dirname(hdf5_path))}/" \
            f"{os.path.basename(hdf5_path)}"

    if args.out is None:
        base = os.path.splitext(os.path.basename(hdf5_path))[0]
        out_dir = os.path.dirname(os.path.abspath(args.score_json))
        args.out = os.path.join(out_dir, f"single_{base}.png")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    _plot(frames, rewards, keys, raw, std, title, args.out, mode=mode)

    if args.video:
        out_mp4 = os.path.splitext(args.out)[0] + ".mp4"
        _animate(frames, rewards, keys, raw, std, title, out_mp4,
                 fps=args.fps, mode=mode)


if __name__ == "__main__":
    main()
