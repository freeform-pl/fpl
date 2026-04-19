"""
Run reward model inference on all preference-folder trajectories.

For each preference folder the script writes two files:
    reward_model_{ckptname}_rollout_A_score.json
    reward_model_{ckptname}_rollout_B_score.json

Each file contains:
    raw              — raw reward scores in [0, 1]
    buckets          — equal-width bucket labels [1-5] over [0, 1]
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

import numpy as np
import torch

from dataset import load_trajectory
from model import RewardModel, DiscountedRewardModel
from tasks import TASKS


N_BUCKETS = 5
BUCKET_EDGES = [i / N_BUCKETS for i in range(N_BUCKETS + 1)]  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BAR_WIDTH = 30


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0


def to_bucket(score: float) -> int:
    """Map a score in [0, 1] to an equal-width bucket label in [1, 5]."""
    return min(int(score * N_BUCKETS) + 1, N_BUCKETS)


def to_quantile_bucket(score: float, edges: list[float]) -> int:
    """Map a score to an equal-frequency bucket label in [1, 5] given quantile edges."""
    for i in range(N_BUCKETS - 1):
        if score < edges[i + 1]:
            return i + 1
    return N_BUCKETS


def score_trajectory(model: RewardModel, hdf5_path: str, args, device: torch.device) -> dict:
    """Return raw scores only; buckets are added after quantile edges are computed."""
    traj = load_trajectory(hdf5_path, args.stride, args.seq_len, (args.img_size, args.img_size), offset=0)
    tp = normalize(traj["third_person"].unsqueeze(0).to(device))  # (1, T, 3, H, W)
    wr = normalize(traj["wrist"].unsqueeze(0).to(device))

    with torch.no_grad():
        rewards = model.encode_trajectory(tp, wr)  # (1, K)

    return {k: float(rewards[0, i]) for i, k in enumerate(args.preference_keys)}


def compute_quantile_edges(all_scores: dict[str, list[float]]) -> dict[str, list[float]]:
    """Compute per-key quantile bucket edges that give equal-frequency bins."""
    percentiles = np.linspace(0, 100, N_BUCKETS + 1)  # [0, 20, 40, 60, 80, 100]
    return {
        key: [float(v) for v in np.percentile(vals, percentiles)]
        for key, vals in all_scores.items()
    }


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

        # Equal-width histogram
        print("\n    Equal-width buckets [0-1 → 5 bins]:")
        counts_ew, _ = np.histogram(arr, bins=BUCKET_EDGES)
        max_count = max(counts_ew) if max(counts_ew) > 0 else 1
        for i, count in enumerate(counts_ew):
            lo, hi = BUCKET_EDGES[i], BUCKET_EDGES[i + 1]
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
    parser.add_argument("--preferences_dir", nargs="+", default=["preferences"])
    # These are read from the checkpoint by default; override if needed
    parser.add_argument("--stride",    type=int, default=None)
    parser.add_argument("--seq_len",   type=int, default=None)
    parser.add_argument("--img_size",  type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    args = parser.parse_args()

    # Support both space-separated and +-separated dirs, e.g.:
    #   --preferences_dir dir1 dir2   OR   --preferences_dir dir1+dir2
    args.preferences_dir = [
        d for entry in args.preferences_dir for d in entry.split("+")
    ]

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
    if model_type == "discounted":
        model = DiscountedRewardModel(
            num_preferences=len(args.preference_keys),
            embed_dim=args.embed_dim,
            gamma=saved_args.get("gamma", 0.99),
        ).to(device)
    else:
        model = RewardModel(
            num_preferences=len(args.preference_keys),
            embed_dim=args.embed_dim,
        ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    prefix = f"reward_model_{ckpt_name}"

    pref_dirs = sorted(
        os.path.join(root, d)
        for root in args.preferences_dir
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )

    # --- Pass 1: score all trajectories, collect raw scores ---
    all_scores = defaultdict(list)
    results = []  # [(pref_dir, raw_a, raw_b)]

    for pref_dir in pref_dirs:
        filenames = os.listdir(pref_dir)
        hdf5_files = []
        for file in filenames:
            if "hdf5" in file:
                hdf5_files.append(file)
        hdf5_files = sorted(hdf5_files)
        print()
        hdf5_a = os.path.join(pref_dir, "demos_A.hdf5")
        hdf5_b = os.path.join(pref_dir, "demos_B.hdf5")

        if not (os.path.exists(hdf5_a) and os.path.exists(hdf5_b)):
            print(f"  [skip] {pref_dir} — missing hdf5 files")
            continue

        raw_a = score_trajectory(model, hdf5_a, args, device)
        raw_b = score_trajectory(model, hdf5_b, args, device)
        results.append((pref_dir, raw_a, raw_b))

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

    if not results:
        print("No valid preference folders found.")
        return

    # --- Compute quantile edges from full distribution ---
    quantile_edges = compute_quantile_edges(all_scores)

    # --- Pass 2: write JSON files with both bucket types ---
    for pref_dir, raw_a, raw_b in results:
        for raw, suffix in [(raw_a, "A"), (raw_b, "B")]:
            out = {
                "raw": raw,
                "buckets": {k: to_bucket(v) for k, v in raw.items()},
                "buckets_quantile": {k: to_quantile_bucket(v, quantile_edges[k]) for k, v in raw.items()},
            }
            out_path = os.path.join(pref_dir, f"{prefix}_rollout_{suffix}_score.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

    print(f"\nDone. Scores written as '{prefix}_rollout_A/B_score.json' in each folder.")

    # --- Stats ---
    print_stats(all_scores, quantile_edges)

    stats_out = {}
    for key, vals in all_scores.items():
        arr = np.array(vals)
        counts_ew, _ = np.histogram(arr, bins=BUCKET_EDGES)
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
                f"{BUCKET_EDGES[i]:.2f}-{BUCKET_EDGES[i+1]:.2f}": int(counts_ew[i])
                for i in range(N_BUCKETS)
            },
            "quantile_edges": qe,
            "histogram_equal_freq": {
                f"{qe[i]:.3f}-{qe[i+1]:.3f}": int(counts_qf[i])
                for i in range(N_BUCKETS)
            },
        }

    stats_dir = args.preferences_dir[0] if len(args.preferences_dir) == 1 else os.path.dirname(args.ckpt)
    stats_path = os.path.join(stats_dir, f"{prefix}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"Stats saved → {stats_path}")


if __name__ == "__main__":
    main()
