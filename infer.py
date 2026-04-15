"""
Run reward model inference on all preference-folder trajectories.

For each preference folder the script writes two files:
    reward_model_{ckptname}_rollout_A_score.json
    reward_model_{ckptname}_rollout_B_score.json

Each file contains a dict mapping preference key → scalar score in [0, 1].

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
from model import RewardModel
from tasks import TASKS


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0


def to_bucket(score: float) -> int:
    """Map a score in [0, 1] to a bucket label in [1, 5]."""
    return min(int(score * N_BUCKETS) + 1, N_BUCKETS)


def score_trajectory(model: RewardModel, hdf5_path: str, args, device: torch.device) -> dict:
    """Return {"raw": {...}, "buckets": {...}} for a single trajectory."""
    traj = load_trajectory(hdf5_path, args.stride, args.seq_len, (args.img_size, args.img_size), offset=0)
    tp = normalize(traj["third_person"].unsqueeze(0).to(device))  # (1, T, 3, H, W)
    wr = normalize(traj["wrist"].unsqueeze(0).to(device))

    with torch.no_grad():
        rewards = model.encode_trajectory(tp, wr)  # (1, K)

    raw = {k: float(rewards[0, i]) for i, k in enumerate(args.preference_keys)}
    buckets = {k: to_bucket(v) for k, v in raw.items()}
    return {"raw": raw, "buckets": buckets}


N_BUCKETS = 5
BUCKET_EDGES = [i / N_BUCKETS for i in range(N_BUCKETS + 1)]  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BAR_WIDTH = 30


def print_stats(all_scores: dict[str, list[float]]) -> None:
    """Print per-key min/max/percentiles and an ASCII histogram."""
    print("\n" + "=" * 70)
    print("SCORE STATISTICS (A and B rollouts combined)")
    print("=" * 70)

    for key, vals in all_scores.items():
        arr = np.array(vals)
        counts, _ = np.histogram(arr, bins=BUCKET_EDGES)
        total = len(arr)

        p = np.percentile(arr, [10, 25, 50, 75, 90])
        print(f"\n  {key}  (n={total})")
        print(f"    min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")
        print(f"    p10={p[0]:.3f}  p25={p[1]:.3f}  p50={p[2]:.3f}  p75={p[3]:.3f}  p90={p[4]:.3f}")
        print()

        max_count = max(counts) if max(counts) > 0 else 1
        for i, count in enumerate(counts):
            lo, hi = BUCKET_EDGES[i], BUCKET_EDGES[i + 1]
            bar_len = int(count / max_count * BAR_WIDTH)
            bar = "█" * bar_len
            pct = 100.0 * count / total
            print(f"    [{lo:.1f}-{hi:.1f})  {bar:<{BAR_WIDTH}}  {count:3d}  ({pct:5.1f}%)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--preferences_dir", default="preferences")
    # These are read from the checkpoint by default; override if needed
    parser.add_argument("--stride",   type=int, default=None)
    parser.add_argument("--seq_len",  type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})

    # Fill in from checkpoint unless overridden on CLI
    for key in ("stride", "seq_len", "img_size", "embed_dim"):
        if getattr(args, key) is None:
            setattr(args, key, saved_args.get(key))
            if getattr(args, key) is None:
                parser.error(f"--{key} not found in checkpoint; pass it explicitly")

    task = saved_args.get("task", "cube_in_three_bowls")
    args.preference_keys = TASKS[task]

    # Build and load model
    model = RewardModel(
        num_preferences=len(args.preference_keys),
        embed_dim=args.embed_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    prefix = f"reward_model_{ckpt_name}"

    # Iterate over preference folders
    pref_dirs = sorted(
        os.path.join(args.preferences_dir, d)
        for d in os.listdir(args.preferences_dir)
        if os.path.isdir(os.path.join(args.preferences_dir, d))
    )

    all_scores = defaultdict(list)

    for pref_dir in pref_dirs:
        hdf5_a = os.path.join(pref_dir, "rollout_A.hdf5")
        hdf5_b = os.path.join(pref_dir, "rollout_B.hdf5")

        if not (os.path.exists(hdf5_a) and os.path.exists(hdf5_b)):
            print(f"  [skip] {pref_dir} — missing hdf5 files")
            continue

        scores_a = score_trajectory(model, hdf5_a, args, device)
        scores_b = score_trajectory(model, hdf5_b, args, device)

        out_a = os.path.join(pref_dir, f"{prefix}_rollout_A_score.json")
        out_b = os.path.join(pref_dir, f"{prefix}_rollout_B_score.json")

        with open(out_a, "w") as f:
            json.dump(scores_a, f, indent=2)
        with open(out_b, "w") as f:
            json.dump(scores_b, f, indent=2)

        for k, v in scores_a["raw"].items():
            all_scores[k].append(v)
        for k, v in scores_b["raw"].items():
            all_scores[k].append(v)

        session = os.path.basename(pref_dir)
        a_str = ", ".join(f"{k}: {v:.3f}" for k, v in scores_a["raw"].items())
        b_str = ", ".join(f"{k}: {v:.3f}" for k, v in scores_b["raw"].items())
        print(f"[{session}]")
        print(f"  A: {a_str}")
        print(f"  B: {b_str}")

    print(f"\nDone. Scores written as '{prefix}_rollout_A/B_score.json' in each folder.")

    if all_scores:
        print_stats(all_scores)

        stats_out = {}
        for key, vals in all_scores.items():
            arr = np.array(vals)
            counts, _ = np.histogram(arr, bins=BUCKET_EDGES)
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
                "histogram": {
                    f"{BUCKET_EDGES[i]:.1f}-{BUCKET_EDGES[i+1]:.1f}": int(counts[i])
                    for i in range(N_BUCKETS)
                },
            }

        stats_path = os.path.join(args.preferences_dir, f"{prefix}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats_out, f, indent=2)
        print(f"\nStats saved → {stats_path}")


if __name__ == "__main__":
    main()
