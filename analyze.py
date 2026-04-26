"""
Interactive analysis of reward model inference results.

Load a .npz file produced by infer.py and explore the scored trajectories.

Usage:
    python analyze.py --npz vis_step000200/reward_model_..._data.npz

Then use the `data` object interactively, or add analysis functions below.

Quick reference:
    data.keys               — preference dimension names
    data.scores             — (N, K) raw reward scores
    data.normalized         — (N, K) min-max scaled to [0, 1]
    data.standardized       — (N, K) z-score
    data.q_buckets          — (N, K) quantile bucket labels [1-5]
    data.hdf5_paths         — (N,) path to each trajectory HDF5
    data.sessions           — (N,) session folder name
    data.rollouts           — (N,) 'A' or 'B'

    data.top(k, n)          — top-n trajectories for dimension k (by name or index)
    data.bottom(k, n)       — bottom-n trajectories for dimension k
    data.bucket(k, b)       — all trajectories in quantile bucket b for dimension k
    data.for_session(s)     — all trajectories from session s
    data.summary()          — print per-dimension statistics
    data.correlation()      — print pairwise reward correlations across dimensions
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


class RewardData:
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.keys         = list(d["preference_keys"])
        self.hdf5_paths   = d["hdf5_paths"]
        self.sessions     = d["sessions"]
        self.rollouts     = d["rollouts"]
        self.scores       = d["scores"]        # (N, K)
        self.normalized   = d["normalized"]    # (N, K)
        self.standardized = d["standardized"]  # (N, K)
        self.q_buckets    = d["q_buckets"]     # (N, K)
        self.stat_min     = d["stat_min"]      # (K,)
        self.stat_max     = d["stat_max"]
        self.stat_mean    = d["stat_mean"]
        self.stat_std     = d["stat_std"]
        self.quantile_edges = d["quantile_edges"]  # (K, N_buckets+1)
        self.N, self.K    = self.scores.shape
        self._npz_path    = npz_path

    def _dim(self, k):
        """Resolve k to a column index (accepts int or str)."""
        if isinstance(k, str):
            return self.keys.index(k)
        return k

    def top(self, k, n: int = 10) -> np.ndarray:
        """Return indices of the n highest-scoring trajectories for dimension k."""
        ki = self._dim(k)
        return np.argsort(self.scores[:, ki])[-n:][::-1]

    def bottom(self, k, n: int = 10) -> np.ndarray:
        """Return indices of the n lowest-scoring trajectories for dimension k."""
        ki = self._dim(k)
        return np.argsort(self.scores[:, ki])[:n]

    def bucket(self, k, b: int) -> np.ndarray:
        """Return indices of trajectories in quantile bucket b (1–5) for dimension k."""
        ki = self._dim(k)
        return np.where(self.q_buckets[:, ki] == b)[0]

    def for_session(self, session: str) -> np.ndarray:
        """Return indices of all trajectories from a given session folder."""
        return np.where(self.sessions == session)[0]

    def row(self, idx: int) -> dict:
        """Return all info for a single trajectory index."""
        return {
            "hdf5":        self.hdf5_paths[idx],
            "session":     self.sessions[idx],
            "rollout":     self.rollouts[idx],
            "scores":      {k: float(self.scores[idx, i])       for i, k in enumerate(self.keys)},
            "normalized":  {k: float(self.normalized[idx, i])   for i, k in enumerate(self.keys)},
            "standardized":{k: float(self.standardized[idx, i]) for i, k in enumerate(self.keys)},
            "q_buckets":   {k: int(self.q_buckets[idx, i])      for i, k in enumerate(self.keys)},
        }

    def summary(self):
        """Print per-dimension statistics."""
        print(f"\n{'='*60}")
        print(f"  {self.N} trajectories  ×  {self.K} dimensions")
        print(f"  Source: {os.path.basename(self._npz_path)}")
        print(f"{'='*60}")
        for i, key in enumerate(self.keys):
            arr = self.scores[:, i]
            edges = self.quantile_edges[i]
            print(f"\n  {key}")
            print(f"    raw    min={self.stat_min[i]:.3f}  max={self.stat_max[i]:.3f}"
                  f"  mean={self.stat_mean[i]:.3f}  std={self.stat_std[i]:.3f}")
            print(f"    p10={np.percentile(arr,10):.3f}  p25={np.percentile(arr,25):.3f}"
                  f"  p50={np.percentile(arr,50):.3f}  p75={np.percentile(arr,75):.3f}"
                  f"  p90={np.percentile(arr,90):.3f}")
            bucket_counts = [int((self.q_buckets[:, i] == b).sum()) for b in range(1, len(edges))]
            bucket_str = "  ".join(f"B{b}({c})" for b, c in enumerate(bucket_counts, 1))
            print(f"    buckets: {bucket_str}")
        print(f"{'='*60}\n")

    def correlation(self):
        """Print pairwise Pearson correlation between reward dimensions."""
        print(f"\n{'='*60}")
        print("  Pairwise reward correlations (raw scores)")
        print(f"{'='*60}")
        C = np.corrcoef(self.scores.T)  # (K, K)
        header = "".join(f"{k[:8]:>10}" for k in self.keys)
        print(f"{'':20}{header}")
        for i, key in enumerate(self.keys):
            row = "".join(f"{C[i,j]:>10.3f}" for j in range(self.K))
            print(f"  {key[:18]:<18}{row}")
        print(f"{'='*60}\n")

    def rank_sessions(self, k, descending: bool = True) -> list:
        """
        Return sessions ranked by their mean score on dimension k.
        Each entry: (session, mean_raw, mean_normalized, n_trajectories).
        """
        ki = self._dim(k)
        session_names = np.unique(self.sessions)
        rows = []
        for s in session_names:
            idx = self.for_session(s)
            rows.append((
                s,
                float(self.scores[idx, ki].mean()),
                float(self.normalized[idx, ki].mean()),
                len(idx),
            ))
        rows.sort(key=lambda r: r[1], reverse=descending)
        return rows

    def print_rank_sessions(self, k):
        """Print sessions ranked by mean score for dimension k."""
        ki = self._dim(k)
        key_name = self.keys[ki]
        ranked = self.rank_sessions(k)
        print(f"\n  Sessions ranked by '{key_name}' (mean raw score):")
        print(f"  {'session':<35} {'mean_raw':>10} {'mean_norm':>10} {'n':>4}")
        print(f"  {'-'*65}")
        for s, mean_raw, mean_norm, n in ranked:
            print(f"  {s:<35} {mean_raw:>10.3f} {mean_norm:>10.3f} {n:>4}")


def plot_scatter_matrix(data: "RewardData", out_path: str):
    """
    K×K scatter matrix using standardized scores.

    - Diagonal  : histogram of standardized scores for that dimension
    - Lower tri : scatter plot (dim_j vs dim_i), alpha=0.25
    - Upper tri : Pearson r value as large centred text

    All cells show proper numeric axis tick labels.
    """
    K = data.K
    z = data.standardized  # (N, K)
    corr = np.corrcoef(z.T)  # (K, K)

    fig, axes = plt.subplots(K, K, figsize=(2.8 * K, 2.8 * K))
    fig.suptitle("Reward dimension scatter matrix  (standardized scores)", fontsize=11, y=1.01)

    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            ax.tick_params(labelsize=6, length=3)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
            # Always show tick labels so numbers are visible on every cell.
            ax.tick_params(labelbottom=True, labelleft=True)

            if i == j:
                # Diagonal — histogram
                ax.hist(z[:, i], bins=30, color="steelblue", alpha=0.75, edgecolor="none")
                ax.set_ylabel("count", fontsize=6)
            elif i > j:
                # Lower triangle — scatter
                ax.scatter(z[:, j], z[:, i], s=4, alpha=0.25, color="steelblue", linewidths=0)
                ax.axhline(0, color="gray", lw=0.4, ls="--")
                ax.axvline(0, color="gray", lw=0.4, ls="--")
            else:
                # Upper triangle — r value
                r = corr[i, j]
                color = plt.cm.RdBu_r(0.5 + 0.5 * r)
                ax.set_facecolor((*color[:3], 0.25))
                ax.text(0.5, 0.5, f"r = {r:.2f}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        color="black" if abs(r) < 0.6 else "darkred")
                ax.set_xticks([])
                ax.set_yticks([])

            # Axis labels on edges only
            if i == K - 1:
                ax.set_xlabel(data.keys[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(data.keys[i], fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter matrix → {out_path}")


def plot_dim_histograms(
    data: "RewardData",
    out_path: str,
    norm_threshold: float = 0.8,
    std_threshold: float = 1.0,
):
    """
    One row per dimension.  Each row has two panels:

    Left  — bar chart: count per quantile bucket (B1–B5) for this dimension.
            For every other dimension, a separate bar overlay shows what fraction
            of trajectories in that bucket exceed `norm_threshold` on the
            normalized score of that other dimension.

    Right — same but using standardized scores and `std_threshold`.
    """
    K = data.K
    n_buckets = 5
    buckets = np.arange(1, n_buckets + 1)
    bar_w = 0.7 / K  # width per overlay bar

    _named_colors = {
        "blue bowl": "royalblue",
        "orange bowl": "darkorange",
        "yellow bowl": "gold",
        "fast": "mediumpurple",
        "smooth": "mediumseagreen",
    }
    dim_colors = [
        _named_colors.get(k.lower(), plt.cm.tab10(i / max(K, 1)))
        for i, k in enumerate(data.keys)
    ]

    fig, axes = plt.subplots(K, 2, figsize=(14, 3.2 * K))
    if K == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"Per-dimension bucket histograms\n"
        f"Overlay: fraction in bucket also exceeding threshold in other dims\n"
        f"(norm>{norm_threshold}  |  std>{std_threshold})",
        fontsize=10, y=1.01,
    )

    for i, key in enumerate(data.keys):
        for panel, (values_all, threshold, label_suffix) in enumerate([
            (data.normalized,   norm_threshold, f"norm>{norm_threshold}"),
            (data.standardized, std_threshold,  f"std>{std_threshold}"),
        ]):
            ax = axes[i, panel]
            ax2 = ax.twinx()

            counts = [(data.q_buckets[:, i] == b).sum() for b in buckets]
            ax.bar(buckets, counts, color="steelblue", alpha=0.55, width=0.55, label="count")
            ax.set_ylabel("count", fontsize=7, color="steelblue")
            ax.tick_params(axis="y", labelcolor="steelblue", labelsize=7)
            ax.set_xticks(buckets)
            ax.set_xticklabels([f"B{b}" for b in buckets], fontsize=8)
            ax.set_title(f"{key}  [{label_suffix}]", fontsize=8)

            # For each OTHER dimension j, show fraction exceeding threshold per bucket.
            other_dims = [j for j in range(K) if j != i]
            for oi, j in enumerate(other_dims):
                fracs = []
                for b in buckets:
                    in_bucket = data.q_buckets[:, i] == b
                    n_b = in_bucket.sum()
                    if n_b == 0:
                        fracs.append(0.0)
                    else:
                        fracs.append((values_all[in_bucket, j] > threshold).sum() / n_b)
                x_off = (oi - len(other_dims) / 2 + 0.5) * bar_w
                ax2.bar(
                    buckets + x_off, fracs, width=bar_w,
                    color=dim_colors[j], alpha=0.8,
                    label=data.keys[j],
                )
                # Show fraction value on top of each bar
                for b_idx, (x, f) in enumerate(zip(buckets + x_off, fracs)):
                    if f > 0.02:
                        ax2.text(x, f + 0.01, f"{f:.2f}", ha="center", va="bottom",
                                 fontsize=5, color=dim_colors[j])

            ax2.set_ylim(0, 1.25)
            ax2.set_ylabel("fraction exceeding threshold", fontsize=7)
            ax2.tick_params(labelsize=7)
            ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

            if i == 0 and panel == 1:
                ax2.legend(fontsize=6, loc="upper right", title="other dim", title_fontsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved dim histograms  → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze reward model inference results.")
    parser.add_argument("--npz", required=True, help="Path to .npz file from infer.py")
    parser.add_argument("--summary", action="store_true", help="Print per-dimension statistics")
    parser.add_argument("--correlation", action="store_true", help="Print pairwise reward correlations")
    parser.add_argument("--rank_sessions", type=str, default=None,
                        help="Rank sessions by mean score for this dimension (name or index)")
    parser.add_argument("--top", type=str, default=None,
                        help="Show top-N for dimension: e.g. 'Blue bowl:10'")
    parser.add_argument("--bottom", type=str, default=None,
                        help="Show bottom-N for dimension: e.g. 'Blue bowl:10'")
    parser.add_argument("--plot", action="store_true",
                        help="Save scatter matrix and per-dim histogram plots next to the npz file")
    parser.add_argument("--norm_threshold", type=float, default=0.8,
                        help="Normalized score threshold for histogram overlays (default 0.8)")
    parser.add_argument("--std_threshold", type=float, default=1.0,
                        help="Standardized score threshold for histogram overlays (default 1.0)")
    args = parser.parse_args()

    data = RewardData(args.npz)

    if args.summary or not any([args.correlation, args.rank_sessions, args.top, args.bottom]):
        data.summary()

    if args.correlation:
        data.correlation()

    if args.plot:
        stem = os.path.splitext(args.npz)[0]
        plot_scatter_matrix(data, stem + "_scatter_matrix.png")
        plot_dim_histograms(data, stem + "_dim_histograms.png",
                            norm_threshold=args.norm_threshold,
                            std_threshold=args.std_threshold)

    if args.rank_sessions is not None:
        k = int(args.rank_sessions) if args.rank_sessions.isdigit() else args.rank_sessions
        data.print_rank_sessions(k)

    for spec, fn in [(args.top, data.top), (args.bottom, data.bottom)]:
        if spec is not None:
            parts = spec.rsplit(":", 1)
            k = int(parts[0]) if parts[0].isdigit() else parts[0]
            n = int(parts[1]) if len(parts) > 1 else 10
            idxs = fn(k, n)
            ki = data._dim(k)
            label = "Top" if fn == data.top else "Bottom"
            print(f"\n  {label}-{n} for '{data.keys[ki]}':")
            print(f"  {'#':<4} {'raw':>8} {'norm':>8} {'z':>8} {'bucket':>7}  session / rollout")
            print(f"  {'-'*70}")
            for rank, idx in enumerate(idxs, 1):
                r = data.row(idx)
                print(f"  {rank:<4} {r['scores'][data.keys[ki]]:>8.3f}"
                      f" {r['normalized'][data.keys[ki]]:>8.3f}"
                      f" {r['standardized'][data.keys[ki]]:>8.3f}"
                      f" {r['q_buckets'][data.keys[ki]]:>7}"
                      f"  {r['session']} / {r['rollout']}")

    # Drop into interactive mode so the user can query `data` directly
    import code
    banner = (
        "\n  RewardData loaded as `data`.\n"
        "  Try: data.summary()  |  data.top('Blue bowl', 5)  |  data.correlation()\n"
        "  Ctrl-D to exit.\n"
    )
    code.interact(banner=banner, local={"data": data, "np": np})


if __name__ == "__main__":
    main()
