"""
Parse results_reward_models_val.txt and compute val_acc statistics across
the top-5 peaks of each seed, then aggregate per method (mean ± std across
seeds of the per-seed mean of top-5).

File format (one section per method):
  method_name
  https://wandb.ai/.../runs/<id>[/overview]
  https://wandb.ai/.../runs/<id>
  ...

Usage:
  python fetch_reward_models_val.py [--results path] [--metric val_acc]
                                    [--peaks 5] [--verbose] [--plot out.png]
"""

import argparse
import os
import re
from collections import defaultdict
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import wandb

DEFAULT_RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results_reward_models_val.txt",
)
DEFAULT_METRIC = "val/acc_mean"
FALLBACK_METRICS = ["val/acc_mean", "val_acc", "val/acc", "val/accuracy", "eval/val_acc"]

URL_RE = re.compile(
    r"https?://wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[A-Za-z0-9]+)"
)


def parse_results(path):
    """Return [(method, entity, project, run_id), ...]."""
    entries = []
    method = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = URL_RE.search(line)
            if m:
                entries.append((
                    method or "(no method)",
                    m["entity"], m["project"], m["run_id"],
                ))
            else:
                # Plain line that isn't a URL → method header.
                method = line
    return entries


def load_rows(run, keys, samples=10000):
    """Fetch only the requested keys via the sampled-history endpoint."""
    try:
        rows = list(run.history(keys=keys, samples=samples, pandas=False))
    except Exception:
        rows = []
    rows.append(dict(run.summary))
    return rows


def top_k_values(rows, keys, k):
    """Return the top-k values for the first key in `keys` that appears.

    `keys` is an ordered fallback list; the first one that has any data wins.
    """
    for key in keys:
        vals = []
        for row in rows:
            v = row.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            vals.sort(reverse=True)
            return vals[:k], key
    return [], None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help="Primary metric key (val_acc by default)")
    parser.add_argument("--peaks", type=int, default=5,
                        help="How many top values per seed to aggregate (default 5)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--plot",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results_reward_models_val_plot.png",
        ),
        help="Output path for the bar chart (set to '' to skip)",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help="Only process entries whose method matches (case- and "
             "whitespace/underscore-insensitive). Repeatable.",
    )
    args = parser.parse_args()

    entries = parse_results(args.results)
    if args.method:
        wanted = {re.sub(r"[\s_]+", "", m.lower()) for m in args.method}
        entries = [e for e in entries
                   if re.sub(r"[\s_]+", "", e[0].lower()) in wanted]
    if not entries:
        print(f"No runs parsed from {args.results}")
        return

    # Build the ordered fallback list, primary metric first.
    metric_chain = [args.metric] + [m for m in FALLBACK_METRICS if m != args.metric]

    api = wandb.Api(timeout=120)
    grouped = defaultdict(list)   # method -> [(run_id, [top-k values], src_key)]

    for method, entity, project, run_id in entries:
        path = f"{entity}/{project}/{run_id}"
        try:
            run = api.run(path)
        except Exception as e:
            print(f"[error] {path}: {e}")
            grouped[method].append((run_id, [], "FAILED"))
            continue

        rows = load_rows(run, metric_chain)
        peaks, src = top_k_values(rows, metric_chain, args.peaks)
        grouped[method].append((run_id, peaks, src or "MISSING"))

        if args.verbose:
            peaks_str = ", ".join(f"{v:.4f}" for v in peaks) if peaks else "-"
            print(f"  {method} / {run_id}: top-{args.peaks} = [{peaks_str}]  ({src})")

    # ---- Per-seed table ----------------------------------------------------
    print()
    print("=" * 110)
    print(f"{'Method':<22} {'Run':<12} {'N_peaks':>7}  "
          f"{'PeakMean':>9}  {'PeakStd':>9}  src")
    print("-" * 110)
    for method, items in grouped.items():
        for run_id, peaks, src in items:
            if not peaks:
                print(f"{method[:22]:<22} {run_id:<12} {'-':>7}  "
                      f"{'-':>9}  {'-':>9}  {src}")
                continue
            pm = mean(peaks)
            ps = stdev(peaks) if len(peaks) > 1 else 0.0
            print(f"{method[:22]:<22} {run_id:<12} {len(peaks):>7}  "
                  f"{pm:>9.4f}  {ps:>9.4f}  {src}")

    # ---- Aggregated table --------------------------------------------------
    print()
    print("=" * 90)
    print(f"{'Method':<22} {'N_seeds':>7}  {'Avg(seed top-k mean)':>22}  {'Std':>10}")
    print("-" * 90)
    agg = []
    for method, items in grouped.items():
        seed_means = [mean(p) for _, p, _ in items if p]
        n = len(seed_means)
        if n == 0:
            agg.append((method, 0, 0.0, 0.0))
            print(f"{method[:22]:<22} {n:>7}  {'-':>22}  {'-':>10}")
            continue
        m = mean(seed_means)
        s = stdev(seed_means) if n > 1 else 0.0
        agg.append((method, n, m, s))
        print(f"{method[:22]:<22} {n:>7}  {m:>22.4f}  {s:>10.4f}")
    print("=" * 90)

    # ---- Plot --------------------------------------------------------------
    if args.plot and agg:
        plotted = [a for a in agg if a[1] > 0]
        if plotted:
            labels = [a[0] for a in plotted]
            means = [a[2] for a in plotted]
            stds = [a[3] for a in plotted]
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

            fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels) + 3), 5))
            x = np.arange(len(labels))
            bars = ax.bar(
                x, means, 0.6, yerr=stds, capsize=4,
                color=colors, edgecolor="black", linewidth=0.5,
            )
            for bar, m, n in zip(bars, means, [a[1] for a in plotted]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    m,
                    f"{m:.3f}\n(n={n})",
                    ha="center", va="bottom", fontsize=8,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_ylabel(f"{args.metric}  (mean of top-{args.peaks} per seed)")
            ax.set_title("Reward-model val accuracy — top-{} peaks".format(args.peaks))
            ax.set_ylim(0, max(1.0, max(m + s for m, s in zip(means, stds)) * 1.1))
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
