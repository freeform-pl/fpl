"""
Compare baselines from slow_fast experiments by fetching metrics from wandb.

Each z-condition becomes its own entry (e.g. rhp_original, rhp_z_pos, rhp_z_neg).

Usage:
  python scripts/plot_wandb_comparison.py \
    --runs rhp=memory_rl/slow_fast_rhp/pb05ve1b \
           demo_only=memory_rl/slow_fast_demo_only/9b2py8np \
    --output_dir plots/
"""

import argparse
import os
import numpy as np
import wandb
import matplotlib.pyplot as plt


METRIC_KEYS = [
    "mean_success", "mean_speed_reward", "mean_smoothness",
    "mean_score", "mean_throughput",
    "mean_success_left", "mean_success_right",
    "mean_speed_left", "mean_speed_right",
    "mean_throughput_left", "mean_throughput_right",
    "mean_score_left", "mean_score_right",
    "left_peg_rate", "right_peg_rate",
]

Z_CONDITIONS = ["original", "z_pos", "z_zero", "z_neg"]


def fetch_run_metrics(api, run_path):
    """Fetch eval metrics from a wandb run. Returns {z_cond: {metric: val}}."""
    print(f"  Fetching {run_path} ...")
    run = api.run(run_path)
    results = {}

    # Strategy 1: comparison/{z}/{m} in summary
    for z in Z_CONDITIONS:
        for m in METRIC_KEYS:
            key = f"comparison/{z}/{m}"
            if key in run.summary and run.summary[key] is not None:
                results.setdefault(z, {})[m] = run.summary[key]

    # Strategy 2: eval/{z}_{m} in summary
    for z in Z_CONDITIONS:
        for m in METRIC_KEYS:
            key = f"eval/{z}_{m}"
            if key in run.summary and run.summary[key] is not None:
                results.setdefault(z, {})[m] = run.summary[key]

    # Strategy 3: scan history for eval/{z}_{m} (max across steps)
    if not results:
        target_keys = [f"eval/{z}_{m}" for z in Z_CONDITIONS for m in METRIC_KEYS]
        history = list(run.scan_history(keys=target_keys, min_step=0))
        for row in history:
            for z in Z_CONDITIONS:
                for m in METRIC_KEYS:
                    key = f"eval/{z}_{m}"
                    if key in row and row[key] is not None:
                        val = float(row[key])
                        prev = results.get(z, {}).get(m, -np.inf)
                        if val > prev:
                            results.setdefault(z, {})[m] = val

    # Strategy 4: test/{m} in summary (training runs)
    if not results:
        for m in METRIC_KEYS:
            key = f"test/{m}"
            if key in run.summary and run.summary[key] is not None:
                results.setdefault("original", {})[m] = run.summary[key]

    return results


def load_and_flatten(name, results):
    """Flatten {z_cond: {metric: val}} into {name_zcond: {metric: val}} entries."""
    flat = {}
    for z_cond, metrics in results.items():
        label = f"{name}_{z_cond}" if z_cond != "original" else name
        flat[label] = metrics
    return flat


def plot_metric(all_entries, metric, output_dir):
    """Bar chart for a single metric across all entries."""
    labels = [k for k in all_entries if metric in all_entries[k]]
    if not labels:
        return
    vals = [all_entries[k][metric] for k in labels]

    color_map = {"z_pos": "#55A868", "z_zero": "#C4C4C4", "z_neg": "#DD8452"}
    colors = []
    for k in labels:
        c = "#4C72B0"
        for suffix, col in color_map.items():
            if k.endswith(suffix):
                c = col
                break
        colors.append(c)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(x, vals, color=colors)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(metric.replace("mean_", "").replace("_", " ").title())
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"{metric}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_combined_core(all_entries, output_dir):
    """Single figure with subplots for core metrics."""
    core = ["mean_success", "mean_speed_reward", "mean_smoothness", "mean_score", "mean_throughput"]
    labels = list(all_entries.keys())

    color_map = {"z_pos": "#55A868", "z_zero": "#C4C4C4", "z_neg": "#DD8452"}
    colors = []
    for k in labels:
        c = "#4C72B0"
        for suffix, col in color_map.items():
            if k.endswith(suffix):
                c = col
                break
        colors.append(c)

    fig, axes = plt.subplots(1, len(core), figsize=(4 * len(core), 5), sharey=False)
    x = np.arange(len(labels))

    for idx, metric in enumerate(core):
        ax = axes[idx]
        vals = [all_entries[k].get(metric, 0.0) for k in labels]
        ax.bar(x, vals, color=colors)
        ax.set_title(metric.replace("mean_", "").replace("_", " ").title(), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(bottom=0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="original"),
        Patch(facecolor="#55A868", label="z_pos"),
        Patch(facecolor="#C4C4C4", label="z_zero"),
        Patch(facecolor="#DD8452", label="z_neg"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.suptitle("Slow/Fast Baseline Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    fname = os.path.join(output_dir, "core_metrics_comparison.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def print_table(all_entries):
    core = ["mean_success", "mean_speed_reward", "mean_smoothness", "mean_score", "mean_throughput"]
    print("\n" + "=" * 100)
    print("METRIC COMPARISON")
    print("=" * 100)
    header = f"{'Baseline':<30s}"
    for m in core:
        header += f" {m.replace('mean_', ''):>12s}"
    print(header)
    print("-" * len(header))
    for label, metrics in all_entries.items():
        row = f"{label:<30s}"
        for m in core:
            val = metrics.get(m, None)
            row += f" {val:>12.3f}" if val is not None else f" {'n/a':>12s}"
        print(row)
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="name=entity/project/run_id pairs, e.g. rhp=memory_rl/slow_fast_rhp/pb05ve1b")
    parser.add_argument("--output_dir", default="plots/slow_fast_comparison")
    args = parser.parse_args()

    experiments = {}
    for r in args.runs:
        if "=" not in r:
            parser.error(f"Invalid format '{r}', expected name=entity/project/run_id")
        name, run_path = r.split("=", 1)
        experiments[name] = run_path

    os.makedirs(args.output_dir, exist_ok=True)
    api = wandb.Api()

    all_entries = {}
    for name, run_path in experiments.items():
        print(f"Fetching {name}...")
        results = fetch_run_metrics(api, run_path)
        all_entries.update(load_and_flatten(name, results))

    if not all_entries:
        print("No data found!")
        return

    print_table(all_entries)

    print("\nGenerating plots...")
    for metric in METRIC_KEYS:
        plot_metric(all_entries, metric, args.output_dir)
    plot_combined_core(all_entries, args.output_dir)

    print(f"\nDone! All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
