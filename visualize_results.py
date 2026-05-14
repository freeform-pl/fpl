"""
Visualize results from wandb for slow/fast mid-range discrete experiments.

Uses run.summary (fast) for final metrics and run.history() for training curves.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ENTITY = "memory_rl"
PROJECT_PREFIX = "slow_fast_medium_no_rollouts_mid_range_discrete"

# ── Experiment definitions ──────────────────────────────────────────────────
# display name -> (project_suffix, run_id, is_conditioned)
EXPERIMENTS = {
    # RHP iterations (conditioned eval with z_pos / z_neg / z_zero)
    "RHP iter0": ("rhp", "8w7hm4id", True),
    "RHP iter1": ("rhp", "9grn1503", True),
    "RHP iter2": ("rhp", "ihj84r3e", True),
    "RHP iter3": ("rhp", "ut5yl1bn", True),
    # Baselines
    "Single Pref": ("single_pref", "gtokf16z", True),
    "Demo Only": ("demo_only", "6l7964nt", False),
    # "Demo Success": ("demo_success", "6l7964nt", False),
    "AWR": ("awr", "y1y33azt", False),
}

METRICS = {
    "Success Rate": "test/mean_success",
    "Time to First Success": "test/mean_first_success_step",
    "Success Rate (Right Peg)": "test/mean_success_right",
    "Success Rate (Left Peg)": "test/mean_success_left",
    "Time to 1st Success (R)": "test/mean_first_success_step_right",
    "Time to 1st Success (L)": "test/mean_first_success_step_left",
    "Speed Reward": "test/mean_speed_reward",
    "Score": "test/mean_score",
}

Z_TARGETS = ["z_pos", "z_neg"]


def get_summary(run, key):
    """Get numeric value from run summary."""
    v = run.summary.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def fetch_history(run, keys):
    """Fetch training curve data by loading each key separately and merging on nearest _step.

    Wandb logs different metrics at different moments and samples differently
    per key, so steps won't match exactly. We fetch each key individually,
    then use pd.merge_asof to join on the nearest step.
    """
    try:
        merged = None
        for key in keys:
            df = run.history(keys=[key, "_step"], samples=10000, pandas=True)
            if df.empty or key not in df.columns:
                continue
            chunk = df[["_step", key]].dropna(subset=[key]).sort_values("_step").reset_index(drop=True)
            if chunk.empty:
                continue
            if merged is None:
                merged = chunk
            else:
                merged = pd.merge_asof(
                    merged.sort_values("_step"),
                    chunk,
                    on="_step",
                    direction="nearest",
                )
        return merged if merged is not None and not merged.empty else None
    except Exception as e:
        print(f"    [warn] history fetch failed: {e}")
        return None


def main():
    api = wandb.Api()
    print("Fetching runs from wandb...\n")

    runs = {}
    for name, (suffix, run_id, is_cond) in EXPERIMENTS.items():
        project = f"{PROJECT_PREFIX}_{suffix}"
        try:
            run = api.run(f"{ENTITY}/{project}/{run_id}")
            runs[name] = (run, is_cond)
            print(f"  {name}: {run.name} (state={run.state})")
        except Exception as e:
            print(f"  [error] {name}: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 1: Bar chart of final metrics
    # ═══════════════════════════════════════════════════════════════════════
    n_metrics = len(METRICS)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 6 * nrows))

    for idx, (metric_title, metric_key) in enumerate(METRICS.items()):
        ax = axes.flat[idx]
        labels, values, colors = [], [], []

        for name, (run, is_cond) in runs.items():
            if is_cond:
                for z in Z_TARGETS:
                    val = get_summary(run, f"{z}/{metric_key}")
                    if val is not None:
                        labels.append(f"{name}\n({z})")
                        values.append(val)
                        colors.append("#4C72B0" if z == "z_pos" else "#DD8452")
            else:
                val = get_summary(run, metric_key)
                if val is not None:
                    labels.append(name)
                    values.append(val)
                    colors.append("#55A868")

        if labels:
            bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(metric_title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Final Metrics (run summary) - Mid Range Discrete", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results_final_metrics.png", dpi=150, bbox_inches="tight")
    print("\nSaved: results_final_metrics.png")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 2: RHP iteration progression
    # ═══════════════════════════════════════════════════════════════════════
    rhp_names = sorted([n for n in runs if n.startswith("RHP")])
    if rhp_names:
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 6 * nrows))

        for idx, (metric_title, metric_key) in enumerate(METRICS.items()):
            ax = axes.flat[idx]

            for z in Z_TARGETS + ["z_zero"]:
                vals, labels = [], []
                for name in rhp_names:
                    run, _ = runs[name]
                    val = get_summary(run, f"{z}/{metric_key}")
                    if val is not None:
                        vals.append(val)
                        labels.append(name.replace("RHP ", ""))

                if vals:
                    style = {"z_pos": ("-o", "#4C72B0"), "z_neg": ("--s", "#DD8452"), "z_zero": (":^", "#937860")}
                    ls, c = style.get(z, ("-o", "gray"))
                    ax.plot(labels, vals, ls, color=c, label=z, markersize=8, linewidth=2)

            ax.set_title(metric_title, fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle("RHP Iteration Progress (z_pos=[0.9,0.9], z_neg=[0.5,0.9])", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("results_rhp_iterations.png", dpi=150, bbox_inches="tight")
        print("Saved: results_rhp_iterations.png")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 3: Training curves (sampled history)
    # ═══════════════════════════════════════════════════════════════════════
    print("\nFetching training histories (sampled)...")
    histories = {}
    for name, (run, is_cond) in runs.items():
        keys = []
        for mk in METRICS.values():
            if is_cond:
                for z in Z_TARGETS:
                    keys.append(f"{z}/{mk}")
            else:
                keys.append(mk)
        hist = fetch_history(run, keys)
        if hist is not None:
            histories[name] = hist
            print(f"  {name}: {len(hist)} rows")
        else:
            print(f"  {name}: no history")

    if histories:
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 6 * nrows))
        tab10 = plt.cm.tab10.colors
        name_colors = {name: tab10[i % len(tab10)] for i, name in enumerate(runs)}

        for idx, (metric_title, metric_key) in enumerate(METRICS.items()):
            ax = axes.flat[idx]

            for name, hist in histories.items():
                _, is_cond = runs[name]
                color = name_colors[name]

                if is_cond:
                    for z in Z_TARGETS:
                        key = f"{z}/{metric_key}"
                        if key in hist.columns:
                            series = hist[["_step", key]].dropna()
                            if not series.empty:
                                ls = "-" if z == "z_pos" else "--"
                                alpha = 1.0 if z == "z_pos" else 0.5
                                ax.plot(series["_step"], series[key],
                                       label=f"{name} ({z})", color=color,
                                       linestyle=ls, alpha=alpha, linewidth=1.5)
                else:
                    if metric_key in hist.columns:
                        series = hist[["_step", metric_key]].dropna()
                        if not series.empty:
                            ax.plot(series["_step"], series[metric_key],
                                   label=name, color=color, linewidth=1.5)

            ax.set_title(metric_title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Step")
            ax.legend(fontsize=6, loc="best")
            ax.grid(True, alpha=0.3)

        plt.suptitle("Training Curves (sampled) - Mid Range Discrete", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("results_training_curves.png", dpi=150, bbox_inches="tight")
        print("Saved: results_training_curves.png")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary table - metrics at step with BEST SUCCESS RATE
    # ═══════════════════════════════════════════════════════════════════════
    def best_success_row(hist, success_key, metric_keys):
        """Find the row with highest success rate and return all metrics from it."""
        if hist is None or success_key not in hist.columns:
            return None
        valid = hist.dropna(subset=[success_key])
        if valid.empty:
            return None
        best_idx = valid[success_key].idxmax()
        best_row = valid.loc[best_idx]
        return {mk: float(best_row[mk]) if mk in best_row and pd.notna(best_row[mk]) else None
                for mk in metric_keys}

    print("\n" + "=" * 130)
    print("BEST METRICS (all metrics from step with highest success rate RIGHT PEG)")
    print("=" * 130)

    header = f"{'Experiment':<30s}"
    for mt in METRICS:
        header += f"  {mt[:20]:>20s}"
    print(header)
    print("-" * 130)

    for name, (run, is_cond) in runs.items():
        hist = histories.get(name)
        if is_cond:
            for z in Z_TARGETS:
                label = f"{name} ({z})"
                success_key = f"{z}/test/mean_success_right"
                metric_keys = [f"{z}/{mk}" for mk in METRICS.values()]
                best = best_success_row(hist, success_key, metric_keys)
                row = f"{label:<30s}"
                for mk in METRICS.values():
                    key = f"{z}/{mk}"
                    val = best[key] if best and best.get(key) is not None else get_summary(run, key)
                    row += f"  {val:>20.4f}" if val is not None else f"  {'N/A':>20s}"
                print(row)
        else:
            label = name
            success_key = "test/mean_success_right"
            metric_keys = list(METRICS.values())
            best = best_success_row(hist, success_key, metric_keys)
            row = f"{label:<30s}"
            for mk in METRICS.values():
                val = best[mk] if best and best.get(mk) is not None else get_summary(run, mk)
                row += f"  {val:>20.4f}" if val is not None else f"  {'N/A':>20s}"
            print(row)

    plt.show()


if __name__ == "__main__":
    main()
