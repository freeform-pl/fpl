"""
Parse results.txt and fetch the best test/mean_strict_success (or
zpos/test/mean_strict_success) per seed from wandb, then print a table
with mean / std grouped by (experiment section, method).

Usage:
  python fetch_results.py                      # uses results.txt in same dir
  python fetch_results.py --results other.txt
  python fetch_results.py --entity memory_rl --metric test/mean_strict_success
"""

import argparse
import os
import re
from collections import defaultdict
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import wandb

DEFAULT_ENTITY = "memory_rl"
DEFAULT_METRIC = "test/mean_strict_success"
FALLBACK_METRIC = "zpos/test/mean_strict_success"

URL_RE = re.compile(
    r"https?://wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[A-Za-z0-9]+)"
)
SECTION_RE = re.compile(r'"(?P<name>[^"]+)"\s*:\s*\{')
METHOD_RE = re.compile(r'"(?P<name>[^"]+)"\s*:')


def parse_results_file(path):
    """Return list of (section, method, entity, project, run_id) tuples.

    The results file is loosely structured:
      "section name": {
          "method name": <url1>
          <url2>          # continuation seeds for the same method
          "next method": <url1>
          ...
      }
    """
    entries = []
    section = None
    method = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            sec_match = SECTION_RE.search(line)
            # A line that starts a section also has a method on the same line
            # only if it ends with `{` — those are top-level sections.
            if sec_match and stripped.endswith("{"):
                section = sec_match.group("name")
                method = None
                continue

            if stripped.startswith("}"):
                section = None
                method = None
                continue

            url_match = URL_RE.search(line)
            method_match = METHOD_RE.search(line)
            if method_match and url_match and method_match.start() < url_match.start():
                method = method_match.group("name")
            elif method_match and not url_match:
                method = method_match.group("name")

            if url_match:
                entries.append((
                    section or "(no section)",
                    method or "(no method)",
                    url_match.group("entity"),
                    url_match.group("project"),
                    url_match.group("run_id"),
                ))
    return entries


def best_metric_from_run(run, metric, fallback):
    """Return the best (max) value of `metric` (or fallback) across the run.

    Try summary first, then scan history.
    """
    for key in (metric, fallback):
        v = run.summary.get(key)
        if isinstance(v, (int, float)):
            return float(v), key, "summary"

    for key in (metric, fallback):
        best = None
        try:
            for row in run.scan_history(keys=[key]):
                v = row.get(key)
                if isinstance(v, (int, float)):
                    if best is None or v > best:
                        best = float(v)
        except Exception:
            continue
        if best is not None:
            return best, key, "history-max"

    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt"),
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY,
                        help="Override the wandb entity parsed from the URL.")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--fallback-metric", default=FALLBACK_METRIC)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--plot",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_plot.png"
        ),
        help="Output path for the bar chart (set to '' to skip)",
    )
    args = parser.parse_args()

    entries = parse_results_file(args.results)
    if not entries:
        print(f"No runs parsed from {args.results}")
        return

    api = wandb.Api()
    grouped = defaultdict(list)   # (section, method) -> [(run_id, best, source_key)]
    per_run_info = []

    for section, method, entity, project, run_id in entries:
        ent = args.entity or entity
        path = f"{ent}/{project}/{run_id}"
        try:
            run = api.run(path)
        except Exception as e:
            print(f"[error] {path}: {e}")
            per_run_info.append((section, method, run_id, None, "FAILED", str(e)))
            continue

        best, source_key, source = best_metric_from_run(
            run, args.metric, args.fallback_metric
        )
        if best is None:
            print(f"[warn] {path}: no value found for "
                  f"'{args.metric}' or '{args.fallback_metric}'")
            per_run_info.append((section, method, run_id, None, "MISSING", ""))
            continue

        grouped[(section, method)].append((run_id, best, source_key))
        per_run_info.append((section, method, run_id, best, source_key, source))
        if args.verbose:
            print(f"  {section} / {method} / {run_id} = {best:.4f}  ({source_key}, {source})")

    # ---- Per-run table -----------------------------------------------------
    print()
    print("=" * 110)
    print(f"{'Section':<40} {'Method':<18} {'Run':<12} {'Best':>10} {'Metric':<35}")
    print("-" * 110)
    for section, method, run_id, best, key, _ in per_run_info:
        best_str = f"{best:.4f}" if isinstance(best, float) else str(best)
        print(f"{section[:40]:<40} {method[:18]:<18} {run_id:<12} {best_str:>10} {str(key)[:35]:<35}")

    # ---- Aggregated table --------------------------------------------------
    print()
    print("=" * 90)
    print(f"{'Section':<40} {'Method':<18} {'N':>3}  {'Mean':>10}  {'Std':>10}")
    print("-" * 90)
    agg = []  # (section, method, n, mean, std)
    for (section, method), vals in sorted(grouped.items()):
        nums = [v for _, v, _ in vals]
        n = len(nums)
        m = mean(nums)
        s = stdev(nums) if n > 1 else 0.0
        agg.append((section, method, n, m, s))
        print(f"{section[:40]:<40} {method[:18]:<18} {n:>3}  {m:>10.4f}  {s:>10.4f}")
    print("=" * 90)

    # ---- Plot --------------------------------------------------------------
    if args.plot and agg:
        sections = sorted({a[0] for a in agg})
        methods = sorted({a[1] for a in agg})
        method_color = {
            m: c for m, c in zip(methods, plt.cm.tab10(np.linspace(0, 1, len(methods))))
        }

        n_methods = len(methods)
        bar_w = 0.8 / max(n_methods, 1)
        x = np.arange(len(sections))

        fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(sections)), 5))
        for i, m in enumerate(methods):
            heights, errs, present = [], [], []
            for s in sections:
                hit = next((a for a in agg if a[0] == s and a[1] == m), None)
                if hit is None:
                    heights.append(0)
                    errs.append(0)
                    present.append(False)
                else:
                    heights.append(hit[3])
                    errs.append(hit[4])
                    present.append(True)
            offset = (i - (n_methods - 1) / 2) * bar_w
            bars = ax.bar(
                x + offset, heights, bar_w, yerr=errs, capsize=3,
                label=m, color=method_color[m],
                edgecolor="black", linewidth=0.5,
            )
            for bar, h, ok, n in zip(
                bars, heights, present,
                [next((a[2] for a in agg if a[0] == s and a[1] == m), 0) for s in sections],
            ):
                if ok:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.01,
                        f"{h:.2f}\n(n={n})",
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(sections, rotation=15, ha="right")
        ax.set_ylabel(f"best {args.metric}")
        ax.set_title("Best strict-success per seed (mean ± std)")
        ax.set_ylim(0, max(1.0, max(a[3] + a[4] for a in agg) * 1.15))
        ax.legend(title="method", loc="best", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
