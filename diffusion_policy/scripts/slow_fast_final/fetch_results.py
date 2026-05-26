"""
Parse results.txt (slow_fast_final) and fetch three metrics per seed from wandb:

  1. best_success_right     = max of  test/mean_success_right
  2. step_at_best           = test/mean_first_success_step_right at the step
                              where best_success_right is reached (if multiple
                              steps tie on success_right, take the smallest step)
  3. throughput             = best_success_right * 300 / step_at_best

Then print mean/std per (section, method) across seeds and save a plot.

Usage:
  python fetch_results.py [--verbose] [--plot out.png]
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
SUCCESS_KEY = "test/mean_success_right"
STEP_KEY = "test/mean_first_success_step_right"
FALLBACK_SUCCESS_KEY = "zpos/test/mean_success_right"
FALLBACK_STEP_KEY = "zpos/test/mean_first_success_step_right"

URL_RE = re.compile(
    r"https?://wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[A-Za-z0-9]+)"
)
SECTION_RE = re.compile(r'"(?P<name>[^"]+)"\s*:\s*\{')
METHOD_RE = re.compile(r'"(?P<name>[^"]+)"\s*:')


def parse_results_file(path):
    """Return list of (section, method, entity, project, run_id) tuples."""
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


def fetch_best_pair(run, success_key, step_key):
    """Scan run history; return (best_success, step_at_best) or (None, None).

    Tie-break: among rows whose success == max success, pick the smallest step.
    Rows missing either metric are skipped.
    """
    best_success = None
    best_step = None
    try:
        rows = list(run.scan_history(keys=[success_key, step_key]))
    except Exception:
        return None, None

    for row in rows:
        s = row.get(success_key)
        t = row.get(step_key)
        if not isinstance(s, (int, float)) or not isinstance(t, (int, float)):
            continue
        s = float(s)
        t = float(t)
        if best_success is None or s > best_success:
            best_success = s
            best_step = t
        elif s == best_success and (best_step is None or t < best_step):
            best_step = t

    return best_success, best_step


def best_for_run(run):
    """Try primary metric pair, fall back to zpos/ variants."""
    s, t = fetch_best_pair(run, SUCCESS_KEY, STEP_KEY)
    if s is not None and t is not None:
        return s, t, SUCCESS_KEY
    s, t = fetch_best_pair(run, FALLBACK_SUCCESS_KEY, FALLBACK_STEP_KEY)
    if s is not None and t is not None:
        return s, t, FALLBACK_SUCCESS_KEY
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt"),
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
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
    # grouped[(section, method)] = list of (run_id, success, step, throughput)
    grouped = defaultdict(list)
    per_run = []

    for section, method, entity, project, run_id in entries:
        ent = args.entity or entity
        path = f"{ent}/{project}/{run_id}"
        try:
            run = api.run(path)
        except Exception as e:
            print(f"[error] {path}: {e}")
            per_run.append((section, method, run_id, None, None, None, "FAILED"))
            continue

        success, step, src = best_for_run(run)
        if success is None or step is None:
            print(f"[warn] {path}: no paired ({SUCCESS_KEY}, {STEP_KEY}) found")
            per_run.append((section, method, run_id, None, None, None, "MISSING"))
            continue

        throughput = (success * 300.0 / step) if step > 0 else 0.0
        grouped[(section, method)].append((run_id, success, step, throughput))
        per_run.append((section, method, run_id, success, step, throughput, src))
        if args.verbose:
            print(f"  {section} / {method} / {run_id} "
                  f"success={success:.4f} step={step:.2f} thr={throughput:.4f} ({src})")

    # ---- Per-run table -----------------------------------------------------
    print()
    print("=" * 110)
    print(f"{'Section':<35} {'Method':<14} {'Run':<12} "
          f"{'Succ_R':>8} {'Step_R':>8} {'Throughput':>11}  Src")
    print("-" * 110)
    for section, method, run_id, succ, step, thr, src in per_run:
        ss = f"{succ:.4f}" if isinstance(succ, float) else "-"
        st = f"{step:.2f}" if isinstance(step, float) else "-"
        tt = f"{thr:.4f}" if isinstance(thr, float) else "-"
        print(f"{section[:35]:<35} {method[:14]:<14} {run_id:<12} "
              f"{ss:>8} {st:>8} {tt:>11}  {src}")

    # ---- Aggregated table --------------------------------------------------
    print()
    print("=" * 110)
    print(f"{'Section':<35} {'Method':<14} {'N':>3}  "
          f"{'Succ μ±σ':>16}  {'Step μ±σ':>16}  {'Throughput μ±σ':>18}")
    print("-" * 110)

    agg = []
    for (section, method), vals in sorted(grouped.items()):
        succs = [v[1] for v in vals]
        steps = [v[2] for v in vals]
        thrs = [v[3] for v in vals]
        n = len(vals)
        sm, ss = mean(succs), (stdev(succs) if n > 1 else 0.0)
        tm, ts = mean(steps), (stdev(steps) if n > 1 else 0.0)
        rm, rs = mean(thrs), (stdev(thrs) if n > 1 else 0.0)
        agg.append((section, method, n, sm, ss, tm, ts, rm, rs))
        print(f"{section[:35]:<35} {method[:14]:<14} {n:>3}  "
              f"{sm:>7.4f}±{ss:<7.4f}  {tm:>7.2f}±{ts:<7.2f}  {rm:>9.4f}±{rs:<7.4f}")
    print("=" * 110)

    # ---- Plot --------------------------------------------------------------
    if args.plot and agg:
        sections = sorted({a[0] for a in agg})
        methods = sorted({a[1] for a in agg})
        method_color = {
            m: c for m, c in zip(methods, plt.cm.tab10(np.linspace(0, 1, len(methods))))
        }

        metric_specs = [
            ("Success Right (best)", 3, 4, None),
            ("First Success Step Right (at best)", 5, 6, None),
            ("Throughput = succ * 300 / step", 7, 8, None),
        ]

        fig, axes = plt.subplots(
            1, 3, figsize=(max(15, 4.5 * len(sections)), 5),
        )
        if len(metric_specs) == 1:
            axes = [axes]

        n_methods = len(methods)
        bar_w = 0.8 / max(n_methods, 1)
        x = np.arange(len(sections))

        for ax, (title, mi, ei, _) in zip(axes, metric_specs):
            for i, m in enumerate(methods):
                heights, errs, present, ns = [], [], [], []
                for s in sections:
                    hit = next((a for a in agg if a[0] == s and a[1] == m), None)
                    if hit is None:
                        heights.append(0); errs.append(0); present.append(False); ns.append(0)
                    else:
                        heights.append(hit[mi]); errs.append(hit[ei])
                        present.append(True); ns.append(hit[2])
                offset = (i - (n_methods - 1) / 2) * bar_w
                bars = ax.bar(
                    x + offset, heights, bar_w, yerr=errs, capsize=3,
                    label=m, color=method_color[m],
                    edgecolor="black", linewidth=0.5,
                )
                for bar, h, ok, n in zip(bars, heights, present, ns):
                    if ok:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + (max(heights) * 0.01 if max(heights) > 0 else 0.01),
                            f"{h:.2f}\n(n={n})",
                            ha="center", va="bottom", fontsize=7,
                        )
            ax.set_xticks(x)
            ax.set_xticklabels(sections, rotation=15, ha="right")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)

        axes[0].set_ylabel("value")
        axes[-1].legend(title="method", loc="best", fontsize=9)
        fig.suptitle("Slow/Fast — best success_right and paired metrics", y=1.02)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
