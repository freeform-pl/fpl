"""
Parse results.txt and fetch BOTH positive- and negative-direction scores per
seed from wandb, then print a table with mean / std per (section, method).

Positive score = mean_strict_success.
Negative score = order_reward * mean_full_success (range [-1, +1], -1 = perfect
inverted-order success).

For each direction we take the best value across the run history:
  - positive: max
  - negative: min (more negative is better)

Conditioned methods (single / rhp / rhp_neg) read from the `z_pos/test/` or
`z_neg/test/` prefix; unconditioned methods (demo only / success only / awr)
read from `test/`.

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

# Methods whose underlying runs are reward-conditioned (z_pos / z_neg eval).
# Matched after lowercasing and collapsing whitespace/underscores.
CONDITIONED_METHODS = {"single", "singlematchinglabels", "rhp", "rhpneg"}


def _is_conditioned(method):
    norm = re.sub(r"[\s_]+", "", (method or "").lower())
    return norm in CONDITIONED_METHODS

URL_RE = re.compile(
    r"https?://wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[A-Za-z0-9]+)"
)
SECTION_RE = re.compile(r'"(?P<name>[^"]+)"\s*:\s*\{')
METHOD_RE = re.compile(r'"(?P<name>[^"]+)"\s*:?')


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


def prefixes_for(method, direction):
    """Return ordered list of wandb key prefixes to try for this method/direction.

    First match wins.
    """
    if _is_conditioned(method):
        if direction == "pos":
            # Train-time logs use `z_pos/test/`; eval_conditioned.py uses
            # `eval/z_pos_`. Bare `test/` is the final fallback.
            return ["z_pos/test/", "zpos/test/", "eval/z_pos_", "test/"]
        else:
            return ["z_neg/test/", "zneg/test/", "eval/z_neg_", "test/"]
    # Unconditioned policies log under `test/` during training and
    # `eval/policy_` during a dedicated eval episode.
    return ["test/", "eval/policy_"]


def candidate_keys(method):
    """All keys this method/run might log for strict_success, order_reward,
    and mean_full_success (across pos / neg prefix variants)."""
    keys = []
    for direction in ("pos", "neg"):
        for prefix in prefixes_for(method, direction):
            keys.append(prefix + "mean_strict_success")
            keys.append(prefix + "order_reward")
            keys.append(prefix + "mean_full_success")
    return list(dict.fromkeys(keys))


def load_rows(run, method, samples=10000):
    """Fetch only the candidate keys via the sampled-history endpoint —
    much faster than `scan_history()` and still outer-joins by step.
    """
    keys = candidate_keys(method)
    try:
        rows = list(run.history(keys=keys, samples=samples, pandas=False))
    except Exception:
        rows = []
    rows.append(dict(run.summary))
    return rows


def best_pos_from_rows(rows, method):
    """Best (max) `mean_strict_success` across rows."""
    for prefix in prefixes_for(method, "pos"):
        k = prefix + "mean_strict_success"
        best = None
        for row in rows:
            v = row.get(k)
            if isinstance(v, (int, float)):
                vv = float(v)
                if best is None or vv > best:
                    best = vv
        if best is not None:
            return best, k
    return None, None


def best_neg_from_rows(rows, method):
    """Best (min) of `order_reward * mean_full_success` across rows.

    Uses each row independently; ignores rows missing either key.
    """
    for prefix in prefixes_for(method, "neg"):
        k_order = prefix + "order_reward"
        k_full = prefix + "mean_full_success"
        best = None
        for row in rows:
            o = row.get(k_order)
            f = row.get(k_full)
            if not isinstance(o, (int, float)) or not isinstance(f, (int, float)):
                continue
            s = float(o) * float(f)
            if best is None or s < best:
                best = s
        if best is not None:
            return best, prefix
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt"),
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help="Only process entries whose method matches (case- and space/underscore-insensitive). "
             "Repeatable. Example: --method 'rhp neg'",
    )
    parser.add_argument(
        "--plot",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_plot.png"
        ),
        help="Output path for the bar chart (set to '' to skip)",
    )
    args = parser.parse_args()

    entries = parse_results_file(args.results)
    if args.method:
        wanted = {re.sub(r"[\s_]+", "", m.lower()) for m in args.method}
        entries = [e for e in entries
                   if re.sub(r"[\s_]+", "", e[1].lower()) in wanted]
    if not entries:
        print(f"No runs parsed from {args.results}")
        return

    api = wandb.Api(timeout=120)
    grouped = defaultdict(list)   # (section, method) -> [(run_id, pos, neg)]
    per_run = []

    for section, method, entity, project, run_id in entries:
        ent = args.entity or entity
        path = f"{ent}/{project}/{run_id}"
        try:
            run = api.run(path)
        except Exception as e:
            print(f"[error] {path}: {e}")
            per_run.append((section, method, run_id, None, None, "FAILED", ""))
            continue

        rows = load_rows(run, method)
        pos, pos_src = best_pos_from_rows(rows, method)
        neg, neg_src = best_neg_from_rows(rows, method)

        grouped[(section, method)].append((run_id, pos, neg))
        per_run.append((section, method, run_id, pos, neg, pos_src, neg_src))
        if args.verbose:
            print(f"  {section} / {method} / {run_id}  "
                  f"pos={pos if pos is None else f'{pos:.4f}'} ({pos_src})  "
                  f"neg={neg if neg is None else f'{neg:.4f}'} ({neg_src})")

    # ---- Per-run table -----------------------------------------------------
    print()
    print("=" * 120)
    print(f"{'Section':<35} {'Method':<14} {'Run':<12} "
          f"{'PosStrict':>10} {'NegScore':>10}  pos_src / neg_src")
    print("-" * 120)
    for section, method, run_id, pos, neg, ps, ns in per_run:
        sp = f"{pos:.4f}" if isinstance(pos, float) else "-"
        sn = f"{neg:.4f}" if isinstance(neg, float) else "-"
        print(f"{section[:35]:<35} {method[:14]:<14} {run_id:<12} "
              f"{sp:>10} {sn:>10}  {ps} / {ns}")

    # ---- Aggregated table --------------------------------------------------
    def stats(vals):
        nums = [v for v in vals if isinstance(v, float)]
        if not nums:
            return 0, 0.0, 0.0
        return len(nums), mean(nums), (stdev(nums) if len(nums) > 1 else 0.0)

    print()
    print("=" * 110)
    print(f"{'Section':<35} {'Method':<14} {'N':>3}  "
          f"{'Pos μ±σ (max strict)':>22}  {'Neg μ±σ (min order*full)':>26}")
    print("-" * 110)
    agg = []
    for (section, method), vals in sorted(grouped.items()):
        pos_n, pos_m, pos_s = stats([v[1] for v in vals])
        neg_n, neg_m, neg_s = stats([v[2] for v in vals])
        n = max(pos_n, neg_n)
        agg.append((section, method, n, pos_n, pos_m, pos_s, neg_n, neg_m, neg_s))
        print(f"{section[:35]:<35} {method[:14]:<14} {n:>3}  "
              f"{pos_m:>10.4f}±{pos_s:<10.4f}  {neg_m:>12.4f}±{neg_s:<12.4f}")
    print("=" * 110)

    # ---- Plot --------------------------------------------------------------
    if args.plot and agg:
        sections = sorted({a[0] for a in agg})
        methods = sorted({a[1] for a in agg})
        method_color = {
            m: c for m, c in zip(methods, plt.cm.tab10(np.linspace(0, 1, len(methods))))
        }

        fig, (ax_pos, ax_neg) = plt.subplots(
            1, 2, figsize=(max(12, 4 * len(sections)), 5)
        )
        n_methods = len(methods)
        bar_w = 0.8 / max(n_methods, 1)
        x = np.arange(len(sections))

        for ax, title, m_idx, s_idx, n_idx in [
            (ax_pos, "Positive: best strict_success", 4, 5, 3),
            (ax_neg, "Negative: best (min) order_reward * full_success", 7, 8, 6),
        ]:
            for i, m in enumerate(methods):
                heights, errs, present, ns = [], [], [], []
                for s in sections:
                    hit = next((a for a in agg if a[0] == s and a[1] == m), None)
                    if hit is None or hit[n_idx] == 0:
                        heights.append(0); errs.append(0); present.append(False); ns.append(0)
                    else:
                        heights.append(hit[m_idx]); errs.append(hit[s_idx])
                        present.append(True); ns.append(hit[n_idx])
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
                            h,
                            f"{h:.2f}\n(n={n})",
                            ha="center",
                            va="bottom" if h >= 0 else "top",
                            fontsize=7,
                        )
            ax.set_xticks(x)
            ax.set_xticklabels(sections, rotation=15, ha="right")
            ax.set_title(title)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.grid(axis="y", alpha=0.3)

        ax_pos.set_ylabel("strict_success (higher is better)")
        ax_neg.set_ylabel("order * full (lower / more negative is better)")
        ax_neg.legend(title="method", loc="best", fontsize=9)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
