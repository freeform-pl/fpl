"""
Parse results.txt (slow_fast_final) and fetch BOTH positive- (right peg) and
negative- (left peg) direction metrics per seed from wandb.

For each direction we compute three numbers per seed:
  best_success    = max  of  test/mean_success_{right|left}
  step_at_best    = test/mean_first_success_step_{right|left} at the step
                     where best_success is reached (tie-break: smallest step)
  throughput      = best_success * 300 / step_at_best

Conditioned methods (single / rhp / rhp_neg) read from `z_pos/test/` (positive)
or `z_neg/test/` (negative); unconditioned methods read from `test/`.

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
    if _is_conditioned(method):
        if direction == "pos":
            # Train-time: `z_pos/test/`; eval_conditioned.py (phase5_eval):
            # `eval/z_pos_`. `test/` is the bare fallback.
            return ["z_pos/test/", "zpos/test/", "eval/z_pos_", "test/"]
        else:
            return ["z_neg/test/", "zneg/test/", "eval/z_neg_", "test/"]
    return ["test/", "eval/policy_"]


def candidate_keys(method):
    """All keys this method/run might log for success and step (both pegs)."""
    keys = []
    for direction in ("pos", "neg"):
        side = "right" if direction == "pos" else "left"
        for prefix in prefixes_for(method, direction):
            keys.append(prefix + f"mean_success_{side}")
            keys.append(prefix + f"mean_first_success_step_{side}")
    return list(dict.fromkeys(keys))  # dedupe, keep order


def load_rows(run, method, samples=10000):
    """Fetch only the keys we care about via the sampled-history endpoint.

    Much faster than `scan_history()` (which streams every key for every
    step). `run.history(keys=K)` outer-joins by step, so we don't lose rows
    where only one of the keys is present. Summary is appended as a final
    row to catch eval-only runs that don't log via history.
    """
    keys = candidate_keys(method)
    try:
        rows = list(run.history(keys=keys, samples=samples, pandas=False))
    except Exception:
        rows = []
    rows.append(dict(run.summary))
    return rows


def best_pair_from_rows(rows, method, direction):
    """Return (best_success, step_at_best, prefix_used).

    Direction = 'pos' -> right peg, 'neg' -> left peg.

    `mean_first_success_step_{side}` is only logged when at least one rollout
    actually succeeds, so it may be missing while `mean_success_{side}` exists.
    We report success regardless and leave step as None when not co-logged.
    """
    side = "right" if direction == "pos" else "left"
    succ_name = f"mean_success_{side}"
    step_name = f"mean_first_success_step_{side}"

    for prefix in prefixes_for(method, direction):
        k_succ = prefix + succ_name
        k_step = prefix + step_name

        best_s = None
        for row in rows:
            s = row.get(k_succ)
            if isinstance(s, (int, float)):
                v = float(s)
                if best_s is None or v > best_s:
                    best_s = v
        if best_s is None:
            continue

        # Among rows tied on best success, pick the smallest step (if any).
        best_t = None
        for row in rows:
            s = row.get(k_succ)
            t = row.get(k_step)
            if not isinstance(s, (int, float)) or float(s) != best_s:
                continue
            if isinstance(t, (int, float)):
                v = float(t)
                if best_t is None or v < best_t:
                    best_t = v
        return best_s, best_t, prefix
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
    # grouped[(section, method)] = list of dicts with pos / neg numbers
    grouped = defaultdict(list)
    per_run = []

    for section, method, entity, project, run_id in entries:
        ent = args.entity or entity
        path = f"{ent}/{project}/{run_id}"
        try:
            run = api.run(path)
        except Exception as e:
            print(f"[error] {path}: {e}")
            per_run.append((section, method, run_id, None, None, None, None, None, None, "FAILED"))
            continue

        rows = load_rows(run, method)
        ps, pt, p_src = best_pair_from_rows(rows, method, "pos")
        ns, nt, n_src = best_pair_from_rows(rows, method, "neg")
        p_thr = (ps * 300.0 / pt) if (ps is not None and pt and pt > 0) else None
        n_thr = (ns * 300.0 / nt) if (ns is not None and nt and nt > 0) else None

        grouped[(section, method)].append((run_id, ps, pt, p_thr, ns, nt, n_thr))
        per_run.append((section, method, run_id, ps, pt, p_thr, ns, nt, n_thr, f"{p_src}|{n_src}"))
        if args.verbose:
            def fmt(x, p=4):
                return "-" if x is None else f"{x:.{p}f}"
            print(f"  {section} / {method} / {run_id}  "
                  f"pos(R succ={fmt(ps)} step={fmt(pt,2)} thr={fmt(p_thr)})  "
                  f"neg(L succ={fmt(ns)} step={fmt(nt,2)} thr={fmt(n_thr)})  "
                  f"src={p_src}|{n_src}")

    # ---- Per-run table -----------------------------------------------------
    def fmt(x, p=4):
        return "-" if x is None else f"{x:.{p}f}"

    print()
    print("=" * 140)
    print(f"{'Section':<27} {'Method':<10} {'Run':<10} "
          f"| {'R_succ':>7} {'R_step':>7} {'R_thr':>7} "
          f"| {'L_succ':>7} {'L_step':>7} {'L_thr':>7} | src(pos|neg)")
    print("-" * 140)
    for section, method, run_id, ps, pt, p_thr, ns, nt, n_thr, src in per_run:
        print(f"{section[:27]:<27} {method[:10]:<10} {run_id:<10} "
              f"| {fmt(ps):>7} {fmt(pt,2):>7} {fmt(p_thr,2):>7} "
              f"| {fmt(ns):>7} {fmt(nt,2):>7} {fmt(n_thr,2):>7} | {src}")

    # ---- Aggregated table --------------------------------------------------
    def stats(vals):
        nums = [v for v in vals if isinstance(v, float)]
        if not nums:
            return 0, 0.0, 0.0
        return len(nums), mean(nums), (stdev(nums) if len(nums) > 1 else 0.0)

    print()
    print("=" * 150)
    print(f"{'Section':<27} {'Method':<10} {'N':>3}  "
          f"| {'PosSucc μ±σ':>14} {'PosStep μ±σ':>14} {'PosThr μ±σ':>14} "
          f"| {'NegSucc μ±σ':>14} {'NegStep μ±σ':>14} {'NegThr μ±σ':>14}")
    print("-" * 150)
    agg = []
    for (section, method), vals in sorted(grouped.items()):
        ps_n, ps_m, ps_s = stats([v[1] for v in vals])
        pt_n, pt_m, pt_s = stats([v[2] for v in vals])
        pr_n, pr_m, pr_s = stats([v[3] for v in vals])
        ns_n, ns_m, ns_s = stats([v[4] for v in vals])
        nt_n, nt_m, nt_s = stats([v[5] for v in vals])
        nr_n, nr_m, nr_s = stats([v[6] for v in vals])
        n = max(ps_n, ns_n)
        agg.append((
            section, method, n,
            ps_m, ps_s, pt_m, pt_s, pr_m, pr_s,
            ns_m, ns_s, nt_m, nt_s, nr_m, nr_s,
        ))
        print(f"{section[:27]:<27} {method[:10]:<10} {n:>3}  "
              f"| {ps_m:>6.4f}±{ps_s:<6.4f} {pt_m:>6.2f}±{pt_s:<6.2f} {pr_m:>6.4f}±{pr_s:<6.4f} "
              f"| {ns_m:>6.4f}±{ns_s:<6.4f} {nt_m:>6.2f}±{nt_s:<6.2f} {nr_m:>6.4f}±{nr_s:<6.4f}")
    print("=" * 150)

    # ---- Plot --------------------------------------------------------------
    if args.plot and agg:
        sections = sorted({a[0] for a in agg})
        methods = sorted({a[1] for a in agg})
        method_color = {
            m: c for m, c in zip(methods, plt.cm.tab10(np.linspace(0, 1, len(methods))))
        }

        # 2 rows (pos / neg) x 3 cols (succ / step / throughput)
        fig, axes = plt.subplots(
            2, 3, figsize=(max(15, 4.5 * len(sections)), 9),
        )

        # (row_label, [(title, mean_idx, std_idx)])
        panels = [
            ("Positive (Right peg)", [
                ("Success Right",                3, 4),
                ("First Success Step Right",     5, 6),
                ("Throughput Right",             7, 8),
            ]),
            ("Negative (Left peg)", [
                ("Success Left",                 9, 10),
                ("First Success Step Left",      11, 12),
                ("Throughput Left",              13, 14),
            ]),
        ]

        n_methods = len(methods)
        bar_w = 0.8 / max(n_methods, 1)
        x = np.arange(len(sections))

        for row_idx, (row_label, specs) in enumerate(panels):
            for col_idx, (title, mi, ei) in enumerate(specs):
                ax = axes[row_idx, col_idx]
                for i, m in enumerate(methods):
                    heights, errs, present = [], [], []
                    for s in sections:
                        hit = next((a for a in agg if a[0] == s and a[1] == m), None)
                        if hit is None or hit[2] == 0:
                            heights.append(0); errs.append(0); present.append(False)
                        else:
                            heights.append(hit[mi]); errs.append(hit[ei]); present.append(True)
                    offset = (i - (n_methods - 1) / 2) * bar_w
                    bars = ax.bar(
                        x + offset, heights, bar_w, yerr=errs, capsize=3,
                        label=m, color=method_color[m],
                        edgecolor="black", linewidth=0.5,
                    )
                    for bar, h, ok in zip(bars, heights, present):
                        if ok:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2, h,
                                f"{h:.2f}",
                                ha="center", va="bottom", fontsize=7,
                            )
                ax.set_xticks(x)
                ax.set_xticklabels(sections, rotation=15, ha="right")
                ax.set_title(f"{row_label} — {title}", fontsize=10)
                ax.grid(axis="y", alpha=0.3)

        axes[0, -1].legend(title="method", loc="best", fontsize=9)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
