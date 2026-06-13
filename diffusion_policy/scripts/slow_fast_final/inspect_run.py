"""
Dump wandb keys for a single run so we can figure out which prefix the
metrics were logged under.

Usage:
  python inspect_run.py <wandb_url_or_path>
  python inspect_run.py https://wandb.ai/memory_rl/slow_fast_final_rhp/runs/wxv009bb
  python inspect_run.py memory_rl/slow_fast_final_rhp/wxv009bb
"""

import re
import sys
import wandb

URL_RE = re.compile(
    r"https?://wandb\.ai/(?P<entity>[^/]+)/(?P<project>[^/]+)/runs/(?P<run_id>[A-Za-z0-9]+)"
)


def parse(arg):
    m = URL_RE.search(arg)
    if m:
        return f"{m['entity']}/{m['project']}/{m['run_id']}"
    return arg.strip()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = parse(sys.argv[1])
    print(f"Inspecting {path}\n")

    api = wandb.Api()
    run = api.run(path)
    print(f"Run name: {run.name}")
    print(f"State:    {run.state}")
    print()

    # --- Summary keys ---
    keys_of_interest = lambda k: any(
        s in k.lower()
        for s in ("success", "step", "order_reward", "full_success",
                  "throughput", "score", "left", "right")
    )

    summary_keys = sorted(k for k in run.summary.keys() if keys_of_interest(k))
    print(f"Summary keys ({len(summary_keys)} of interest):")
    for k in summary_keys:
        v = run.summary[k]
        if isinstance(v, (int, float)):
            print(f"  {k} = {v:.4f}")
        else:
            print(f"  {k} = <{type(v).__name__}>")

    # --- History keys (sample a few rows to see what's there) ---
    print("\nHistory: sampling 5 rows to discover keys...")
    try:
        hist_rows = list(run.history(samples=5, pandas=False))
    except Exception as e:
        print(f"  history() failed: {e}")
        hist_rows = []

    hist_keys = set()
    for row in hist_rows:
        for k in row.keys():
            if keys_of_interest(k):
                hist_keys.add(k)

    print(f"History keys ({len(hist_keys)} of interest):")
    for k in sorted(hist_keys):
        print(f"  {k}")

    # --- Group keys by prefix to make the structure obvious ---
    all_keys = set(summary_keys) | hist_keys
    print(f"\nPrefixes used (heuristic — first segment up to last '/' or known '_label_'):")
    prefixes = set()
    for k in all_keys:
        if "/" in k:
            prefixes.add(k.rsplit("/", 1)[0] + "/")
        else:
            for label in ("z_pos_", "z_neg_", "z_zero_", "policy_"):
                idx = k.find(label)
                if idx >= 0:
                    prefixes.add(k[: idx + len(label)])
                    break
    for p in sorted(prefixes):
        print(f"  {p}")


if __name__ == "__main__":
    main()
