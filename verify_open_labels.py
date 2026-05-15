"""Quick sanity check that OpenPreferenceDataset returns the right axis label
paired with the right preference value, by cross-checking against the raw JSON.

Usage:
    python verify_open_labels.py --task setup_table \
        --preferences_dir /iris/u/am208/droid-robot/preferences_setup \
        --n 20
"""
import argparse
import json
import os

from dataset import make_datasets, OpenPreferenceDataset
from tasks import TASKS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="setup_table")
    p.add_argument("--preferences_dir", required=True)
    p.add_argument("--n", type=int, default=20, help="Number of items to check")
    args = p.parse_args()

    preference_keys = TASKS[args.task]
    print(f"TASKS[{args.task}] order:")
    for k, key in enumerate(preference_keys):
        print(f"  [{k}] {key}")
    print()

    train_ds, _ = make_datasets(
        task=args.task,
        preferences_dir=args.preferences_dir.split(","),
        val_fraction=0.2,
        stride=20, seq_len=20, img_size=(128, 128),
        seed=0, preload=False, action_chunk_size=0,
    )

    open_ds = OpenPreferenceDataset(train_ds, preference_keys)

    mismatches = 0
    checked = 0
    for idx in range(min(args.n, len(open_ds))):
        # peek without invoking full image loading by going through .items
        base_idx, axis_idx, axis_name = open_ds.items[idx]
        sample = train_ds.samples[base_idx]
        label_val = float(sample["labels"][axis_idx].item())
        session = sample.get("session", "")

        # Locate the JSON for this session. Sessions are subdirs of preferences_dir.
        json_path = None
        for pdir in args.preferences_dir.split(","):
            cand = os.path.join(pdir, session, "preference.json")
            if os.path.exists(cand):
                json_path = cand
                break
        if json_path is None:
            print(f"[{idx}] session={session}  axis={axis_name!r}  label={label_val}  (no JSON found — likely cross-pair)")
            continue

        with open(json_path) as f:
            meta = json.load(f)
        raw = meta["preferences"].get(axis_name, "Equal")
        expected = {"A": 1.0, "B": 0.0}.get(raw, 0.5)
        ok = "OK" if abs(label_val - expected) < 1e-6 else "MISMATCH"
        if ok != "OK":
            mismatches += 1
        checked += 1
        print(f"[{idx}] {ok}  session={session}  axis={axis_name!r:42s}  "
              f"label={label_val}  json={raw!r}  expected={expected}")

    print(f"\nChecked {checked}, mismatches: {mismatches}")


if __name__ == "__main__":
    main()
