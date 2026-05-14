"""Diagnostic: scan source HDF5s referenced by score JSONs and report image shapes.

Reports files whose agent_view / wrist images are not already (H, W) = (224, 224).
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import tyro
from tqdm import tqdm


def find_score_jsons(scores_dir: Path) -> list[Path]:
    entries = list(sorted(scores_dir.glob("*_score*.json")))
    for subdir in sorted(scores_dir.iterdir()):
        if subdir.is_dir():
            entries.extend(sorted(subdir.glob("*_score*.json")))
    return entries


def main(scores_dir: str, expected_h: int = 224, expected_w: int = 224):
    sd = Path(scores_dir)
    score_files = find_score_jsons(sd)
    print(f"Found {len(score_files)} score files")

    # Collect unique source HDF5 paths
    sources = []
    seen = set()
    for sf in score_files:
        try:
            data = json.loads(sf.read_text())
        except Exception as e:
            print(f"  [warn] cannot read {sf}: {e}")
            continue
        s = data.get("source_hdf5")
        if s and s not in seen:
            seen.add(s)
            sources.append(s)

    print(f"Unique source HDF5s: {len(sources)}\n")

    shape_counter: Counter = Counter()       # (cam, shape_tuple) -> count of demos
    bad_files: defaultdict = defaultdict(list)  # path -> list of (demo, cam, shape)
    missing = []
    errored = []

    for src in tqdm(sources, desc="Scanning HDF5s"):
        p = Path(src)
        if not p.exists():
            missing.append(src)
            continue
        try:
            with h5py.File(p, "r") as f:
                demo_keys = list(f["data"].keys())
                for dk in demo_keys:
                    obs = f[f"data/{dk}/obs"]
                    for cam in ("agent_view", "wrist"):
                        if cam not in obs:
                            continue
                        shape = tuple(obs[cam].shape)  # (T, H, W, C)
                        shape_counter[(cam, shape[1:])] += 1
                        if shape[1] != expected_h or shape[2] != expected_w:
                            bad_files[src].append((dk, cam, shape))
        except Exception as e:
            errored.append((src, str(e)))

    print("\n=== Shape distribution (cam, (H,W,C)) -> count of demos ===")
    for (cam, shp), n in sorted(shape_counter.items(), key=lambda x: -x[1]):
        marker = " <-- BAD" if (shp[0] != expected_h or shp[1] != expected_w) else ""
        print(f"  {cam:12s} {shp}  -> {n} demos{marker}")

    print(f"\n=== Files with non-{expected_h}x{expected_w} frames: {len(bad_files)} ===")
    for src, entries in bad_files.items():
        print(f"\n  {src}")
        # group by (cam, shape) within file for compactness
        per_shape: defaultdict = defaultdict(list)
        for dk, cam, shape in entries:
            per_shape[(cam, shape)].append(dk)
        for (cam, shape), dks in per_shape.items():
            print(f"    {cam} shape={shape} demos={dks}")

    if missing:
        print(f"\n=== Missing source HDF5s: {len(missing)} ===")
        for m in missing:
            print(f"  {m}")

    if errored:
        print(f"\n=== Errors opening files: {len(errored)} ===")
        for src, err in errored:
            print(f"  {src}: {err}")

    # Save the bad-file list for easy downstream use
    out = sd / "bad_image_shapes.json"
    payload = {
        "expected_hw": [expected_h, expected_w],
        "shape_distribution": {f"{cam}:{shp}": n for (cam, shp), n in shape_counter.items()},
        "bad_files": {src: [{"demo": dk, "cam": cam, "shape": list(shape)} for dk, cam, shape in entries] for src, entries in bad_files.items()},
        "missing": missing,
        "errored": [{"src": s, "err": e} for s, e in errored],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    tyro.cli(main)
