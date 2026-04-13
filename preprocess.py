"""
Preprocessing script: extract frames from MP4 videos and write them into the
existing HDF5 files as uint8 image arrays.

Run once before training:
    python preprocess.py [--preferences_dir preferences] [--img_size 128] [--jobs 4]

Adds to each rollout_A/B.hdf5:
    observation/third_person  (N, H, W, 3)  uint8
    observation/wrist         (N, H, W, 3)  uint8
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import h5py
import numpy as np


def extract_frames(video_path: str, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Read all frames from video_path, split each frame into third_person (left)
    and wrist (right), resize to (img_size, img_size).

    Returns:
        third_person: (N, img_size, img_size, 3) uint8
        wrist:        (N, img_size, img_size, 3) uint8
    """
    cap = cv2.VideoCapture(video_path)
    tp_frames, wr_frames = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        half_w = frame.shape[1] // 2
        tp = cv2.resize(frame[:, :half_w], (img_size, img_size), interpolation=cv2.INTER_AREA)
        wr = cv2.resize(frame[:, half_w:], (img_size, img_size), interpolation=cv2.INTER_AREA)
        tp_frames.append(tp)
        wr_frames.append(wr)
    cap.release()
    return np.stack(tp_frames), np.stack(wr_frames)


def process_session(session_dir: str, img_size: int) -> str:
    for rollout in ("rollout_A", "rollout_B"):
        hdf5_path = os.path.join(session_dir, f"{rollout}.hdf5")
        video_path = os.path.join(session_dir, f"{rollout}.mp4")

        if not os.path.exists(video_path):
            continue

        tp, wr = extract_frames(video_path, img_size)

        with h5py.File(hdf5_path, "a") as f:
            obs = f.require_group("observation")
            for key, arr in (("third_person", tp), ("wrist", wr)):
                if key in obs:
                    del obs[key]
                obs.create_dataset(key, data=arr, compression="lzf")

    return session_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preferences_dir", default="preferences")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()

    session_dirs = sorted(
        d for d in glob.glob(os.path.join(args.preferences_dir, "*"))
        if os.path.isdir(d)
    )
    print(f"Processing {len(session_dirs)} sessions at {args.img_size}x{args.img_size}...")

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(process_session, d, args.img_size): d for d in session_dirs}
        for i, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                fut.result()
                print(f"[{i}/{len(session_dirs)}] {os.path.basename(d)}")
            except Exception as e:
                print(f"[{i}/{len(session_dirs)}] ERROR {os.path.basename(d)}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
