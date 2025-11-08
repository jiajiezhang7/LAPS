from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import numpy as np


def _frame_hof(gray_prev, gray_curr, bins: int = 16) -> np.ndarray:
    """Compute magnitude-weighted orientation histogram (HOF) between two gray frames.
    Returns (bins,) float32 vector L1-normalized later at clip aggregation stage.
    """
    import cv2  # local import to avoid hard dependency at module import time
    # Dense optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)
    # Orientation range [0, 2pi)
    ang = np.mod(ang, 2.0 * np.pi)
    # Histogram of orientations weighted by magnitude
    hist, _ = np.histogram(ang, bins=bins, range=(0.0, 2.0 * np.pi), weights=mag)
    return hist.astype(np.float32)


def extract_hof_features_for_video(
    video_path: Path,
    clip_duration: float = 2.0,
    clip_stride: float = 0.4,
    bins: int = 16,
) -> Optional[np.ndarray]:
    """Extract clip-wise HOF features for a video.

    Args:
        video_path: input video file
        clip_duration: window length in seconds
        clip_stride: step in seconds
        bins: number of orientation bins
    Returns:
        X: (N_clips, bins) float32 array; None on failure
    """
    import cv2  # local import (see _frame_hof comment)

    p = Path(video_path)
    if not p.exists():
        print(f"[HOF] Video not found: {p}")
        return None

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        print(f"[HOF] Failed to open video: {p}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0  # fallback
    dt = 1.0 / float(fps)

    # Read first frame
    ok, prev = cap.read()
    if not ok or prev is None:
        cap.release()
        print(f"[HOF] Empty or unreadable video: {p}")
        return None
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Per-flow (between consecutive frames) HOF and their timestamps (aligned to 'curr' frame time)
    flow_hists: List[np.ndarray] = []
    flow_times: List[float] = []
    t_curr = dt  # time of frame index 1 (second frame)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            h = _frame_hof(prev_gray, gray, bins=bins)
        except Exception as e:
            print(f"[HOF] Flow failed at t={t_curr:.3f}s in {p.name}: {e}")
            prev_gray = gray
            t_curr += dt
            continue
        flow_hists.append(h)
        flow_times.append(t_curr)
        prev_gray = gray
        t_curr += dt

    cap.release()

    if len(flow_hists) == 0:
        print(f"[HOF] No flow computed for {p.name}")
        return None

    H = np.stack(flow_hists, axis=0)  # (F-1, bins)
    times = np.asarray(flow_times, dtype=np.float32)  # (F-1,)
    T_total = float(times[-1]) if times.size > 0 else 0.0

    # Slide windows over time
    feats: List[np.ndarray] = []
    start = 0.0
    eps = 1e-6
    # Ensure at least one clip even for very short videos
    if T_total < eps:
        feats.append(np.zeros((bins,), dtype=np.float32))
    else:
        while start <= T_total + eps:  # include tail window
            end = start + float(clip_duration)
            mask = (times >= start) & (times < end)
            if np.any(mask):
                v = H[mask].sum(axis=0)
                s = float(v.sum())
                if s > 0:
                    v = v / s  # L1 normalize
                feats.append(v.astype(np.float32))
            else:
                feats.append(np.zeros((bins,), dtype=np.float32))
            start += float(clip_stride)

    X = np.stack(feats, axis=0)
    return X


if __name__ == "__main__":
    # quick CLI for a single video
    import argparse
    parser = argparse.ArgumentParser("Extract HOF features for one video")
    parser.add_argument("video", type=str)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--clip-duration", type=float, default=2.0)
    parser.add_argument("--clip-stride", type=float, default=0.4)
    parser.add_argument("--bins", type=int, default=16)
    args = parser.parse_args()

    X = extract_hof_features_for_video(Path(args.video), args.clip_duration, args.clip_stride, args.bins)
    if X is None:
        raise SystemExit(2)
    print(f"[HOF] features shape: {X.shape}")
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.save(outp, X)
        print(f"[HOF] saved to {outp}")

