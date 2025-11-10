#!/usr/bin/env python3
"""
Convert GTEA frame-level labels to segment-level JSON with seconds.

Inputs:
  --videos-dir: directory containing original videos (e.g., .../gtea/Videos)
  --gt-dir: directory containing frame-wise labels (.txt)
  --split-file: a bundle file listing gt filenames (e.g., train.split1.bundle/test.split1.bundle)
  --out-dir: output directory for segment JSON files

Output JSON schema (per video):
{
  "video": "<stem>.mp4",
  "video_duration_sec": <float>,
  "segments": [ {"start_sec": <float>, "end_sec": <float>} , ... ]
}

Notes:
- We compute time using original video fps and frame count via OpenCV.
- If the label length and video frame count mismatch, we clamp to min(len(labels), n_frames).
- Labels are not used further (class-agnostic evaluation), only boundaries matter.
"""
from __future__ import annotations
import argparse
import os
import json
from typing import List, Tuple

import cv2


def read_lines(p: str) -> List[str]:
    with open(p, 'r') as f:
        return [ln.strip() for ln in f.read().splitlines() if ln.strip()]


def get_video_meta(video_path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        fps = 30.0  # sensible fallback
    return fps, n_frames


def labels_to_segments(labels: List[str], n_frames: int, fps: float) -> List[Tuple[float, float]]:
    # compress runs into segments in seconds, using clamped length N
    N = min(len(labels), int(n_frames) if n_frames and n_frames > 0 else len(labels))
    if N <= 0:
        return []
    segs: List[Tuple[float, float]] = []
    s = 0
    for i in range(1, N + 1):
        if i == N or labels[i] != labels[i - 1]:
            segs.append((s / fps, i / fps))
            s = i
    return segs


def process_split(videos_dir: str, gt_dir: str, split_file: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    items = read_lines(split_file)
    for name in items:
        # Lines are like: S1_Coffee_C1.txt
        stem = name[:-4] if name.endswith('.txt') else name
        gt_path = os.path.join(gt_dir, f"{stem}.txt")
        video_path = os.path.join(videos_dir, f"{stem}.mp4")
        if not os.path.exists(gt_path):
            print(f"[WARN] GT not found: {gt_path}, skip")
            continue
        if not os.path.exists(video_path):
            print(f"[WARN] Video not found: {video_path}, skip")
            continue
        labels = read_lines(gt_path)
        fps, n_frames = get_video_meta(video_path)
        duration = (n_frames / fps) if fps > 0 and n_frames > 0 else (len(labels) / max(fps, 1.0))
        segs = labels_to_segments(labels, n_frames, fps)
        out = {
            "video": f"{stem}.mp4",
            "video_duration_sec": float(duration),
            "segments": [ {"start_sec": float(s), "end_sec": float(e)} for (s, e) in segs ],
        }
        out_path = os.path.join(out_dir, f"{stem}_segments.json")
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"[OK] Wrote {out_path} (segments={len(segs)}, fps={fps:.2f}, frames={n_frames})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos-dir', required=True)
    ap.add_argument('--gt-dir', required=True)
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    process_split(args.videos_dir, args.gt_dir, args.split_file, args.out_dir)


if __name__ == '__main__':
    main()

