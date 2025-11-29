#!/usr/bin/env python3
"""
Convert Breakfast frame-level groundTruth (.txt) to segment-level JSON with seconds.
- Videos are 15 fps (BreakfastII_15fps_qvga_sync). We try to read actual fps and frame count via OpenCV; if unavailable, fallback to fps=15 and frame_count=len(labels).
- Input split file lines look like: P03_cam01_P03_cereals.txt
- Raw videos (symlinked) are expected under:
    ./online_datasets/breakfast/breakfast/Videos_test.split1/{stem}.avi
- GroundTruth files under:
    ./online_datasets/breakfast/breakfast/groundTruth/{stem}.txt
- Output JSON per video to:
    ./online_datasets/breakfast/gt_segments_json/test.split1/{stem}_segments.json
Output JSON schema (class-agnostic):
{
  "video": "<stem>.avi",
  "video_duration_sec": <float>,
  "segments": [
    {"start_sec": <float>, "end_sec": <float>}, ...
  ]
}
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import json
from typing import List, Tuple

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # fallback later

ROOT = Path('.').resolve()  # Change to your workspace root
VID_DIR_DEFAULT = ROOT / 'online_datasets/breakfast/breakfast/Videos_test.split1'
GT_DIR_DEFAULT = ROOT / 'online_datasets/breakfast/breakfast/groundTruth'
SPLIT_DEFAULT = ROOT / 'online_datasets/breakfast/breakfast/splits/test.split1.bundle'
OUT_DEFAULT = ROOT / 'online_datasets/breakfast/gt_segments_json/test.split1'

def read_lines(p: Path) -> List[str]:
    with open(p, 'r', encoding='utf-8') as f:
        return [ln.strip().rstrip('\r') for ln in f if ln.strip()]


def labels_to_segments(labels: List[str], fps: float) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    if not labels:
        return segs
    cur = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            segs.append((start / fps, i / fps))
            cur = labels[i]
            start = i
    segs.append((start / fps, len(labels) / fps))
    # ensure strictly increasing and non-negative
    clean = []
    for s, e in segs:
        s = max(0.0, float(s))
        e = max(s, float(e))
        clean.append((s, e))
    return clean


def get_video_meta(video_path: Path, num_labels: int) -> Tuple[float, int, float]:
    # returns: (fps, frame_count, duration_sec)
    if cv2 is not None:
        try:
            cap = cv2.VideoCapture(str(video_path))
            ok = cap.isOpened()
            if ok:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                if fps > 0.0 and frame_count > 0:
                    return fps, frame_count, frame_count / fps
        except Exception:
            pass
    # fallback to 15 fps
    fps = 15.0
    frame_count = int(num_labels)
    return fps, frame_count, frame_count / fps


def process_split(videos_dir: Path, gt_dir: Path, split_file: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = [ln[:-4] if ln.endswith('.txt') else ln for ln in read_lines(split_file)]
    for stem in stems:
        video_path = videos_dir / f'{stem}.avi'
        gt_path = gt_dir / f'{stem}.txt'
        if not gt_path.exists():
            print(f'[WARN] Missing GT: {gt_path}')
            continue
        labels = read_lines(gt_path)
        fps, frame_count, duration = get_video_meta(video_path, len(labels))
        segs = labels_to_segments(labels, fps=fps)
        obj = {
            'video': f'{stem}.avi',
            'video_duration_sec': float(duration),
            'segments': [
                {'start_sec': float(s), 'end_sec': float(e)} for s, e in segs
            ],
        }
        out_path = out_dir / f'{stem}_segments.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f)
        print(f'[OK] {out_path} n_segments={len(segs)} fps={fps:.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos-dir', default=str(VID_DIR_DEFAULT))
    ap.add_argument('--gt-dir', default=str(GT_DIR_DEFAULT))
    ap.add_argument('--split-file', default=str(SPLIT_DEFAULT))
    ap.add_argument('--out-dir', default=str(OUT_DEFAULT))
    args = ap.parse_args()
    process_split(Path(args.videos_dir), Path(args.gt_dir), Path(args.split_file), Path(args.out_dir))

if __name__ == '__main__':
    main()

