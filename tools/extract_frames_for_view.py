#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract frames from raw mp4 videos for a given VIEW into OTAS frames layout.
- Input raw dir (e.g., datasets/gt_raw_videos/D02)
- Output frames dir (e.g., comapred_algorithm/OTAS/data/breakfast/frames)
- Each video D02_sample_1_seg001.mp4 -> frames/D02_D02_sample_1_seg001/Frame_%06d.jpg

Usage:
  conda run -n laps python tools/extract_frames_for_view.py \
    --raw-dir datasets/gt_raw_videos/D02 \
    --view D02 \
    --out-dir comapred_algorithm/OTAS/data/breakfast/frames
"""

import argparse
import os
from pathlib import Path
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', required=True)
    p.add_argument('--view', required=True, help='D01 or D02')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing frames if present')
    return p.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = sorted(raw_dir.glob('*.mp4'))
    if not vids:
        print(f'[ERR] No mp4 files found under {raw_dir}')
        return 1

    for v in vids:
        stem = v.stem  # e.g., D02_sample_1_seg001
        parts = stem.split('_')
        if parts and parts[0] in ('D01', 'D02'):
            act = '_'.join(parts[1:])
        else:
            act = stem
        dir_name = f'{args.view}_{args.view}_{act}'
        dest = out_dir / dir_name
        dest.mkdir(parents=True, exist_ok=True)

        # Skip if frames already exist (unless overwrite)
        first_frame = dest / 'Frame_000001.jpg'
        if first_frame.exists() and not args.overwrite:
            print(f'[SKIP] {v.name} -> {dest} (exists)')
            continue

        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f'[WARN] Cannot open {v}')
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        print(f'[EXTRACT] {v.name} -> {dest} (frames~{total}, fps={fps:.2f})')
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            out_path = dest / f'Frame_{n:06d}.jpg'
            cv2.imwrite(str(out_path), frame)
            if n % 300 == 0:
                print(f'  wrote {n}/{total} frames...')
        cap.release()
        print(f'[DONE] {v.name}: wrote {n} frames')

    print('[OK] Extraction completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

