import argparse
import json
import os
import shutil
from typing import List, Tuple, Dict
import numpy as np

# Offline segmentation from energy JSONL using LAPS-like state machine
# - Input: per-video energy JSONL, threshold JSON from threshold_search_with_gt.py
# - Output: {output_root}/{stem}/segmented_videos/{stem}_segments.json
#           and copy stream_energy_*.jsonl into {output_root}/{stem}/


def load_energy_jsonl(path: str) -> List[float]:
    if not os.path.exists(path):
        return []
    items = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                w = int(obj.get("window"))
                e = float(obj.get("energy"))
                items.append((w, e))
            except Exception:
                continue
    items.sort(key=lambda x: x[0])
    return [float(e) for (_w, e) in items]


def aggregate_stride(values: List[float], stride: int) -> List[float]:
    if stride <= 1:
        return list(values)
    out = []
    n = len(values)
    i = 0
    while i < n:
        j = min(n, i + stride)
        if j > i:
            out.append(float(np.mean(values[i:j])))
        i = j
    return out


def segment_series(series: List[float], thr_on: float, hysteresis_ratio: float, up_count: int, down_count: int, cooldown_windows: int, max_duration_windows: int) -> List[Tuple[int, int]]:
    thr_off = float(thr_on) * float(hysteresis_ratio)
    segs = []
    active = False
    pos_run = 0
    neg_run = 0
    cooldown = 0
    start_idx = -1
    length = 0
    for i, e in enumerate(series):
        if cooldown > 0:
            cooldown -= 1
        if not active:
            if e >= thr_on:
                pos_run += 1
            else:
                pos_run = 0
            if cooldown == 0 and pos_run >= max(1, int(up_count)):
                active = True
                start_idx = i
                length = 0
                pos_run = 0
                neg_run = 0
        else:
            if e < thr_off:
                neg_run += 1
            else:
                neg_run = 0
            length += 1
            if neg_run >= max(1, int(down_count)):
                end_idx = max(start_idx, i)
                segs.append((start_idx, end_idx))
                active = False
                cooldown = int(cooldown_windows)
                start_idx = -1
                length = 0
                pos_run = 0
                neg_run = 0
            elif max_duration_windows > 0 and length >= int(max_duration_windows):
                end_idx = i
                segs.append((start_idx, end_idx))
                active = False
                cooldown = int(cooldown_windows)
                start_idx = -1
                length = 0
                pos_run = 0
                neg_run = 0
    if active and start_idx >= 0:
        segs.append((start_idx, len(series) - 1))
    return segs


def read_threshold(thr_json_path: str) -> float:
    with open(thr_json_path, 'r') as f:
        d = json.load(f)
    # Preferred location
    try:
        return float(d['optical_flow_mag_mean_best']['best_f1']['thr'])
    except Exception:
        pass
    # Fallbacks
    for k in ('best', 'best_f1', 'quantized_token_diff_best'):
        try:
            v = d[k]
            if isinstance(v, dict) and 'thr' in v:
                return float(v['thr'])
            if isinstance(v, dict) and 'best_f1' in v and 'thr' in v['best_f1']:
                return float(v['best_f1']['thr'])
        except Exception:
            continue
    # Last resort: top-level thr
    if 'thr' in d:
        return float(d['thr'])
    raise KeyError('Cannot find threshold value in JSON')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', required=True, choices=['D01', 'D02'])
    ap.add_argument('--energy-root', required=True)
    ap.add_argument('--threshold-json', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--source', default='optical_flow')
    ap.add_argument('--mode', default='mag_mean')
    ap.add_argument('--target-fps', type=float, default=10.0)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--hysteresis-ratio', type=float, default=0.95)
    ap.add_argument('--up-count', type=int, default=2)
    ap.add_argument('--down-count', type=int, default=2)
    ap.add_argument('--cooldown-windows', type=int, default=1)
    ap.add_argument('--max-duration-seconds', type=float, default=2.0)
    args = ap.parse_args()

    thr_on = read_threshold(args.threshold_json)
    dt = float(args.stride) / float(args.target_fps) if args.stride > 0 else 1.0 / float(args.target_fps)
    max_dur_w = int(round(float(args.max_duration_seconds) / dt)) if args.max_duration_seconds > 0 else 0

    os.makedirs(args.output_root, exist_ok=True)

    # iterate videos under energy root
    for stem in sorted(os.listdir(args.energy_root)):
        vid_dir = os.path.join(args.energy_root, stem)
        if not os.path.isdir(vid_dir):
            continue
        energy_name = f'stream_energy_{args.source}_{args.mode}.jsonl'
        energy_path = os.path.join(vid_dir, energy_name)
        if not os.path.exists(energy_path):
            continue
        e = load_energy_jsonl(energy_path)
        if not e:
            continue
        ew = aggregate_stride(e, int(args.stride))

        seg_idx = segment_series(
            ew,
            thr_on=float(thr_on),
            hysteresis_ratio=float(args.hysteresis_ratio),
            up_count=int(args.up_count),
            down_count=int(args.down_count),
            cooldown_windows=int(args.cooldown_windows),
            max_duration_windows=max_dur_w,
        )

        # map to seconds
        segments = [ {'start_sec': i0 * dt, 'end_sec': (i1 + 1) * dt, 'label': f'segment_{k+1}'} for k, (i0, i1) in enumerate(seg_idx) ]
        video_duration_sec = float(len(e)) / float(args.target_fps)

        # write outputs
        out_dir = os.path.join(args.output_root, stem)
        seg_dir = os.path.join(out_dir, 'segmented_videos')
        os.makedirs(seg_dir, exist_ok=True)
        out_json = os.path.join(seg_dir, f'{stem}_segments.json')
        meta = {
            'video': f'{stem}.mp4',
            'video_duration_sec': float(video_duration_sec),
            'segments': segments,
            'segmentation_params': {
                'target_fps': float(args.target_fps),
                'stride': int(args.stride),
                'hysteresis_ratio': float(args.hysteresis_ratio),
                'up_count': int(args.up_count),
                'down_count': int(args.down_count),
                'cooldown_windows': int(args.cooldown_windows),
                'max_duration_seconds': float(args.max_duration_seconds),
                'threshold_on': float(thr_on),
                'threshold_off': float(thr_on) * float(args.hysteresis_ratio),
                'source': args.source,
                'mode': args.mode,
            }
        }
        with open(out_json, 'w') as f:
            json.dump(meta, f, indent=2)

        # copy energy file for confidence computation
        try:
            shutil.copy2(energy_path, os.path.join(out_dir, energy_name))
        except Exception:
            pass

        print(f'[Seg] Wrote {out_json} with {len(segments)} segments')


if __name__ == '__main__':
    main()

