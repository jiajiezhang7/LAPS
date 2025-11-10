#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline segmentation from pre-computed energy JSONL files.

- Reads per-video energy JSONL (e.g., stream_energy_quantized_token_diff_l2_mean.jsonl or energy_split1)
- Applies causal smoothing (optional) and hysteresis-based state machine segmentation
- Writes prediction JSON under: {pred_root}/{stem}/segmented_videos/{stem}_segments.json
- Optionally creates a symlink to the energy JSONL under the same pred folder for mAP confidence

Example:
  conda run -n laps python tools/offline_segment_from_energy.py \
    --energy-root /home/johnny/action_ws/output/gtea \
    --pred-root /home/johnny/action_ws/output/gtea/segments_train_split1 \
    --threshold-json /home/johnny/action_ws/output/gtea/thresholds/split1/best_threshold_quantized_token_diff.json \
    --threshold-key optical_flow_mag_mean_best.best_f1.thr \
    --target-fps 10 --stride 4 --hysteresis-ratio 0.95 --up-count 2 --down-count 2 --cooldown-windows 1 \
    --max-duration-seconds 2.0 --stem-prefixes S2_ S3_ S4_ --use-smoothing --smooth-method ema --smooth-alpha 0.7 --smooth-window 3
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone

import numpy as np
import sys
# Ensure repository root on sys.path so that `video_action_segmenter` can be imported when running as a script
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
except Exception:
    pass


# Reuse smoothing util from package
from video_action_segmenter.stream_utils import apply_smoothing_1d


def read_energy_series(path: Path) -> Tuple[List[int], List[float]]:
    """Read JSONL energy: returns (windows_sorted, energies_sorted)."""
    items: List[Tuple[int, float]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                w = int(obj.get('window'))
                e = float(obj.get('energy'))
                items.append((w, e))
            except Exception:
                continue
    items.sort(key=lambda x: x[0])
    if not items:
        return [], []
    ws = [int(w) for (w, _e) in items]
    es = [float(e) for (_w, e) in items]
    return ws, es


def find_energy_file(stem_dir: Path) -> Optional[Path]:
    """Find energy file within {energy_root}/{stem}/.
    Priority:
      - stream_energy_quantized_token_diff_l2_mean.jsonl
      - energy_split1 (jsonl without extension)
      - any stream_energy_*.jsonl (first match)
    """
    cand = [
        stem_dir / 'stream_energy_quantized_token_diff_l2_mean.jsonl',
        stem_dir / 'energy_split1',
    ]
    for p in cand:
        if p.exists():
            return p
    # fallback
    for p in sorted(stem_dir.glob('stream_energy_*.jsonl')):
        return p
    return None


def segment_series(series: List[float], thr_on: float, hysteresis_ratio: float, up_count: int, down_count: int,
                   cooldown_windows: int, max_duration_windows: int) -> List[Tuple[int, int]]:
    """State machine segmentation over 1D series; returns inclusive index pairs (start_idx, end_idx)."""
    thr_off = float(thr_on) * float(hysteresis_ratio)
    segs: List[Tuple[int, int]] = []
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


def load_threshold(thr_json: Path, key: str, fallback: float) -> float:
    try:
        with thr_json.open('r', encoding='utf-8') as f:
            rep = json.load(f)
        cur = rep
        for k in key.split('.'):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                cur = None
                break
        if isinstance(cur, (int, float)):
            return float(cur)
    except Exception:
        pass
    return float(fallback)


def ensure_symlink(src: Path, dst: Path):
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            try:
                if dst.resolve() == src.resolve():
                    return
            except Exception:
                pass
            try:
                dst.unlink()
            except Exception:
                pass
        os.symlink(src, dst)
    except Exception:
        # best-effort, ignore
        pass


def main():
    ap = argparse.ArgumentParser(description='Offline segmentation from energy JSONL')
    ap.add_argument('--energy-root', type=str, required=True)
    ap.add_argument('--pred-root', type=str, required=True)
    ap.add_argument('--threshold-json', type=str, required=True)
    ap.add_argument('--threshold-key', type=str, default='optical_flow_mag_mean_best.best_f1.thr')
    ap.add_argument('--target-fps', type=float, default=10.0)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--hysteresis-ratio', type=float, default=0.95)
    ap.add_argument('--up-count', type=int, default=2)
    ap.add_argument('--down-count', type=int, default=2)
    ap.add_argument('--cooldown-windows', type=int, default=1)
    ap.add_argument('--max-duration-seconds', type=float, default=2.0)
    ap.add_argument('--min-len-windows', type=int, default=2)
    ap.add_argument('--segments-json-suffix', type=str, default='_segments')
    ap.add_argument('--stem-prefixes', nargs='+', type=str, default=['S2_', 'S3_', 'S4_'])
    ap.add_argument('--use-smoothing', action='store_true')
    ap.add_argument('--smooth-method', type=str, default='ema', choices=['ema', 'ma'])
    ap.add_argument('--smooth-alpha', type=float, default=0.7)
    ap.add_argument('--smooth-window', type=int, default=3)
    args = ap.parse_args()

    energy_root = Path(args.energy_root).resolve()
    pred_root = Path(args.pred_root).resolve()
    thr_json = Path(args.threshold_json).resolve()

    dt = float(args.stride) / float(args.target_fps) if args.target_fps > 0 else 0.4
    max_dur_win = int(round(float(args.max_duration_seconds) / dt)) if args.max_duration_seconds > 0 else 0
    min_save_windows = max(int(args.min_len_windows), 3)

    thr = load_threshold(thr_json, args.threshold_key, fallback=0.0)
    print(f"[OfflineSeg] threshold = {thr} (from {thr_json}, key='{args.threshold_key}')")
    print(f"[OfflineSeg] dt={dt:.3f}s stride={args.stride} target_fps={args.target_fps} max_duration_windows={max_dur_win}")

    # list stems
    stems: List[str] = []
    if energy_root.exists() and energy_root.is_dir():
        for name in sorted(os.listdir(energy_root)):
            if not any(name.startswith(pfx) for pfx in args.stem_prefixes):
                continue
            d = energy_root / name
            if not d.is_dir():
                continue
            p = find_energy_file(d)
            if p is not None:
                stems.append(name)
    if not stems:
        raise RuntimeError(f"No stems found under {energy_root} with prefixes {args.stem_prefixes}")

    total = 0
    ok = 0
    skipped = 0

    for stem in stems:
        total += 1
        stem_dir = energy_root / stem
        efile = find_energy_file(stem_dir)
        if efile is None:
            print(f"[OfflineSeg][WARN] energy not found for {stem}")
            continue
        try:
            wins, energies = read_energy_series(efile)
        except Exception as e:
            print(f"[OfflineSeg][WARN] read energy failed for {stem}: {e}")
            continue
        if len(energies) == 0:
            print(f"[OfflineSeg][WARN] empty energy for {stem}")
            continue

        # series aligned by observed windows (assume consecutive or near-consecutive)
        series = np.asarray(energies, dtype=np.float32)
        if args.use_smoothing:
            try:
                series = apply_smoothing_1d(series, method=args.smooth_method, alpha=args.smooth_alpha, window=args.smooth_window)
            except Exception:
                pass
        series = series.tolist()

        idx_pairs = segment_series(series, thr_on=float(thr), hysteresis_ratio=float(args.hysteresis_ratio),
                                   up_count=int(args.up_count), down_count=int(args.down_count),
                                   cooldown_windows=int(args.cooldown_windows), max_duration_windows=int(max_dur_win))
        # filter by min_save_windows
        idx_pairs = [p for p in idx_pairs if (p[1] - p[0] + 1) >= int(min_save_windows)]

        # map to seconds using dt and (i1+1)*dt for end
        segs_json = []
        for k, (i0, i1) in enumerate(idx_pairs):
            start_s = float(i0) * float(dt)
            end_s = float(i1 + 1) * float(dt)
            if end_s < start_s:
                end_s = start_s
            segs_json.append({
                "start_sec": float(start_s),
                "end_sec": float(end_s),
                "label": f"segment_{k}"
            })

        # write prediction JSON
        out_dir = pred_root / stem / 'segmented_videos'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}{args.segments_json_suffix}.json"
        meta = {
            "video": f"{stem}.mp4",
            "segments": segs_json,
            "video_duration_sec": float(len(series)) * float(dt),
            "fps": None,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "segmentation_params": {
                "threshold": float(thr),
                "mode": "report",
                "hysteresis_ratio": float(args.hysteresis_ratio),
                "up_count": int(args.up_count),
                "down_count": int(args.down_count),
                "cooldown_windows": int(args.cooldown_windows),
                "max_duration_seconds": float(args.max_duration_seconds),
                "min_len_windows": int(args.min_len_windows),
                "stride": int(args.stride),
                "target_fps": float(args.target_fps),
                "dt": float(dt),
                "orig_fps": None,
            },
        }
        try:
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            ok += 1
            print(f"[OfflineSeg] Written {out_path} (n_segments={len(segs_json)})")
        except Exception as e:
            print(f"[OfflineSeg][WARN] write failed for {stem}: {e}")
            continue

        # also drop a symlink to energy jsonl into pred folder to enable mAP confidence
        # prefer a canonical name: stream_energy_quantized_token_diff_l2_mean.jsonl
        try:
            energy_target = out_dir.parent / 'stream_energy_quantized_token_diff_l2_mean.jsonl'
            ensure_symlink(efile, energy_target)
        except Exception:
            pass

    print(f"[OfflineSeg] Done. stems={len(stems)} total={total} ok={ok} skipped={skipped}")


if __name__ == '__main__':
    main()

