#!/usr/bin/env python3
"""
Aggregate per-video best thresholds into a view-level threshold JSON.

For each video_name that has both:
  - quantized JSONL at {quantized_root}/{video_name}/stream_energy_quantized_token_diff_l2_mean.jsonl
  - velocity  JSONL at {velocity_root}/{video_name}/stream_energy_velocity_token_diff_l2_mean.jsonl

We compute per-video best thresholds (best_j and best_f1) using the same
logic as compute_best_threshold.py, then aggregate across videos using a
robust statistic (median by default).

Usage:
  python -m video_action_segmenter.aggregate_thresholds \
    --view D01 \
    --quantized-root ./datasets/output/segmentation_outputs/D01 \
    --velocity-root  ./datasets/output/energy_sweep_out/D01 \
    --label-threshold auto \
    --output-json    ./datasets/output/energy_sweep_report/D01/best_threshold_quantized_token_diff.json

Notes:
- This script aligns windows by their integer index present in each JSONL.
- Optional smoothing can be applied on the target (quantized) series prior to threshold search.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

from video_action_segmenter.compute_best_threshold import (
    load_energy_jsonl,
    align_by_windows,
    gen_label_threshold,
    search_best_thresholds,
)
try:
    from video_action_segmenter.stream_utils import apply_smoothing_1d
except Exception:
    def apply_smoothing_1d(x, method='ema', alpha=0.4, window=3):
        return x


def find_video_names(root: Path) -> List[str]:
    names = []
    if root.exists() and root.is_dir():
        for p in sorted(root.iterdir()):
            if p.is_dir():
                names.append(p.name)
    return names


def load_pair(quant_path: Path, vel_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    wq, x = load_energy_jsonl(quant_path)
    wv, v = load_energy_jsonl(vel_path)
    x, v = align_by_windows(wq, x, wv, v)
    return x, v


def main():
    ap = argparse.ArgumentParser(description='Aggregate per-video thresholds into a view-level threshold JSON')
    ap.add_argument('--view', type=str, required=True, help='D01 or D02')
    ap.add_argument('--quantized-root', type=str, required=True)
    ap.add_argument('--velocity-root', type=str, required=True)
    ap.add_argument('--label-threshold', type=str, default='auto', help='auto | quantile:0.7 | value:0.02')
    ap.add_argument('--output-json', type=str, required=True)
    ap.add_argument('--smooth', action='store_true', help='Apply smoothing to quantized series before threshold search')
    ap.add_argument('--smooth-method', type=str, default='ema', choices=['ema', 'ma'])
    ap.add_argument('--smooth-alpha', type=float, default=0.4)
    ap.add_argument('--smooth-window', type=int, default=3)
    ap.add_argument('--aggregate', type=str, default='median', choices=['median','mean'])
    args = ap.parse_args()

    q_root = Path(args.quantized_root).resolve()
    v_root = Path(args.velocity_root).resolve()

    q_names = set(find_video_names(q_root))
    v_names = set(find_video_names(v_root))
    names = sorted(q_names & v_names)

    details: Dict[str, Dict[str, float]] = {}
    thr_best_j: List[float] = []
    thr_best_f1: List[float] = []

    for name in names:
        qp = q_root / name / 'stream_energy_quantized_token_diff_l2_mean.jsonl'
        vp = v_root / name / 'stream_energy_velocity_token_diff_l2_mean.jsonl'
        if not (qp.exists() and vp.exists()):
            continue
        x, v = load_pair(qp, vp)
        if x.size == 0:
            continue
        if args.smooth:
            try:
                x = apply_smoothing_1d(x, method=args.smooth_method, alpha=args.smooth_alpha, window=args.smooth_window).astype(np.float64)
            except Exception:
                pass
        thr_v = gen_label_threshold(v, args.label_threshold)
        y = (v > thr_v).astype(np.int64)
        res = search_best_thresholds(x, y)
        # record per-video
        bj_thr = float(res.get('best_j', {}).get('thr', np.nan))
        bf_thr = float(res.get('best_f1', {}).get('thr', np.nan))
        if np.isfinite(bj_thr): thr_best_j.append(bj_thr)
        if np.isfinite(bf_thr): thr_best_f1.append(bf_thr)
        details[name] = {
            'label_threshold_on_velocity': float(thr_v),
            'best_j_threshold_on_quantized': bj_thr,
            'best_f1_threshold_on_quantized': bf_thr,
        }

    if not thr_best_j and not thr_best_f1:
        out = {
            'view': args.view,
            'label_spec': args.label_threshold,
            'n_videos_considered': len(names),
            'message': 'No valid per-video threshold could be computed.'
        }
    else:
        if args.aggregate == 'median':
            agg_j = float(np.median(thr_best_j)) if thr_best_j else float('nan')
            agg_f1 = float(np.median(thr_best_f1)) if thr_best_f1 else float('nan')
        else:
            agg_j = float(np.mean(thr_best_j)) if thr_best_j else float('nan')
            agg_f1 = float(np.mean(thr_best_f1)) if thr_best_f1 else float('nan')
        out = {
            'view': args.view,
            'label_spec': args.label_threshold,
            'aggregate_method': args.aggregate,
            'n_videos_considered': len(names),
            'per_video': details,
            'quantized_token_diff_best': {
                'best_j': {
                    'thr': agg_j,
                    'source': 'aggregate_over_videos'
                },
                'best_f1': {
                    'thr': agg_f1,
                    'source': 'aggregate_over_videos'
                }
            }
        }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('Saved aggregated threshold JSON:', str(out_path), flush=True)


if __name__ == '__main__':
    main()

