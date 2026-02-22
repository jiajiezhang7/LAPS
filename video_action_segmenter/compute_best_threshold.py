#!/usr/bin/env python3
"""
Compute the best threshold for quantized/token_diff_l2_mean energy using
velocity/token_diff_l2_mean as the reference for labeling.

Usage (defaults assume files under energy_sweep_out/):
  python -m video_action_segmenter.compute_best_threshold \
    --quantized-jsonl ./video_action_segmenter/energy_sweep_out/D02_20250811064933/d02_stream_energy_quantized_token_diff_l2_mean.jsonl \
    --velocity-jsonl  ./video_action_segmenter/energy_sweep_out/D02_20250811064933/d02_stream_energy_velocity_token_diff_l2_mean.jsonl \
    --label-threshold auto \
    --output-json ./video_action_segmenter/energy_sweep_report/d02_best_threshold_quantized_token_diff.json

It prints and saves two recommended thresholds:
- Best Youden's J (TPR - FPR) threshold
- Best F1 threshold
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from video_action_segmenter.stream_utils import apply_smoothing_1d


def load_energy_jsonl(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    wins, vals = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            w = o.get('window')
            v = o.get('energy')
            if w is None or v is None:
                continue
            wins.append(int(w))
            vals.append(float(v))
    wins = np.asarray(wins, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    if wins.size == 0:
        return wins, vals
    idx = np.argsort(wins)
    return wins[idx], vals[idx]


def align_by_windows(w1: np.ndarray, x1: np.ndarray, w2: np.ndarray, x2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if w1.size == 0 or w2.size == 0:
        return np.array([]), np.array([])
    common = np.intersect1d(w1, w2)
    if common.size == 0:
        return np.array([]), np.array([])
    i1 = np.searchsorted(w1, common)
    i2 = np.searchsorted(w2, common)
    return x1[i1], x2[i2]


def otsu_threshold(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float('nan')
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return vmin
    hist, edges = np.histogram(x, bins=256, range=(vmin, vmax))
    prob = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(prob)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    return float(centers[k])


def gen_label_threshold(v: np.ndarray, spec: str) -> float:
    if spec == 'auto':
        return otsu_threshold(v)
    if spec.startswith('quantile:'):
        q = float(spec.split(':', 1)[1])
        return float(np.quantile(v, q))
    if spec.startswith('value:'):
        return float(spec.split(':', 1)[1])
    return otsu_threshold(v)


def metrics_at_thr(x: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (x > thr)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    prec = tp / (tp + fp + 1e-12)
    rec = tpr
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    acc = (tp + tn) / (len(y) + 1e-12)
    return dict(thr=float(thr), tp=tp, tn=tn, fp=fp, fn=fn, tpr=float(tpr), fpr=float(fpr), precision=float(prec), recall=float(rec), f1=float(f1), acc=float(acc))


def search_best_thresholds(x: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
    if len(x) == 0 or len(y) == 0:
        return {'best_j': {}, 'best_f1': {}}
    # Use sorted unique values and their midpoints for robust search
    xs = np.unique(x)
    if xs.size > 1:
        mids = (xs[:-1] + xs[1:]) / 2.0
        cands = np.concatenate([xs, mids])
    else:
        cands = xs
    best_j = (-1.0, None)
    best_f1 = (-1.0, None)
    best_j_m = None
    best_f1_m = None
    for thr in cands:
        m = metrics_at_thr(x, y, float(thr))
        J = m['tpr'] - m['fpr']
        if J > best_j[0]:
            best_j = (J, float(thr))
            best_j_m = m
        if m['f1'] > best_f1[0]:
            best_f1 = (m['f1'], float(thr))
            best_f1_m = m
    return {'best_j': {'J': best_j[0], **best_j_m}, 'best_f1': {'F1': best_f1[0], **best_f1_m}}


def main():
    ap = argparse.ArgumentParser(description='Compute best thresholds for quantized/token_diff using velocity/token_diff as labels')
    dflt_root = Path(__file__).with_name('energy_sweep_out')
    ap.add_argument('--quantized-jsonl', type=str, default=str(dflt_root / 'stream_energy_quantized_token_diff_l2_mean.jsonl'))
    ap.add_argument('--velocity-jsonl', type=str, default=str(dflt_root / 'stream_energy_velocity_token_diff_l2_mean.jsonl'))
    ap.add_argument('--label-threshold', type=str, default='auto', help='auto | quantile:0.7 | value:0.02')
    ap.add_argument('--output-json', type=str, default=str(Path(__file__).with_name('energy_sweep_report') / 'best_threshold_quantized_token_diff.json'))
    # Optional smoothing (applied to the target series x before searching thresholds)
    ap.add_argument('--smooth', action='store_true', help='Enable causal smoothing on target series before threshold search')
    ap.add_argument('--smooth-method', type=str, default='ema', choices=['ema', 'ma'])
    ap.add_argument('--smooth-alpha', type=float, default=0.4)
    ap.add_argument('--smooth-window', type=int, default=3)
    args = ap.parse_args()

    q_path = Path(args.quantized_jsonl).resolve()
    v_path = Path(args.velocity_jsonl).resolve()

    wq, x = load_energy_jsonl(q_path)
    wv, v = load_energy_jsonl(v_path)

    x, v = align_by_windows(wq, x, wv, v)
    if x.size == 0:
        print('[ERR] No aligned windows between quantized and velocity files.', flush=True)
        return

    # Optional smoothing on target series (quantized) prior to threshold search
    if args.smooth:
        try:
            x = apply_smoothing_1d(x, method=args.smooth_method, alpha=args.smooth_alpha, window=args.smooth_window).astype(np.float64)
        except Exception:
            pass

    thr_v = gen_label_threshold(v, args.label_threshold)
    y = (v > thr_v).astype(np.int64)

    res = search_best_thresholds(x, y)

    out = {
        'label_spec': args.label_threshold,
        'label_threshold_on_velocity_token_diff': float(thr_v),
        'n_samples': int(x.size),
        'n_pos': int(y.sum()),
        'n_neg': int((y == 0).sum()),
        'quantized_token_diff_best': res,
        'quantized_stats': {
            'min': float(np.min(x)), 'max': float(np.max(x)), 'mean': float(np.mean(x)),
            'q25': float(np.quantile(x, 0.25)), 'q50': float(np.quantile(x, 0.50)), 'q75': float(np.quantile(x, 0.75)), 'q90': float(np.quantile(x, 0.90)),
        },
        'smoothing': {
            'enabled': bool(args.smooth),
            'method': str(args.smooth_method),
            'alpha': float(args.smooth_alpha),
            'window': int(args.smooth_window),
            'applied_to': 'quantized_series_only'
        }
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print('Baseline velocity/token_diff threshold (spec={}): {:.8f}'.format(args.label_threshold, thr_v), flush=True)
    print('Aligned samples = {}, pos = {}, neg = {}'.format(out['n_samples'], out['n_pos'], out['n_neg']), flush=True)
    print('\nBest Youden\'s J threshold:', flush=True)
    print(json.dumps(res['best_j'], indent=2), flush=True)
    print('\nBest F1 threshold:', flush=True)
    print(json.dumps(res['best_f1'], indent=2), flush=True)
    print('\nSaved:', str(out_path), flush=True)


if __name__ == '__main__':
    main()
