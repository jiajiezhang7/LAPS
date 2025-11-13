#!/usr/bin/env python3
"""
Sweep theta_on thresholds for hysteresis segmentation on E_action and evaluate vs y_pseudo (unsupervised).
- Energy (E_action): stream_energy_quantized_token_diff_l2_mean.jsonl
- Labels (y_pseudo): from velocity energy via Otsu/quantile/value (no GT)

Outputs a PDF (F1/J-Index vs theta_on) and optional CSV.
"""
import os
import argparse
import json
from typing import List, Tuple, Dict

import math
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np


def list_sample_dirs(sample_roots: List[str]) -> List[str]:
    out = []
    for root in sample_roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                out.append(p)
    return out


def load_energy_jsonl(path: str) -> List[float]:
    energies: Dict[int, float] = {}
    max_idx = -1
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # window index may be 0- or 1-based; if missing, append sequentially
            idx = int(d.get('window', len(energies)))
            # Normalize to 0-based if appears 1-based
            if idx > 0 and len(energies) == 0 and idx == 1:
                # we'll shift later if we detect first index is 1
                pass
            energies[idx] = float(d['energy'])
            if idx > max_idx:
                max_idx = idx
    if not energies:
        return []
    # detect 1-based indexing
    is_one_based = (0 not in energies) and (1 in energies)
    length = max_idx + (0 if not is_one_based else 0) + 1
    arr = [0.0] * length
    for idx, val in energies.items():
        j = idx if not is_one_based else idx  # already contiguous from 1..N; index 0 stays 0.0
        if j < len(arr):
            arr[j] = val
    # If is_one_based, arr[0] may be 0.0 placeholder; drop it
    if is_one_based:
        arr = arr[1:]
    return arr


def gt_path_for_sample(sample_dir: str, gt_dir: str) -> str:
    sample_name = os.path.basename(sample_dir)
    # Parent directory name D01/D02
    dataset = os.path.basename(os.path.dirname(sample_dir))
    return os.path.join(gt_dir, dataset, f"{sample_name}_segments.json")


def load_gt_windows(ann_path: str, n_windows: int) -> Tuple[List[int], float]:
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    target_fps = float(ann.get('segmentation_params', {}).get('target_fps', 10.0))
    y = [0] * n_windows
    for seg in ann.get('segments', []):
        s = float(seg['start_sec'])
        e = float(seg['end_sec'])
        start_idx = int(math.floor(s * target_fps))
        end_idx_excl = int(math.ceil(e * target_fps))
        if end_idx_excl <= start_idx:
            end_idx_excl = start_idx + 1
        start_idx = max(0, start_idx)
        end_idx_excl = min(n_windows, end_idx_excl)
        for i in range(start_idx, end_idx_excl):
            y[i] = 1
    return y, target_fps


def hysteresis_predict(energies: List[float], theta_on: float, hysteresis_ratio: float = 0.95,
                       up_count: int = 2, down_count: int = 2) -> List[int]:
    theta_off = theta_on * hysteresis_ratio
    state = 0
    up = 0
    down = 0
    out = []
    for e in energies:
        if state == 0:
            if e >= theta_on:
                up += 1
            else:
                up = 0
            if up >= up_count:
                state = 1
                down = 0
        else:  # state == 1
            if e < theta_off:
                down += 1
            else:
                down = 0
            if down >= down_count:
                state = 0
                up = 0
        out.append(state)
    return out


def f1_and_jaccard(y_true: List[int], y_pred: List[int]) -> Tuple[float, float]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    j = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return f1, j


def frange(start: float, stop: float, step: float) -> List[float]:
    n = int(round((stop - start) / step))
    vals = [start + i * step for i in range(n + 1)]
    return [float(f"{v:.6f}") for v in vals]


# --- Unsupervised thresholding utilities for y_pseudo generation ---

def otsu_threshold(values, nbins: int = 256) -> float:
    import numpy as _np
    x = _np.asarray(values, dtype=float)
    x = x[_np.isfinite(x)]
    if x.size == 0:
        return 0.0
    vmin = float(x.min())
    vmax = float(x.max())
    if vmin == vmax:
        return vmin
    hist, bin_edges = _np.histogram(x, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(float)
    prob = hist / (hist.sum() + 1e-12)
    omega = _np.cumsum(prob)
    mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mu = _np.cumsum(prob * mids)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(_np.nanargmax(sigma_b2))
    thr = float((bin_edges[idx] + bin_edges[idx + 1]) / 2.0)
    return thr


def resolve_threshold(values, thr_spec: str = 'auto') -> float:
    spec = (thr_spec or 'auto').strip().lower()
    if spec in ('auto', 'otsu'):
        return otsu_threshold(values)
    if spec.startswith('quantile:'):
        try:
            q = float(spec.split(':', 1)[1])
        except Exception:
            q = 0.9
        import numpy as _np
        x = _np.asarray(values, dtype=float)
        x = x[_np.isfinite(x)]
        if x.size == 0:
            return 0.0
        return float(_np.quantile(x, q))
    if spec.startswith('value:'):
        try:
            return float(spec.split(':', 1)[1])
        except Exception:
            return float(values[0]) if values else 0.0
    # default
    return otsu_threshold(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_roots', nargs='+', required=True,
                    help='One or more directories whose subfolders are samples, e.g., datasets/output/energy_sweep_out/D01 D02')
    ap.add_argument('--gt_dir', default='datasets/gt_annotations')  # unused in pseudo-label mode
    ap.add_argument('--energy_file', default='stream_energy_quantized_token_diff_l2_mean.jsonl',
                    help='E_action energy file name relative to each sample dir')
    ap.add_argument('--proxy_file', default='stream_energy_velocity_token_diff_l2_mean.jsonl',
                    help='Proxy (velocity) energy file name relative to each sample dir')
    ap.add_argument('--pseudo_thr', default='auto',
                    help='Threshold spec for y_pseudo: auto|quantile:q|value:v')
    ap.add_argument('--theta_min', type=float, default=0.5)
    ap.add_argument('--theta_max', type=float, default=3.0)
    ap.add_argument('--theta_step', type=float, default=0.1)
    ap.add_argument('--hysteresis_ratio', type=float, default=0.95)
    ap.add_argument('--up_count', type=int, default=2)
    ap.add_argument('--down_count', type=int, default=2)
    ap.add_argument('--outpdf', required=True)
    ap.add_argument('--outcsv', default=None)
    args = ap.parse_args()

    samples = list_sample_dirs(args.sample_roots)
    if not samples:
        raise SystemExit(f"No samples found under: {args.sample_roots}")

    thetas = frange(args.theta_min, args.theta_max, args.theta_step)
    f1s: List[float] = []
    js: List[float] = []

    # Pre-load all sequences (E_action) and pseudo-labels (from velocity) to avoid repeated IO in sweep
    seqs = []
    for sd in samples:
        eact_path = os.path.join(sd, args.energy_file)
        vproxy_path = os.path.join(sd, args.proxy_file)
        if not (os.path.isfile(eact_path) and os.path.isfile(vproxy_path)):
            continue
        try:
            eact = load_energy_jsonl(eact_path)
            vproxy = load_energy_jsonl(vproxy_path)
        except Exception:
            continue
        if not eact or not vproxy:
            continue
        # align by length
        n = min(len(eact), len(vproxy))
        eact = eact[:n]
        vproxy = vproxy[:n]
        # generate y_pseudo by unsupervised threshold on velocity
        thr = resolve_threshold(vproxy, args.pseudo_thr)
        y_pseudo = [1 if v >= thr else 0 for v in vproxy]
        seqs.append((eact, y_pseudo))

    if not seqs:
        raise SystemExit("No valid (E_action, y_pseudo) pairs found.")

    for th in thetas:
        # aggregate across all samples by concatenation
        all_true: List[int] = []
        all_pred: List[int] = []
        for energies, y_true in seqs:
            y_pred = hysteresis_predict(energies, th, args.hysteresis_ratio, args.up_count, args.down_count)
            all_true.extend(y_true)
            all_pred.extend(y_pred)
        f1, j = f1_and_jaccard(all_true, all_pred)
        f1s.append(f1)
        js.append(j)


    # report best theta_on by F1 vs y_pseudo
    if f1s:
        best_idx = max(range(len(f1s)), key=lambda i: f1s[i])
        print(f"Best theta_on (F1 vs y_pseudo): {thetas[best_idx]:.4f}, F1={f1s[best_idx]:.4f}")

    # optional CSV
    if args.outcsv:
        os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
        with open(args.outcsv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['theta_on', 'f1', 'j_index'])
            for t, f1, j in zip(thetas, f1s, js):
                w.writerow([t, f1, j])

    # plot
    os.makedirs(os.path.dirname(args.outpdf), exist_ok=True)
    plt.figure(figsize=(6.0, 4.0), dpi=160)
    plt.plot(thetas, f1s, label='F1', color='#1f77b4')
    plt.plot(thetas, js, label='J-Index', color='#ff7f0e')
    plt.xlabel('theta_on')
    plt.ylabel('Score')
    # plt.title('E_action vs y_pseudo: F1/J vs theta_on')  # disabled per request
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outpdf)
    print(f"Saved {args.outpdf}")


if __name__ == '__main__':
    main()

