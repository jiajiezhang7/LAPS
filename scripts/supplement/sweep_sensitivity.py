#!/usr/bin/env python3
"""
Figure S5: Sensitivity sweep for hysteresis/debounce parameters.
- Fix theta_on to the best value from a previous sweep CSV.
- Sweep hysteresis_ratio in {0.7..1.0}, up_count in {1..5}, down_count in {1..5}.
- Compute F1 and J-Index per dataset (e.g., D01, D02), then macro-average to avoid mixing.
- Save a CSV of results and two single-page PDFs (F1, J-Index heatmaps separately).
"""
import os
import argparse
import json
import math
import csv
from typing import List, Tuple, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



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
            idx = int(d.get('window', len(energies)))
            energies[idx] = float(d['energy'])
            if idx > max_idx:
                max_idx = idx
    if not energies:
        return []
    is_one_based = (0 not in energies) and (1 in energies)
    length = max_idx + 1
    arr = [0.0] * length
    for idx, val in energies.items():
        j = idx
        if j < len(arr):
            arr[j] = val
    if is_one_based:
        arr = arr[1:]
    return arr


def gt_path_for_sample(sample_dir: str, gt_dir: str) -> str:
    sample_name = os.path.basename(sample_dir)
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
        else:
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


def load_best_theta_from_csv(theta_csv: str) -> float:
    best_f1 = -1.0
    best_theta = None
    with open(theta_csv, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            th = float(row['theta_on'])
            f1 = float(row['f1'])
            if f1 > best_f1:
                best_f1 = f1
                best_theta = th
    if best_theta is None:
        raise RuntimeError(f"No theta found in {theta_csv}")
    return best_theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_roots', nargs='+', required=True,
                    help='Directories whose subfolders are samples (e.g., .../D01 .../D02)')
    ap.add_argument('--gt_dir', default='datasets/gt_annotations')
    ap.add_argument('--energy_file', default='stream_energy_optical_flow_mag_mean.jsonl')
    ap.add_argument('--theta_csv', default='supplement_output/segmentor/fig_S4_theta_sweep.csv')
    ap.add_argument('--theta_on', type=float, default=None, help='Override theta_on; if None, read from theta_csv')
    ap.add_argument('--hysteresis_ratio', nargs='*', type=float,
                    default=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    ap.add_argument('--up_count', nargs='*', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--down_count', nargs='*', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--outpdf_f1', required=True)
    ap.add_argument('--outpdf_j', required=True)
    ap.add_argument('--outcsv', required=True)
    args = ap.parse_args()

    samples = list_sample_dirs(args.sample_roots)
    if not samples:
        raise SystemExit(f"No samples found under: {args.sample_roots}")

    theta_on = args.theta_on if args.theta_on is not None else load_best_theta_from_csv(args.theta_csv)
    print(f"Using theta_on={theta_on}")

    # Pre-load sequences, grouped by dataset (e.g., D01, D02)
    seqs_by_ds: Dict[str, List[Tuple[List[float], List[int]]]] = {}
    for sd in samples:
        energy_path = os.path.join(sd, args.energy_file)
        if not os.path.isfile(energy_path):
            continue
        try:
            energies = load_energy_jsonl(energy_path)
        except Exception:
            continue
        if not energies:
            continue
        ann_path = gt_path_for_sample(sd, args.gt_dir)
        if not os.path.isfile(ann_path):
            continue
        try:
            y_true, _ = load_gt_windows(ann_path, len(energies))
        except Exception:
            continue
        if len(y_true) != len(energies):
            n = min(len(y_true), len(energies))
            energies = energies[:n]
            y_true = y_true[:n]
        ds_name = os.path.basename(os.path.dirname(sd))
        seqs_by_ds.setdefault(ds_name, []).append((energies, y_true))

    total_pairs = sum(len(v) for v in seqs_by_ds.values())
    if total_pairs == 0:
        raise SystemExit("No valid (energy, GT) pairs found.")

    hrs = args.hysteresis_ratio
    ups = args.up_count
    downs = args.down_count

    # Sweep and collect results (compute per dataset, then macro-average)
    rows = []
    best = {'f1': -1.0}
    for hr in hrs:
        for u in ups:
            for d in downs:
                f1_list = []
                j_list = []
                for ds_name, seqs in seqs_by_ds.items():
                    all_true: List[int] = []
                    all_pred: List[int] = []
                    for energies, y_true in seqs:
                        y_pred = hysteresis_predict(energies, theta_on, hr, u, d)
                        all_true.extend(y_true)
                        all_pred.extend(y_pred)
                    f1_ds, j_ds = f1_and_jaccard(all_true, all_pred)
                    f1_list.append(f1_ds)
                    j_list.append(j_ds)
                if f1_list:
                    f1 = sum(f1_list) / len(f1_list)
                    j = sum(j_list) / len(j_list)
                else:
                    f1 = 0.0
                    j = 0.0
                rows.append({'hysteresis_ratio': hr, 'up_count': u, 'down_count': d, 'f1': f1, 'j_index': j})
                if f1 > best['f1']:
                    best = {'hysteresis_ratio': hr, 'up_count': u, 'down_count': d, 'f1': f1, 'j_index': j}

    # Write CSV
    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    with open(args.outcsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['hysteresis_ratio', 'up_count', 'down_count', 'f1', 'j_index'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved {args.outcsv}")
    print(f"Best params: hr={best['hysteresis_ratio']}, up={best['up_count']}, down={best['down_count']}, f1={best['f1']:.4f}, j={best['j_index']:.4f}")

    # Prepare heatmaps per hysteresis_ratio
    def plot_heatmaps(metric_key: str, title_prefix: str, save_path: str):
        import numpy as np
        n_hr = len(hrs)
        ncols = min(4, n_hr) if n_hr > 0 else 1
        nrows = int(math.ceil(n_hr / ncols)) if n_hr > 0 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 2.4*nrows), dpi=160, squeeze=False)
        for idx, hr in enumerate(hrs):
            grid = np.zeros((len(ups), len(downs)), dtype=float)
            for i, u in enumerate(ups):
                for j, d in enumerate(downs):
                    for r in rows:
                        if r['hysteresis_ratio'] == hr and r['up_count'] == u and r['down_count'] == d:
                            grid[i, j] = r[metric_key]
                            break
            r_i = idx // ncols
            c_i = idx % ncols
            ax = axes[r_i][c_i]
            im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis')
            ax.set_title(f"hr={hr}")
            ax.set_xlabel('down_count')
            ax.set_ylabel('up_count')
            ax.set_xticks(range(len(downs)))
            ax.set_xticklabels([str(x) for x in downs])
            ax.set_yticks(range(len(ups)))
            ax.set_yticklabels([str(x) for x in ups])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Hide unused axes
        total_axes = nrows * ncols
        for idx in range(n_hr, total_axes):
            r_i = idx // ncols
            c_i = idx % ncols
            ax = axes[r_i][c_i]
            ax.axis('off')
        # fig.suptitle(f"{title_prefix} (theta_on={theta_on})")  # disabled per request
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    plot_heatmaps('f1', 'F1 heatmaps', args.outpdf_f1)
    print(f"Saved {args.outpdf_f1}")
    plot_heatmaps('j_index', 'J-Index heatmaps', args.outpdf_j)
    print(f"Saved {args.outpdf_j}")


if __name__ == '__main__':
    main()

