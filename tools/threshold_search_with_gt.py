import argparse
import json
import os
import glob
from typing import List, Tuple, Dict, Any
import numpy as np

# Threshold search for energy-based segmentation using GT boundaries
# - Input: per-video energy JSONL and GT segments
# - Search: quantile-based candidates + midpoints
# - Segmentation: hysteresis + debouncing + cooldown (parameters aligned with LAPS)
# - Objective: maximize mean F1@tolerance_sec across videos


def load_energy_jsonl(path: str) -> List[float]:
    """Load energy series from jsonl as list sorted by window index.
    Each line: {"window": int, "energy": float}
    """
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
    """Aggregate per-step energy into non-overlapping stride windows by mean.
    Falls back to trailing partial window if exists.
    """
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


def boundary_list(segs: List[Tuple[float, float]], video_duration: float) -> List[float]:
    bs = []
    for s, e in segs:
        if s > 0:
            bs.append(float(s))
        if e < video_duration:
            bs.append(float(e))
    return sorted(bs)


def f1_at_tolerance(pred_segs: List[Tuple[float, float]], gt_segs: List[Tuple[float, float]], video_duration: float, tol_sec: float) -> Tuple[float, float, float, int, int, int]:
    pb = boundary_list(pred_segs, video_duration)
    gb = boundary_list(gt_segs, video_duration)
    num_det = len(pb)
    num_pos = len(gb)
    if num_pos == 0:
        return (1.0 if num_det == 0 else 0.0), 1.0, (1.0 if num_det == 0 else 0.0), 0, num_pos, num_det
    used = set()
    tp = 0
    for g in gb:
        best_j = -1
        best_off = 1e18
        for j, p in enumerate(pb):
            if j in used:
                continue
            off = abs(p - g)
            if off < best_off:
                best_off = off
                best_j = j
        if best_j >= 0 and best_off <= tol_sec:
            tp += 1
            used.add(best_j)
    fp = num_det - tp
    fn = num_pos - tp
    prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    rec = 1.0 if num_pos == 0 else tp / (tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec, tp, num_pos, num_det


def segment_series(
    series: List[float],
    thr_on: float,
    hysteresis_ratio: float,
    up_count: int,
    down_count: int,
    cooldown_windows: int,
    max_duration_windows: int,
) -> List[Tuple[int, int]]:
    """Run state machine over stride-aggregated series; return list of (start_idx, end_idx) inclusive.
    """
    thr_off = float(thr_on) * float(hysteresis_ratio)
    segs = []
    active = False
    pos_run = 0
    neg_run = 0
    cooldown = 0
    start_idx = -1
    length = 0
    for i, e in enumerate(series):
        # cooldown
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
            # active
            if e < thr_off:
                neg_run += 1
            else:
                neg_run = 0
            length += 1
            # end by hysteresis
            if neg_run >= max(1, int(down_count)):
                end_idx = max(start_idx, i)  # close at current index
                segs.append((start_idx, end_idx))
                active = False
                cooldown = int(cooldown_windows)
                start_idx = -1
                length = 0
                pos_run = 0
                neg_run = 0
            # end by max duration
            elif max_duration_windows > 0 and length >= int(max_duration_windows):
                end_idx = i
                segs.append((start_idx, end_idx))
                active = False
                cooldown = int(cooldown_windows)
                start_idx = -1
                length = 0
                pos_run = 0
                neg_run = 0
    # tail: if still active, close at end
    if active and start_idx >= 0:
        segs.append((start_idx, len(series) - 1))
    return segs


def load_gt_map(gt_dir: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load GT segments; support *_segments.json and *_segments_noisy.json; prefer non-noisy if both exist.
    Returns: {video_stem -> [(start_sec, end_sec), ...]}
    """
    paths = []
    for pat in ('*_segments.json', '*_segments_noisy.json'):
        paths.extend(glob.glob(os.path.join(gt_dir, pat)))
    if not paths:
        return {}
    # choose per stem
    chosen: Dict[str, str] = {}
    for p in sorted(paths):
        base = os.path.basename(p)
        if base.endswith('_segments_noisy.json'):
            stem = base[:-len('_segments_noisy.json')]
            if stem not in chosen:
                chosen[stem] = p
        elif base.endswith('_segments.json'):
            stem = base[:-len('_segments.json')]
            chosen[stem] = p
    out: Dict[str, List[Tuple[float, float]]] = {}
    for stem, p in chosen.items():
        try:
            with open(p, 'r') as f:
                d = json.load(f)
            segs = [(float(s['start_sec']), float(s['end_sec'])) for s in d.get('segments', [])]
            out[stem] = segs
        except Exception:
            continue
    return out


def collect_energy(energy_root: str, source: str, mode: str) -> Dict[str, List[float]]:
    """Collect per-video energy series keyed by video stem (folder name)."""
    series_map: Dict[str, List[float]] = {}
    if not os.path.isdir(energy_root):
        return series_map
    for name in sorted(os.listdir(energy_root)):
        d = os.path.join(energy_root, name)
        if not os.path.isdir(d):
            continue
        p = os.path.join(d, f'stream_energy_{source}_{mode}.jsonl')
        if os.path.exists(p):
            series_map[name] = load_energy_jsonl(p)
    return series_map


def gen_threshold_candidates(all_values: np.ndarray) -> List[float]:
    if all_values.size == 0:
        return []
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    vals = np.quantile(all_values, qs).astype(float)
    cand = set(float(v) for v in vals)
    # midpoints between sorted unique vals
    sv = np.unique(vals)
    mids = (sv[:-1] + sv[1:]) / 2.0 if sv.size > 1 else sv
    for v in mids:
        cand.add(float(v))
    # add extremes
    cand.add(float(np.min(all_values)))
    cand.add(float(np.max(all_values)))
    return sorted(cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', required=True, choices=['D01', 'D02'])
    ap.add_argument('--energy-root', required=True)
    ap.add_argument('--gt-dir', required=True)
    ap.add_argument('--source', default='optical_flow')
    ap.add_argument('--mode', default='mag_mean')
    ap.add_argument('--target-fps', type=float, default=10.0)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--hysteresis-ratio', type=float, default=0.95)
    ap.add_argument('--up-count', type=int, default=2)
    ap.add_argument('--down-count', type=int, default=2)
    ap.add_argument('--cooldown-windows', type=int, default=1)
    ap.add_argument('--max-duration-seconds', type=float, default=2.0)
    ap.add_argument('--tolerance-sec', type=float, default=2.0)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    gt_map = load_gt_map(args.gt_dir)
    energy_map_raw = collect_energy(args.energy_root, args.source, args.mode)

    # intersect by stem
    stems = sorted(set(gt_map.keys()) & set(energy_map_raw.keys()))
    if not stems:
        raise RuntimeError(f'No overlapping videos between energy_root={args.energy_root} and gt_dir={args.gt_dir}')

    # aggregate to stride windows
    dt = float(args.stride) / float(args.target_fps) if args.stride > 0 else 1.0 / float(args.target_fps)
    series_map = {}
    durations = {}
    all_vals = []
    for stem in stems:
        e = energy_map_raw[stem]
        ew = aggregate_stride(e, int(args.stride))
        series_map[stem] = ew
        durations[stem] = float(len(e)) / float(args.target_fps)  # duration derived from raw sampling
        all_vals.extend(list(ew))
    all_vals = np.array(all_vals, dtype=float)

    cands = gen_threshold_candidates(all_vals)
    if not cands:
        raise RuntimeError('No threshold candidates generated')

    results_per_thr = []
    for thr in cands:
        f1s = []
        precs = []
        recs = []
        for stem in stems:
            seg_idx = segment_series(
                series_map[stem],
                thr_on=float(thr),
                hysteresis_ratio=float(args.hysteresis_ratio),
                up_count=int(args.up_count),
                down_count=int(args.down_count),
                cooldown_windows=int(args.cooldown_windows),
                max_duration_windows=int(round(float(args.max_duration_seconds) / dt)) if args.max_duration_seconds > 0 else 0,
            )
            # map to seconds
            pred = [ (i0 * dt, (i1 + 1) * dt) for (i0, i1) in seg_idx ]
            gt_segs = gt_map.get(stem, [])
            dur = durations.get(stem, len(series_map[stem]) * dt)
            f1, prec, rec, tp, num_pos, num_det = f1_at_tolerance(pred, gt_segs, dur, float(args.tolerance_sec))
            f1s.append(f1)
            precs.append(prec)
            recs.append(rec)
        results_per_thr.append({
            'thr': float(thr),
            'F1_mean': float(np.mean(f1s) if f1s else 0.0),
            'Precision_mean': float(np.mean(precs) if precs else 0.0),
            'Recall_mean': float(np.mean(recs) if recs else 0.0),
        })

    # select best by F1_mean, then by Precision_mean
    best = sorted(results_per_thr, key=lambda r: (r['F1_mean'], r['Precision_mean']))[-1]

    out = {
        'label_spec': 'gt_boundaries',
        'n_videos': len(stems),
        'n_candidates': len(cands),
        'energy': {'source': args.source, 'mode': args.mode},
        'segmentation_params': {
            'target_fps': float(args.target_fps),
            'stride': int(args.stride),
            'dt': float(dt),
            'hysteresis_ratio': float(args.hysteresis_ratio),
            'up_count': int(args.up_count),
            'down_count': int(args.down_count),
            'cooldown_windows': int(args.cooldown_windows),
            'max_duration_seconds': float(args.max_duration_seconds),
            'tolerance_sec': float(args.tolerance_sec),
        },
        'optical_flow_mag_mean_best': {
            'best_f1': best
        }
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(best, indent=2))


if __name__ == '__main__':
    main()

