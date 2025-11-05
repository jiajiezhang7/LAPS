import argparse
import json
import os
import glob
from typing import List, Tuple, Dict, Any
import numpy as np

# Class-agnostic temporal segment evaluation
# - F1@tolerance_sec (boundary matching in seconds)
# - mAP@IoU thresholds (segment IoU, confidence = mean energy within segment if available)
#
# Expected prediction layout per video under --pred-root:
#   {pred_root}/{video_stem}/segmented_videos/{video_stem}_segments.json
#   {pred_root}/{video_stem}/stream_energy_*.jsonl  (optional, used to compute confidences)
#
# Ground truth under --gt-dir:
#   {gt_dir}/{video_stem}_segments.json


def load_segments_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def segments_to_list(d: Dict[str, Any]) -> List[Tuple[float, float]]:
    return [(float(s["start_sec"]), float(s["end_sec"])) for s in d.get("segments", [])]


def load_energy_jsonl(path: str) -> Dict[int, float]:
    """Return mapping: window_index -> energy.
    JSONL lines each include: {"window": int, "energy": float}
    """
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                w = int(obj.get("window"))
                e = float(obj.get("energy"))
                mapping[w] = e
            except Exception:
                continue
    return mapping


def segment_confidence(seg: Tuple[float, float], seg_meta: Dict[str, Any], energy_dir: str) -> float:
    """Compute confidence as mean energy over windows inside the segment.
    Fall back to segment duration if energy is unavailable.
    """
    start_s, end_s = seg
    duration = max(0.0, end_s - start_s)
    params = seg_meta.get("segmentation_params", {})
    stride = int(params.get("stride", 4))
    target_fps = float(params.get("target_fps", 10.0))

    # energy file priority: prefer quantized token_diff; fallback to any jsonl in directory
    candidate_files = sorted(glob.glob(os.path.join(energy_dir, "stream_energy_*.jsonl")))
    if not candidate_files:
        return duration

    # If multiple energies exist, average across all available files
    energies_list = [load_energy_jsonl(p) for p in candidate_files]
    if not energies_list:
        return duration

    # Map second -> window index using center of window approximation
    def window_center_time(w_idx: int) -> float:
        return (w_idx * stride) / target_fps

    # Collect energies whose window centers fall inside the segment
    vals = []
    for energies in energies_list:
        for w_idx, e in energies.items():
            t = window_center_time(w_idx)
            if start_s <= t <= end_s:
                vals.append(float(e))
    if not vals:
        return duration
    return float(np.mean(vals))


def temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-8, (a1 - a0) + (b1 - b0) - inter)
    return inter / union


def boundary_list(segs: List[Tuple[float, float]], video_duration: float) -> List[float]:
    # use both boundaries; ignore 0 and video end to avoid trivial matches
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
        return 1.0 if num_det == 0 else 0.0, 1.0, 1.0 if num_det == 0 else 0.0, 0, num_pos, num_det

    used = set()
    tp = 0
    for g in gb:
        # find nearest unmatched prediction
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


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    # Standard area under PR curve (monotonic envelope)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Compute area under curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def evaluate_map_iou(all_preds: List[Tuple[str, Tuple[float, float], float]],
                     all_gts: Dict[str, List[Tuple[float, float]]],
                     iou_thrs: List[float]) -> Dict[str, float]:
    # For each IoU threshold, compute AP across all videos
    results = {}
    # Prepare GT used flags per video
    gt_flags = {vid: np.zeros(len(segs), dtype=bool) for vid, segs in all_gts.items()}

    for thr in iou_thrs:
        # Sort predictions by confidence desc
        preds_sorted = sorted(all_preds, key=lambda x: x[2], reverse=True)
        tp = []
        fp = []
        npos = sum(len(segs) for segs in all_gts.values())
        # Reset flags each threshold
        for vid in gt_flags:
            gt_flags[vid][:] = False
        for vid, seg, conf in preds_sorted:
            best_iou = 0.0
            best_idx = -1
            gt_list = all_gts.get(vid, [])
            flags = gt_flags.get(vid)
            for i, g in enumerate(gt_list):
                if flags[i]:
                    continue
                iou = temporal_iou(seg, g)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= thr and best_idx >= 0:
                tp.append(1)
                fp.append(0)
                flags[best_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        if npos == 0:
            results[f"mAP@{thr}"] = 0.0
            continue
        tp = np.array(tp)
        fp = np.array(fp)
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / float(npos)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap = average_precision(recalls, precisions)
        results[f"mAP@{thr}"] = float(ap)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-root', required=True, help='Root of prediction outputs per video')
    parser.add_argument('--gt-dir', required=True, help='Directory of GT JSON files')
    parser.add_argument('--iou-thrs', nargs='+', type=float, default=[0.5, 0.75])
    parser.add_argument('--tolerance-sec', type=float, default=2.0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, '*_segments.json')))
    if not gt_files:
        raise FileNotFoundError(f'No GT files under {args.gt_dir}')

    per_video = {}
    all_preds = []  # (video_stem, (s,e), conf)
    all_gts: Dict[str, List[Tuple[float, float]]] = {}

    for gt_path in gt_files:
        gt = load_segments_json(gt_path)
        video = gt.get('video')
        stem = os.path.splitext(video)[0]
        gt_segs = segments_to_list(gt)
        all_gts[stem] = gt_segs

        pred_seg_path = os.path.join(args.pred_root, stem, 'segmented_videos', f'{stem}_segments.json')
        if not os.path.exists(pred_seg_path):
            # try alternative naming (some pipelines may store under VIEW subgroup)
            alt = glob.glob(os.path.join(args.pred_root, '*', stem, 'segmented_videos', f'{stem}_segments.json'))
            if alt:
                pred_seg_path = alt[0]
        if not os.path.exists(pred_seg_path):
            # no prediction for this video
            per_video[stem] = {
                'f1_at_tol': 0.0, 'precision_at_tol': 0.0, 'recall_at_tol': 0.0,
                'num_gt_boundaries': len(boundary_list(gt_segs, gt.get('video_duration_sec', 1e9))),
                'num_pred_boundaries': 0,
            }
            continue

        pred = load_segments_json(pred_seg_path)
        pred_segs = segments_to_list(pred)
        duration = float(gt.get('video_duration_sec', pred.get('video_duration_sec', 0.0)))
        f1, prec, rec, tp, num_pos, num_det = f1_at_tolerance(pred_segs, gt_segs, duration, args.tolerance_sec)
        per_video[stem] = {
            'f1_at_tol': f1,
            'precision_at_tol': prec,
            'recall_at_tol': rec,
            'num_gt_boundaries': num_pos,
            'num_pred_boundaries': num_det,
        }

        # collect predictions with confidences for mAP
        energy_dir = os.path.dirname(pred_seg_path).replace('segmented_videos', '')
        energy_dir = energy_dir if energy_dir.endswith(os.sep) else energy_dir + os.sep
        # parent folder of video: {pred_root}/{stem}/
        energy_dir = os.path.dirname(os.path.dirname(pred_seg_path))
        for seg in pred_segs:
            conf = segment_confidence(seg, pred, energy_dir)
            all_preds.append((stem, seg, conf))

    map_results = evaluate_map_iou(all_preds, all_gts, args.iou_thrs)

    # aggregate F1
    f1_list = [v['f1_at_tol'] for v in per_video.values()]
    prec_list = [v['precision_at_tol'] for v in per_video.values()]
    rec_list = [v['recall_at_tol'] for v in per_video.values()]
    summary = {
        'num_videos': len(per_video),
        'F1@{:.1f}s_mean'.format(args.tolerance_sec): float(np.mean(f1_list) if f1_list else 0.0),
        'Precision@{:.1f}s_mean'.format(args.tolerance_sec): float(np.mean(prec_list) if prec_list else 0.0),
        'Recall@{:.1f}s_mean'.format(args.tolerance_sec): float(np.mean(rec_list) if rec_list else 0.0),
    }
    summary.update(map_results)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'per_video': per_video, 'summary': summary}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

