from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Optional
import os
import sys
import numpy as np

from .abd_core import abd_offline, ABDResult


# -------- Helpers --------

def _list_videos(input_dir: Path) -> List[Path]:
    vids = []
    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        vids.extend(sorted(input_dir.glob(f"*{ext}")))
    return vids


def _infer_view_from_path(p: Path) -> Optional[str]:
    parts = [s for s in p.parts]
    for s in parts:
        if s in {"D01", "D02"}:
            return s
    return None


def _save_segments_json(out_dir: Path, video_path: Path, boundaries: List[int], stride: int, target_fps: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    b = [int(x) for x in boundaries]
    segs = []
    for i in range(len(b) - 1):
        s_win, e_win = b[i], b[i + 1]
        start_sec = float(s_win * stride) / float(max(target_fps, 1))
        end_sec = float(e_win * stride) / float(max(target_fps, 1))
        if end_sec <= start_sec:
            continue
        segs.append({"start_sec": round(start_sec, 4), "end_sec": round(end_sec, 4)})
    meta = {
        "video": str(video_path),
        "segments": segs,
        "fps": None,
        "processed_at": None,
    }
    out_path = out_dir / f"{video_path.stem}_segments.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_path


def _load_features_npy(features_dir: Path, video_path: Path) -> Optional[np.ndarray]:
    # Expect naming: {video_stem}.npy
    cand = features_dir / f"{video_path.stem}.npy"
    if cand.exists():
        try:
            return np.load(cand)
        except Exception:
            return None
    return None


def _extract_rgb_mean_features(video_path: Path, target_fps: int = 10, stride: int = 4, T: int = 16) -> Optional[np.ndarray]:
    """Very light-weight fallback features for minimal E2E validation.
    Returns (N, 3) array of mean RGB per window of ~T frames, step ~stride.
    """
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap or not cap.isOpened():
        return None
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_fps = float(orig_fps) if orig_fps and orig_fps > 0 else 30.0
    # frames per window and step in frames
    win_frames = max(1, int(round(T * orig_fps / float(max(target_fps, 1)))))
    step_frames = max(1, int(round(stride * orig_fps / float(max(target_fps, 1)))))
    feats: List[np.ndarray] = []
    frame_idx = 0
    buf = []
    def _flush(buf_list):
        if len(buf_list) == 0:
            return None
        arr = np.stack(buf_list, axis=0)  # (F, H, W, 3)
        m = arr.reshape(-1, 3).mean(axis=0)
        return m
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step_frames == 0:
            # start a new window: read next win_frames frames including current
            frames = [frame]
            for _ in range(win_frames - 1):
                r2, f2 = cap.read()
                if not r2:
                    break
                frames.append(f2)
            m = _flush([cv2.resize(f, (64, 64)) for f in frames])
            if m is not None:
                feats.append(m.astype(np.float32))
        frame_idx += 1
    cap.release()
    if len(feats) < 2:
        return None
    X = np.stack(feats, axis=0)  # (N, 3)
    # Normalize per-dim
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    return X.astype(np.float32)


# -------- Main --------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser("ABD offline runner")
    parser.add_argument("--view", type=str, default=None, help="D01 or D02 (optional)")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of input videos")
    parser.add_argument("--output-dir", type=str, required=True, help="Root output dir for ABD results")
    parser.add_argument("--features-dir", type=str, default=None, help="Optional directory of per-video npy features ({stem}.npy)")
    parser.add_argument("--feature-source", type=str, default="npy", choices=["npy", "i3d", "rgb_mean"], help="Feature source")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--k", type=str, default="5", help="K segments (int) or 'auto'")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--target-fps", type=int, default=10)
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    features_dir = Path(args.features_dir).resolve() if args.features_dir else None

    vids = _list_videos(input_dir)
    if len(vids) == 0:
        print(f"[ABD] No videos found in {input_dir}")
        return 1

    # Determine K
    def _resolve_k(view: Optional[str]) -> int:
        if isinstance(args.k, str) and args.k.lower() == "auto":
            # Try to read gt annotations to estimate average segment count
            if view in {"D01", "D02"}:
                gt_dir = Path(f"/home/johnny/action_ws/datasets/gt_annotations/{view}")
                if gt_dir.exists():
                    counts = []
                    for p in gt_dir.glob("*.json"):
                        try:
                            with open(p, "r", encoding="utf-8") as f:
                                obj = json.load(f)
                            segs = obj.get("segments", [])
                            if isinstance(segs, list) and len(segs) > 0:
                                counts.append(len(segs))
                        except Exception:
                            pass
                    if len(counts) > 0:
                        k_est = int(round(float(sum(counts)) / float(len(counts))))
                        return max(1, k_est)
            # Fallback default
            return 5
        try:
            return max(1, int(args.k))
        except Exception:
            return 5

    # Process each video
    exit_code = 0
    for vp in vids:
        view = args.view or _infer_view_from_path(vp)
        K_val = _resolve_k(view)
        out_dir = output_root / (view or "UNK") / vp.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load features
        X: Optional[np.ndarray] = None
        if args.feature_source == "npy":
            if features_dir is None:
                print(f"[ABD][WARN] features_dir is required for npy source; skipping {vp.name}")
                exit_code = 2
                continue
            X = _load_features_npy(features_dir, vp)
        elif args.feature_source == "i3d":
            try:
                from .features_i3d import extract_i3d_features_for_video
                X = extract_i3d_features_for_video(vp)
            except Exception as e:
                print(f"[ABD][ERROR] I3D feature extraction failed for {vp.name}: {e}")
                exit_code = 3
                continue
        elif args.feature_source == "rgb_mean":
            X = _extract_rgb_mean_features(vp, target_fps=int(args.target_fps), stride=int(args.stride), T=16)
        if X is None:
            print(f"[ABD][WARN] Features not available for {vp.name}; skip")
            exit_code = 2
            continue
        if X.ndim != 2 or X.shape[0] < 2:
            print(f"[ABD][WARN] Invalid feature shape for {vp.name}: {getattr(X, 'shape', None)}; skip")
            exit_code = 2
            continue

        # Run ABD
        res: ABDResult = abd_offline(X, K=K_val, alpha=float(args.alpha))

        # Save segments json
        seg_json = _save_segments_json(out_dir, vp, res.boundaries, stride=int(args.stride), target_fps=int(args.target_fps))
        print(f"[ABD] Saved segments: {seg_json}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

