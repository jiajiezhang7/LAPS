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


def _save_segments_json(out_dir: Path, video_path: Path, boundaries: List[int], clip_stride: float, meta_params: dict) -> Path:
    # Ensure output directory and 'segmented_videos' subdir
    sv_dir = out_dir / "segmented_videos"
    sv_dir.mkdir(parents=True, exist_ok=True)

    b = [int(x) for x in boundaries]
    segs = []
    for i in range(len(b) - 1):
        s_win, e_win = b[i], b[i + 1]
        start_sec = float(s_win) * float(clip_stride)
        end_sec = float(e_win) * float(clip_stride)
        if end_sec <= start_sec:
            continue
        segs.append({"start_sec": round(start_sec, 4), "end_sec": round(end_sec, 4)})

    duration = float(b[-1]) * float(clip_stride) if len(b) > 0 else 0.0
    from datetime import datetime, timezone
    meta = {
        "video": str(video_path.name),
        "segments": segs,
        "fps": None,
        "video_duration_sec": round(duration, 4),
        "segmentation_params": meta_params,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = sv_dir / f"{video_path.stem}_segments.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_path






# -------- Main --------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser("ABD offline runner")
    parser.add_argument("--view", type=str, default=None, help="D01 or D02 (optional)")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of input videos")
    parser.add_argument("--output-dir", type=str, required=True, help="Root output dir for ABD results")
    parser.add_argument("--features-dir", type=str, default=None, help="Optional directory of precomputed features (e.g., I3D/HOF) as {stem}.npy")
    parser.add_argument("--feature-source", type=str, default="i3d", choices=["i3d", "hof"], help="Feature source (i3d or hof); affects metadata only")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--k", type=str, default="5", help="K segments (int) or 'auto'")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--clip-duration", type=float, default=2.0, help="I3D clip duration (seconds)")
    parser.add_argument("--clip-stride", type=float, default=0.4, help="I3D clip stride (seconds)")
    parser.add_argument("--i3d-device", type=str, default="cuda:0", help="Device for I3D feature extraction")
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
                gt_dir = Path(f"./datasets/gt_annotations/{view}")
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
        out_dir = output_root / vp.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load features (prefer precomputed under --features-dir; fallback to on-the-fly extraction)
        X: Optional[np.ndarray] = None
        if 'features_dir' in locals() and features_dir is not None:
            pre = features_dir / f"{vp.stem}.npy"
            if pre.exists():
                try:
                    X = np.load(pre)
                except Exception as e:
                    print(f"[ABD][WARN] Failed to load precomputed features for {vp.name}: {e}")
        if X is None:
            if str(args.feature_source).lower() == "i3d":
                try:
                    from .features_i3d import extract_i3d_features_for_video
                    X = extract_i3d_features_for_video(
                        vp,
                        device=str(args.i3d_device),
                        clip_duration=float(args.clip_duration),
                        clip_stride=float(args.clip_stride),
                        target_fps=int(args.target_fps),
                    )
                except Exception as e:
                    print(f"[ABD][ERROR] I3D feature extraction failed for {vp.name}: {e}")
                    exit_code = 3
                    continue
            else:
                print(f"[ABD][WARN] No precomputed features for {vp.name} and feature_source='{args.feature_source}'; skip")
                exit_code = 2
                continue
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

        # Save segments json (time mapping by clip_stride)
        meta_params = {
            "source": str(args.feature_source),
            "alpha": float(args.alpha),
            "k": int(K_val),
            "stride": int(args.stride),
            "target_fps": int(args.target_fps),
            "view": str(view) if view else None,
            "clip_duration": float(args.clip_duration),
            "clip_stride": float(args.clip_stride),
        }
        seg_json = _save_segments_json(out_dir, vp, res.boundaries, clip_stride=float(args.clip_stride), meta_params=meta_params)
        print(f"[ABD] Saved segments: {seg_json}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

