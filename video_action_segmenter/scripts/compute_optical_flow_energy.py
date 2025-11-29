#!/usr/bin/env python3
"""
Compute optical flow energy (Dual TV-L1) for each video in a directory and export per-video JSONL.

- Resample frames to target_fps (default: 10)
- Energy per sampled step = mean( sqrt(u^2 + v^2) ) over the whole image
- Optional EMA smoothing (alpha=0.7 by default) applied to the reported energy
- Output JSONL per video at: <output_root>/<video_stem>/stream_energy_optical_flow_mag_mean.jsonl
- JSONL record: {"window": int, "energy": float, "source": "optical_flow", "mode": "mag_mean"}

Run example:
  conda run -n laps python -m video_action_segmenter.scripts.compute_optical_flow_energy \
    --view D01 \
    --input-dir ./datasets/gt_raw_videos/D01 \
    --output-root ./datasets/output/energy_sweep_out/D01 \
    --target-fps 10 --ema-alpha 0.7 --resize-shorter 480
"""
from pathlib import Path
import argparse
import sys
import math
from typing import List, Optional

import numpy as np

# Lazy-import cv2 to allow --help to work even if contrib not installed yet
# Other helpers from our repo
from video_action_segmenter.stream_utils.stream_io import append_energy_jsonl
from video_action_segmenter.stream_utils.paths import compute_per_video_energy_jsonl_path
from video_action_segmenter.stream_utils.video import resize_shorter_keep_aspect


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def _require_tvl1():
    """Create and return a Dual TV-L1 optical flow instance (cv2.optflow).
    Raises a RuntimeError with installation hint if contrib module is missing.
    """
    try:
        import cv2  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "OpenCV (cv2) not available in this environment. Please install opencv-contrib-python"
        ) from e
    if not hasattr(sys.modules["cv2"], "optflow"):
        raise RuntimeError(
            "cv2.optflow not available. Please install opencv-contrib-python (contrib modules required)."
        )
    cv2 = sys.modules["cv2"]
    try:
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    except Exception:
        # alternative API name in some builds
        tvl1 = cv2.optflow.createOptFlow_DualTVL1()
    return tvl1


def _iter_videos(input_dir: Path) -> List[Path]:
    vids: List[Path] = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p)
    return vids


def _should_skip_video(jsonl_path: Path) -> bool:
    """Check if video has already been processed (JSONL exists and is non-empty).
    
    Returns True if the video should be skipped.
    """
    if not jsonl_path.exists():
        return False
    try:
        return jsonl_path.stat().st_size > 0
    except Exception:
        return False


def _sample_and_compute(
    video_path: Path,
    output_root: Path,
    target_fps: float,
    ema_alpha: Optional[float],
    resize_shorter: int,
) -> bool:
    """Process a single video, writing optical_flow energy JSONL.

    Returns True if at least one energy sample was written.
    """
    import cv2

    # Compute output path early to check if already processed
    jsonl_path = compute_per_video_energy_jsonl_path(
        base_path=output_root / "stream_energy.jsonl",
        video_name=video_path.stem,
        energy_source="optical_flow",
        energy_mode="mag_mean",
        seg_enable=False,
        seg_output_dir=output_root,
    )
    
    if _should_skip_video(jsonl_path):
        print(f"[SKIP] {video_path.name} (already processed)")
        return True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Unable to open: {video_path}")
        return False

    # read input fps, fallback if unavailable
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    fps = float(fps)
    if target_fps <= 0:
        target_fps = 10.0

    # Floating-step scheduler: emit when frame_index >= next_emit
    emit_every = fps / float(target_fps)
    next_emit: float = 0.0

    tvl1 = _require_tvl1()

    prev_gray = None
    energy_smooth: Optional[float] = None
    frame_idx = 0
    window = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if resize_shorter and int(resize_shorter) > 0:
                frame = resize_shorter_keep_aspect(frame, int(resize_shorter))

            if frame_idx + 1e-9 >= next_emit:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = tvl1.calc(prev_gray, gray, None)
                    # flow: (H, W, 2)
                    mag = np.sqrt(flow[..., 0].astype(np.float32) ** 2 + flow[..., 1].astype(np.float32) ** 2)
                    energy = float(np.mean(mag))
                    # EMA smoothing if requested
                    if ema_alpha is not None and 0.0 < float(ema_alpha) < 1.0:
                        if energy_smooth is None:
                            energy_smooth = energy
                        else:
                            energy_smooth = float(ema_alpha) * energy + (1.0 - float(ema_alpha)) * energy_smooth
                        e_write = energy_smooth
                    else:
                        e_write = energy
                    append_energy_jsonl(jsonl_path, window=window, energy=e_write, source="optical_flow", mode="mag_mean")
                    window += 1
                prev_gray = gray
                next_emit += emit_every
            frame_idx += 1
    finally:
        cap.release()

    if window == 0:
        print(f"[WARN] No energy samples written for: {video_path.name}")
        return False
    print(f"[OK] {video_path.name} -> {jsonl_path} | windows={window} | fps_in={fps:.2f} | target_fps={target_fps}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Compute optical flow (Dual TV-L1) energy per video and export JSONL")
    ap.add_argument("--view", type=str, default=None, help="Optional view tag (D01/D02), only for logging")
    ap.add_argument("--input-dir", type=str, required=True, help="Directory containing input videos")
    ap.add_argument("--output-root", type=str, required=True, help="Output root (per-video subfolders will be created here)")
    ap.add_argument("--target-fps", type=float, default=10.0, help="Target FPS for resampling")
    ap.add_argument("--ema-alpha", type=float, default=0.7, help="EMA alpha for smoothing (0<alpha<1); set <=0 or >=1 to disable")
    ap.add_argument("--resize-shorter", type=int, default=480, help="Resize shorter side to this size (<=0 to disable)")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N videos (0 = no limit)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    vids = _iter_videos(in_dir)
    if args.limit and args.limit > 0:
        vids = vids[: int(args.limit)]

    if not vids:
        print(f"[ERR] No videos found under: {in_dir}")
        sys.exit(1)

    # Verify TV-L1 availability now, for earlier failure with clear message
    try:
        _ = _require_tvl1()
    except RuntimeError as e:
        print(f"[ERR] {e}")
        sys.exit(2)

    ok_count = 0
    for v in vids:
        ok = _sample_and_compute(
            video_path=v,
            output_root=out_root,
            target_fps=float(args.target_fps),
            ema_alpha=float(args.ema_alpha) if (0.0 < float(args.ema_alpha) < 1.0) else None,
            resize_shorter=int(args.resize_shorter) if args.resize_shorter is not None else 0,
        )
        ok_count += int(ok)

    print(f"[DONE] view={args.view or '-'} | processed={ok_count}/{len(vids)} | output_root={out_root}")


if __name__ == "__main__":
    main()

