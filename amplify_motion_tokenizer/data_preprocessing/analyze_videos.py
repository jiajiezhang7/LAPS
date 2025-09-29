import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import cv2
import yaml


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration_sec: float
    frame_count: int
    size_bytes: int


def _ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def _parse_fraction(frac: str) -> Optional[float]:
    try:
        if "/" in frac:
            num, den = frac.split("/", 1)
            num, den = float(num), float(den)
            return num / den if den != 0 else None
        return float(frac)
    except Exception:
        return None


def probe_with_ffprobe(path: Path) -> Optional[VideoInfo]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,duration",
        "-of", "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8"))
        streams = data.get("streams", [])
        if not streams:
            return None
        s0 = streams[0]
        width = int(s0.get("width", 0))
        height = int(s0.get("height", 0))
        fps = _parse_fraction(s0.get("avg_frame_rate") or s0.get("r_frame_rate") or "0") or 0.0
        duration = float(s0.get("duration") or 0.0)
        size_bytes = path.stat().st_size
        frame_count = int(duration * fps) if (duration > 0 and fps > 0) else 0
        return VideoInfo(path, width, height, fps, duration, frame_count, size_bytes)
    except Exception:
        return None


def probe_with_cv2(path: Path) -> Optional[VideoInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if (fps > 0 and frame_count > 0) else 0.0
        size_bytes = path.stat().st_size
        return VideoInfo(path, width, height, fps, duration, frame_count, size_bytes)
    finally:
        cap.release()


def gather_videos(root: Path, exts: List[str]) -> List[Path]:
    exts = [e.lower() for e in exts]
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def analyze(config: Dict, limit: Optional[int] = None) -> Tuple[List[VideoInfo], Dict[str, float]]:
    data_cfg = config['data']
    pre_cfg = config.get('preprocess', {})
    root = Path(data_cfg['video_source_dir'])
    exts = pre_cfg.get('video_exts', [".mp4", ".mov", ".avi", ".mkv"])

    paths = gather_videos(root, exts)
    if limit is not None:
        paths = paths[:limit]

    infos: List[VideoInfo] = []
    for p in paths:
        vi = None
        if _ffprobe_available():
            vi = probe_with_ffprobe(p)
        if vi is None:
            vi = probe_with_cv2(p)
        if vi is not None:
            infos.append(vi)

    # aggregate stats
    if infos:
        fps_med = median([i.fps for i in infos if i.fps > 0])
        shorter_sides = [min(i.width, i.height) for i in infos if i.width > 0 and i.height > 0]
        shorter_med = median(shorter_sides) if shorter_sides else 0
        duration_med = median([i.duration_sec for i in infos if i.duration_sec > 0])
    else:
        fps_med = 0
        shorter_med = 0
        duration_med = 0

    sugg: Dict[str, float] = {}
    # target_fps
    if fps_med >= 50:
        sugg['target_fps'] = 15
    elif fps_med >= 25:
        sugg['target_fps'] = 15
    elif fps_med >= 15:
        sugg['target_fps'] = 15
    else:
        sugg['target_fps'] = max(10, round(fps_med)) if fps_med > 0 else 15

    # resize_shorter
    if shorter_med >= 1080:
        sugg['resize_shorter'] = 480
    elif shorter_med >= 720:
        sugg['resize_shorter'] = 480
    elif shorter_med >= 480:
        sugg['resize_shorter'] = 480
    else:
        sugg['resize_shorter'] = int(shorter_med) if shorter_med > 0 else 480

    # window_stride (balance data volume vs. redundancy)
    sugg['window_stride'] = 4

    # segment_minutes
    sugg['segment_minutes'] = 5

    # parallel workers
    sugg['parallel_workers'] = min(os.cpu_count() or 4, 8)

    return infos, sugg


def print_report(infos: List[VideoInfo], sugg: Dict[str, float], T: int):
    print("Video Analysis Report")
    print("=====================")
    for vi in infos:
        print(f"- {vi.path}")
        print(f"  Size: {vi.size_bytes/1e6:.2f} MB | {vi.width}x{vi.height} | FPS={vi.fps:.2f} | Duration={vi.duration_sec/60:.2f} min | Frames≈{vi.frame_count}")
    if not infos:
        print("No videos found.")
        return
    print("\nSuggested Parameters")
    print("--------------------")
    for k, v in sugg.items():
        print(f"- {k}: {v}")

    # estimate windows per hour using suggestion
    fps = sugg['target_fps']
    stride = sugg['window_stride']
    per_hour_frames = fps * 3600
    windows_per_hour = max(0, math.floor((per_hour_frames - T) / stride) + 1)
    print(f"\nEst. windows per hour @ target_fps={fps}, stride={stride}, T={T}: {windows_per_hour:,}")


def main():
    parser = argparse.ArgumentParser(description="Analyze videos and suggest preprocessing parameters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to analyze")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    infos, sugg = analyze(config, args.limit)
    print_report(infos, sugg, T=int(config['data']['sequence_length']))


if __name__ == "__main__":
    main()
