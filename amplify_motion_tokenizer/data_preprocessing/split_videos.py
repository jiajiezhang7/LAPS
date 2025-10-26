import argparse
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import yaml


def gather_videos(root: Path, exts: List[str]) -> List[Path]:
    exts = [e.lower() for e in exts]
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def ffmpeg_supports_encoder(name: str) -> bool:
    """Return True if ffmpeg binary supports the given encoder (e.g., 'h264_nvenc')."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-v", "error", "-encoders"], text=True)
        return name in out
    except Exception:
        return False


def choose_gpu_backend(requested: str) -> str:
    """
    Choose the GPU backend to use.
    Currently supports: 'cuda' (NVENC) or 'none'.
    If requested == 'auto', pick 'cuda' if NVENC is available, else 'none'.
    """
    if requested == "none":
        return "none"
    if requested == "cuda":
        return "cuda" if ffmpeg_supports_encoder("h264_nvenc") else "none"
    # auto
    return "cuda" if ffmpeg_supports_encoder("h264_nvenc") else "none"


def split_one(
    input_path: Path,
    out_dir: Path,
    minutes: int,
    reencode: bool = False,
    video_codec: Optional[str] = None,
    audio_codec: str = "copy",
    use_gpu_backend: str = "none",
    gpu_preset: Optional[str] = None,
    force_key_frames: bool = True,
    hwaccel: Optional[str] = None,
    extra_video_args: Optional[List[str]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_seconds = max(1, minutes * 60)
    out_template = out_dir / f"{input_path.stem}_%05d{input_path.suffix}"

    cmd: List[str] = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
    ]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(input_path), "-map", "0"]

    if not reencode:
        # Original fast path: stream copy
        cmd += ["-c", "copy"]
    else:
        # Start from copying all streams, then override per-stream codecs
        cmd += ["-c", "copy"]
        vcodec = video_codec or ("h264_nvenc" if use_gpu_backend == "cuda" else "libx264")
        cmd += ["-c:v", vcodec]
        if audio_codec:
            cmd += ["-c:a", audio_codec]
        if use_gpu_backend == "cuda" and gpu_preset:
            # NVENC preset p1..p7; p4 ~= medium
            cmd += ["-preset", gpu_preset]
        if force_key_frames and seg_seconds > 0:
            # Ensure split boundaries align with keyframes when re-encoding
            cmd += ["-force_key_frames", f"expr:gte(t,n_forced*{seg_seconds})"]
        if extra_video_args:
            cmd += list(extra_video_args)

    cmd += [
        "-f", "segment", "-segment_time", str(seg_seconds),
        "-reset_timestamps", "1",
        str(out_template),
    ]

    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Split long videos into segments using ffmpeg")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--minutes", type=int, default=None, help="Segment length in minutes (override config)")
    parser.add_argument("--jobs", type=int, default=None, help="Parallel workers for splitting")
    # New options for re-encoding and GPU
    parser.add_argument("--reencode", action="store_true", help="Re-encode segments instead of stream copy (enables GPU usage)")
    parser.add_argument("--use-gpu", type=str, default=None, choices=["auto", "none", "cuda"], help="GPU backend to use (only effective with --reencode). Defaults to config or auto.")
    parser.add_argument("--video-codec", type=str, default=None, help="Video codec when re-encoding (e.g., h264_nvenc, libx264)")
    parser.add_argument("--audio-codec", type=str, default=None, help="Audio codec when re-encoding (default: copy)")
    parser.add_argument("--gpu-concurrency", type=int, default=None, help="Max concurrent GPU encodes (default from config or 1)")
    parser.add_argument("--gpu-preset", type=str, default=None, help="Encoder preset (NVENC: p1..p7). If unset, ffmpeg default is used")
    parser.add_argument("--no-force-key-frames", action="store_true", help="Do not force key frames at segment boundaries when re-encoding")
    # New: skip already-processed videos if output segments exist
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip a video if its output segment folder already contains segments")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    pre_cfg = config.get('preprocess', {})

    video_root = Path(data_cfg['video_source_dir'])
    segments_root = Path(pre_cfg.get('segments_dir', './amplify_motion_tokenizer/data/video_segments'))
    minutes = args.minutes or int(pre_cfg.get('segment_minutes', 5))
    exts = pre_cfg.get('video_exts', [".mp4", ".mov", ".avi", ".mkv"])
    jobs = args.jobs or int(pre_cfg.get('parallel_workers', 4))

    # GPU-related config (all optional; safe defaults)
    gpu_cfg = pre_cfg.get('gpu', {})
    reencode = bool(args.reencode or gpu_cfg.get('reencode', False))
    use_gpu_requested = args.use_gpu if args.use_gpu is not None else gpu_cfg.get('use', 'auto')
    video_codec = args.video_codec or gpu_cfg.get('video_codec')
    audio_codec = args.audio_codec or gpu_cfg.get('audio_codec', 'copy')
    gpu_concurrency = int(args.gpu_concurrency or gpu_cfg.get('concurrency', 1))
    gpu_preset = args.gpu_preset or gpu_cfg.get('preset')
    force_key_frames = (gpu_cfg.get('force_key_frames', True) and (not args.no_force_key_frames))
    # New: config-level default for skipping
    skip_if_exists = bool(args.skip_if_exists or pre_cfg.get('skip_if_exists', False))

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg 未安装或不可用，请先安装 ffmpeg 再运行该脚本。")

    chosen_backend = choose_gpu_backend(use_gpu_requested if reencode else "none")
    if reencode:
        if chosen_backend == "cuda":
            print("[INFO] Using GPU backend: CUDA (NVENC)")
        else:
            print("[INFO] GPU unavailable or disabled; falling back to CPU re-encode (libx264)")
    else:
        print("[INFO] Using stream copy (no re-encode). GPU is not used.")

    videos = gather_videos(video_root, exts)
    if not videos:
        print(f"未在 {video_root} 下找到指定后缀的视频：{exts}")
        return

    print(f"将对 {len(videos)} 个视频进行分割，每段时长约 {minutes} 分钟，输出到 {segments_root}")

    # Limit concurrent GPU encodes if needed
    gpu_sema = threading.Semaphore(max(1, gpu_concurrency)) if (reencode and chosen_backend == "cuda") else None

    futures = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for v in videos:
            out_dir = segments_root / v.stem
            # Skip if exists: if any segment file already present, skip this video
            if skip_if_exists and out_dir.exists():
                pattern = f"{v.stem}_*{v.suffix}"
                if any(out_dir.glob(pattern)):
                    print(f"[SKIP] {v.name} -> segments already exist in {out_dir}")
                    continue
            if gpu_sema is None:
                futures.append(ex.submit(
                    split_one, v, out_dir, minutes,
                    reencode, video_codec, audio_codec,
                    chosen_backend, gpu_preset, force_key_frames,
                    "cuda" if chosen_backend == "cuda" else None,
                    None,
                ))
            else:
                def _task(v=v, out_dir=out_dir):
                    gpu_sema.acquire()
                    try:
                        split_one(
                            v, out_dir, minutes,
                            reencode, video_codec, audio_codec,
                            chosen_backend, gpu_preset, force_key_frames,
                            "cuda", None,
                        )
                    finally:
                        gpu_sema.release()
                futures.append(ex.submit(_task))
        for fut in as_completed(futures):
            try:
                fut.result()
            except subprocess.CalledProcessError as e:
                print(f"ffmpeg 分割失败：{e}")

    print("分割完成。")


if __name__ == "__main__":
    main()
