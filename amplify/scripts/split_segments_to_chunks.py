#!/usr/bin/env python3
"""将嵌套目录中的视频批量切割成固定时长的小片段。"""
# TODO： 附加断点继续 - 跳过目标文件夹下已重复的文件
import argparse
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def discover_videos(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    if dry_run:
        print("[DRY-RUN]", " ".join(cmd))
        return
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"命令执行失败：{' '.join(cmd)}\n"
            f"stdout:\n{result.stdout.decode(errors='ignore')}\n"
            f"stderr:\n{result.stderr.decode(errors='ignore')}"
        )


def probe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"读取时长失败：{video_path}\n"
            f"stdout:\n{result.stdout.decode(errors='ignore')}\n"
            f"stderr:\n{result.stderr.decode(errors='ignore')}"
        )
    try:
        return float(result.stdout.decode().strip())
    except ValueError as exc:
        raise RuntimeError(f"无法解析视频时长：{video_path}") from exc


def format_seconds(sec: float) -> str:
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def split_video(
    video_path: Path,
    input_root: Path,
    out_root: Path,
    chunk_seconds: float,
    dry_run: bool,
) -> None:
    duration = probe_duration(video_path)
    if duration <= chunk_seconds + 1e-2:
        print(f"跳过{video_path}（时长{duration:.2f}s，短于或等于切割时长）")
        return

    rel_path = video_path.relative_to(input_root)
    out_dir = out_root / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    total_parts = math.ceil(duration / chunk_seconds)
    print(f"处理{video_path}，时长{duration:.2f}s，预计输出{total_parts}段")

    for idx in range(total_parts):
        start = idx * chunk_seconds
        if start >= duration:
            break
        remaining = duration - start
        segment = min(chunk_seconds, remaining)
        start_ts = format_seconds(start)
        segment_ts = format_seconds(segment)
        out_name = f"{video_path.stem}_part{idx + 1:02d}{video_path.suffix}"
        out_path = out_dir / out_name

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            start_ts,
            "-i",
            str(video_path),
            "-t",
            segment_ts,
            "-c",
            "copy",
            str(out_path),
        ]
        run_cmd(cmd, dry_run=dry_run)


def process_video(
    video_path: Path,
    input_root: Path,
    output_root: Path,
    chunk_seconds: float,
    dry_run: bool,
) -> Optional[Tuple[Path, str]]:
    try:
        split_video(
            video_path,
            input_root,
            output_root,
            chunk_seconds,
            dry_run=dry_run,
        )
        return None
    except Exception as exc:  # pylint: disable=broad-except
        return video_path, str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量将视频切割为固定时长片段")
    parser.add_argument("input_root", type=Path, help="输入根目录，包含嵌套视频文件")
    parser.add_argument("output_root", type=Path, help="输出根目录，将复刻原目录结构")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=40.0,
        help="切割片段时长（秒，默认 40）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的命令，不实际写文件",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="并行线程数（默认取 min(4, CPU 核心数)）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_seconds = args.chunk_seconds

    videos = list(discover_videos(args.input_root))
    if not videos:
        print(f"在{args.input_root}下未发现视频文件（支持扩展名：{', '.join(sorted(VIDEO_EXTS))}）")
        return

    print(
        f"共发现{len(videos)}个视频，开始切割，目标片段长度 {args.chunk_seconds} 秒，"
        f"并行线程数 {args.workers}"
    )

    failures: List[Tuple[Path, str]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_path = {
                executor.submit(
                    process_video,
                    video_path,
                    args.input_root,
                    args.output_root,
                    chunk_seconds,
                    args.dry_run,
                ): video_path
                for video_path in videos
            }
            for future in as_completed(future_to_path):
                result = future.result()
                if result is not None:
                    failures.append(result)
    except KeyboardInterrupt:
        print("检测到中断信号，等待正在运行的任务结束…")
        raise

    if failures:
        print("以下视频处理失败：")
        for path, err in failures:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
