#!/usr/bin/env python3
"""统计目录中的视频片段数量，并清理损坏或静止的 mp4 文件（支持并行处理）。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Iterable, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


class ProcessResult(NamedTuple):
    """封装单个文件的处理结果"""

    status: str  # 'ok', 'corrupt', 'static', 'zero_size', 'error'
    path: Path
    message: str = ""
    metrics: Optional[dict] = None


BAD_KEYWORDS = (
    "invalid data found when processing input",
    "error reading header",
    "contradictory stsc and stco",
    "contradictionary stsc and stco",
    "moov atom not found",
    "invalid nal unit size",
    "error splitting the input into nal units",
)

DEFAULT_ROOT = "/media/johnny/Data/data_motion_tokenizer/whole_d01_videos_segments_40s"
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "cfg"
    / "preprocessing"
    / "check_and_clean_segments.yaml"
)


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("[FATAL] 未检测到 ffmpeg，可通过 \"sudo apt install ffmpeg\" 安装。", file=sys.stderr)
        sys.exit(2)


def gather_mp4_files(root: Path) -> Iterable[Path]:
    for file_path in root.rglob("*.mp4"):
        if file_path.is_file():
            yield file_path


def load_config(path: Path) -> dict:
    if path is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            print(f"[WARN] 配置文件格式非法（需为映射）：{path}", file=sys.stderr)
            return {}
        return data
    except FileNotFoundError:
        print(f"[WARN] 未找到配置文件：{path}，使用内置默认参数", file=sys.stderr)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] 读取配置文件失败：{path} -> {exc}，使用内置默认参数", file=sys.stderr)
    return {}


def ffmpeg_probe(path: Path, timeout: float) -> Tuple[int, str]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        "-t",
        "0.1",  # 只处理0.1秒，快速检测
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as exc:  # pylint: disable=broad-except
        return 1, f"exception: {exc}"
    return proc.returncode, proc.stderr or ""


def looks_corrupt(returncode: int, stderr: str) -> bool:
    if returncode != 0:
        return True
    low = stderr.lower()
    return any(keyword in low for keyword in BAD_KEYWORDS)


def _compute_sample_indices(frame_count: int, desired: int) -> Sequence[int]:
    if frame_count <= 0:
        return []
    desired = max(3, min(desired, frame_count))
    idxs = np.linspace(0, frame_count - 1, num=desired, dtype=int)
    return sorted(set(int(i) for i in idxs))


def _resize_keep_shorter(frame: np.ndarray, shorter: int) -> np.ndarray:
    if shorter <= 0:
        return frame
    h, w = frame.shape[:2]
    if min(h, w) <= shorter:
        return frame
    scale = shorter / float(min(h, w))
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _to_gray_f32(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def _compute_diffs(
    cap: cv2.VideoCapture,
    sample_indices: Sequence[int],
    resize_shorter: int,
    diff_threshold: float,
    early_stop_ratio: Optional[float],
    check_interval: int,
    seek_threshold: int,
) -> Tuple[Sequence[float], int, int, bool]:
    if not sample_indices:
        return [], 0, 0, False

    diffs: list[float] = []
    last_frame: Optional[np.ndarray] = None
    motion_hits = 0
    frames_processed = 0
    current_idx = -1
    target_ratio = early_stop_ratio if early_stop_ratio and early_stop_ratio > 0 else None
    check_interval = max(1, check_interval)
    seek_threshold = max(0, seek_threshold)

    for target_idx in sample_indices:
        skip = target_idx - current_idx - 1
        if skip < 0 or skip > seek_threshold:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_idx)):
                break
            current_idx = target_idx - 1
            skip = 0

        if skip > 0:
            for _ in range(skip):
                if not cap.grab():
                    return diffs, frames_processed, motion_hits, False

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        current_idx = target_idx
        frames_processed += 1

        frame = _resize_keep_shorter(frame, resize_shorter)
        gray = _to_gray_f32(frame)

        if last_frame is not None:
            diff = float(np.mean(np.abs(gray - last_frame)))
            diffs.append(diff)
            if diff > diff_threshold:
                motion_hits += 1
            if target_ratio is not None and len(diffs) >= check_interval:
                motion_ratio = motion_hits / len(diffs)
                if motion_ratio >= target_ratio:
                    return diffs, frames_processed, motion_hits, True

        last_frame = gray

    return diffs, frames_processed, motion_hits, False


def analyze_static_video(
    path: Path,
    sample_count: int,
    sample_seconds: float,
    resize_shorter: int,
    diff_threshold: float,
    min_motion_ratio: float,
    max_mean_diff: float,
    max_diff: float,
    early_stop_ratio: float,
    check_interval: int,
    seek_threshold: int,
) -> Optional[dict[str, float | bool]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        sample_seconds = float(sample_seconds or 0.0)
        if sample_seconds > 0:
            if fps > 0:
                approx_frames = int(round(fps * sample_seconds))
            else:
                approx_frames = int(round(30.0 * sample_seconds))
            approx_frames = max(1, approx_frames)
            frame_limit = approx_frames if frame_count <= 0 else min(frame_count, approx_frames)
        else:
            frame_limit = frame_count

        indices = _compute_sample_indices(frame_limit, sample_count)
        if not indices:
            if frame_limit > 0:
                step = max(1, frame_limit // max(sample_count, 1))
                indices = list({min(i * step, frame_limit - 1) for i in range(sample_count)})
                indices.sort()
            else:
                step = max(1, int(round(fps)) if fps > 0 else 30)
                indices = [i * step for i in range(sample_count)]

        diffs, frames_processed, motion_hits, early_exit = _compute_diffs(
            cap,
            indices,
            resize_shorter=resize_shorter,
            diff_threshold=diff_threshold,
            early_stop_ratio=early_stop_ratio if early_stop_ratio > 0 else None,
            check_interval=check_interval,
            seek_threshold=seek_threshold,
        )

        if frames_processed <= 1:
            return None

        if diffs:
            mean_diff = float(np.mean(diffs))
            max_diff_val = float(np.max(diffs))
            motion_ratio = float(motion_hits / len(diffs))
        else:
            mean_diff = 0.0
            max_diff_val = 0.0
            motion_ratio = 0.0

        is_static = (
            (not early_exit)
            and motion_ratio <= min_motion_ratio
            and mean_diff <= max_mean_diff
            and max_diff_val <= max_diff
        )

        return {
            "is_static": is_static,
            "motion_ratio": motion_ratio,
            "mean_diff": mean_diff,
            "max_diff": max_diff_val,
            "samples": float(frames_processed),
            "fps": fps,
            "frame_count": float(frame_count),
            "early_exit": early_exit,
        }
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计目录中的 mp4 文件数量，并删除损坏或静止的视频片段",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="要扫描的根目录（默认读取配置文件 root，若无配置则使用内置默认路径）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"配置文件路径（默认 {CONFIG_PATH}）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="ffmpeg 探针超时时间（秒）",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否仅预演（支持 --dry-run / --no-dry-run）",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否打印 ffmpeg 探针与静止检测的详细信息",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="最多处理多少个文件，0 表示全部",
    )
    parser.add_argument(
        "--skip-static-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否跳过静止检测（支持 --skip-static-check / --no-skip-static-check）",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="静止检测抽样帧数量（至少 3）",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=None,
        help="静止检测仅在视频开始处采样的时长（秒），<=0 表示不限制",
    )
    parser.add_argument(
        "--resize-shorter",
        type=int,
        default=None,
        help="静止检测时调整短边至该像素（0 表示不缩放）",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=None,
        help="像素差阈值(0~1)，超过该值视为存在运动",
    )
    parser.add_argument(
        "--min-motion-ratio",
        type=float,
        default=None,
        help="判定静止时允许的最大运动比例",
    )
    parser.add_argument(
        "--max-mean-diff",
        type=float,
        default=None,
        help="判定静止时允许的最大平均像素差",
    )
    parser.add_argument(
        "--max-diff",
        type=float,
        default=None,
        help="判定静止时允许的最大单次像素差",
    )
    parser.add_argument(
        "--early-stop-ratio",
        type=float,
        default=None,
        help="若运动比例达到该值则提前判定为非静止，<=0 禁用",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=None,
        help="计算运动比例进行早停所需的最少帧差个数",
    )
    parser.add_argument(
        "--seek-threshold",
        type=int,
        default=None,
        help="抽样帧间距超过该值时使用 seek() 而非逐帧 grab",
    )
    parser.add_argument(
        "--static-verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否打印静止检测阶段的指标（支持 --static-verbose / --no-static-verbose）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行处理的工作进程数（默认使用所有CPU核心）",
    )
    return parser.parse_args()


def process_file(path: Path, cfg: dict) -> ProcessResult:
    """处理单个文件，返回 ProcessResult"""
    try:
        size = path.stat().st_size
    except OSError as exc:
        return ProcessResult("error", path, f"无法读取文件大小: {exc}")

    if size == 0:
        return ProcessResult("zero_size", path, "文件大小为 0")

    rc, err = ffmpeg_probe(path, cfg["timeout"])
    if cfg["verbose"]:
        preview = " | ".join((err or "").strip().splitlines()[:2])
        print(f"[FFMPEG] file={path} rc={rc} msg={preview}")

    if looks_corrupt(rc, err):
        preview = " | ".join((err or "").strip().splitlines()[:2])
        return ProcessResult("corrupt", path, f"rc={rc}; err={preview}")

    if cfg["skip_static_check"]:
        if cfg["verbose"]:
            print(f"[KEEP] {path}（已通过 ffmpeg 探针，静止检测被跳过）")
        return ProcessResult("ok", path, "ffmpeg ok, skipped static check")

    metrics = analyze_static_video(
        path=path,
        sample_count=cfg["sample_count"],
        sample_seconds=cfg["sample_seconds"],
        resize_shorter=cfg["resize_shorter"],
        diff_threshold=cfg["diff_threshold"],
        min_motion_ratio=cfg["min_motion_ratio"],
        max_mean_diff=cfg["max_mean_diff"],
        max_diff=cfg["max_diff"],
        early_stop_ratio=cfg["early_stop_ratio"],
        check_interval=cfg["check_interval"],
        seek_threshold=cfg["seek_threshold"],
    )

    if metrics is None:
        return ProcessResult("error", path, "静止检测失败")

    if bool(metrics["is_static"]):
        msg = (
            f"motion_ratio={metrics['motion_ratio']:.3f} "
            f"mean_diff={metrics['mean_diff']:.4f} "
            f"max_diff={metrics['max_diff']:.4f}"
        )
        return ProcessResult("static", path, msg, metrics)

    if cfg["verbose"] or cfg["static_verbose"]:
        status = "ACTIVE" if metrics.get("early_exit") else "MOTION"
        print(
            f"[{status}] {path} | motion_ratio={metrics['motion_ratio']:.3f} "
            f"mean_diff={metrics['mean_diff']:.4f} max_diff={metrics['max_diff']:.4f}"
        )
    return ProcessResult("ok", path, "ok", metrics)


def main() -> None:
    args = parse_args()

    ensure_ffmpeg()

    config_path = args.config if args.config is not None else CONFIG_PATH
    config = load_config(config_path)

    # 将命令行参数覆盖到配置字典中，方便传递
    cli_args = {
        k: v
        for k, v in vars(args).items()
        if v is not None
    }
    config.update(cli_args)

    root_value = config.get("root", DEFAULT_ROOT)
    root = Path(root_value).expanduser()
    if not root.is_dir():
        print(f"[FATAL] 目录不存在或不可访问：{root}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] 扫描目录：{root}")

    files = sorted(gather_mp4_files(root))
    max_files_value = int(config.get("max_files", 0) or 0)
    if max_files_value > 0:
        files = files[:max_files_value]
    total = len(files)
    print(f"[INFO] 共发现 {total} 个 mp4 文件")

    # 填充默认值
    config.setdefault("timeout", 20.0)
    config.setdefault("dry_run", False)
    config.setdefault("skip_static_check", False)
    config.setdefault("verbose", False)
    config.setdefault("static_verbose", False)
    config.setdefault("sample_count", 120)
    config.setdefault("sample_seconds", 4.0)
    config.setdefault("resize_shorter", 256)
    config.setdefault("diff_threshold", 0.02)
    config.setdefault("min_motion_ratio", 0.05)
    config.setdefault("max_mean_diff", 0.015)
    config.setdefault("max_diff", 0.05)
    config.setdefault("early_stop_ratio", 0.25)
    config.setdefault("check_interval", 6)
    config.setdefault("seek_threshold", 600)
    config.setdefault("workers", os.cpu_count())

    results = []
    worker_fn = partial(process_file, cfg=config)
    max_workers = config.get("workers")

    print(f"[INFO] 使用 {max_workers or 'all'} 个工作进程进行并行处理...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 使用 executor.map 并通过 tqdm 显示进度
        results_iterator = executor.map(worker_fn, files)
        results = list(tqdm(results_iterator, total=total, desc="正在处理"))

    corrupt_files = [r for r in results if r.status in ("corrupt", "zero_size")]
    static_files = [r for r in results if r.status == "static"]
    error_files = [r for r in results if r.status == "error"]

    deleted_corrupt = 0
    deleted_static = 0
    delete_failures = 0
    dry_run = config["dry_run"]

    print("\n[处理结果]")
    for res in corrupt_files:
        if dry_run:
            print(f"[BAD][DRY] 将删除: {res.path}; 原因: {res.message}")
        else:
            try:
                res.path.unlink()
                deleted_corrupt += 1
                print(f"[BAD] 已删除: {res.path}")
            except OSError as exc:
                delete_failures += 1
                print(f"[WARN] 删除失败: {res.path} -> {exc}")

    if not config["skip_static_check"]:
        for res in static_files:
            if dry_run:
                print(f"[STATIC][DRY] 将删除: {res.path} | {res.message}")
            else:
                try:
                    res.path.unlink()
                    deleted_static += 1
                    print(f"[STATIC] 已删除: {res.path} | {res.message}")
                except OSError as exc:
                    delete_failures += 1
                    print(f"[WARN] 删除失败: {res.path} -> {exc}")

    print("\n[SUMMARY]")
    print(f"  总数: {total}")
    print(f"  判坏: {len(corrupt_files)}")
    if not config["skip_static_check"]:
        print(f"  判静止: {len(static_files)}")
    if error_files:
        print(f"  处理错误: {len(error_files)}")

    if dry_run:
        planned = len(corrupt_files) + (
            len(static_files) if not config["skip_static_check"] else 0
        )
        print(f"  预演模式: 拟删除 {planned} 个文件")
        print(f"  保留（预演）: {total - planned}")
    else:
        deleted_total = deleted_corrupt + deleted_static
        print(
            f"  已删除: {deleted_total}（损坏 {deleted_corrupt}｜静止 {deleted_static}）"
        )
        if delete_failures:
            print(f"  删除失败: {delete_failures}")
        print(f"  保留: {total - deleted_total}")


if __name__ == "__main__":
    main()
