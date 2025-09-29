#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检测长时间完全静止的工业监控视频。

脚本会在每个视频中均匀抽样若干帧，通过相邻抽样帧之间的像素差判断是否存在明显运动。
若运动比例与强度均低于阈值，则判定该视频为“空场景”（无人/基本静止）。

优化点：
- 大步长时自动 seek，避免逐帧解码整段视频。
- 支持运动比例早停，对明显有人的视频立即跳过后续抽样。
- 可调缩放、帧差阈值与比例阈值，兼顾精度与速度。

示例用法：
    python detect_static_videos.py \
        --root /media/johnny/Data/data_motion_tokenizer/raw_videos_d01_910 \
        --output-json empty_videos.json

建议在运行前激活 conda 环境 amplify_mt：
    conda activate amplify_mt
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


@dataclass
class VideoMetrics:
    path: Path
    samples: int
    mean_diff: float
    max_diff: float
    motion_ratio: float
    is_static: bool
    fps: float
    duration_sec: float


@dataclass
class DeleteStats:
    total: int
    deleted: int
    skipped: int
    missing: int


def _iter_videos(root: Path, exts: Sequence[str]) -> Iterable[Path]:
    lowers = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            p = Path(dirpath) / fname
            try:
                if p.suffix.lower() in lowers:
                    yield p
            except Exception:
                continue


def _compute_sample_indices(frame_count: int, desired: int) -> List[int]:
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


def _compute_diffs(cap: cv2.VideoCapture,
                   sample_indices: Sequence[int],
                   resize_shorter: int,
                   diff_threshold: float,
                   early_stop_ratio: Optional[float] = None,
                   check_interval: int = 1,
                   seek_threshold: int = 600) -> Tuple[List[float], int]:
    if not sample_indices:
        return [], 0

    diffs: List[float] = []
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
                    return diffs, frames_processed

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
                    last_frame = gray
                    return diffs, frames_processed

        last_frame = gray

    return diffs, frames_processed


def _analyze_video_worker(payload: Tuple[str, dict]) -> Optional[VideoMetrics]:
    path_str, options = payload
    path = Path(path_str)
    return analyze_video(path=path, **options)


def analyze_video(path: Path,
                  sample_count: int,
                  resize_shorter: int,
                  diff_threshold: float,
                  min_motion_ratio: float,
                  max_mean_diff: float,
                  max_diff: float,
                  early_stop_ratio: Optional[float],
                  check_interval: int,
                  seek_threshold: int) -> Optional[VideoMetrics]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = frame_count / fps if (fps > 0 and frame_count > 0) else 0.0

        indices = _compute_sample_indices(frame_count, sample_count)
        if not indices:
            step = max(1, int(round(fps)) if fps > 0 else 30)
            indices = [i * step for i in range(sample_count)]

        diffs, frames_processed = _compute_diffs(
            cap,
            indices,
            resize_shorter=resize_shorter,
            diff_threshold=diff_threshold,
            early_stop_ratio=early_stop_ratio,
            check_interval=check_interval,
            seek_threshold=seek_threshold,
        )

        if frames_processed <= 1:
            return None

        if diffs:
            mean_diff = float(np.mean(diffs))
            max_diff_val = float(np.max(diffs))
            motion_ratio = float(np.mean([d > diff_threshold for d in diffs]))
        else:
            mean_diff = 0.0
            max_diff_val = 0.0
            motion_ratio = 0.0

        is_static = (
            motion_ratio <= min_motion_ratio and
            mean_diff <= max_mean_diff and
            max_diff_val <= max_diff
        )
        return VideoMetrics(
            path=path,
            samples=frames_processed,
            mean_diff=mean_diff,
            max_diff=max_diff_val,
            motion_ratio=motion_ratio,
            is_static=is_static,
            fps=fps,
            duration_sec=duration_sec,
        )
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检测完全静止/空场景的视频")
    parser.add_argument("--root", type=str,
                        default="/media/johnny/Data/data_motion_tokenizer/raw_videos_d01_910",
                        help="视频根目录")
    parser.add_argument("--exts", type=str, default=",".join(DEFAULT_EXTS),
                        help="逗号分隔的视频扩展名列表")
    parser.add_argument("--sample-count", type=int, default=180,
                        help="每个视频抽样帧数量（至少 3）")
    parser.add_argument("--resize-shorter", type=int, default=256,
                        help="抽样帧短边缩放目标，0 表示不缩放")
    parser.add_argument("--diff-threshold", type=float, default=0.02,
                        help="像素差阈值(0~1)，超过该值视为存在运动")
    parser.add_argument("--min-motion-ratio", type=float, default=0.05,
                        help="判定非空场景所需的最小运动比例")
    parser.add_argument("--max-mean-diff", type=float, default=0.015,
                        help="判定为空时的最大平均差值(0~1)")
    parser.add_argument("--max-diff", type=float, default=0.05,
                        help="判定为空时允许的最大单次差值(0~1)")
    parser.add_argument("--early-stop-ratio", type=float, default=0.25,
                        help="当运动比例达到该值时提前判定为非静止并结束分析；<=0 表示关闭早停")
    parser.add_argument("--check-interval", type=int, default=6,
                        help="计算运动比例并判定早停所需的最小帧差数量")
    parser.add_argument("--seek-threshold", type=int, default=600,
                        help="相邻抽样帧跨度超过该值时改用 seek() 而非逐帧 grab")
    parser.add_argument("--max-files", type=int, default=0,
                        help="最多处理多少个视频，0 表示全部")
    parser.add_argument("--output-json", type=str, default=None,
                        help="若指定则写入检测结果 JSON 文件")
    parser.add_argument("--verbose", action="store_true",
                        help="输出每个视频的详细指标")
    parser.add_argument("--workers", type=int, default=64,
                        help="并行进程数（<=0 自动取 CPU 核数，1 表示串行）")
    parser.add_argument("--delete", action="store_true",
                        help="检测完成后删除符合条件的空场景视频")
    parser.add_argument("--dry-run", action="store_true",
                        help="删除阶段仅预览目标，不实际删除")
    parser.add_argument("--delete-include-nonstatic", action="store_true",
                        help="允许删除判定为有活动但满足阈值的视频")
    parser.add_argument("--delete-min-motion-ratio", type=float, default=None,
                        help="删除时额外要求 motion_ratio <= 该值")
    parser.add_argument("--delete-max-mean-diff", type=float, default=None,
                        help="删除时额外要求 mean_diff <= 该值")
    parser.add_argument("--delete-verbose", action="store_true",
                        help="删除阶段输出详细信息")
    return parser.parse_args()


def _format_metric(metric: VideoMetrics) -> str:
    return (
        f"path={metric.path} | is_static={metric.is_static} | "
        f"motion_ratio={metric.motion_ratio:.3f} | mean_diff={metric.mean_diff:.4f}"
    )


def _filter_for_deletion(metrics: Iterable[VideoMetrics],
                         include_nonstatic: bool,
                         min_motion_ratio: Optional[float],
                         max_mean_diff: Optional[float]) -> List[VideoMetrics]:
    selected: List[VideoMetrics] = []
    for metric in metrics:
        if not include_nonstatic and not metric.is_static:
            continue
        if min_motion_ratio is not None and metric.motion_ratio > min_motion_ratio:
            continue
        if max_mean_diff is not None and metric.mean_diff > max_mean_diff:
            continue
        selected.append(metric)
    return selected


def _delete_video_files(metrics: Sequence[VideoMetrics],
                        do_delete: bool,
                        verbose: bool) -> DeleteStats:
    deleted = skipped = missing = 0
    metrics_list = list(metrics)
    with tqdm(total=len(metrics_list), desc="删除视频", unit="file") as pbar:
        for metric in metrics_list:
            if verbose:
                pbar.write(f"[TARGET] {_format_metric(metric)}")
            path = metric.path
            if not path.exists():
                missing += 1
                if verbose:
                    pbar.write(f"[MISS] 文件不存在: {path}")
                pbar.update(1)
                continue
            if do_delete:
                try:
                    os.remove(path)
                    deleted += 1
                    if verbose:
                        pbar.write(f"[DELETE] 已删除: {path}")
                except Exception as exc:
                    skipped += 1
                    pbar.write(f"[ERROR] 删除失败: {path} -> {exc}")
            else:
                skipped += 1
                if verbose:
                    pbar.write(f"[DRY-RUN] 预览删除: {path}")
            pbar.update(1)
    return DeleteStats(
        total=len(metrics_list),
        deleted=deleted,
        skipped=skipped,
        missing=missing,
    )


def main() -> None:
    args = parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"[FATAL] 无效目录: {root}")

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    videos = list(_iter_videos(root, exts))
    total_found = len(videos)
    if args.max_files > 0:
        videos = videos[: args.max_files]

    metrics: List[VideoMetrics] = []
    empty_videos: List[VideoMetrics] = []

    options = {
        "sample_count": args.sample_count,
        "resize_shorter": args.resize_shorter,
        "diff_threshold": args.diff_threshold,
        "min_motion_ratio": args.min_motion_ratio,
        "max_mean_diff": args.max_mean_diff,
        "max_diff": args.max_diff,
        "early_stop_ratio": (args.early_stop_ratio if args.early_stop_ratio > 0 else None),
        "check_interval": max(1, args.check_interval),
        "seek_threshold": max(0, args.seek_threshold),
    }

    worker_count = args.workers if args.workers is not None else 0
    if worker_count <= 0:
        worker_count = os.cpu_count() or 1

    if worker_count <= 1 or len(videos) <= 1:
        with tqdm(total=len(videos), desc="扫描视频", unit="video") as pbar:
            for path in videos:
                pbar.set_postfix(empty=len(empty_videos), processed=len(metrics))
                result = analyze_video(path=path, **options)
                if result is None:
                    if args.verbose:
                        pbar.write(f"[WARN] 无法分析: {path}")
                    pbar.update(1)
                    continue

                metrics.append(result)
                if result.is_static:
                    empty_videos.append(result)
                if args.verbose:
                    status = "EMPTY" if result.is_static else "ACTIVE"
                    duration_min = result.duration_sec / 60 if result.duration_sec > 0 else 0.0
                    pbar.write(
                        f"[{status}] {path} | samples={result.samples} | "
                        f"mean_diff={result.mean_diff:.4f} | max_diff={result.max_diff:.4f} | "
                        f"motion_ratio={result.motion_ratio:.3f} | fps={result.fps:.2f} | "
                        f"duration_min={duration_min:.1f}"
                    )
                pbar.update(1)
    else:
        worker_count = min(worker_count, len(videos))
        payloads = [(str(path), options) for path in videos]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_path = {
                executor.submit(_analyze_video_worker, payload): Path(payload[0])
                for payload in payloads
            }
            with tqdm(total=len(videos), desc="扫描视频", unit="video") as pbar:
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - 保护性日志
                        result = None
                        if args.verbose:
                            pbar.write(f"[ERROR] 分析失败: {path} | {exc}")

                    if result is None:
                        if args.verbose:
                            pbar.write(f"[WARN] 无法分析: {path}")
                    else:
                        metrics.append(result)
                        if result.is_static:
                            empty_videos.append(result)
                        if args.verbose:
                            status = "EMPTY" if result.is_static else "ACTIVE"
                            duration_min = result.duration_sec / 60 if result.duration_sec > 0 else 0.0
                            pbar.write(
                                f"[{status}] {path} | samples={result.samples} | "
                                f"mean_diff={result.mean_diff:.4f} | max_diff={result.max_diff:.4f} | "
                                f"motion_ratio={result.motion_ratio:.3f} | fps={result.fps:.2f} | "
                                f"duration_min={duration_min:.1f}"
                            )

                    pbar.set_postfix(empty=len(empty_videos), processed=len(metrics))
                    pbar.update(1)

    print("\n[SUMMARY]")
    print(f"  输入视频数: {total_found}")
    print(f"  实际处理: {len(metrics)}")
    print(f"  判定为空: {len(empty_videos)}")
    print(f"  判定为有活动: {len(metrics) - len(empty_videos)}")

    run_deletion = args.delete or args.dry_run
    if run_deletion:
        include_nonstatic = args.delete_include_nonstatic
        targets = _filter_for_deletion(
            metrics,
            include_nonstatic=include_nonstatic,
            min_motion_ratio=args.delete_min_motion_ratio,
            max_mean_diff=args.delete_max_mean_diff,
        )

        if not targets:
            print("[DELETE] 没有符合删除条件的视频。")
        else:
            do_delete = args.delete and not args.dry_run
            if not do_delete:
                print("[INFO] 当前为 dry-run 模式或未传入 --delete，文件不会被删除。")
            stats = _delete_video_files(
                targets,
                do_delete=do_delete,
                verbose=args.delete_verbose,
            )
            print("\n[DELETE SUMMARY]")
            print(f"  待处理: {stats.total}")
            print(f"  已删除: {stats.deleted}")
            if not do_delete:
                print("  dry-run，未执行删除。")
            print(f"  已跳过/未删: {stats.skipped}")
            print(f"  未找到: {stats.missing}")

    if args.output_json:
        out_path = Path(args.output_json)
        data = [
            {
                "path": str(m.path),
                "samples": m.samples,
                "mean_diff": m.mean_diff,
                "max_diff": m.max_diff,
                "motion_ratio": m.motion_ratio,
                "is_static": m.is_static,
                "fps": m.fps,
                "duration_sec": m.duration_sec,
            }
            for m in metrics
        ]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 结果已写入: {out_path}")


if __name__ == "__main__":
    main()
