import json
import csv
import shutil
from pathlib import Path
from datetime import datetime
import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "批量修正 segmentation_outputs 下 D01_LAPS 与 D02_LAPS 目录中的 *_segments.json 时间戳："
            "按 correction_factor = target_fps / orig_fps 缩放 start_sec/end_sec；"
            "默认原地覆盖并在写入前自动备份；输出修正报告 CSV。"
        )
    )
    p.add_argument("--dry-run", action="store_true", help="只统计与输出报告，不写回文件与备份")
    p.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="备份根目录（默认：datasets/output/segmentation_outputs_backup_<timestamp>）",
    )
    p.add_argument(
        "--report-path",
        type=str,
        default=None,
        help=(
            "修正报告 CSV 输出路径（默认：video_action_segmenter/inference_outputs/"
            "segmentation_outputs_timebase_correction_report_<timestamp>.csv）"
        ),
    )
    return p.parse_args()


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_with_dirs(src: Path, dst: Path):
    ensure_parent_dir(dst)
    shutil.copy2(src, dst)


def safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def resolve_fps_fields(meta: dict):
    """
    解析 orig_fps / target_fps / video_duration_sec。
    兼容字段位置：
      - 顶层 "orig_fps"
      - 顶层 "fps" 作为 orig_fps 的别名
      - segmentation_params.{orig_fps,target_fps}
      - 顶层 "target_fps"
      - 顶层 "video_duration_sec"
    """
    segp = meta.get("segmentation_params", {}) or {}

    orig_fps = safe_float(meta.get("orig_fps"))
    if orig_fps is None:
        orig_fps = safe_float(meta.get("fps"))
    if orig_fps is None:
        orig_fps = safe_float(segp.get("orig_fps"))

    target_fps = safe_float(meta.get("target_fps"))
    if target_fps is None:
        target_fps = safe_float(segp.get("target_fps"))

    video_duration_sec = safe_float(meta.get("video_duration_sec"))

    return orig_fps, target_fps, video_duration_sec


def find_segment_files(roots):
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            continue
        # 仅匹配路径中包含 segmented_videos 的 *_segments.json
        for p in root.rglob("**/segmented_videos/*_segments.json"):
            if p.is_file():
                yield p


def try_relative_to(path: Path, base: Path):
    try:
        return path.relative_to(base)
    except Exception:
        return path.name


def main():
    args = parse_args()

    # 固定的两个根目录
    roots = [
        Path("datasets/output/segmentation_outputs/D01_LAPS").resolve(),
        Path("datasets/output/segmentation_outputs/D02_LAPS").resolve(),
    ]

    existing_roots = [r for r in roots if r.exists()]
    if not existing_roots:
        print("[ERR] 未找到任何根目录：D01_LAPS / D02_LAPS 均不存在")
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.report_path:
        report_path = Path(args.report_path).resolve()
    else:
        report_path = Path("video_action_segmenter/inference_outputs").resolve() / (
            f"segmentation_outputs_timebase_correction_report_{ts}.csv"
        )
    ensure_parent_dir(report_path)

    if args.backup_dir:
        backup_root = Path(args.backup_dir).resolve()
    else:
        backup_root = Path(f"datasets/output/segmentation_outputs_backup_{ts}").resolve()
    backup_root.mkdir(parents=True, exist_ok=True)

    # 用于生成相对路径，备份与日志更清晰
    common_base = Path("datasets/output/segmentation_outputs").resolve()

    files = list(find_segment_files(existing_roots))
    total = len(files)
    print(f"[Info] 待处理文件数：{total}")

    processed = 0
    skipped = 0
    warned = 0
    errored = 0

    rows = []

    for idx, fpath in enumerate(files, 1):
        try:
            rel_to_common = try_relative_to(fpath.resolve(), common_base)
        except Exception:
            rel_to_common = fpath.name

        status = "ok"
        warn_msgs = []

        # 读取 json
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            status = f"error: load fail: {e}"
            errored += 1
            rows.append({
                "file": str(rel_to_common),
                "video": None,
                "orig_fps": None,
                "target_fps": None,
                "correction_factor": None,
                "video_duration_sec": None,
                "min_start_old": None,
                "max_end_old": None,
                "min_start_new": None,
                "max_end_new": None,
                "status": status,
                "warnings": None,
            })
            print(f"[ERR] ({idx}/{total}) {rel_to_common}: {status}")
            continue

        video_name = meta.get("video")
        segs = meta.get("segments")
        if not isinstance(segs, list):
            status = "skip: segments missing or not list"
            skipped += 1
            rows.append({
                "file": str(rel_to_common),
                "video": video_name,
                "orig_fps": None,
                "target_fps": None,
                "correction_factor": None,
                "video_duration_sec": meta.get("video_duration_sec", None),
                "min_start_old": None,
                "max_end_old": None,
                "min_start_new": None,
                "max_end_new": None,
                "status": status,
                "warnings": None,
            })
            print(f"[SKIP] ({idx}/{total}) {rel_to_common}: {status}")
            continue

        orig_fps, target_fps, video_duration_sec = resolve_fps_fields(meta)
        if orig_fps is None or orig_fps <= 0:
            status = "skip: invalid orig_fps"
            skipped += 1
            rows.append({
                "file": str(rel_to_common),
                "video": video_name,
                "orig_fps": orig_fps,
                "target_fps": target_fps,
                "correction_factor": None,
                "video_duration_sec": video_duration_sec,
                "min_start_old": None,
                "max_end_old": None,
                "min_start_new": None,
                "max_end_new": None,
                "status": status,
                "warnings": None,
            })
            print(f"[SKIP] ({idx}/{total}) {rel_to_common}: {status}")
            continue
        if target_fps is None or target_fps <= 0:
            status = "skip: invalid target_fps"
            skipped += 1
            rows.append({
                "file": str(rel_to_common),
                "video": video_name,
                "orig_fps": orig_fps,
                "target_fps": target_fps,
                "correction_factor": None,
                "video_duration_sec": video_duration_sec,
                "min_start_old": None,
                "max_end_old": None,
                "min_start_new": None,
                "max_end_new": None,
                "status": status,
                "warnings": None,
            })
            print(f"[SKIP] ({idx}/{total}) {rel_to_common}: {status}")
            continue

        factor = float(target_fps) / float(orig_fps)

        # 统计修正前范围
        try:
            old_starts = [float(s.get("start_sec", 0.0)) for s in segs if isinstance(s, dict)]
            old_ends = [float(s.get("end_sec", 0.0)) for s in segs if isinstance(s, dict)]
            min_start_old = min(old_starts) if old_starts else None
            max_end_old = max(old_ends) if old_ends else None
        except Exception:
            min_start_old = None
            max_end_old = None

        # 应用修正
        for s in segs:
            if not isinstance(s, dict):
                continue
            if "start_sec" in s and s["start_sec"] is not None:
                try:
                    s["start_sec"] = float(s["start_sec"]) * factor
                except Exception:
                    pass
            if "end_sec" in s and s["end_sec"] is not None:
                try:
                    s["end_sec"] = float(s["end_sec"]) * factor
                except Exception:
                    pass

        # 修正后范围与校验
        try:
            new_starts = [float(s.get("start_sec", 0.0)) for s in segs if isinstance(s, dict)]
            new_ends = [float(s.get("end_sec", 0.0)) for s in segs if isinstance(s, dict)]
            min_start_new = min(new_starts) if new_starts else None
            max_end_new = max(new_ends) if new_ends else None
        except Exception:
            min_start_new = None
            max_end_new = None

        if video_duration_sec is not None and max_end_new is not None:
            if max_end_new > float(video_duration_sec) + 1e-6:
                warn = (
                    f"max_end_new ({max_end_new:.6f}s) > video_duration_sec ({video_duration_sec:.6f}s)"
                )
                warn_msgs.append(warn)
                warned += 1
                print(f"[WARN] ({idx}/{total}) {rel_to_common}: {warn}")

        # 原地覆盖（写前先备份）
        if not args.dry_run:
            backup_path = backup_root / try_relative_to(fpath.resolve(), common_base)
            copy_with_dirs(fpath, backup_path)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        processed += 1
        rows.append({
            "file": str(rel_to_common),
            "video": video_name,
            "orig_fps": orig_fps,
            "target_fps": target_fps,
            "correction_factor": factor,
            "video_duration_sec": video_duration_sec,
            "min_start_old": min_start_old,
            "max_end_old": max_end_old,
            "min_start_new": min_start_new,
            "max_end_new": max_end_new,
            "status": status,
            "warnings": " | ".join(warn_msgs) if warn_msgs else "",
        })
        print(
            f"[OK]   ({idx}/{total}) {rel_to_common}: factor={factor:.6f} old[max_end]={max_end_old} -> new[max_end]={max_end_new}"
        )

    # 写报告
    fieldnames = [
        "file",
        "video",
        "orig_fps",
        "target_fps",
        "correction_factor",
        "video_duration_sec",
        "min_start_old",
        "max_end_old",
        "min_start_new",
        "max_end_new",
        "status",
        "warnings",
    ]
    ensure_parent_dir(report_path)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\n=== 修正完成 ===")
    print(f"处理成功: {processed}, 跳过: {skipped}, 警告: {warned}, 错误: {errored}")
    print(f"报告已写入: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())

