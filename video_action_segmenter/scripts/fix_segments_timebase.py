import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import shutil
import csv


def parse_args():
    p = argparse.ArgumentParser(
        description="批量修正 *_segments.json 的时间戳：按 correction_factor = target_fps / orig_fps 缩放 start_sec/end_sec"
    )
    p.add_argument(
        "--root",
        type=str,
        default="datasets/gt_annotations/true_gt",
        help="待遍历的根目录（递归查找 *_segments.json）",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="原地覆盖原文件（默认写入到新的输出目录）",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="datasets/gt_annotations/true_gt_corrected",
        help="输出目录（非 in-place 模式下生效）",
    )
    p.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="原地覆盖前的备份根目录（未指定则自动创建 datasets/gt_annotations/true_gt_backup_<ts>）",
    )
    p.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="修正报告 CSV 输出路径（未指定则写入 video_action_segmenter/inference_outputs/segments_timebase_correction_report_<ts>.csv）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅输出报告与控制台信息，不写入任何文件",
    )
    return p.parse_args()


def safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def resolve_fps_fields(meta: dict):
    """从 json 元数据中解析 orig_fps、target_fps、video_duration_sec。
    兼容不同写法：顶层的 fps/orig_fps，或 segmentation_params 内的 orig_fps/target_fps。
    """
    segp = meta.get("segmentation_params", {}) or {}

    orig_fps = safe_float(meta.get("orig_fps"))
    if orig_fps is None:
        # 兼容早期写法：顶层 "fps" 即原视频 fps
        orig_fps = safe_float(meta.get("fps"))
    if orig_fps is None:
        orig_fps = safe_float(segp.get("orig_fps"))

    target_fps = safe_float(meta.get("target_fps"))
    if target_fps is None:
        target_fps = safe_float(segp.get("target_fps"))

    video_duration_sec = safe_float(meta.get("video_duration_sec"))

    return orig_fps, target_fps, video_duration_sec


def iter_segment_files(root: Path):
    for p in root.rglob("*_segments.json"):
        if p.is_file():
            yield p


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_with_dirs(src: Path, dst: Path):
    ensure_parent_dir(dst)
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERR] 根目录不存在: {root}")
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.report_path:
        report_path = Path(args.report_path).resolve()
    else:
        report_path = Path("video_action_segmenter/inference_outputs").resolve() / f"segments_timebase_correction_report_{ts}.csv"
    ensure_parent_dir(report_path)

    if args.in_place:
        if args.backup_dir:
            backup_root = Path(args.backup_dir).resolve()
        else:
            backup_root = Path("datasets/gt_annotations/true_gt_backup_" + ts).resolve()
        backup_root.mkdir(parents=True, exist_ok=True)
        out_root = None  # 不使用
    else:
        out_root = Path(args.output_dir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        backup_root = None

    rows = []
    files = list(iter_segment_files(root))
    total = len(files)
    print(f"[Info] 将处理 {total} 个文件（根目录: {root}）")

    processed = 0
    skipped = 0
    warned = 0
    errored = 0

    for idx, fpath in enumerate(files, 1):
        rel = fpath.relative_to(root)
        status = "ok"
        warn_msgs = []
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            status = f"error: load fail: {e}"
            errored += 1
            rows.append({
                "file": str(rel), "video": None, "orig_fps": None, "target_fps": None,
                "correction_factor": None, "video_duration_sec": None,
                "min_start_old": None, "max_end_old": None,
                "min_start_new": None, "max_end_new": None,
                "status": status, "warnings": None,
            })
            print(f"[ERR] ({idx}/{total}) {rel}: {status}")
            continue

        video_name = meta.get("video", None)
        segs = meta.get("segments", None)
        if not isinstance(segs, list):
            status = "skip: segments missing or not list"
            skipped += 1
            rows.append({
                "file": str(rel), "video": video_name, "orig_fps": None, "target_fps": None,
                "correction_factor": None, "video_duration_sec": meta.get("video_duration_sec", None),
                "min_start_old": None, "max_end_old": None,
                "min_start_new": None, "max_end_new": None,
                "status": status, "warnings": None,
            })
            print(f"[SKIP] ({idx}/{total}) {rel}: {status}")
            continue

        orig_fps, target_fps, video_duration_sec = resolve_fps_fields(meta)
        if orig_fps is None or orig_fps <= 0:
            status = "skip: invalid orig_fps"
            skipped += 1
            rows.append({
                "file": str(rel), "video": video_name, "orig_fps": orig_fps, "target_fps": target_fps,
                "correction_factor": None, "video_duration_sec": video_duration_sec,
                "min_start_old": None, "max_end_old": None,
                "min_start_new": None, "max_end_new": None,
                "status": status, "warnings": None,
            })
            print(f"[SKIP] ({idx}/{total}) {rel}: {status}")
            continue
        if target_fps is None or target_fps <= 0:
            status = "skip: invalid target_fps"
            skipped += 1
            rows.append({
                "file": str(rel), "video": video_name, "orig_fps": orig_fps, "target_fps": target_fps,
                "correction_factor": None, "video_duration_sec": video_duration_sec,
                "min_start_old": None, "max_end_old": None,
                "min_start_new": None, "max_end_new": None,
                "status": status, "warnings": None,
            })
            print(f"[SKIP] ({idx}/{total}) {rel}: {status}")
            continue

        factor = float(target_fps) / float(orig_fps)

        # 统计修正前后范围
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
                print(f"[WARN] ({idx}/{total}) {rel}: {warn}")

        # 写回或输出
        if not args.dry_run:
            if args.in_place:
                # 先备份
                if backup_root:
                    backup_path = backup_root / rel
                    copy_with_dirs(fpath, backup_path)
                # 覆盖保存
                out_path = fpath
                ensure_parent_dir(out_path)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            else:
                out_path = out_root / rel
                ensure_parent_dir(out_path)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

        processed += 1
        rows.append({
            "file": str(rel),
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
            f"[OK]   ({idx}/{total}) {rel}: factor={factor:.6f} old[max_end]={max_end_old} -> new[max_end]={max_end_new}"
        )

    # 写报告
    fieldnames = [
        "file", "video", "orig_fps", "target_fps", "correction_factor", "video_duration_sec",
        "min_start_old", "max_end_old", "min_start_new", "max_end_new", "status", "warnings",
    ]
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

