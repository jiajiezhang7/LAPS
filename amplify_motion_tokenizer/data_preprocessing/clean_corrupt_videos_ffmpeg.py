#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 ffmpeg 检测并删除损坏/不可读的视频文件。"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple

from tqdm import tqdm

KEYWORDS = [
    "invalid data found when processing input",
    "error reading header",
    "contradictory stsc and stco",
    "contradictionary stsc and stco",
]

DEFAULT_EXTS = ("mp4", "mov", "m4v", "avi", "mkv", "3gp")


def ensure_ffmpeg():
    if not shutil.which("ffmpeg"):
        print("[FATAL] 未找到 ffmpeg，请先安装并加入 PATH。", file=sys.stderr)
        sys.exit(2)


def ffmpeg_probe_first_frame(path: Path, timeout: float = 15.0) -> Tuple[int, str]:
    cmd = [
        "ffmpeg", "-v", "error", "-hide_banner", "-nostats", "-nostdin",
        "-i", str(path), "-frames:v", "1", "-f", "null", "-"
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return p.returncode, p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as e:
        return 1, f"exception: {e}"


def looks_corrupt(returncode: int, stderr: str) -> bool:
    if returncode != 0:
        return True
    s = (stderr or "").lower()
    return any(k in s for k in KEYWORDS)


def iter_videos(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    """遍历目录，生成所有符合扩展名的视频文件路径。"""
    lowers = {("." + e.lower().lstrip(".")) for e in exts}

    def _onerror(e):
        print(f"[WARN] walk 访问错误: {e}", flush=True)

    for dirpath, _, filenames in os.walk(root, onerror=_onerror):
        for fname in filenames:
            p = Path(dirpath) / fname
            try:
                if p.suffix.lower() in lowers:
                    yield p
            except Exception as e:
                print(f"[WARN] 跳过异常文件名: {p} -> {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="用 ffmpeg 检测并删除损坏/不可读的视频文件")
    parser.add_argument("--root", type=str, default="/media/johnny/Data/data_motion_tokenizer/raw_videos_d02_910", help="要遍历的根目录")
    parser.add_argument("--exts", type=str, default=",".join(DEFAULT_EXTS), help="匹配扩展名，逗号分隔")
    parser.add_argument("--timeout", type=float, default=15.0, help="ffmpeg 单文件超时秒数")
    parser.add_argument("--dry-run", action="store_true", help="预演：只打印不删除")
    parser.add_argument("--verbose", action="store_true", help="输出每个文件的检测过程")
    parser.add_argument("--max-files", type=int, default=0, help="最多处理多少个文件（0 表示不限制）")
    args = parser.parse_args()

    ensure_ffmpeg()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"[FATAL] 非法目录: {root}", file=sys.stderr)
        sys.exit(2)

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    total = ok = bad = deleted = zero = 0

    print(f"[INFO] 扫描目录: {root}", flush=True)
    print(f"[INFO] 扩展名: {exts}", flush=True)
    print(
        f"[INFO] 参数: timeout={args.timeout}s, dry_run={args.dry_run}, verbose={args.verbose}, "
        f"max_files={args.max_files}",
        flush=True,
    )

    print("[INFO] 正在收集文件列表...", flush=True)
    video_files = list(iter_videos(root, exts))
    total = len(video_files)
    print(f"[INFO] 发现 {total} 个视频文件，开始处理...", flush=True)

    if args.max_files and args.max_files > 0:
        video_files = video_files[:args.max_files]
        print(f"[INFO] 已根据 max_files={args.max_files} 限制处理文件数量。", flush=True)

    try:
        with tqdm(total=len(video_files), desc="检查视频", unit="file") as pbar:
            for f in video_files:
                pbar.set_postfix(ok=ok, bad=bad, zero=zero, deleted=deleted)

                if args.verbose:
                    pbar.write(f"[SCAN] {f}", file=sys.stderr)

                try:
                    size = f.stat().st_size
                except Exception:
                    size = -1

                if size == 0:
                    zero += 1
                    bad += 1
                    if args.dry_run:
                        pbar.write(f"[ZERO][DRY] 将删除: {f}", file=sys.stderr)
                    else:
                        try:
                            os.remove(f)
                            deleted += 1
                            pbar.write(f"[ZERO] 已删除: {f}", file=sys.stderr)
                        except Exception as e:
                            pbar.write(f"[WARN] 删除失败: {f} -> {e}", file=sys.stderr)
                    pbar.update(1)
                    continue

                rc, err = ffmpeg_probe_first_frame(f, args.timeout)
                if args.verbose:
                    preview = " | ".join((err or "").strip().splitlines()[:2])
                    pbar.write(f"[FFMPEG] file={f} rc={rc} err={preview}", file=sys.stderr)

                if looks_corrupt(rc, err):
                    bad += 1
                    if args.dry_run:
                        preview = " | ".join((err or "").strip().splitlines()[:2])
                        pbar.write(f"[BAD][DRY] 将删除: {f}; rc={rc}; err={preview}", file=sys.stderr)
                    else:
                        try:
                            os.remove(f)
                            deleted += 1
                            pbar.write(f"[BAD] 已删除: {f}", file=sys.stderr)
                        except Exception as e:
                            pbar.write(f"[WARN] 删除失败: {f} -> {e}", file=sys.stderr)
                else:
                    ok += 1
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] 收到 Ctrl-C，中断扫描，输出当前统计...", flush=True)

    print("\n[SUMMARY]", flush=True)
    print(f"  总计: {total}", flush=True)
    print(f"  正常: {ok}", flush=True)
    print(f"  判坏: {bad} (其中 0 字节: {zero})", flush=True)
    if args.dry_run:
        print("  预演模式：未实际删除。", flush=True)
    else:
        print(f"  已删除: {deleted}", flush=True)


if __name__ == "__main__":
    main()
