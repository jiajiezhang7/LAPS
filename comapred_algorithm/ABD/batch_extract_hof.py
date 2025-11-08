#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from typing import List

from .features_hof import extract_hof_features_for_video


def list_videos(input_dir: Path) -> List[Path]:
    vids = []
    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        vids.extend(sorted(input_dir.glob(f"*{ext}")))
    return vids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("批量提取 HOF 特征 (CPU/OpenCV)")
    parser.add_argument("--view", type=str, choices=["D01", "D02"], default=None, help="数据视角（可选）")
    parser.add_argument("--input-dir", type=str, default=None, help="输入视频目录（默认依据 view 推断）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出特征目录（默认存放到 ABD/hof_features/{view}）")
    parser.add_argument("--clip-duration", type=float, default=2.0, help="每个 clip 的时长（秒）")
    parser.add_argument("--clip-stride", type=float, default=0.4, help="clip 之间的步长（秒）")
    parser.add_argument("--bins", type=int, default=16, help="HOF 方向直方图的 bin 数")
    args = parser.parse_args(argv)

    # Resolve directories
    if args.input_dir is None:
        if args.view is None:
            print("必须提供 --view 或 --input-dir 之一")
            return 2
        input_dir = Path(f"/home/johnny/action_ws/datasets/gt_raw_videos/{args.view}")
    else:
        input_dir = Path(args.input_dir)

    if args.output_dir is None:
        if args.view is None:
            print("必须提供 --view 或 --output-dir 之一")
            return 2
        output_dir = Path(f"/home/johnny/action_ws/comapred_algorithm/ABD/hof_features/{args.view}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(input_dir)
    if len(videos) == 0:
        print(f"❌ 没有找到视频文件: {input_dir}")
        return 1

    print(f"发现 {len(videos)} 个视频文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

    ok_cnt = 0
    fail_cnt = 0
    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] 处理: {vp.name}")
        out_npy = output_dir / f"{vp.stem}.npy"
        if out_npy.exists():
            print(f"  ⏭️  跳过（已存在）: {out_npy}")
            ok_cnt += 1
            continue
        try:
            X = extract_hof_features_for_video(vp, args.clip_duration, args.clip_stride, args.bins)
            if X is None:
                print("  ❌ 提取失败")
                fail_cnt += 1
                continue
            np.save(out_npy, X.astype(np.float32))
            print(f"  ✅ 成功保存: {out_npy}  形状: {X.shape}")
            ok_cnt += 1
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            fail_cnt += 1
            import traceback
            traceback.print_exc()

    print("=" * 80)
    print(f"完成: 成功 {ok_cnt} / 失败 {fail_cnt}")
    return 0 if fail_cnt == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())

