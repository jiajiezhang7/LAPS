#!/usr/bin/env python3
"""
批量提取 I3D 特征
Run in abd_env: conda run -n abd_env python comapred_algorithm/ABD/batch_extract_i3d.py --view D01
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
from features_i3d import extract_i3d_features_for_video


def list_videos(input_dir: Path) -> list[Path]:
    """列出目录中的所有视频文件"""
    vids = []
    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        vids.extend(sorted(input_dir.glob(f"*{ext}")))
    return vids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("批量提取 I3D 特征")
    parser.add_argument("--view", type=str, required=True, choices=["D01", "D02"], help="视角")
    parser.add_argument("--input-dir", type=str, default=None, help="输入视频目录（默认自动推断）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出特征目录（默认自动推断）")
    parser.add_argument("--device", type=str, default="cuda:0", help="PyTorch 设备")
    parser.add_argument("--clip-duration", type=float, default=2.0, help="每个 clip 的时长（秒）")
    parser.add_argument("--clip-stride", type=float, default=0.4, help="clip 之间的步长（秒）")
    args = parser.parse_args(argv)
    
    # 确定输入输出目录
    if args.input_dir is None:
        input_dir = Path(f"./datasets/gt_raw_videos/{args.view}")
    else:
        input_dir = Path(args.input_dir)
    
    if args.output_dir is None:
        output_dir = Path(f"./comapred_algorithm/ABD/i3d_features/{args.view}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 列出所有视频
    videos = list_videos(input_dir)
    if len(videos) == 0:
        print(f"❌ 没有找到视频文件: {input_dir}")
        return 1
    
    print(f"发现 {len(videos)} 个视频文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    
    # 批量提取
    success_count = 0
    fail_count = 0
    
    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] 处理: {video_path.name}")
        
        # 检查是否已存在
        output_path = output_dir / f"{video_path.stem}.npy"
        if output_path.exists():
            print(f"  ⏭️  跳过（特征已存在）: {output_path}")
            success_count += 1
            continue
        
        # 提取特征
        try:
            features = extract_i3d_features_for_video(
                video_path=video_path,
                device=args.device,
                clip_duration=args.clip_duration,
                clip_stride=args.clip_stride,
            )
            
            if features is None:
                print(f"  ❌ 提取失败")
                fail_count += 1
                continue
            
            # 保存特征
            np.save(output_path, features)
            print(f"  ✅ 成功保存: {output_path}")
            print(f"     特征形状: {features.shape}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            fail_count += 1
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 80)
    print(f"批量提取完成:")
    print(f"  成功: {success_count}/{len(videos)}")
    print(f"  失败: {fail_count}/{len(videos)}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
