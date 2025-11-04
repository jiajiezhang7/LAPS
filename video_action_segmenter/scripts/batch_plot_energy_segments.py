#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量调用 plot_energy_segment_from_jsonl_for_paperteaser.py 生成多张能量曲线图。

用于生成多个随机片段的能量曲线图,以便从中选择最佳的三个动作分割曲线。

用法示例:
  python -m video_action_segmenter.scripts.batch_plot_energy_segments \
    --jsonl /path/to/stream_energy.jsonl \
    --params ./video_action_segmenter/params_d02.yaml \
    --segment-length 100 \
    --num-plots 20 \
    --output-dir ./figures/batch_output

建议在 conda 环境 amplify_mt 中运行:
  conda activate amplify_mt
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_single_plot(
    jsonl_path: str,
    params_path: str,
    segment_length: int,
    seed: int,
    output_path: str,
    viz_style: str = "enhanced",
    theme: str = "academic_blue",
    plot_width: int = 800,
    plot_height: int = 240,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> bool:
    """调用单次绘图脚本。
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        sys.executable,
        "-m",
        "video_action_segmenter.scripts.plot_energy_segment_from_jsonl_for_paperteaser",
        "--jsonl", jsonl_path,
        "--params", params_path,
        "--segment-length", str(segment_length),
        "--seed", str(seed),
        "--output", output_path,
        "--viz-style", viz_style,
        "--theme", theme,
        "--plot-width", str(plot_width),
        "--plot-height", str(plot_height),
    ]
    
    if y_min is not None:
        cmd.extend(["--y-min", str(y_min)])
    if y_max is not None:
        cmd.extend(["--y-max", str(y_max)])
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Seed {seed} failed:")
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="批量生成能量曲线图片（论文风格）"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="energy JSONL 路径",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="./video_action_segmenter/params.yaml",
        help="params.yaml 路径",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=100,
        help="连续窗口片段长度（固定值）",
    )
    parser.add_argument(
        "--num-plots",
        type=int,
        default=20,
        help="生成图片的数量",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="输出目录（默认为 video_action_segmenter/figures/batch_YYYYMMDD_HHMMSS）",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000,
        help="起始随机种子",
    )
    parser.add_argument(
        "--viz-style",
        type=str,
        default="enhanced",
        help="可视化风格",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="academic_blue",
        help="主题风格",
    )
    parser.add_argument(
        "--plot-width",
        type=int,
        default=800,
        help="图片宽度",
    )
    parser.add_argument(
        "--plot-height",
        type=int,
        default=240,
        help="图片高度",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Y轴最小值",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Y轴最大值",
    )
    
    args = parser.parse_args()
    
    # 验证输入文件
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")
    
    params_path = Path(args.params)
    if not params_path.exists():
        raise FileNotFoundError(f"参数文件不存在: {params_path}")
    
    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figures_base = Path(__file__).resolve().parents[1] / "figures"
        output_dir = figures_base / f"batch_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("批量生成能量曲线图")
    print("=" * 60)
    print(f"JSONL 文件    : {jsonl_path}")
    print(f"参数文件      : {params_path}")
    print(f"片段长度      : {args.segment_length}")
    print(f"生成数量      : {args.num_plots}")
    print(f"输出目录      : {output_dir}")
    print(f"起始种子      : {args.seed_start}")
    print(f"可视化风格    : {args.viz_style}")
    print(f"主题          : {args.theme}")
    print(f"图片尺寸      : {args.plot_width}x{args.plot_height}")
    if args.y_min is not None or args.y_max is not None:
        print(f"Y轴范围       : [{args.y_min}, {args.y_max}]")
    print("=" * 60)
    
    # 批量生成
    success_count = 0
    failed_seeds = []
    
    for i in range(args.num_plots):
        seed = args.seed_start + i
        output_name = f"seg_len{args.segment_length}_seed{seed:04d}.png"
        output_path = output_dir / output_name
        
        print(f"\n[{i+1}/{args.num_plots}] 生成图片 (seed={seed})...", end=" ")
        
        success = run_single_plot(
            jsonl_path=str(jsonl_path),
            params_path=str(params_path),
            segment_length=args.segment_length,
            seed=seed,
            output_path=str(output_path),
            viz_style=args.viz_style,
            theme=args.theme,
            plot_width=args.plot_width,
            plot_height=args.plot_height,
            y_min=args.y_min,
            y_max=args.y_max,
        )
        
        if success:
            print(f"✓ {output_name}")
            success_count += 1
        else:
            print(f"✗ 失败")
            failed_seeds.append(seed)
    
    # 总结
    print("\n" + "=" * 60)
    print("批量生成完成")
    print("=" * 60)
    print(f"成功: {success_count}/{args.num_plots}")
    if failed_seeds:
        print(f"失败的种子: {failed_seeds}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 生成索引文件
    index_file = output_dir / "index.txt"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(f"批量生成能量曲线图\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"JSONL 文件: {jsonl_path}\n")
        f.write(f"参数文件: {params_path}\n")
        f.write(f"片段长度: {args.segment_length}\n")
        f.write(f"生成数量: {args.num_plots}\n")
        f.write(f"成功数量: {success_count}\n")
        f.write(f"起始种子: {args.seed_start}\n")
        f.write(f"\n生成的文件:\n")
        for i in range(args.num_plots):
            seed = args.seed_start + i
            output_name = f"seg_len{args.segment_length}_seed{seed:04d}.png"
            status = "✓" if seed not in failed_seeds else "✗"
            f.write(f"{status} {output_name} (seed={seed})\n")
    
    print(f"索引文件已保存: {index_file}")


if __name__ == "__main__":
    main()
