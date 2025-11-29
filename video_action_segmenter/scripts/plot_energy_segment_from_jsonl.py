#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从离线 energy JSONL 中随机抽取连续片段并绘制能量曲线（论文风格）。

- 默认读取：./data/YOUR_DATA_PATH
- 片段长度：2000 个窗口（若不足则使用最长的连续片段）
- 绘图风格与参数参照 params.yaml::energy（viz_style/theme/y_min/y_max/plot_size/smoothing 等）
- 使用现有绘图函数：draw_energy_plot_enhanced / draw_energy_plot_enhanced_dual

用法示例：
  python -m video_action_segmenter.scripts.plot_energy_segment_from_jsonl \
    --jsonl ./data/YOUR_DATA_PATH
    --params ./video_action_segmenter/params_d02.yaml \
    --segment-length 50 \
    --seed 100

建议在 conda 环境 amplify_mt 中运行：
  conda activate amplify_mt
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from video_action_segmenter.stream_utils import (
    load_config,
    apply_smoothing_1d,
    draw_energy_plot_enhanced,
    draw_energy_plot_enhanced_dual,
)


def read_energy_jsonl(jsonl_path: Path) -> Tuple[List[int], List[float], Optional[str], Optional[str]]:
    """读取 JSONL 文件，返回 (windows, energies, source, mode)。

    若 JSONL 中存在不同的 source/mode，将返回遇到的第一组非空值。
    """
    windows: List[int] = []
    energies: List[float] = []
    src: Optional[str] = None
    mode: Optional[str] = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                w = int(rec.get("window"))
                e = float(rec.get("energy"))
                if np.isfinite(e):
                    windows.append(w)
                    energies.append(e)
                if src is None and rec.get("source"):
                    src = str(rec.get("source"))
                if mode is None and rec.get("mode"):
                    mode = str(rec.get("mode"))
            except Exception:
                # 忽略解析失败的行
                continue
    return windows, energies, src, mode


def find_consecutive_runs(windows: List[int]) -> List[Tuple[int, int]]:
    """在 window id 列表中查找连续自增（+1）的区间，返回 (start_idx, length) 列表。"""
    runs: List[Tuple[int, int]] = []
    if not windows:
        return runs
    start = 0
    for i in range(1, len(windows)):
        if windows[i] != windows[i - 1] + 1:
            runs.append((start, i - start))
            start = i
    runs.append((start, len(windows) - start))
    return runs


def select_random_segment(
    windows: List[int],
    segment_len: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """基于 window id 的连续性，随机选择一个长度为 segment_len 的连续片段。

    返回的是在数组中的 [start_idx, end_idx) 下标（左闭右开）。
    若没有任意一个区间长度 >= segment_len，则退化为选择“最长连续区间”。
    """
    runs = find_consecutive_runs(windows)
    if not runs:
        return 0, min(len(windows), segment_len)

    # 所有满足长度的区间
    candidates = [r for r in runs if r[1] >= segment_len]
    if candidates:
        start_run_idx, run_len = rng.choice(candidates)
        max_offset = run_len - segment_len
        offset = rng.randint(0, max_offset)
        start_idx = start_run_idx + offset
        end_idx = start_idx + segment_len
        return start_idx, end_idx

    # 否则选择最长区间
    start_run_idx, run_len = max(runs, key=lambda x: x[1])
    return start_run_idx, start_run_idx + run_len


def main():
    parser = argparse.ArgumentParser(description="Plot a random consecutive segment of energy curve from JSONL (paper style)")
    parser.add_argument(
        "--jsonl",
        type=str,
        default="./data/YOUR_DATA_PATH",
        help="energy JSONL 路径",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "params.yaml"),
        help="用于读取能量曲线与平滑配置的 params.yaml 路径",
    )
    parser.add_argument("--segment-length", type=int, default=2000, help="连续窗口片段长度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default="", help="输出图片路径（默认保存在 JSONL 同目录）")
    parser.add_argument("--title", type=str, default="", help="图标题（留空则自动生成）")
    parser.add_argument("--start-index", type=int, default=-1, help="可选：指定片段起始的数组下标（调试用）")

    args = parser.parse_args()

    rng = random.Random(args.seed)

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL 不存在: {jsonl_path}")

    windows, energies, src, mode = read_energy_jsonl(jsonl_path)
    if len(energies) == 0:
        raise RuntimeError("JSONL 中没有可用的 energy 记录")

    # 读取 params.yaml 的 energy 配置
    cfg = load_config(args.params)
    energy_cfg = cfg.get("energy", {}) if isinstance(cfg.get("energy", {}), dict) else {}
    seg_cfg = cfg.get("segmentation", {}) if isinstance(cfg.get("segmentation", {}), dict) else {}

    viz_style = str(energy_cfg.get("viz_style", energy_cfg.get("energy_viz_style", "enhanced"))).lower()
    theme = str(energy_cfg.get("theme", energy_cfg.get("energy_theme", "academic_blue")))

    # 轴范围与画布大小
    y_min = energy_cfg.get("y_min", None)
    y_max = energy_cfg.get("y_max", None)
    try:
        y_min = float(y_min) if y_min is not None else None
    except Exception:
        y_min = None
    try:
        y_max = float(y_max) if y_max is not None else None
    except Exception:
        y_max = None
    width = int(energy_cfg.get("plot_width", 800))
    height = int(energy_cfg.get("plot_height", 240))
    height = max(120, int(height * 0.85))

    # 平滑配置
    smoothing_cfg = energy_cfg.get("smoothing", {}) if isinstance(energy_cfg.get("smoothing", {}), dict) else {}
    smoothing_enable = bool(smoothing_cfg.get("enable", False))
    smoothing_method = str(smoothing_cfg.get("method", "ema")).lower()
    try:
        smoothing_alpha = float(smoothing_cfg.get("alpha", 0.4))
    except Exception:
        smoothing_alpha = 0.4
    try:
        smoothing_window = int(smoothing_cfg.get("window", 3))
    except Exception:
        smoothing_window = 3
    smoothing_visualize_both = bool(smoothing_cfg.get("visualize_both", True))

    # 阈值线配置（动作启动/结束阈值）
    threshold_lines = []
    try:
        thr_on = float(seg_cfg.get("threshold", float("nan")))
    except Exception:
        thr_on = float("nan")
    try:
        hysteresis_ratio = float(seg_cfg.get("hysteresis_ratio", float("nan")))
    except Exception:
        hysteresis_ratio = float("nan")
    if np.isfinite(thr_on):
        threshold_lines.append({
            "value": thr_on,
            "label": "Thr_on",
            "color": (40, 220, 255),  # 浅黄（BGR）
            "thickness": 2,
            "style": "dashed",
        })
        if np.isfinite(hysteresis_ratio):
            thr_off = thr_on * hysteresis_ratio
            if np.isfinite(thr_off):
                threshold_lines.append({
                    "value": thr_off,
                    "label": "Thr_off",
                    "color": (40, 220, 255),
                    "thickness": 2,
                    "style": "dashed",
                })

    # 选择随机连续片段
    if args.start_index is not None and args.start_index >= 0:
        start_idx = int(args.start_index)
        end_idx = min(len(energies), start_idx + int(args.segment_length))
    else:
        start_idx, end_idx = select_random_segment(windows, int(args.segment_length), rng)

    energies_np = np.asarray(energies, dtype=np.float32)

    # 为了因果平滑具有“预热”效果，先对整条序列做平滑，然后再切片
    smooth_np: Optional[np.ndarray] = None
    if smoothing_enable:
        smooth_np = apply_smoothing_1d(energies_np, method=smoothing_method, alpha=smoothing_alpha, window=smoothing_window)

    seg_raw = energies_np[start_idx:end_idx]
    seg_smooth = smooth_np[start_idx:end_idx] if smoothing_enable and smooth_np is not None else None

    # 构造标题
    if args.title:
        title = args.title
    else:
        window_start = windows[start_idx] if start_idx < len(windows) else start_idx
        window_end = windows[end_idx - 1] if end_idx - 1 < len(windows) else end_idx - 1
        total_windows = end_idx - start_idx
        title = f"Energy Curve, window={window_start}-{window_end}, totally {total_windows}"

    # 绘图（增强风格）
    if viz_style in ("enhanced", "paper", "academic"):
        if smoothing_enable and smoothing_visualize_both and seg_smooth is not None:
            img = draw_energy_plot_enhanced_dual(
                raw_values=seg_raw.tolist(),
                smooth_values=seg_smooth.tolist(),
                width=width,
                height=height,
                y_min=y_min,
                y_max=y_max,
                theme=theme,
                show_grid=True,
                show_labels=True,
                show_legend=True,
                show_statistics=True,
                title=title,
                threshold_lines=threshold_lines,
            )
        else:
            values_to_draw = seg_smooth if (smoothing_enable and seg_smooth is not None) else seg_raw
            img = draw_energy_plot_enhanced(
                values=values_to_draw.tolist(),
                width=width,
                height=height,
                y_min=y_min,
                y_max=y_max,
                theme=theme,
                show_grid=True,
                show_labels=True,
                show_statistics=True,
                title=title,
                threshold_lines=threshold_lines,
            )
    else:
        # 退化到基础样式（本脚本主打增强样式，基础样式仅作为兜底）
        from video_action_segmenter.stream_utils import draw_energy_plot
        img = draw_energy_plot(values=seg_raw.tolist(), width=width, height=height, y_min=y_min, y_max=y_max)

    # 生成输出路径
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        figures_dir = Path(__file__).resolve().parents[1] / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        jstem = jsonl_path.stem
        out_name = f"{jstem}_seg_{windows[start_idx]:06d}-{windows[end_idx-1]:06d}_n{end_idx-start_idx}.png"
        out_path = figures_dir / out_name

    # 保存图像（BGR）
    cv2.imwrite(str(out_path), img)

    print("================ Plot Energy Segment ================")
    print(f"JSONL : {jsonl_path}")
    print(f"Range : windows {windows[start_idx]} - {windows[end_idx-1]} (n={end_idx-start_idx})  idx[{start_idx}:{end_idx}]")
    print(f"Style : {viz_style} | theme={theme} | size=({width}x{height}) | ylim=({y_min},{y_max})")
    if smoothing_enable:
        print(f"Smooth: enable method={smoothing_method} alpha={smoothing_alpha} window={smoothing_window} visualize_both={smoothing_visualize_both}")
    else:
        print("Smooth: disabled")
    print(f"Output: {out_path}")
    print("=====================================================")


if __name__ == "__main__":
    main()
