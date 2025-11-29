#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_hw.py
- 递归扫描 root_dir 寻找原始视频，读取分辨率 (H, W)
- 若找不到视频，则尝试从 HDF5 读取：
  - 优先 'video' 数据集 (THWC)
  - 否则从 'root/<view>/tracks' 的像素坐标估计 (ceil(max)+1)
- 将检测到的分辨率写回到 cfg_yaml 的 img_shape 字段

用法：
  python amplify/tools/extract_hw.py \
    --root_dir ./data/YOUR_DATA_PATH
    --config amplify/cfg/train_motion_tokenizer.yaml
"""
import argparse
import glob
import math
import os
from typing import Optional, Tuple

import yaml

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]


def find_first_video(root_dir: str) -> Optional[str]:
    for ext in VIDEO_EXTS:
        paths = sorted(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
        if paths:
            return paths[0]
    return None


def get_hw_from_video(video_path: str) -> Optional[Tuple[int, int]]:
    try:
        import cv2  # opencv-python
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width > 0 and height > 0:
            return height, width
    except Exception:
        pass

    # fallback: imageio 读取第一帧
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        frame = reader.get_data(0)
        # frame: H, W, C
        h, w = frame.shape[:2]
        reader.close()
        return h, w
    except Exception:
        return None


def find_first_h5(root_dir: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.hdf5"), recursive=True))
    return paths[0] if paths else None


def get_hw_from_h5(h5_path: str) -> Optional[Tuple[int, int]]:
    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            # 1) 直接找 'video' 数据集（顶层或 root/<view>/video）
            if "video" in f:
                data = f["video"]
                if data.ndim >= 3:
                    h, w = int(data.shape[-3]), int(data.shape[-2])
                    return h, w
            if "root" in f:
                root = f["root"]
                for view in root.keys():
                    if f"root/{view}/video" in f:
                        data = f[f"root/{view}/video"]
                        if data.ndim >= 3:
                            h, w = int(data.shape[-3]), int(data.shape[-2])
                            return h, w

            # 2) 用 'tracks' 像素坐标估计 H、W（reinit: (T, horizon, N, 2)）
            if "root" in f:
                for view in f["root"].keys():
                    key = f"root/{view}/tracks"
                    if key in f:
                        dset = f[key]
                        # 为避免大 IO，抽样若干时间步
                        T = dset.shape[0]
                        step = max(1, T // 64)
                        max_row = 0.0
                        max_col = 0.0
                        for t in range(0, T, step):
                            arr = dset[t]  # (horizon, N, 2)
                            if arr.size == 0:
                                continue
                            # arr[..., 0]: row, arr[..., 1]: col
                            max_row = max(max_row, float(arr[..., 0].max()))
                            max_col = max(max_col, float(arr[..., 1].max()))
                        # 估计高度与宽度（像素坐标从 0 开始）
                        h = int(math.ceil(max_row + 1)) if max_row > 0 else None
                        w = int(math.ceil(max_col + 1)) if max_col > 0 else None
                        if h is not None and w is not None:
                            return h, w
    except Exception:
        return None
    return None


def write_img_shape_to_yaml(cfg_path: str, h: int, w: int) -> None:
    """
    优先进行文本替换以保留注释/格式：
      - 处理形如: img_shape: [128, 128]
      - 次选处理: 多行列表样式
    若匹配失败，再回退到 YAML 解析写回（可能丢失注释）。
    """
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        updated = False
        for i, line in enumerate(lines):
            # 去掉前后空白便于匹配，但保留原缩进
            stripped = line.strip()
            if stripped.startswith("img_shape:"):
                indent = line[: line.find("i")]  # 估算原缩进
                # 情况1：同一行内联列表
                if "[" in line and "]" in line:
                    prefix = line[: line.find("[")]
                    lines[i] = f"{prefix}[{int(h)}, {int(w)}]\n"
                    updated = True
                    break
                # 情况2：多行列表，例如：
                # img_shape:
                #   - 128
                #   - 128
                # 尝试覆盖接下来的两行
                next_i = i + 1
                if next_i < len(lines):
                    # 计算下一行缩进（若存在）
                    next_indent = lines[next_i][: len(lines[next_i]) - len(lines[next_i].lstrip(" "))]
                    # 写两行，若没有足够行则追加
                    if next_i + 1 >= len(lines):
                        lines.append("")
                    lines[next_i] = f"{next_indent}- {int(h)}\n"
                    lines[next_i + 1] = f"{next_indent}- {int(w)}\n"
                    updated = True
                    break

        if updated:
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"Updated {cfg_path} (text replace): img_shape -> [{h}, {w}]")
            return
    except Exception:
        pass

    # 回退：YAML 解析写回（可能丢失注释）
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["img_shape"] = [int(h), int(w)]
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"Updated {cfg_path} (yaml dump): img_shape -> [{h}, {w}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="包含原始视频或HDF5的根目录（递归扫描）")
    parser.add_argument("--config", type=str, default="amplify/cfg/train_motion_tokenizer.yaml", help="要写入的训练配置YAML路径")
    args = parser.parse_args()

    print(f"Scanning root_dir: {args.root_dir}")

    # 1) 优先从原始视频读取
    vid = find_first_video(args.root_dir)
    if vid is not None:
        print(f"Found video: {vid}")
        hw = get_hw_from_video(vid)
        if hw is not None:
            h, w = hw
            print(f"Video resolution: H={h}, W={w}")
            write_img_shape_to_yaml(args.config, h, w)
            return
        else:
            print("Failed to read video resolution, fallback to HDF5...")

    # 2) HDF5 读取
    h5 = find_first_h5(args.root_dir)
    if h5 is not None:
        print(f"Found hdf5: {h5}")
        hw = get_hw_from_h5(h5)
        if hw is not None:
            h, w = hw
            print(f"Estimated resolution from HDF5: H={h}, W={w}")
            write_img_shape_to_yaml(args.config, h, w)
            return
        else:
            print("Failed to infer resolution from HDF5.")

    raise SystemExit("ERROR: 未能从 root_dir 中找到视频或可用于估计分辨率的 HDF5 数据集。")


if __name__ == "__main__":
    main()
