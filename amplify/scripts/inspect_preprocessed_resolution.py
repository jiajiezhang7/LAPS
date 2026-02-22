#!/usr/bin/env python
"""Inspect preprocessed Motion Tokenizer data to infer the working image size.

Given either a preprocessing output directory or a single HDF5 file, the script:

1. Reads the accompanying `config.yaml` (if present) to show parameters such as
   `resize_shorter` and `target_fps`.
2. Samples one or more `.hdf5` files and reports the raw track tensor shape
   as well as the observed coordinate ranges, which indicate the effective
   image height/width used during preprocessing.

Example:
    python amplify/scripts/inspect_preprocessed_resolution.py \
        --path /media/.../preprocessed_data_d01_train_partly \
        --samples 3 --frames 8 --view default
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import h5py
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


def _load_config(config_path: Path) -> Optional[dict]:
    if not config_path.exists():
        return None
    if yaml is None:
        print(f"[WARN] PyYAML 未安装，跳过读取 {config_path}", file=sys.stderr)
        return None
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_h5_files(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() in {".h5", ".hdf5"}:
        yield root
        return
    for path in sorted(root.rglob("*.hdf5")):
        if path.is_file():
            yield path


def _inspect_h5(path: Path, view: Optional[str], frames: int) -> dict:
    info: dict = {"file": str(path)}
    with h5py.File(path, "r") as f:
        if "root" not in f:
            info["error"] = "缺少 root 组"
            return info
        available_views: List[str] = list(f["root"].keys())
        if not available_views:
            info["error"] = "root 下没有视角"
            return info
        use_view = view or available_views[0]
        if use_view not in available_views:
            info["error"] = f"视角 {use_view} 不存在，可选: {available_views}"
            return info
        info["view"] = use_view

        grp = f[f"root/{use_view}"]
        if "tracks" not in grp:
            info["error"] = "缺少 tracks 数据集"
            return info
        dset = grp["tracks"]
        info["tracks_shape"] = tuple(dset.shape)

        take_t = min(frames, dset.shape[0])
        sample = dset[:take_t]
        rows = sample[..., 0]
        cols = sample[..., 1]
        row_min, row_max = float(rows.min()), float(rows.max())
        col_min, col_max = float(cols.min()), float(cols.max())
        approx_height = int(np.ceil(row_max + 1.0))
        approx_width = int(np.ceil(col_max + 1.0))
        info["row_range"] = [row_min, row_max]
        info["col_range"] = [col_min, col_max]
        info["approx_height"] = approx_height
        info["approx_width"] = approx_width
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="检查预处理后的轨迹分辨率")
    parser.add_argument("--path", required=True, type=Path,
                        help="单个 HDF5 文件或包含 config.yaml 的预处理输出目录")
    parser.add_argument("--view", default=None, help="指定视角，默认为文件中第一个视角")
    parser.add_argument("--samples", type=int, default=1,
                        help="随机/顺序抽检的 HDF5 数量；<=0 表示检查全部")
    parser.add_argument("--frames", type=int, default=16,
                        help="每个 HDF5 采样前多少帧用于统计坐标范围")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果，便于脚本化处理")

    args = parser.parse_args()
    path: Path = args.path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

    result: dict = {"root": str(path)}

    # Config.yaml
    config_path = path / "config.yaml" if path.is_dir() else path.parent / "config.yaml"
    config = _load_config(config_path)
    if config is not None:
        result["config"] = {
            "resize_shorter": config.get("resize_shorter"),
            "target_fps": config.get("target_fps"),
            "view_name": config.get("view_name"),
            "source": config.get("source"),
        }
    else:
        result["config"] = None

    files = list(_iter_h5_files(path))
    if not files:
        inspections: List[dict] = []
    else:
        limit = None if args.samples <= 0 else max(1, args.samples)
        chosen = files if limit is None else files[:limit]
        inspections = []
        for fp in chosen:
            try:
                inspections.append(_inspect_h5(fp, args.view, args.frames))
            except Exception as exc:  # noqa: BLE001
                inspections.append({"file": str(fp), "error": str(exc)})
    result["h5_files"] = inspections

    # 汇总统计
    heights = [item["approx_height"] for item in inspections if "approx_height" in item]
    widths = [item["approx_width"] for item in inspections if "approx_width" in item]
    if heights and widths:
        result["summary"] = {
            "count": len(heights),
            "height_min": int(min(heights)),
            "height_max": int(max(heights)),
            "height_mean": float(np.mean(heights)),
            "width_min": int(min(widths)),
            "width_max": int(max(widths)),
            "width_mean": float(np.mean(widths)),
        }
    else:
        result["summary"] = None

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"检查根目录/文件: {result['root']}")
        if result["config"]:
            cfg = result["config"]
            print("配置摘要:")
            print(f"  resize_shorter: {cfg['resize_shorter']}")
            print(f"  target_fps:    {cfg['target_fps']}")
            print(f"  view_name:     {cfg['view_name']}")
            print(f"  source:        {cfg['source']}")
        else:
            print("未找到 config.yaml 或无法解析（缺少 PyYAML）。")

        if not result["h5_files"]:
            print("未找到任何 HDF5 文件。")
        else:
            if result["summary"]:
                summary = result["summary"]
                print("尺寸摘要:")
                print(f"  样本数: {summary['count']}")
                print(f"  高度 min/max/mean: {summary['height_min']} / {summary['height_max']} / {summary['height_mean']:.2f}")
                print(f"  宽度 min/max/mean: {summary['width_min']} / {summary['width_max']} / {summary['width_mean']:.2f}")
            print("样本 HDF5 检查:")
            for item in result["h5_files"]:
                print(f"- 文件: {item['file']}")
                if "error" in item:
                    print(f"  错误: {item['error']}")
                    continue
                print(f"  视角: {item['view']}")
                print(f"  tracks 形状: {item['tracks_shape']}")
                print(f"  行坐标范围: {item['row_range']}")
                print(f"  列坐标范围: {item['col_range']}")
                print(f"  估计高度: {item['approx_height']}")
                print(f"  估计宽度: {item['approx_width']}")


if __name__ == "__main__":
    main()
