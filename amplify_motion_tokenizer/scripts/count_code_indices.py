#!/usr/bin/env python3
"""统计指定目录下所有 `.json` 文件数量的脚本。

用法示例：
    python amplify_motion_tokenizer/scripts/count_code_indices.py \
      --root /media/johnny/Data/data_motion_tokenizer/online_inference_results

可以使用 `--list` 列出所有文件的绝对路径。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple


def iter_json_files(root: Path) -> Iterable[Path]:
    """遍历 `root` 下所有以 `.json` 结尾的文件。"""
    if not root.exists():
        return []

    return (path for path in root.rglob("*.json") if path.is_file())


def count_json_files(root: Path) -> Tuple[int, list[Path]]:
    """返回数量和文件列表。"""
    files = list(iter_json_files(root))
    return len(files), files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计指定目录下的 JSON 文件数量"
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="待统计的根目录"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="是否打印所有找到的文件路径"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count, files = count_json_files(args.root)

    print(f"在 {args.root} 下共找到 {count} 个 JSON 文件。")

    if args.list and files:
        print("\n文件列表：")
        for path in files:
            print(path)


if __name__ == "__main__":
    main()
