#!/usr/bin/env python3
"""
从 D02 数据集中随机抽取部分 HDF5 文件，使总样本数与 D01 相当。
保持原始文件不变，将选中的文件复制到新目录。
"""

import h5py
import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict, Tuple


def analyze_hdf5_file(hdf5_path: Path) -> Dict:
    """分析单个 HDF5 文件的样本数"""
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'root' in f and 'default' in f['root']:
                default_group = f['root']['default']
                if 'tracks' in default_group:
                    num_samples = default_group['tracks'].shape[0]
                else:
                    num_samples = 0
            else:
                num_samples = 0
            
            return {
                'path': hdf5_path,
                'num_samples': num_samples,
                'success': True
            }
    except Exception as e:
        return {
            'path': hdf5_path,
            'num_samples': 0,
            'success': False,
            'error': str(e)
        }


def get_all_hdf5_files_with_samples(root_dir: Path) -> List[Dict]:
    """获取所有 HDF5 文件及其样本数"""
    hdf5_files = list(root_dir.rglob("*.hdf5"))
    
    print(f"扫描目录: {root_dir}")
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")
    
    file_info_list = []
    for hdf5_file in tqdm(hdf5_files, desc="分析文件"):
        info = analyze_hdf5_file(hdf5_file)
        if info['success'] and info['num_samples'] > 0:
            file_info_list.append(info)
    
    total_samples = sum(f['num_samples'] for f in file_info_list)
    print(f"有效文件数: {len(file_info_list)}")
    print(f"总样本数: {total_samples:,}")
    
    return file_info_list


def select_files_to_match_target(
    file_info_list: List[Dict],
    target_samples: int,
    seed: int = 42
) -> Tuple[List[Dict], int]:
    """
    随机选择文件，使总样本数接近目标值。
    使用贪心算法：优先选择样本数较少的文件，以便更精确地控制总样本数。
    """
    random.seed(seed)
    
    # 按样本数排序（从小到大）
    sorted_files = sorted(file_info_list, key=lambda x: x['num_samples'])
    
    # 随机打乱顺序（保持样本数分布的随机性）
    random.shuffle(sorted_files)
    
    selected_files = []
    current_samples = 0
    
    for file_info in sorted_files:
        if current_samples >= target_samples:
            break
        
        # 如果添加这个文件不会超出目标太多，就添加
        new_total = current_samples + file_info['num_samples']
        
        # 策略：如果当前已经接近目标（>95%），则只添加小文件
        if current_samples > target_samples * 0.95:
            if file_info['num_samples'] > (target_samples - current_samples):
                continue
        
        selected_files.append(file_info)
        current_samples += file_info['num_samples']
    
    return selected_files, current_samples


def copy_files_to_new_directory(
    selected_files: List[Dict],
    source_root: Path,
    target_root: Path,
    dry_run: bool = False
) -> Dict:
    """
    将选中的文件复制到新目录，保持相对路径结构。
    """
    if not dry_run:
        target_root.mkdir(parents=True, exist_ok=True)
    
    copy_stats = {
        'total_files': len(selected_files),
        'total_samples': sum(f['num_samples'] for f in selected_files),
        'copied_files': 0,
        'failed_files': 0,
        'file_list': []
    }
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}复制文件到: {target_root}")
    
    for file_info in tqdm(selected_files, desc="复制文件"):
        source_path = file_info['path']
        
        # 计算相对路径
        try:
            relative_path = source_path.relative_to(source_root)
        except ValueError:
            # 如果无法计算相对路径，使用文件名
            relative_path = source_path.name
        
        target_path = target_root / relative_path
        
        if not dry_run:
            try:
                # 创建目标目录
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 复制文件
                shutil.copy2(source_path, target_path)
                
                copy_stats['copied_files'] += 1
                copy_stats['file_list'].append({
                    'source': str(source_path),
                    'target': str(target_path),
                    'num_samples': file_info['num_samples']
                })
            except Exception as e:
                print(f"\n错误: 复制 {source_path} 失败: {e}")
                copy_stats['failed_files'] += 1
        else:
            copy_stats['copied_files'] += 1
            copy_stats['file_list'].append({
                'source': str(source_path),
                'target': str(target_path),
                'num_samples': file_info['num_samples']
            })
    
    return copy_stats


def main():
    parser = argparse.ArgumentParser(
        description="从 D02 数据集中随机抽取文件，使总样本数与 D01 相当"
    )
    parser.add_argument(
        '--d01-dir',
        type=str,
        default='/media/johnny/48FF-AA60/preprocessed_data_d01_m10',
        help='D01 数据集目录（用于确定目标样本数）'
    )
    parser.add_argument(
        '--d02-dir',
        type=str,
        default='/media/johnny/48FF-AA60/preprocessed_data_d02_m10',
        help='D02 数据集目录（源目录）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/media/johnny/48FF-AA60/preprocessed_data_d02_m10_balanced',
        help='输出目录（新的平衡后的 D02 数据集）'
    )
    parser.add_argument(
        '--target-samples',
        type=int,
        default=None,
        help='目标样本数（默认使用 D01 的样本数）'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.02,
        help='允许的样本数偏差比例（默认 2%）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅模拟运行，不实际复制文件'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='保存统计信息到 JSON 文件'
    )
    
    args = parser.parse_args()
    
    d01_dir = Path(args.d01_dir)
    d02_dir = Path(args.d02_dir)
    output_dir = Path(args.output_dir)
    
    # 检查目录
    if not d01_dir.exists():
        print(f"错误: D01 目录不存在 - {d01_dir}")
        return
    
    if not d02_dir.exists():
        print(f"错误: D02 目录不存在 - {d02_dir}")
        return
    
    if output_dir.exists() and not args.dry_run:
        response = input(f"警告: 输出目录已存在 - {output_dir}\n是否继续？(y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    print("=" * 80)
    print("D02 数据集子采样工具")
    print("=" * 80)
    
    # 1. 分析 D01 数据集（获取目标样本数）
    if args.target_samples is None:
        print("\n步骤 1/4: 分析 D01 数据集...")
        d01_files = get_all_hdf5_files_with_samples(d01_dir)
        target_samples = sum(f['num_samples'] for f in d01_files)
        print(f"目标样本数: {target_samples:,}")
    else:
        target_samples = args.target_samples
        print(f"\n使用指定的目标样本数: {target_samples:,}")
    
    # 2. 分析 D02 数据集
    print("\n步骤 2/4: 分析 D02 数据集...")
    d02_files = get_all_hdf5_files_with_samples(d02_dir)
    d02_total_samples = sum(f['num_samples'] for f in d02_files)
    
    if d02_total_samples < target_samples:
        print(f"\n警告: D02 总样本数 ({d02_total_samples:,}) 小于目标样本数 ({target_samples:,})")
        print("将使用所有 D02 文件")
        selected_files = d02_files
        selected_samples = d02_total_samples
    else:
        # 3. 选择文件
        print("\n步骤 3/4: 选择文件...")
        selected_files, selected_samples = select_files_to_match_target(
            d02_files, target_samples, args.seed
        )
    
    # 计算偏差
    deviation = abs(selected_samples - target_samples) / target_samples
    
    print(f"\n选择结果:")
    print(f"  选中文件数: {len(selected_files)} / {len(d02_files)}")
    print(f"  选中样本数: {selected_samples:,}")
    print(f"  目标样本数: {target_samples:,}")
    print(f"  偏差: {deviation:.2%}")
    
    if deviation > args.tolerance:
        print(f"\n警告: 偏差 ({deviation:.2%}) 超过容忍度 ({args.tolerance:.2%})")
        print("建议调整随机种子或容忍度")
    
    # 4. 复制文件
    print("\n步骤 4/4: 复制文件...")
    copy_stats = copy_files_to_new_directory(
        selected_files, d02_dir, output_dir, args.dry_run
    )
    
    # 打印统计
    print("\n" + "=" * 80)
    print("完成统计")
    print("=" * 80)
    print(f"总文件数: {copy_stats['total_files']}")
    print(f"总样本数: {copy_stats['total_samples']:,}")
    print(f"成功复制: {copy_stats['copied_files']}")
    print(f"失败: {copy_stats['failed_files']}")
    
    if args.dry_run:
        print("\n[DRY RUN] 未实际复制文件")
    else:
        print(f"\n文件已复制到: {output_dir}")
    
    # 保存统计信息
    if args.output_json:
        output_data = {
            'd01_target_samples': target_samples,
            'd02_original_samples': d02_total_samples,
            'd02_original_files': len(d02_files),
            'selected_samples': selected_samples,
            'selected_files': len(selected_files),
            'deviation': deviation,
            'tolerance': args.tolerance,
            'seed': args.seed,
            'copy_stats': copy_stats
        }
        
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n统计信息已保存到: {args.output_json}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
