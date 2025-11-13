#!/usr/bin/env python3
"""
统计 HDF5 数据集的样本量和文件数量
用于分析 Motion Tokenizer 训练数据集的规模差异
"""

import h5py
import argparse
from pathlib import Path
from tqdm import tqdm
import json


def analyze_hdf5_file(hdf5_path):
    """分析单个 HDF5 文件的样本数"""
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Motion Tokenizer 的数据结构: root/default/tracks
            # tracks 的形状是 (num_samples, horizon, num_points, 2)
            if 'root' in f and 'default' in f['root']:
                default_group = f['root']['default']
                if 'tracks' in default_group:
                    num_samples = default_group['tracks'].shape[0]
                elif 'images' in default_group:
                    num_samples = default_group['images'].shape[0]
                else:
                    # 获取第一个数据集的第一维
                    keys = list(default_group.keys())
                    if keys:
                        num_samples = default_group[keys[0]].shape[0]
                    else:
                        num_samples = 0
                all_keys = list(default_group.keys())
            # 备用：直接在根目录查找
            elif 'tracks' in f:
                num_samples = f['tracks'].shape[0]
                all_keys = list(f.keys())
            elif 'images' in f:
                num_samples = f['images'].shape[0]
                all_keys = list(f.keys())
            else:
                # 获取第一个数据集的长度
                keys = list(f.keys())
                if keys:
                    first_item = f[keys[0]]
                    if hasattr(first_item, 'shape'):
                        num_samples = first_item.shape[0]
                    else:
                        num_samples = 0
                else:
                    num_samples = 0
                all_keys = keys
            
            return {
                'num_samples': num_samples,
                'keys': all_keys,
                'success': True
            }
    except Exception as e:
        return {
            'num_samples': 0,
            'keys': [],
            'success': False,
            'error': str(e)
        }


def analyze_dataset(root_dir):
    """分析整个数据集目录"""
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"错误: 目录不存在 - {root_dir}")
        return None
    
    # 递归查找所有 .hdf5 文件
    hdf5_files = list(root_path.rglob("*.hdf5"))
    
    if not hdf5_files:
        print(f"警告: 在 {root_dir} 中未找到 .hdf5 文件")
        return None
    
    print(f"\n正在分析: {root_dir}")
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")
    print("-" * 80)
    
    total_samples = 0
    failed_files = []
    file_details = []
    
    for hdf5_file in tqdm(hdf5_files, desc="处理文件"):
        result = analyze_hdf5_file(hdf5_file)
        
        if result['success']:
            total_samples += result['num_samples']
            file_details.append({
                'path': str(hdf5_file.relative_to(root_path)),
                'num_samples': result['num_samples'],
                'keys': result['keys']
            })
        else:
            failed_files.append({
                'path': str(hdf5_file.relative_to(root_path)),
                'error': result['error']
            })
    
    # 统计结果
    stats = {
        'root_dir': str(root_dir),
        'total_hdf5_files': len(hdf5_files),
        'successful_files': len(file_details),
        'failed_files': len(failed_files),
        'total_samples': total_samples,
        'avg_samples_per_file': total_samples / len(file_details) if file_details else 0,
        'file_details': file_details,
        'failed_file_details': failed_files
    }
    
    return stats


def print_summary(stats):
    """打印统计摘要"""
    print("\n" + "=" * 80)
    print("数据集统计摘要")
    print("=" * 80)
    print(f"数据集路径: {stats['root_dir']}")
    print(f"HDF5 文件总数: {stats['total_hdf5_files']}")
    print(f"成功读取文件数: {stats['successful_files']}")
    print(f"读取失败文件数: {stats['failed_files']}")
    print(f"总样本数: {stats['total_samples']:,}")
    print(f"平均每文件样本数: {stats['avg_samples_per_file']:.2f}")
    
    if stats['failed_files'] > 0:
        print("\n失败的文件:")
        for failed in stats['failed_file_details'][:10]:  # 只显示前10个
            print(f"  - {failed['path']}: {failed['error']}")
        if stats['failed_files'] > 10:
            print(f"  ... 还有 {stats['failed_files'] - 10} 个失败文件")
    
    print("=" * 80)


def compare_datasets(stats1, stats2):
    """比较两个数据集"""
    print("\n" + "=" * 80)
    print("数据集对比")
    print("=" * 80)
    
    d1_name = Path(stats1['root_dir']).name
    d2_name = Path(stats2['root_dir']).name
    
    print(f"\n{'指标':<30} {d1_name:<20} {d2_name:<20} {'比例':<15}")
    print("-" * 85)
    
    # 文件数对比
    files_ratio = stats2['total_hdf5_files'] / stats1['total_hdf5_files'] if stats1['total_hdf5_files'] > 0 else 0
    print(f"{'HDF5 文件数':<30} {stats1['total_hdf5_files']:<20,} {stats2['total_hdf5_files']:<20,} {files_ratio:<15.2f}x")
    
    # 样本数对比
    samples_ratio = stats2['total_samples'] / stats1['total_samples'] if stats1['total_samples'] > 0 else 0
    print(f"{'总样本数':<30} {stats1['total_samples']:<20,} {stats2['total_samples']:<20,} {samples_ratio:<15.2f}x")
    
    # 平均样本数对比
    avg_ratio = stats2['avg_samples_per_file'] / stats1['avg_samples_per_file'] if stats1['avg_samples_per_file'] > 0 else 0
    print(f"{'平均每文件样本数':<30} {stats1['avg_samples_per_file']:<20.2f} {stats2['avg_samples_per_file']:<20.2f} {avg_ratio:<15.2f}x")
    
    print("=" * 80)
    
    # 训练步数分析
    print("\n训练步数分析:")
    print("-" * 80)
    
    # 假设 batch_size=8, gpu_max_bs=8
    batch_size = 8
    
    # 每个 epoch 使用 80% 的数据作为训练集
    train_ratio = 0.8
    
    d1_train_samples = int(stats1['total_samples'] * train_ratio)
    d2_train_samples = int(stats2['total_samples'] * train_ratio)
    
    # 每个 epoch 的步数 = 训练样本数 / batch_size
    d1_steps_per_epoch = d1_train_samples / batch_size
    d2_steps_per_epoch = d2_train_samples / batch_size
    
    print(f"{d1_name} 训练样本数 (80%): {d1_train_samples:,}")
    print(f"{d1_name} 每个 epoch 理论步数: {d1_steps_per_epoch:,.0f}")
    print(f"{d1_name} 5 epochs 理论总步数: {d1_steps_per_epoch * 5:,.0f}")
    print(f"{d1_name} 实际 5 epochs 步数: 458,277")
    print(f"{d1_name} 实际/理论比例: {458277 / (d1_steps_per_epoch * 5):.2%}")
    
    print()
    
    print(f"{d2_name} 训练样本数 (80%): {d2_train_samples:,}")
    print(f"{d2_name} 每个 epoch 理论步数: {d2_steps_per_epoch:,.0f}")
    print(f"{d2_name} 5 epochs 理论总步数: {d2_steps_per_epoch * 5:,.0f}")
    print(f"{d2_name} 实际 5 epochs 步数: 1,078,527")
    print(f"{d2_name} 实际/理论比例: {1078527 / (d2_steps_per_epoch * 5):.2%}")
    
    print("\n步数比例: {:.2f}x".format(1078527 / 458277))
    print("样本数比例: {:.2f}x".format(samples_ratio))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="统计 HDF5 数据集的样本量")
    parser.add_argument('--d01-dir', type=str, 
                       default='/media/johnny/48FF-AA60/preprocessed_data_d01_m10',
                       help='D01 数据集目录')
    parser.add_argument('--d02-dir', type=str,
                       default='/media/johnny/48FF-AA60/preprocessed_data_d02_m10',
                       help='D02 数据集目录')
    parser.add_argument('--output-json', type=str, default=None,
                       help='输出 JSON 文件路径（可选）')
    
    args = parser.parse_args()
    
    # 分析 D01
    stats_d01 = analyze_dataset(args.d01_dir)
    if stats_d01:
        print_summary(stats_d01)
    
    # 分析 D02
    stats_d02 = analyze_dataset(args.d02_dir)
    if stats_d02:
        print_summary(stats_d02)
    
    # 对比两个数据集
    if stats_d01 and stats_d02:
        compare_datasets(stats_d01, stats_d02)
    
    # 保存到 JSON（可选）
    if args.output_json and stats_d01 and stats_d02:
        output_data = {
            'd01': stats_d01,
            'd02': stats_d02
        }
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n详细统计已保存到: {args.output_json}")


if __name__ == '__main__':
    main()
