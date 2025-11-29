#!/usr/bin/env python3
"""
对比 GTEA 和 D01/D02 数据集的 HDF5 文件和样本量
"""
import h5py
import os
from pathlib import Path
from collections import defaultdict
import json

def analyze_hdf5_files(directory, dataset_name):
    """分析指定目录下的所有 HDF5 文件"""
    h5_files = list(Path(directory).rglob("*.hdf5")) + list(Path(directory).rglob("*.h5"))
    
    if not h5_files:
        print(f"❌ {dataset_name}: 未找到 HDF5 文件在 {directory}")
        return None
    
    total_samples = 0
    total_files = len(h5_files)
    file_details = []
    
    print(f"\n📊 {dataset_name} 数据集分析")
    print(f"   目录: {directory}")
    print(f"   HDF5 文件数: {total_files}")
    print("\n   文件详情:")
    
    for h5_file in sorted(h5_files)[:10]:  # 显示前10个文件
        try:
            with h5py.File(h5_file, 'r') as f:
                # 获取第一个数据集的样本数
                keys = list(f.keys())
                if keys:
                    first_key = keys[0]
                    if isinstance(f[first_key], h5py.Dataset):
                        num_samples = f[first_key].shape[0]
                    else:
                        # 如果是 Group，尝试找到数据集
                        num_samples = 0
                        for subkey in f[first_key].keys():
                            if isinstance(f[first_key][subkey], h5py.Dataset):
                                num_samples = f[first_key][subkey].shape[0]
                                break
                    
                    total_samples += num_samples
                    file_details.append({
                        'file': h5_file.name,
                        'samples': num_samples
                    })
                    print(f"      {h5_file.name}: {num_samples} 样本")
        except Exception as e:
            print(f"      ❌ {h5_file.name}: 读取失败 - {e}")
    
    if len(h5_files) > 10:
        print(f"      ... 还有 {len(h5_files) - 10} 个文件")
    
    # 计算总样本数（基于采样）
    if file_details:
        avg_samples_per_file = total_samples / len(file_details)
        estimated_total = int(avg_samples_per_file * total_files)
    else:
        estimated_total = 0
    
    return {
        'dataset_name': dataset_name,
        'directory': directory,
        'total_files': total_files,
        'sampled_files': len(file_details),
        'sampled_total_samples': total_samples,
        'avg_samples_per_file': total_samples / len(file_details) if file_details else 0,
        'estimated_total_samples': estimated_total
    }

def main():
    print("=" * 80)
    print("GTEA vs D01/D02 数据集对比分析")
    print("=" * 80)
    
    results = {}
    
    # 分析 GTEA 数据集
    gtea_dir = "./data/preprocessed_gtea_m10"
    if os.path.exists(gtea_dir):
        results['GTEA'] = analyze_hdf5_files(gtea_dir, "GTEA")
    else:
        print(f"❌ GTEA 目录不存在: {gtea_dir}")
    
    # 分析 D01 数据集（假设在外部存储）
    d01_candidates = [
        "./data/YOUR_DATA_PATH",
        "/mnt/D01",
        "/data/D01",
        "./data/D01"
    ]
    
    d01_found = False
    for d01_dir in d01_candidates:
        if os.path.exists(d01_dir):
            results['D01'] = analyze_hdf5_files(d01_dir, "D01")
            d01_found = True
            break
    
    if not d01_found:
        print(f"\n❌ D01 数据集未找到，尝试的位置: {d01_candidates}")
    
    # 分析 D02 数据集
    d02_candidates = [
        "./data/YOUR_DATA_PATH",
        "/mnt/D02",
        "/data/D02",
        "./data/D02"
    ]
    
    d02_found = False
    for d02_dir in d02_candidates:
        if os.path.exists(d02_dir):
            results['D02'] = analyze_hdf5_files(d02_dir, "D02")
            d02_found = True
            break
    
    if not d02_found:
        print(f"\n❌ D02 数据集未找到，尝试的位置: {d02_candidates}")
    
    # 汇总对比
    print("\n" + "=" * 80)
    print("📈 数据集对比总结")
    print("=" * 80)
    
    comparison_table = []
    for name, info in results.items():
        if info:
            comparison_table.append({
                '数据集': name,
                'HDF5文件数': info['total_files'],
                '平均样本/文件': f"{info['avg_samples_per_file']:.1f}",
                '估计总样本数': f"{info['estimated_total_samples']:,}",
                '目录': info['directory']
            })
    
    if comparison_table:
        print("\n{:<10} {:<15} {:<20} {:<20}".format('数据集', 'HDF5文件数', '平均样本/文件', '估计总样本数'))
        print("-" * 70)
        for row in comparison_table:
            print("{:<10} {:<15} {:<20} {:<20}".format(
                row['数据集'],
                str(row['HDF5文件数']),
                row['平均样本/文件'],
                row['估计总样本数']
            ))
    
    # 计算量级差距
    if 'GTEA' in results and 'D01' in results:
        gtea_samples = results['GTEA']['estimated_total_samples']
        d01_samples = results['D01']['estimated_total_samples']
        if gtea_samples > 0 and d01_samples > 0:
            ratio = d01_samples / gtea_samples
            print(f"\n📊 D01 vs GTEA 量级: {ratio:.1f}x")
    
    if 'GTEA' in results and 'D02' in results:
        gtea_samples = results['GTEA']['estimated_total_samples']
        d02_samples = results['D02']['estimated_total_samples']
        if gtea_samples > 0 and d02_samples > 0:
            ratio = d02_samples / gtea_samples
            print(f"📊 D02 vs GTEA 量级: {ratio:.1f}x")
    
    if 'D01' in results and 'D02' in results:
        d01_samples = results['D01']['estimated_total_samples']
        d02_samples = results['D02']['estimated_total_samples']
        if d01_samples > 0 and d02_samples > 0:
            ratio = d02_samples / d01_samples
            print(f"📊 D02 vs D01 量级: {ratio:.1f}x")
    
    # 保存结果
    output_file = "./supplement_output/dataset_comparison.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
