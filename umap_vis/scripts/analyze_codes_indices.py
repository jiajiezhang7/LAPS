#!/usr/bin/env python3
"""
分析 codes_indices JSON 文件的脚本
用于理解 codes 和 quantized_windows 的数据结构、维度和取值范围
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import glob


def analyze_single_json(json_path: str) -> Dict[str, Any]:
    """分析单个 JSON 文件的结构和内容"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    analysis = {
        'file': json_path,
        'top_level_keys': list(data.keys()),
        'codes_windows': {
            'num_windows': len(data['codes_windows']),
            'codes_per_window': [len(w) for w in data['codes_windows']],
            'codes_range': [min(min(w) for w in data['codes_windows']), 
                           max(max(w) for w in data['codes_windows'])],
            'total_codes': sum(len(w) for w in data['codes_windows']),
        },
        'quantized_windows': {
            'num_windows': len(data['quantized_windows']),
            'vectors_per_window': [len(w) for w in data['quantized_windows']],
            'vector_dimension': len(data['quantized_windows'][0][0]) if data['quantized_windows'] and data['quantized_windows'][0] else 0,
            'total_vectors': sum(len(w) for w in data['quantized_windows']),
        },
        'selected_win_idxs': data.get('selected_win_idxs', []),
        'metadata': {
            'overlap_ratio_threshold': data.get('overlap_ratio_threshold'),
            'segment': data.get('segment'),
            'window': data.get('window'),
            'align': data.get('align'),
            'source': data.get('source'),
            'allow_overlap': data.get('allow_overlap'),
        }
    }
    
    # 计算 quantized_windows 的统计信息
    if data['quantized_windows']:
        all_values = []
        for window in data['quantized_windows']:
            for vector in window:
                all_values.extend(vector)
        
        analysis['quantized_windows']['value_range'] = [min(all_values), max(all_values)]
        analysis['quantized_windows']['mean'] = np.mean(all_values)
        analysis['quantized_windows']['std'] = np.std(all_values)
    
    return analysis


def analyze_all_json_files(folder_path: str) -> Dict[str, Any]:
    """分析文件夹下所有 JSON 文件"""
    json_files = glob.glob(f"{folder_path}/**/*.codes.json", recursive=True)
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    all_analyses = []
    for json_file in json_files[:10]:  # 分析前10个文件
        try:
            analysis = analyze_single_json(json_file)
            all_analyses.append(analysis)
        except Exception as e:
            print(f"错误处理 {json_file}: {e}")
    
    # 汇总统计
    summary = {
        'total_files_analyzed': len(all_analyses),
        'total_files_found': len(json_files),
        'avg_codes_windows': np.mean([a['codes_windows']['num_windows'] for a in all_analyses]),
        'avg_codes_per_window': np.mean([np.mean(a['codes_windows']['codes_per_window']) for a in all_analyses]),
        'avg_vector_dimension': np.mean([a['quantized_windows']['vector_dimension'] for a in all_analyses]),
        'codes_range_global': [
            min(a['codes_windows']['codes_range'][0] for a in all_analyses),
            max(a['codes_windows']['codes_range'][1] for a in all_analyses),
        ],
        'quantized_value_range_global': [
            min(a['quantized_windows']['value_range'][0] for a in all_analyses if 'value_range' in a['quantized_windows']),
            max(a['quantized_windows']['value_range'][1] for a in all_analyses if 'value_range' in a['quantized_windows']),
        ],
    }
    
    return {
        'summary': summary,
        'individual_analyses': all_analyses,
        'all_json_files': json_files,
    }


def print_detailed_analysis(analysis: Dict[str, Any]):
    """打印详细的分析结果"""
    print("\n" + "="*80)
    print("codes_indices JSON 文件详细分析")
    print("="*80)
    
    summary = analysis['summary']
    print(f"\n【汇总统计】")
    print(f"  分析的文件数: {summary['total_files_analyzed']} / {summary['total_files_found']}")
    print(f"  平均 codes_windows 数: {summary['avg_codes_windows']:.2f}")
    print(f"  平均每窗 codes 数: {summary['avg_codes_per_window']:.2f}")
    print(f"  平均向量维度: {summary['avg_vector_dimension']:.0f}")
    print(f"  codes 全局取值范围: [{summary['codes_range_global'][0]}, {summary['codes_range_global'][1]}]")
    print(f"  quantized 值全局范围: [{summary['quantized_value_range_global'][0]:.4f}, {summary['quantized_value_range_global'][1]:.4f}]")
    
    # 详细分析第一个文件
    if analysis['individual_analyses']:
        first = analysis['individual_analyses'][0]
        print(f"\n【第一个文件详细信息】: {Path(first['file']).name}")
        print(f"  顶级键: {first['top_level_keys']}")
        print(f"  codes_windows:")
        print(f"    - 窗口数: {first['codes_windows']['num_windows']}")
        print(f"    - 每窗 codes 数: {first['codes_windows']['codes_per_window']}")
        print(f"    - codes 取值范围: {first['codes_windows']['codes_range']}")
        print(f"  quantized_windows:")
        print(f"    - 窗口数: {first['quantized_windows']['num_windows']}")
        print(f"    - 每窗向量数: {first['quantized_windows']['vectors_per_window']}")
        print(f"    - 向量维度: {first['quantized_windows']['vector_dimension']}")
        print(f"    - 值范围: {first['quantized_windows']['value_range']}")
        print(f"    - 均值: {first['quantized_windows']['mean']:.6f}")
        print(f"    - 标准差: {first['quantized_windows']['std']:.6f}")
        print(f"  元数据:")
        for key, val in first['metadata'].items():
            print(f"    - {key}: {val}")


def extract_umap_vectors(json_path: str, flatten: bool = True) -> np.ndarray:
    """
    从 JSON 文件中提取用于 UMAP 可视化的向量
    
    Args:
        json_path: JSON 文件路径
        flatten: 是否将所有窗口的向量展平为单个向量
    
    Returns:
        numpy 数组，形状为 (num_vectors, vector_dim) 或 (1, total_dim) 如果 flatten=True
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    quantized_windows = data['quantized_windows']
    
    if flatten:
        # 将所有向量展平为单个向量
        all_vectors = []
        for window in quantized_windows:
            for vector in window:
                all_vectors.extend(vector)
        return np.array([all_vectors], dtype=np.float32)
    else:
        # 返回所有向量，每个向量一行
        all_vectors = []
        for window in quantized_windows:
            for vector in window:
                all_vectors.append(vector)
        return np.array(all_vectors, dtype=np.float32)


def batch_extract_umap_vectors(folder_path: str, flatten: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    批量提取所有 JSON 文件中的向量用于 UMAP
    
    Args:
        folder_path: 包含 JSON 文件的文件夹
        flatten: 是否为每个文件生成单个展平向量
    
    Returns:
        (向量数组, 文件名列表)
    """
    json_files = sorted(glob.glob(f"{folder_path}/**/*.codes.json", recursive=True))
    
    all_vectors = []
    file_names = []
    
    for json_file in json_files:
        try:
            vectors = extract_umap_vectors(json_file, flatten=flatten)
            all_vectors.append(vectors)
            file_names.append(Path(json_file).stem)
        except Exception as e:
            print(f"错误处理 {json_file}: {e}")
    
    if all_vectors:
        result = np.vstack(all_vectors)
        return result, file_names
    else:
        return np.array([]), []


if __name__ == "__main__":
    # 分析输出文件夹
    output_folder = "./data/YOUR_DATA_PATH"
    
    print(f"分析文件夹: {output_folder}")
    analysis = analyze_all_json_files(output_folder)
    print_detailed_analysis(analysis)
    
    # 示例：提取用于 UMAP 的向量
    print("\n" + "="*80)
    print("UMAP 向量提取示例")
    print("="*80)
    
    if analysis['individual_analyses']:
        first_file = analysis['individual_analyses'][0]['file']
        print(f"\n从文件提取向量: {Path(first_file).name}")
        
        # 方式1：每个向量单独一行
        vectors_unflatten = extract_umap_vectors(first_file, flatten=False)
        print(f"  未展平形状: {vectors_unflatten.shape}")
        print(f"  前3个向量的前5个值:")
        print(f"    {vectors_unflatten[:3, :5]}")
        
        # 方式2：展平为单个向量
        vectors_flatten = extract_umap_vectors(first_file, flatten=True)
        print(f"  展平形状: {vectors_flatten.shape}")
        
        # 批量提取
        print(f"\n批量提取所有文件的向量...")
        all_vectors, file_names = batch_extract_umap_vectors(output_folder, flatten=False)
        print(f"  总向量数: {all_vectors.shape[0]}")
        print(f"  向量维度: {all_vectors.shape[1]}")
        print(f"  处理的文件数: {len(file_names)}")

