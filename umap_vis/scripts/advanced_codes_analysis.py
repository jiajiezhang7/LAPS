#!/usr/bin/env python3
"""
高级分析脚本：codes_indices 的可解释性分析和 UMAP 可视化
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import glob
from collections import Counter
import matplotlib.pyplot as plt


class CodesIndicesAnalyzer:
    """codes_indices 分析器"""
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.json_files = sorted(glob.glob(f"{folder_path}/**/*.codes.json", recursive=True))
        self.data_cache = {}
        print(f"找到 {len(self.json_files)} 个 JSON 文件")
    
    def load_all_data(self):
        """加载所有 JSON 文件"""
        for json_file in self.json_files:
            try:
                with open(json_file, 'r') as f:
                    self.data_cache[json_file] = json.load(f)
            except Exception as e:
                print(f"错误加载 {json_file}: {e}")
    
    def extract_all_vectors(self, flatten: bool = False) -> Tuple[np.ndarray, List[str]]:
        """提取所有向量"""
        all_vectors = []
        file_names = []
        
        for json_file in self.json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if flatten:
                    # 片段级别展平
                    all_values = []
                    for window in data['quantized_windows']:
                        for vector in window:
                            all_values.extend(vector)
                    all_vectors.append(all_values)
                else:
                    # 时间步级别
                    for window in data['quantized_windows']:
                        for vector in window:
                            all_vectors.append(vector)
                
                file_names.append(Path(json_file).stem)
            except Exception as e:
                print(f"错误处理 {json_file}: {e}")
        
        return np.array(all_vectors, dtype=np.float32), file_names
    
    def extract_all_codes(self) -> Tuple[np.ndarray, List[int]]:
        """提取所有码"""
        all_codes = []
        
        for json_file in self.json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for window in data['codes_windows']:
                    all_codes.extend(window)
            except Exception as e:
                print(f"错误处理 {json_file}: {e}")
        
        return np.array(all_codes, dtype=np.int32), all_codes
    
    def analyze_code_distribution(self) -> Dict:
        """分析码的分布"""
        _, all_codes = self.extract_all_codes()
        code_counts = Counter(all_codes)
        
        return {
            'total_codes': len(all_codes),
            'unique_codes': len(code_counts),
            'code_range': [min(all_codes), max(all_codes)],
            'most_common_10': code_counts.most_common(10),
            'least_common_10': code_counts.most_common()[-10:],
            'mean_frequency': np.mean(list(code_counts.values())),
            'std_frequency': np.std(list(code_counts.values())),
        }
    
    def analyze_vector_statistics(self) -> Dict:
        """分析向量的统计特性"""
        vectors, _ = self.extract_all_vectors(flatten=False)
        
        return {
            'total_vectors': vectors.shape[0],
            'vector_dimension': vectors.shape[1],
            'mean': np.mean(vectors, axis=0),
            'std': np.std(vectors, axis=0),
            'min': np.min(vectors, axis=0),
            'max': np.max(vectors, axis=0),
            'global_mean': np.mean(vectors),
            'global_std': np.std(vectors),
            'global_min': np.min(vectors),
            'global_max': np.max(vectors),
        }
    
    def print_comprehensive_report(self):
        """打印综合分析报告"""
        print("\n" + "="*80)
        print("codes_indices 综合分析报告")
        print("="*80)
        
        # 码分布分析
        print("\n【码分布分析】")
        code_dist = self.analyze_code_distribution()
        print(f"  总码数: {code_dist['total_codes']}")
        print(f"  唯一码数: {code_dist['unique_codes']}")
        print(f"  码取值范围: [{code_dist['code_range'][0]}, {code_dist['code_range'][1]}]")
        print(f"  平均频率: {code_dist['mean_frequency']:.2f}")
        print(f"  频率标准差: {code_dist['std_frequency']:.2f}")
        print(f"  最常用的10个码:")
        for code, count in code_dist['most_common_10']:
            print(f"    码 {code}: {count} 次")
        
        # 向量统计分析
        print("\n【向量统计分析】")
        vec_stats = self.analyze_vector_statistics()
        print(f"  总向量数: {vec_stats['total_vectors']}")
        print(f"  向量维度: {vec_stats['vector_dimension']}")
        print(f"  全局均值: {vec_stats['global_mean']:.6f}")
        print(f"  全局标准差: {vec_stats['global_std']:.6f}")
        print(f"  全局最小值: {vec_stats['global_min']:.6f}")
        print(f"  全局最大值: {vec_stats['global_max']:.6f}")
        print(f"  前5维的均值: {vec_stats['mean'][:5]}")
        print(f"  前5维的标准差: {vec_stats['std'][:5]}")
    
    def plot_code_distribution(self, output_path: str = 'code_distribution.png'):
        """绘制码分布直方图"""
        _, all_codes = self.extract_all_codes()
        code_counts = Counter(all_codes)
        
        codes = sorted(code_counts.keys())
        counts = [code_counts[c] for c in codes]
        
        plt.figure(figsize=(14, 6))
        plt.bar(codes, counts, width=1.0, alpha=0.7)
        plt.xlabel('Code Index')
        plt.ylabel('Frequency')
        plt.title('Code Distribution in codes_indices')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"码分布图已保存: {output_path}")
        plt.close()
    
    def plot_vector_statistics(self, output_path: str = 'vector_statistics.png'):
        """绘制向量统计信息"""
        vectors, _ = self.extract_all_vectors(flatten=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 向量值分布
        axes[0, 0].hist(vectors.flatten(), bins=100, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Vector Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Vector Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 向量范数分布
        norms = np.linalg.norm(vectors, axis=1)
        axes[0, 1].hist(norms, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Vector Norm (L2)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Vector Norms')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 前10维的均值
        mean_per_dim = np.mean(vectors, axis=0)[:10]
        axes[1, 0].bar(range(10), mean_per_dim, alpha=0.7)
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].set_title('Mean Value per Dimension (First 10)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 前10维的标准差
        std_per_dim = np.std(vectors, axis=0)[:10]
        axes[1, 1].bar(range(10), std_per_dim, alpha=0.7)
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Std Dev')
        axes[1, 1].set_title('Std Dev per Dimension (First 10)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"向量统计图已保存: {output_path}")
        plt.close()
    
    def plot_umap_2d(self, output_path: str = 'umap_2d.png', n_neighbors: int = 15):
        """绘制 UMAP 2D 可视化"""
        try:
            import umap
        except ImportError:
            print("需要安装 umap-learn: pip install umap-learn")
            return
        
        print("提取向量...")
        vectors, _ = self.extract_all_vectors(flatten=False)
        
        print("运行 UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
        embedding = reducer.fit_transform(vectors)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=20, c=np.arange(len(embedding)), cmap='viridis')
        plt.colorbar(label='Vector Index')
        plt.title('UMAP 2D Visualization of Quantized Vectors')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"UMAP 2D 图已保存: {output_path}")
        plt.close()
    
    def plot_umap_3d(self, output_path: str = 'umap_3d.html', n_neighbors: int = 15):
        """绘制 UMAP 3D 可视化（交互式）"""
        try:
            import umap
            import plotly.graph_objects as go
        except ImportError:
            print("需要安装: pip install umap-learn plotly")
            return
        
        print("提取向量...")
        vectors, _ = self.extract_all_vectors(flatten=False)
        
        print("运行 UMAP...")
        reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
        embedding = reducer.fit_transform(vectors)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=np.arange(len(embedding)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Vector Index")
            )
        )])
        
        fig.update_layout(
            title='UMAP 3D Visualization of Quantized Vectors',
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            ),
            width=1000,
            height=800
        )
        
        fig.write_html(output_path)
        print(f"UMAP 3D 交互式图已保存: {output_path}")


if __name__ == "__main__":
    output_folder = "./data/YOUR_DATA_PATH"
    
    analyzer = CodesIndicesAnalyzer(output_folder)
    
    # 打印综合报告
    analyzer.print_comprehensive_report()
    
    # 生成可视化
    print("\n生成可视化图表...")
    analyzer.plot_code_distribution('code_distribution.png')
    analyzer.plot_vector_statistics('vector_statistics.png')
    analyzer.plot_umap_2d('umap_2d.png')
    
    # 尝试生成 3D 交互式图
    try:
        analyzer.plot_umap_3d('umap_3d.html')
    except Exception as e:
        print(f"3D 可视化生成失败: {e}")
    
    print("\n分析完成！")

