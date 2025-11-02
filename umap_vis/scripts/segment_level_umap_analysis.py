#!/usr/bin/env python3
"""
改进的 UMAP 分析脚本：片段级别的向量平均和可视化
目标：观察片段间是否存在自然的结构关系或聚集模式
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class SegmentLevelAnalyzer:
    """片段级别的分析器"""

    def __init__(self, folder_path: str, output_dir: str = None):
        self.folder_path = folder_path
        self.json_files = sorted(glob.glob(f"{folder_path}/**/*.codes.json", recursive=True))
        print(f"找到 {len(self.json_files)} 个 JSON 文件")

        # 设置输出目录
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "figure"
        else:
            output_dir = Path(output_dir)

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {self.output_dir}")
    
    def compute_segment_average_vectors(self) -> Tuple[np.ndarray, List[str]]:
        """
        计算每个片段的平均向量
        
        Returns:
            (avg_vectors, segment_names)
            - avg_vectors: (n_segments, 768) 的 numpy 数组
            - segment_names: 片段名称列表
        """
        segment_vectors = []
        segment_names = []
        segment_info = []
        
        for json_file in self.json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # 提取所有向量
                all_vectors = []
                for window in data['quantized_windows']:
                    for vector in window:
                        all_vectors.append(vector)
                
                if all_vectors:
                    # 计算平均向量
                    avg_vector = np.mean(all_vectors, axis=0)
                    segment_vectors.append(avg_vector)
                    
                    segment_name = Path(json_file).stem
                    segment_names.append(segment_name)
                    
                    # 记录元数据
                    segment_info.append({
                        'name': segment_name,
                        'num_vectors': len(all_vectors),
                        'num_windows': len(data['quantized_windows']),
                        'video_path': data.get('video_segment_path', 'unknown'),
                    })
            except Exception as e:
                print(f"错误处理 {json_file}: {e}")
        
        return np.array(segment_vectors, dtype=np.float32), segment_names, segment_info
    
    def apply_umap(self, vectors: np.ndarray, n_components: int = 2,
                   n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        """应用 UMAP 降维"""
        try:
            import umap
        except ImportError:
            print("需要安装 umap-learn: pip install umap-learn")
            return None

        print(f"应用 UMAP 降维到 {n_components}D...")
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='euclidean'
        )
        embedding = reducer.fit_transform(vectors)
        return embedding
    
    def plot_umap_2d(self, embedding_2d: np.ndarray, segment_indices: np.ndarray = None,
                     output_path: str = None):
        """绘制 2D UMAP 图，仅显示数据点，无标签"""
        if output_path is None:
            output_path = self.output_dir / 'umap_segment_2d.png'

        plt.figure(figsize=(12, 10))

        # 使用文件索引作为颜色（渐变色）
        if segment_indices is None:
            segment_indices = np.arange(len(embedding_2d))

        # 绘制散点图，仅显示数据点
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                             c=segment_indices, cmap='viridis', s=50, alpha=0.6,
                             edgecolors='none')

        plt.colorbar(scatter, label='Segment Index')
        plt.title('UMAP 2D: Segment-Level Average Vectors', fontsize=14)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"2D UMAP 图已保存: {output_path}")
        plt.close()
    
    def plot_umap_3d(self, embedding_3d: np.ndarray, segment_indices: np.ndarray = None,
                     output_path: str = None):
        """绘制 3D UMAP 交互式图，仅显示数据点，无标签"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("需要安装 plotly: pip install plotly")
            return

        if output_path is None:
            output_path = self.output_dir / 'umap_segment_3d.html'

        print("生成 3D UMAP 交互式图...")

        # 使用文件索引作为颜色（渐变色）
        if segment_indices is None:
            segment_indices = np.arange(len(embedding_3d))

        fig = go.Figure(data=[go.Scatter3d(
            x=embedding_3d[:, 0],
            y=embedding_3d[:, 1],
            z=embedding_3d[:, 2],
            mode='markers',  # 仅显示标记，不显示文本
            marker=dict(
                size=5,
                color=segment_indices,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Segment Index"),
                line=dict(color='rgba(0,0,0,0)', width=0)  # 无边框
            )
        )])

        fig.update_layout(
            title='UMAP 3D: Segment-Level Average Vectors',
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            ),
            width=1200,
            height=900
        )

        fig.write_html(str(output_path))
        print(f"3D UMAP 交互式图已保存: {output_path}")
    
    def print_vector_statistics(self, vectors: np.ndarray):
        """打印向量统计信息"""
        print("\n" + "="*80)
        print("向量统计信息")
        print("="*80)

        print(f"\n向量数量: {len(vectors)}")
        print(f"向量维度: {vectors.shape[1]}")
        print(f"向量均值: {np.mean(vectors, axis=0)[:5]}...")
        print(f"向量方差: {np.var(vectors, axis=0)[:5]}...")
        print(f"向量最小值: {np.min(vectors, axis=0)[:5]}...")
        print(f"向量最大值: {np.max(vectors, axis=0)[:5]}...")
    
    def compute_similarity_matrix(self, vectors: np.ndarray, segment_names: List[str]):
        """计算相似性矩阵并找出最相似的片段对"""
        print("\n" + "="*80)
        print("相似性分析")
        print("="*80)
        
        similarity_matrix = cosine_similarity(vectors)
        
        # 找出最相似的片段对（排除自己与自己的比较）
        similar_pairs = []
        for i in range(len(segment_names)):
            for j in range(i+1, len(segment_names)):
                similar_pairs.append((similarity_matrix[i, j], i, j))
        
        similar_pairs.sort(reverse=True)
        
        print("\n最相似的 10 个片段对:")
        for sim, i, j in similar_pairs[:10]:
            print(f"  {segment_names[i]} <-> {segment_names[j]}: {sim:.4f}")
        
        print("\n最不相似的 10 个片段对:")
        for sim, i, j in similar_pairs[-10:]:
            print(f"  {segment_names[i]} <-> {segment_names[j]}: {sim:.4f}")
        
        return similarity_matrix
    
    def run_full_analysis(self):
        """执行完整分析流程"""
        print("\n" + "="*80)
        print("片段级别 UMAP 分析")
        print("="*80)

        # 步骤 1: 计算平均向量
        print("\n【步骤 1】计算每个片段的平均向量...")
        avg_vectors, segment_names, segment_info = self.compute_segment_average_vectors()
        print(f"  得到 {len(avg_vectors)} 个片段的平均向量")
        print(f"  向量维度: {avg_vectors.shape[1]}")

        # 步骤 2: 标准化向量
        print("\n【步骤 2】标准化向量...")
        scaler = StandardScaler()
        avg_vectors_scaled = scaler.fit_transform(avg_vectors)

        # 步骤 3: UMAP 降维 (2D)
        print("\n【步骤 3】UMAP 降维到 2D...")
        embedding_2d = self.apply_umap(avg_vectors_scaled, n_components=2, n_neighbors=15, min_dist=0.1)

        # 步骤 4: UMAP 降维 (3D)
        print("\n【步骤 4】UMAP 降维到 3D...")
        embedding_3d = self.apply_umap(avg_vectors_scaled, n_components=3, n_neighbors=15, min_dist=0.1)

        # 步骤 5: 可视化
        print("\n【步骤 5】生成可视化...")
        segment_indices = np.arange(len(avg_vectors))
        self.plot_umap_2d(embedding_2d, segment_indices)
        self.plot_umap_3d(embedding_3d, segment_indices)

        # 步骤 6: 向量统计
        print("\n【步骤 6】向量统计信息...")
        self.print_vector_statistics(avg_vectors_scaled)

        # 步骤 7: 相似性分析
        print("\n【步骤 7】相似性分析...")
        similarity_matrix = self.compute_similarity_matrix(avg_vectors, segment_names)

        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
        print(f"\n输出文件位置:")
        print(f"  - 2D UMAP: {self.output_dir / 'umap_segment_2d.png'}")
        print(f"  - 3D UMAP: {self.output_dir / 'umap_segment_3d.html'}")

        return {
            'avg_vectors': avg_vectors,
            'avg_vectors_scaled': avg_vectors_scaled,
            'segment_names': segment_names,
            'embedding_2d': embedding_2d,
            'embedding_3d': embedding_3d,
            'similarity_matrix': similarity_matrix,
        }


if __name__ == "__main__":
    # 分析输出文件夹
    output_folder = "/media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector"

    # 获取脚本所在目录的上级目录的 figure 子目录
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "figure"

    analyzer = SegmentLevelAnalyzer(output_folder, output_dir=str(output_dir))
    results = analyzer.run_full_analysis()

