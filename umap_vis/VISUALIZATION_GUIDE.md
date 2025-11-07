# UMAP 可视化论文级优化指南

## 概述
已优化 `sequence_model_embedding.py` 中的 UMAP 可视化函数，使其输出达到 **CVPR 论文级别**的美观性和专业性。

## 主要改进

### 2D 图优化
| 项目 | 原始 | 优化后 | 说明 |
|------|------|--------|------|
| 分辨率 | 150 DPI | 300 DPI | 适合论文印刷 |
| 图表大小 | 10×8 | 12×10 | 更清晰的细节展示 |
| 点大小 | 36 | 80 | 更易区分簇 |
| 点边框 | 无 | 黑色 0.5pt | 增强视觉对比 |
| 配色 | 单一 tab20 | 专业 tab10 | 离散簇更清晰 |
| 网格 | 无 | 虚线网格 | 便于读数 |
| 字体 | 默认 | 加粗 12-14pt | 更专业 |

### 3D 图优化
| 项目 | 原始 | 优化后 | 说明 |
|------|------|--------|------|
| 点大小 | 5 | 6 | 更清晰 |
| 点边框 | 无 | 半透明黑 | 增强深度感 |
| 图表尺寸 | 默认 | 1000×900 | 高质量输出 |
| 标题 | 小 | 16pt 加粗 | 更突出 |
| 网格线 | 无 | 可见 | 便于读数 |
| 图例 | 无 | 带边框 | 支持交互切换 |
| 相机角度 | 默认 | 预设最优 | 最佳视角 |

## 使用方法

### 基础运行（使用最优配置）
```bash
conda activate laps
python umap_vis/scripts/sequence_model_embedding.py \
  --data-dir /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector \
  --fig-dir umap_vis/figure --stats-dir umap_vis/statistics \
  --metric cosine --neighbors 15 --min-dist 0.1 \
  --use-best-grid-config
```

### 网格搜索（生成所有配置的对比图）
```bash
conda activate laps
python umap_vis/scripts/sequence_model_embedding.py \
  --data-dir /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector \
  --fig-dir umap_vis/figure --stats-dir umap_vis/statistics \
  --metric cosine --neighbors 15 --min-dist 0.1 \
  --grid-search
```

## 输出文件

### 2D 图（PNG 格式）
- **分辨率**：300 DPI
- **格式**：PNG（无损压缩）
- **用途**：直接用于论文主文
- **文件名示例**：
  - `umap_2d_seq_model_grid_best.png`
  - `umap_2d_best_config_k3.png`
  - `umap_2d_best_config_k5.png`

### 3D 图（HTML 格式）
- **格式**：交互式 HTML
- **用途**：补充材料、在线展示
- **交互功能**：
  - 鼠标拖动旋转视角
  - 滚轮缩放
  - 点击图例切换簇显示/隐藏
  - Hover 显示坐标值
- **文件名示例**：
  - `umap_3d_seq_model_grid_best.html`
  - `umap_3d_best_config_k3.html`
  - `umap_3d_best_config_k5.html`

## 配色方案

### 离散簇（≤10个）
使用 Matplotlib 的 tab10 专业色板：
- 蓝色、橙色、绿色、红色、紫色、棕色、粉色、灰色、黄绿、青色

### 连续标签
使用 Viridis 色板（从紫色到黄色的渐变）

## 论文集成建议

### 主文配图
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/umap_2d_seq_model_grid_best.png}
  \caption{动作片段在 UMAP 空间中的聚类结果。采用轻量级 Transformer 编码器提取段级嵌入，
           通过 KMeans 聚类（k=3）得到 3 个动作类别。}
  \label{fig:umap_clustering}
\end{figure}
```

### 补充材料
在补充材料中包含 HTML 文件，读者可以交互式探索 3D 聚类结果。

## 自定义选项

### 修改点大小
编辑 `plot_umap_2d` 中的 `s=80` 或 `plot_umap_3d` 中的 `size=6`

### 修改配色
编辑 `plot_umap_2d` 中的 `cmap='tab10'` 或 `plot_umap_3d` 中的 `discrete_colors` 列表

### 修改分辨率
编辑 `plot_umap_2d` 中的 `dpi=300`

### 修改图表大小
编辑 `plot_umap_2d` 中的 `figsize=(12, 10)` 或 `plot_umap_3d` 中的 `width=1000, height=900`

## 质量检查清单

- [ ] 2D 图清晰可读，簇边界明显
- [ ] 3D 图交互流畅，无卡顿
- [ ] 颜色对比度足够（特别是打印时）
- [ ] 字体大小适中，易于阅读
- [ ] 图例清晰，标签准确
- [ ] 轴标签和标题信息完整

## 常见问题

**Q: 如何导出为其他格式？**
A: 修改 `plot_umap_2d` 中的 `savefig` 参数，支持 PNG、PDF、SVG 等格式。

**Q: 如何改变簇的颜色顺序？**
A: 修改 `discrete_colors` 列表中的颜色顺序。

**Q: 3D 图太大了怎么办？**
A: 修改 `plot_umap_3d` 中的 `width` 和 `height` 参数。

**Q: 如何在论文中嵌入 3D 图？**
A: 在补充材料中提供 HTML 文件链接，或转换为视频演示。
