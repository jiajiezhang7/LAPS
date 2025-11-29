# codes_indices 分析工具使用指南

## 概述

本指南介绍如何使用提供的分析脚本来理解和可视化 `codes_indices` 文件夹下的 JSON 数据。

---

## 文件清单

### 1. 分析脚本

#### `analyze_codes_indices.py`
基础分析脚本，用于：
- 查看 JSON 文件的结构
- 分析 codes 和 quantized_windows 的维度和取值范围
- 提取用于 UMAP 的向量数据

**运行方式**:
```bash
conda run -n laps python analyze_codes_indices.py
```

**输出**:
- 控制台打印详细的分析结果
- 包括汇总统计、第一个文件的详细信息、UMAP 向量提取示例

#### `advanced_codes_analysis.py`
高级分析脚本，用于：
- 码分布分析
- 向量统计分析
- 生成可视化图表（码分布、向量统计）
- 生成 UMAP 2D 和 3D 可视化

**运行方式**:
```bash
conda run -n laps python advanced_codes_analysis.py
```

**输出文件**:
- `code_distribution.png` - 码频率分布直方图
- `vector_statistics.png` - 向量统计信息（4个子图）
- `umap_2d.png` - UMAP 2D 可视化
- `umap_3d.html` - UMAP 3D 交互式可视化

### 2. 文档

#### `CODES_INDICES_ANALYSIS.md`
详细的数据格式文档，包括：
- JSON 文件整体结构说明
- 各字段的详细说明
- codes 和 quantized_windows 的对应关系
- 推荐的读取和解析方式
- UMAP 可视化数据提取方法
- 可解释性分析建议

---

## 快速开始

### 步骤 1: 运行基础分析

```bash
conda run -n laps python analyze_codes_indices.py
```

这会输出：
- JSON 文件的顶级键
- codes_windows 的统计信息
- quantized_windows 的统计信息
- UMAP 向量提取示例

### 步骤 2: 运行高级分析和可视化

```bash
conda run -n laps python advanced_codes_analysis.py
```

这会生成：
- 码分布直方图
- 向量统计图表
- UMAP 2D 可视化
- UMAP 3D 交互式可视化

### 步骤 3: 查看结果

- **码分布**: 打开 `code_distribution.png` 查看码的使用频率
- **向量统计**: 打开 `vector_statistics.png` 查看向量的统计特性
- **UMAP 2D**: 打开 `umap_2d.png` 查看向量的 2D 投影
- **UMAP 3D**: 在浏览器中打开 `umap_3d.html` 查看交互式 3D 投影

---

## 数据结构速查表

### JSON 文件顶级字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `codes_windows` | List[List[int]] | 码索引窗口列表 |
| `quantized_windows` | List[List[List[float]]] | 量化向量窗口列表 |
| `selected_win_idxs` | List[int] | 选中的窗口索引 |
| `overlap_ratio_threshold` | float | 重叠比例阈值 |
| `segment` | dict | 片段元数据 |
| `window` | dict | 窗口配置 |
| `align` | str | 对齐方式 |
| `video_segment_path` | str | 视频片段路径 |
| `source` | str | 数据来源 |
| `allow_overlap` | bool | 是否允许重叠 |

### 关键统计数据

| 指标 | 值 |
|------|-----|
| 总 JSON 文件数 | 46 |
| 总向量数 | 2422 |
| 向量维度 | 768 |
| 唯一码数 | 395 |
| 码取值范围 | [44, 1883] |
| 向量值范围 | [-1.387, 1.423] |
| 平均每文件窗口数 | 3.4 |
| 平均每窗码数 | 14 |

---

## 代码示例

### 示例 1: 读取单个 JSON 文件

```python
import json
import numpy as np

# 读取文件
with open('segment_0000_startwin_000081.codes.json', 'r') as f:
    data = json.load(f)

# 访问码
codes_windows = data['codes_windows']
print(f"窗口数: {len(codes_windows)}")
print(f"第一个窗口的码: {codes_windows[0]}")

# 访问向量
quantized_windows = data['quantized_windows']
print(f"第一个向量的维度: {len(quantized_windows[0][0])}")
```

### 示例 2: 提取所有向量用于 UMAP

```python
import json
import numpy as np
import glob

# 提取所有向量
all_vectors = []
for json_file in glob.glob("code_indices/**/*.codes.json", recursive=True):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for window in data['quantized_windows']:
        for vector in window:
            all_vectors.append(vector)

vectors_array = np.array(all_vectors, dtype=np.float32)
print(f"向量数组形状: {vectors_array.shape}")  # (2422, 768)
```

### 示例 3: 分析码分布

```python
from collections import Counter
import json
import glob

# 统计码的分布
all_codes = []
for json_file in glob.glob("code_indices/**/*.codes.json", recursive=True):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for window in data['codes_windows']:
        all_codes.extend(window)

code_counts = Counter(all_codes)
print(f"使用的码数: {len(code_counts)}")
print(f"最常用的10个码: {code_counts.most_common(10)}")
```

### 示例 4: UMAP 可视化

```python
import umap
import numpy as np
import matplotlib.pyplot as plt

# 假设已有 vectors_array (2422, 768)
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(vectors_array)

plt.figure(figsize=(12, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=10)
plt.title('UMAP Visualization')
plt.savefig('umap_result.png', dpi=150)
plt.show()
```

---

## 关键发现

### 码分布特性

- **总码数**: 2422（来自 46 个文件）
- **唯一码数**: 395（占码本的 19.3%）
- **最常用码**: 1188 和 1347（各 58 次）
- **码频率分布**: 不均匀，遵循幂律分布

### 向量特性

- **维度**: 768（模型隐藏层维度）
- **值范围**: [-1.387, 1.423]（量化后的浮点值）
- **均值**: -0.003（接近 0，标准化）
- **标准差**: 0.321（相对紧凑的分布）

### UMAP 可视化洞察

- 向量在 UMAP 空间中形成多个聚类
- 相邻的向量通常来自同一片段
- 不同片段的向量分布存在明显差异

---

## 常见问题

**Q: 为什么要在 laps 环境下运行？**
A: laps 环境包含了所有必要的依赖（torch, numpy, umap-learn 等）。

**Q: 如何修改 UMAP 的参数？**
A: 编辑 `advanced_codes_analysis.py` 中的 `plot_umap_2d` 和 `plot_umap_3d` 方法，修改 `n_neighbors` 和 `min_dist` 参数。

**Q: 向量维度为什么是 768？**
A: 这是模型配置中的 `hidden_dim` 参数，对应 MotionTokenizer 的隐藏层维度。

**Q: 如何使用这些数据进行动作识别？**
A: 可以使用码序列作为离散符号进行分类，或使用向量进行连续表示学习。

---

## 后续分析建议

1. **聚类分析**: 使用 K-means 对向量进行聚类，分析每个聚类的码分布
2. **时间序列分析**: 分析码序列的时间演变模式
3. **相似性分析**: 计算片段间的向量相似性
4. **动作识别**: 使用码或向量作为特征进行动作分类
5. **异常检测**: 识别异常的码或向量模式

---

## 文件位置

所有分析脚本和文档位于:
```
./
├── analyze_codes_indices.py          # 基础分析脚本
├── advanced_codes_analysis.py        # 高级分析脚本
├── CODES_INDICES_ANALYSIS.md         # 详细文档
├── USAGE_GUIDE.md                    # 本文件
├── code_distribution.png             # 码分布图
├── vector_statistics.png             # 向量统计图
├── umap_2d.png                       # UMAP 2D 图
└── umap_3d.html                      # UMAP 3D 交互式图
```

数据位置:
```
./data/YOUR_DATA_PATH
└── epochs5_complete500_d02_m10_cb2048_stride4_vector/
    └── D02_20250811075805/
        └── code_indices/             # 46 个 JSON 文件
```

---

## 联系和支持

如有问题，请参考 `CODES_INDICES_ANALYSIS.md` 中的详细说明。

