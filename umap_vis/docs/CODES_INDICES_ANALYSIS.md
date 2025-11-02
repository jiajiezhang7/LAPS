# codes_indices JSON 文件详细分析文档

## 目录
1. [概述](#1-概述)
2. [JSON 文件整体结构](#2-json-文件整体结构)
3. [核心字段详细说明](#3-核心字段详细说明)
4. [数据统计汇总](#4-数据统计汇总)
5. [codes 和 quantized_windows 的对应关系](#5-codes-和-quantized_windows-的对应关系)
6. [推荐的读取和解析方式](#6-推荐的读取和解析方式)
7. [UMAP 可视化数据提取](#7-umap-可视化数据提取)
8. [可解释性分析建议](#8-可解释性分析建议)
9. [常见问题解答](#9-常见问题解答)
10. [参考信息](#10-参考信息)

---


# codes_indices JSON 文件详细分析文档

## 1. 概述

`codes_indices` 文件夹下的 JSON 文件是流式推理过程中生成的**片段级别的编码索引和量化向量**的输出。每个 JSON 文件对应一个视频片段，包含该片段内所有选中窗口的码索引和量化向量表示。

### 文件命名规则
```
segment_{segment_id}_startwin_{start_window_id}.codes.json
```
- `segment_id`: 视频片段的序号
- `start_window_id`: 该片段对应的起始窗口编号

---

## 2. JSON 文件整体结构

### 2.1 顶级字段

```json
{
  "codes_windows": [...],              // 码索引窗口列表
  "quantized_windows": [...],          // 量化向量窗口列表
  "selected_win_idxs": [...],          // 选中的窗口索引列表
  "overlap_ratio_threshold": 0.25,     // 重叠比例阈值
  "segment": {...},                    // 片段元数据
  "window": {...},                     // 窗口配置
  "align": "center",                   // 对齐方式
  "video_segment_path": "...",         // 视频片段路径
  "source": "stream_online",           // 数据来源
  "allow_overlap": true                // 是否允许窗口重叠
}
```

---

## 3. 核心字段详细说明

### 3.1 `codes_windows` 字段

**用途**: 存储每个选中窗口的码索引序列

**数据结构**:
```
codes_windows: List[List[int]]
  - 外层列表: 所有选中的窗口
  - 内层列表: 每个窗口内的码索引序列
```

**维度信息**:
- **窗口数**: 通常 2-5 个（取决于片段长度和重叠配置）
- **每窗码数**: 固定为 14（对应窗口时间长度 T=15 帧，stride=4）
- **码的取值范围**: [109, 1819]（全局范围）

**数据类型**: `int` (整数)

**示例**:
```json
"codes_windows": [
  [1164, 1221, 1157, 1284, 1356, 908, 1356, 988, 932, 988, 868, 804, 1188, 804],
  [1221, 1220, 1284, 1347, 1412, 1412, 1419, 1427, 1379, 1379, 1315, 1252, 1251, 1251],
  [1155, 1603, 1667, 1347, 1411, 1411, 1419, 1491, 1435, 1371, 995, 1315, 923, 1307]
]
```

**生成逻辑**:
1. 模型对输入帧序列进行编码得到潜在表示 `z`
2. 使用 FSQ (Finite Scalar Quantization) 量化得到码索引
3. FSQ 将多个量化数字合并为单个码 ID
4. 码 ID 范围由 codebook_size 决定（本例为 2048）

---

### 3.2 `quantized_windows` 字段

**用途**: 存储每个选中窗口的量化向量表示（用于UMAP可视化）

**数据结构**:
```
quantized_windows: List[List[List[float]]]
  - 外层列表: 所有选中的窗口
  - 中层列表: 每个窗口内的向量序列
  - 内层列表: 每个向量的浮点数分量
```

**维度信息**:
- **窗口数**: 与 `codes_windows` 相同
- **每窗向量数**: 14（与码数相同，一一对应）
- **向量维度**: 768（隐藏层维度）
- **值范围**: [-1.3809, 1.4229]（全局范围）
- **均值**: 约 -0.003（接近0，标准化）
- **标准差**: 约 0.312

**数据类型**: `float` (浮点数)

**示例** (部分):
```json
"quantized_windows": [
  [
    [-0.0352783203125, 0.26806640625, -0.6025390625, ..., 0.01080322265625],  // 向量1
    [-0.08392334, 0.21984863, -0.68847656, ..., 0.00604248046875],           // 向量2
    ...
  ],
  ...
]
```

**生成逻辑**:
1. 编码后的潜在表示 `z` 经过 FSQ 量化
2. 量化后的向量 `quantized` 保留了连续值（不是离散的码）
3. 这些向量被转换为 float32 numpy 数组，再序列化为 JSON

---

### 3.3 `selected_win_idxs` 字段

**用途**: 记录选中窗口在全局窗口序列中的索引

**数据结构**: `List[int]`

**示例**:
```json
"selected_win_idxs": [81, 82, 83]
```

**说明**:
- 这些索引对应于流式推理过程中的全局窗口编号
- 用于追踪哪些窗口被选中用于该片段的码导出

---

### 3.4 元数据字段

#### `segment` (片段信息)
```json
"segment": {
  "start_frame": 332,      // 片段起始帧（重采样后）
  "end_frame": 355,        // 片段结束帧（重采样后）
  "start_win": 81          // 片段对应的起始窗口
}
```

#### `window` (窗口配置)
```json
"window": {
  "T": 15,                 // 窗口时间长度（帧数）
  "stride": 4,             // 窗口步长（帧数）
  "target_fps": 10         // 目标帧率（Hz）
}
```

#### 其他字段
- `overlap_ratio_threshold`: 0.25 - 窗口与片段的最小重叠比例
- `align`: "center" - 对齐方式
- `source`: "stream_online" - 数据来源
- `allow_overlap`: true - 是否允许窗口重叠

---

## 4. 数据统计汇总

基于分析的 46 个 JSON 文件：

| 指标 | 值 |
|------|-----|
| 平均 codes_windows 数 | 3.40 |
| 平均每窗 codes 数 | 14.00 |
| 平均向量维度 | 768 |
| codes 全局取值范围 | [109, 1819] |
| quantized 值全局范围 | [-1.3809, 1.4229] |
| 总向量数（所有文件） | 2422 |

---

## 5. codes 和 quantized_windows 的对应关系

### 5.1 一一对应关系

```
codes_windows[i][j]  <-->  quantized_windows[i][j]
  (第i个窗口的第j个码)  <-->  (第i个窗口的第j个向量)
```

- 每个码索引对应一个 768 维的量化向量
- 码是离散的整数（0-2047 范围内）
- 向量是连续的浮点数表示

### 5.2 语义含义

- **码索引**: 表示该时间步在码本中的离散位置
- **量化向量**: 表示该时间步在隐藏空间中的连续表示
- 两者都编码了相同时间步的动作信息，只是表示形式不同

---

## 6. 推荐的读取和解析方式

### 6.1 基础读取

```python
import json
import numpy as np

# 读取 JSON 文件
with open('segment_0000_startwin_000081.codes.json', 'r') as f:
    data = json.load(f)

# 访问码索引
codes_windows = data['codes_windows']  # List[List[int]]
print(f"窗口数: {len(codes_windows)}")
print(f"第一个窗口的码: {codes_windows[0]}")

# 访问量化向量
quantized_windows = data['quantized_windows']  # List[List[List[float]]]
print(f"第一个窗口的向量数: {len(quantized_windows[0])}")
print(f"第一个向量的维度: {len(quantized_windows[0][0])}")
```

### 6.2 转换为 NumPy 数组

```python
# 将所有向量转换为 NumPy 数组
all_vectors = []
for window in quantized_windows:
    for vector in window:
        all_vectors.append(vector)

vectors_array = np.array(all_vectors, dtype=np.float32)
print(f"向量数组形状: {vectors_array.shape}")  # (42, 768)
```

### 6.3 批量处理多个文件

```python
import glob

json_files = glob.glob("code_indices/**/*.codes.json", recursive=True)

all_vectors = []
all_codes = []

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 收集向量
    for window in data['quantized_windows']:
        for vector in window:
            all_vectors.append(vector)

    # 收集码
    for window in data['codes_windows']:
        all_codes.extend(window)

vectors_array = np.array(all_vectors, dtype=np.float32)
codes_array = np.array(all_codes, dtype=np.int32)
```

---

## 7. UMAP 可视化数据提取

### 7.1 用于 UMAP 的向量提取

UMAP 需要输入形状为 `(n_samples, n_features)` 的数据。有两种提取方式：

#### 方式1: 每个向量单独一行（推荐用于细粒度分析）

```python
def extract_vectors_unflatten(json_path):
    """每个时间步的向量单独一行"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    vectors = []
    for window in data['quantized_windows']:
        for vector in window:
            vectors.append(vector)

    return np.array(vectors, dtype=np.float32)

# 使用示例
vectors = extract_vectors_unflatten('segment_0000_startwin_000081.codes.json')
print(vectors.shape)  # (42, 768)
```

#### 方式2: 每个片段展平为单个向量（推荐用于片段级别分析）

```python
def extract_vectors_flatten(json_path):
    """将整个片段的所有向量展平为单个向量"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_values = []
    for window in data['quantized_windows']:
        for vector in window:
            all_values.extend(vector)

    return np.array([all_values], dtype=np.float32)

# 使用示例
vectors = extract_vectors_flatten('segment_0000_startwin_000081.codes.json')
print(vectors.shape)  # (1, 32256)
```

### 7.2 批量提取用于 UMAP

```python
def batch_extract_for_umap(folder_path, flatten=False):
    """批量提取所有文件的向量用于 UMAP"""
    json_files = sorted(glob.glob(f"{folder_path}/**/*.codes.json", recursive=True))

    all_vectors = []
    file_names = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if flatten:
            # 片段级别
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

    return np.array(all_vectors, dtype=np.float32), file_names

# 使用示例
vectors, names = batch_extract_for_umap('code_indices', flatten=False)
print(f"向量形状: {vectors.shape}")  # (2422, 768)
```

### 7.3 UMAP 可视化完整示例

```python
import umap
import matplotlib.pyplot as plt

# 提取向量
vectors, file_names = batch_extract_for_umap('code_indices', flatten=False)

# 运行 UMAP
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(vectors)

# 可视化
plt.figure(figsize=(12, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=10)
plt.title('UMAP Visualization of Quantized Vectors')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('umap_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. 可解释性分析建议

### 8.1 码分布分析

```python
from collections import Counter

# 统计码的分布
all_codes = []
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    for window in data['codes_windows']:
        all_codes.extend(window)

code_counts = Counter(all_codes)
print(f"使用的码数: {len(code_counts)}")
print(f"最常用的10个码: {code_counts.most_common(10)}")
```

### 8.2 向量聚类分析

```python
from sklearn.cluster import KMeans

# 对向量进行聚类
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(vectors)

# 分析每个聚类中的码分布
for cluster_id in range(10):
    cluster_mask = clusters == cluster_id
    cluster_codes = [all_codes[i] for i in range(len(all_codes)) if cluster_mask[i]]
    print(f"Cluster {cluster_id}: {Counter(cluster_codes).most_common(5)}")
```

### 8.3 时间序列分析

```python
# 分析码序列的时间演变
for json_file in json_files[:3]:
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 连接所有窗口的码
    full_sequence = []
    for window in data['codes_windows']:
        full_sequence.extend(window)

    print(f"File: {Path(json_file).name}")
    print(f"  码序列: {full_sequence}")
    print(f"  序列长度: {len(full_sequence)}")
```

---

## 9. 常见问题解答

**Q: 为什么 codes 和 quantized_windows 的数量相同？**
A: 因为它们是同一时间步的两种不同表示。码是离散的，向量是连续的。

**Q: 向量维度为什么是 768？**
A: 这是模型的隐藏层维度（hidden_dim），在配置中定义。

**Q: 码的取值范围为什么是 [109, 1819]？**
A: 这取决于 codebook_size（本例为 2048）和 FSQ 的量化方式。

**Q: 如何使用这些数据进行动作识别？**
A: 可以使用码序列作为离散符号序列进行分类，或使用向量进行连续表示学习。

---

## 10. 参考信息

- **模型**: MotionTokenizer (基于 FSQ 量化)
- **编码维度**: 768
- **码本大小**: 2048
- **窗口配置**: T=15, stride=4, target_fps=10
- **输出格式**: JSON (UTF-8 编码)
- **生成来源**: stream_inference.py (在线流式推理)

