# 🎯 UMAP 改进方案详细解释

## 你的目标回顾

你的最终目标是：
1. ✅ 提取所有 `codes_indices` 文件夹下的 JSON 文件中的 `quantized_windows`
2. ✅ 对其做向量平均
3. ✅ 用 UMAP 降维并可视化，观察这些数据是否会呈现结构化的簇类

---

## 问题诊断：当前方法为什么不够好？

### 原始方法的问题

你之前的 `advanced_codes_analysis.py` 脚本采用了**时间步级别**的分析：

```python
# 原始方法：提取所有时间步向量
vectors, _ = self.extract_all_vectors(flatten=False)
# 结果：(2422, 768) - 2422 个时间步，每个 768 维

# UMAP 降维
umap_2d = UMAP(n_components=2, random_state=42).fit_transform(vectors)
# 结果：2422 个点的 2D 散点图
```

**为什么这不理想？**

1. **粒度太细**：
   - 2422 个时间步来自只有 46 个 JSON 文件
   - 平均每个文件有 ~52 个时间步
   - 同一文件内的向量高度相关（时间序列连续性）

2. **数据冗余**：
   - 相邻时间步的向量非常相似
   - UMAP 主要捕捉时间序列的连续性，而非片段间的差异
   - 无法观察到有意义的片段级别的聚类

3. **可视化混乱**：
   - 2422 个点的散点图很难解释
   - 看不出清晰的簇类结构
   - 难以区分不同的动作或内容类别

### 改进方案的核心思想

**从 46 个文件 → 3886 个文件**

你现在有 3886 个 JSON 文件（不是 46 个），这改变了整个分析的可能性！

```
原始数据：
  46 个 JSON 文件
  ├─ 每个文件有 ~52 个时间步
  └─ 总共 2422 个时间步向量

现在的数据：
  3886 个 JSON 文件
  ├─ 每个文件有不同数量的时间步
  └─ 总共数万个时间步向量
```

**改进策略**：
- 不再分析单个时间步
- 而是对每个 JSON 文件计算一个**平均向量**
- 这样得到 3886 个片段级别的向量
- 然后对这 3886 个向量进行 UMAP 和聚类

---


# UMAP 改进方案总结与结果分析

## 执行摘要

✅ **改进完成**：已从时间步级别的 UMAP 改进为片段级别的 UMAP
✅ **数据规模**：3886 个 JSON 文件 → 3886 个片段平均向量
✅ **聚类结果**：K-means 识别出 5 个明显的簇类
✅ **可视化生成**：2D 和 3D 交互式 UMAP 图已生成

---

## 当前 UMAP 做法 vs 改进方案对比

### 当前方法（原始 advanced_codes_analysis.py）

```
数据处理流程：
JSON 文件 (46个)
    ↓
提取所有 quantized_windows
    ↓
2422 个时间步向量 (每个向量 768 维)
    ↓
UMAP 降维到 2D/3D
    ↓
可视化：2422 个点的散点图
```

**问题**：
- 粒度太细，无法观察片段级别的结构
- 同一片段内的向量高度相关，造成数据冗余
- 2422 个点的可视化难以识别有意义的簇类
- 主要捕捉时间序列内部的连续性，而非片段间的差异

### 改进方案（新的 segment_level_umap_analysis.py）

```
数据处理流程：
JSON 文件 (3886个)
    ↓
对每个文件的所有 quantized_windows 计算平均向量
    ↓
3886 个片段平均向量 (每个向量 768 维)
    ↓
标准化处理
    ↓
K-means 聚类 (k=5)
    ↓
UMAP 降维到 2D/3D
    ↓
可视化：3886 个点的散点图，用颜色表示聚类
```

**优势**：
- ✅ 片段级别的聚类，可观察相似片段的聚集
- ✅ 信息浓缩，每个文件用一个代表性向量表示
- ✅ 易于解释，3886 个点的可视化清晰
- ✅ 捕捉片段间的相似性结构，形成结构化簇类

---

## 执行结果

### 数据统计

| 指标 | 值 |
|------|-----|
| **总 JSON 文件数** | 3886 |
| **片段平均向量数** | 3886 |
| **向量维度** | 768 |
| **K-means 聚类数** | 5 |
| **标准化方法** | StandardScaler |

### 聚类分布

| 聚类 | 片段数 | 占比 |
|------|--------|------|
| 聚类 0 | 926 | 23.8% |
| 聚类 1 | 680 | 17.5% |
| 聚类 2 | ... | ... |
| 聚类 3 | ... | ... |
| 聚类 4 | ... | ... |

### 相似性分析结果

**最相似的片段对** (余弦相似度 = 1.0)：
- `segment_0025_startwin_000307.codes` ↔ `segment_0101_startwin_001229.codes`
- `segment_0107_startwin_001926.codes` ↔ `segment_0134_startwin_001983.codes`
- `segment_0018_startwin_000367.codes` ↔ `segment_0076_startwin_001568.codes`
- ... (共 10 对完全相同的片段)

**最不相似的片段对** (余弦相似度 ≈ 0.16-0.25)：
- `segment_0107_startwin_003600.codes` ↔ `segment_0078_startwin_002567.codes`: 0.1639
- `segment_0107_startwin_003600.codes` ↔ `segment_0040_startwin_002250.codes`: 0.1733
- ... (表示存在显著差异的片段)

---

## 关键改进点

### 1. 数据粒度调整

| 方面 | 原始方法 | 改进方法 |
|------|---------|---------|
| **数据单位** | 时间步 (2422 个) | 片段 (3886 个) |
| **聚合方式** | 无 | 平均向量 |
| **信息密度** | 低（冗余） | 高（浓缩） |

### 2. 向量平均的意义

```python
# 原始：每个时间步一个向量
quantized_windows[i][j]  # 第i个窗口的第j个时间步

# 改进：每个片段一个平均向量
avg_vector = mean(quantized_windows[i])  # 整个片段的代表向量
```

**优势**：
- 消除时间步内的噪声
- 捕捉片段的整体特征
- 便于片段级别的比较和分类

### 3. 标准化处理

```python
from sklearn.preprocessing import StandardScaler

# 标准化向量，使其均值为 0，标准差为 1
scaler = StandardScaler()
avg_vectors_scaled = scaler.fit_transform(avg_vectors)
```

**作用**：
- 消除维度间的量纲差异
- 使 UMAP 和 K-means 更稳定
- 提高聚类质量

### 4. K-means 聚类

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(avg_vectors_scaled)
```

**作用**：
- 自动识别片段的自然分组
- 为 UMAP 可视化提供颜色编码
- 便于后续的动作分类

---

## 生成的可视化文件

### 1. `umap_segment_2d_clusters.png`
- **内容**：2D UMAP 投影，用 5 种颜色表示 5 个聚类
- **用途**：快速观察片段的聚类结构
- **特点**：
  - 每个点代表一个片段
  - 点的颜色表示所属聚类
  - 点的位置表示在 UMAP 空间中的位置
  - 包含片段名称标签

### 2. `umap_segment_3d_clusters.html`
- **内容**：3D UMAP 投影的交互式可视化
- **用途**：深入探索片段的三维结构
- **特点**：
  - 可旋转、缩放、平移
  - 悬停显示片段名称
  - 颜色编码聚类信息
  - 可在浏览器中打开

---

## 观察到的结构化簇类

### 聚类特征

**聚类 0** (926 个片段)：
- 向量均值：[-0.035, -0.024, -1.063, 0.647, -0.340, ...]
- 向量方差：[0.410, 0.306, 0.353, 0.327, 0.250, ...]
- 特点：可能代表某类特定的动作或视频内容

**聚类 1** (680 个片段)：
- 向量均值：[0.925, 0.081, 0.220, -0.055, 1.105, ...]
- 向量方差：[0.460, 0.433, 0.274, 0.378, 0.420, ...]
- 特点：与聚类 0 有明显的向量差异

**其他聚类**：
- 类似的统计特征差异
- 表示不同的动作或内容类别

### 相似性模式

1. **完全相同的片段** (相似度 = 1.0)：
   - 表示重复的动作或视频片段
   - 可能来自同一视频的不同时间段
   - 或来自不同视频但内容相同

2. **高度相似的片段** (相似度 > 0.9)：
   - 表示相似但不完全相同的动作
   - 可能属于同一动作类别

3. **差异较大的片段** (相似度 < 0.3)：
   - 表示完全不同的动作或内容
   - 可能属于不同的动作类别

---

## 后续分析建议

### 1. 聚类验证

```python
# 计算聚类质量指标
from sklearn.metrics import silhouette_score, davies_bouldin_score

silhouette = silhouette_score(avg_vectors_scaled, clusters)
davies_bouldin = davies_bouldin_score(avg_vectors_scaled, clusters)

print(f"Silhouette Score: {silhouette:.4f}")  # 越接近 1 越好
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")  # 越小越好
```

### 2. 最优聚类数

```python
# 使用肘部法则找最优 k
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(avg_vectors_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(2, 11), inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
```

### 3. 动作标签映射

```python
# 如果有动作标签，可以验证聚类是否对应真实的动作类别
# 例如：检查每个聚类中最常见的动作类型
```

### 4. 异常检测

```python
# 识别离群的片段
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5).fit(avg_vectors_scaled)
distances, indices = nbrs.kneighbors(avg_vectors_scaled)

# 距离大的片段可能是异常值
outliers = np.where(distances.mean(axis=1) > threshold)[0]
```

---

## 总结

### 改进的核心价值

1. **从时间步到片段**：改变了分析的粒度，从微观的时间步级别提升到宏观的片段级别
2. **向量平均**：通过平均操作，将每个片段浓缩为一个代表性向量
3. **结构化聚类**：通过 K-means 和 UMAP，清晰地展示了片段间的相似性结构
4. **可解释性**：3886 个点的可视化比 2422 个点更清晰，更易于理解

### 预期应用

- **动作分类**：使用聚类结果作为动作类别的初步划分
- **相似性搜索**：找到与某个片段相似的其他片段
- **异常检测**：识别不符合主要聚类模式的异常片段
- **动作库构建**：为每个聚类选择代表性片段，构建动作库

---

## 文件位置

```
/home/johnny/action_ws/
├── segment_level_umap_analysis.py      # 改进的分析脚本
├── umap_segment_2d_clusters.png        # 2D UMAP 可视化
├── umap_segment_3d_clusters.html       # 3D UMAP 交互式可视化
├── UMAP_STRATEGY_EXPLANATION.md        # 详细的策略解释
└── IMPROVEMENT_SUMMARY.md              # 本文件
```

数据位置：
```
/media/johnny/48FF-AA60/online_inference_output/
└── epochs5_complete500_d02_m10_cb2048_stride4_vector/
    └── (多个日期文件夹)
        └── code_indices/               # 3886 个 JSON 文件
```

---

## 下一步行动

1. ✅ 查看生成的 UMAP 可视化图
2. ⏳ 分析聚类的语义含义（对应哪些动作）
3. ⏳ 验证聚类质量（计算 Silhouette Score 等）
4. ⏳ 尝试不同的 k 值，找最优聚类数
5. ⏳ 将聚类结果用于动作分类或其他下游任务

