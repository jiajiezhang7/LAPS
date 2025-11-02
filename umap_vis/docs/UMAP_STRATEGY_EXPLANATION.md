# UMAP 可视化策略详细解释与改进方案

## 你的目标

1. 提取所有 `codes_indices` 文件夹下 JSON 文件中的 `quantized_windows`
2. 对其做**向量平均**（每个文件/片段平均为一个向量）
3. 用 UMAP 降维并可视化，观察是否呈现**结构化的簇类**

---

## 当前 UMAP 做法的问题

### 当前方法（advanced_codes_analysis.py）

```python
# 当前做法：提取所有向量，每个向量单独一行
vectors, _ = extract_all_vectors(flatten=False)
# 结果形状: (2422, 768)
# 即：2422 个时间步的向量，每个 768 维

reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(vectors)
```

### 问题分析

| 问题 | 影响 |
|------|------|
| **粒度太细** | 2422 个向量来自 46 个文件，每个文件内的向量高度相关，导致 UMAP 主要捕捉**时间序列内部的连续性**而非**文件间的差异** |
| **缺乏聚类信息** | 无法观察到**片段级别的结构化簇类**，只能看到时间步级别的连续流形 |
| **信息冗余** | 同一片段内的 14 个向量非常相似，造成数据冗余 |
| **难以解释** | 2422 个点的可视化很难识别出有意义的簇类 |

### 当前 UMAP 可视化的实际含义

```
当前 UMAP 2D 图显示的是：
- X 轴、Y 轴：768 维向量空间的 2D 投影
- 每个点：一个时间步的向量
- 点的分布：反映时间步向量的相似性结构
- 问题：无法看出"哪些片段相似"，只能看出"哪些时间步相似"
```

---

## 改进方案：片段级别的向量平均

### 核心思想

```
原始数据结构：
File 1: [v1, v2, v3, ..., v14]  (14 个向量)
File 2: [v1, v2, v3, ..., v14]  (14 个向量)
...
File 46: [v1, v2, v3, ..., v14] (14 个向量)
总计：2422 个向量

改进后：
File 1: mean([v1, v2, ..., v14]) = avg_vector_1  (1 个向量)
File 2: mean([v1, v2, ..., v14]) = avg_vector_2  (1 个向量)
...
File 46: mean([v1, v2, ..., v14]) = avg_vector_46 (1 个向量)
总计：46 个向量
```

### 优势

1. **片段级别的聚类** - 可以观察到相似片段的聚集
2. **信息浓缩** - 每个文件用一个代表性向量表示
3. **易于解释** - 46 个点的可视化清晰易读
4. **结构化簇类** - 如果存在动作类别，会形成明显的簇

### 改进后 UMAP 的含义

```
改进后 UMAP 2D 图显示的是：
- X 轴、Y 轴：片段平均向量的 2D 投影
- 每个点：一个视频片段的平均表示
- 点的分布：反映片段间的相似性结构
- 优势：可以清晰看出"哪些片段相似"，形成簇类
```

---

## 实现步骤

### 步骤 1: 计算每个文件的平均向量

```python
def compute_segment_average_vectors(folder_path):
    """计算每个片段的平均向量"""
    json_files = sorted(glob.glob(f"{folder_path}/**/*.codes.json", recursive=True))
    
    segment_vectors = []
    segment_names = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 提取所有向量
        all_vectors = []
        for window in data['quantized_windows']:
            for vector in window:
                all_vectors.append(vector)
        
        # 计算平均向量
        avg_vector = np.mean(all_vectors, axis=0)
        segment_vectors.append(avg_vector)
        segment_names.append(Path(json_file).stem)
    
    return np.array(segment_vectors, dtype=np.float32), segment_names
```

### 步骤 2: 应用 UMAP 降维

```python
import umap

# 获取平均向量 (46, 768)
avg_vectors, segment_names = compute_segment_average_vectors(folder_path)

# UMAP 降维到 2D
reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=15,      # 考虑 15 个最近邻
    min_dist=0.1,        # 最小距离
    metric='euclidean'
)
embedding_2d = reducer.fit_transform(avg_vectors)

# 也可以降维到 3D
reducer_3d = umap.UMAP(
    n_components=3,
    random_state=42,
    n_neighbors=15,
    min_dist=0.1
)
embedding_3d = reducer_3d.fit_transform(avg_vectors)
```

### 步骤 3: 可视化

```python
import matplotlib.pyplot as plt

# 2D 可视化
plt.figure(figsize=(12, 8))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=100, alpha=0.6, c=range(46), cmap='tab20')
for i, name in enumerate(segment_names):
    plt.annotate(name, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=8)
plt.title('UMAP 2D: Segment-Level Average Vectors')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True, alpha=0.3)
plt.savefig('umap_segment_average_2d.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 预期结果

### 如果存在结构化簇类

```
UMAP 图中会显示：
- 多个明显的点群（簇）
- 簇内的点紧密聚集
- 簇间的点相距较远
- 可能对应不同的动作类别或视频内容
```

### 如果没有明显簇类

```
UMAP 图中会显示：
- 点均匀分散
- 没有明显的聚集模式
- 表示片段间的向量差异较大，没有明显的重复模式
```

---

## 进一步分析建议

### 1. 聚类分析

```python
from sklearn.cluster import KMeans

# 对平均向量进行 K-means 聚类
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(avg_vectors)

# 在 UMAP 图上用颜色表示聚类结果
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters, cmap='viridis', s=100)
```

### 2. 相似性分析

```python
from sklearn.metrics.pairwise import cosine_similarity

# 计算片段间的相似性
similarity_matrix = cosine_similarity(avg_vectors)

# 找出最相似的片段对
for i in range(46):
    for j in range(i+1, 46):
        if similarity_matrix[i, j] > 0.95:
            print(f"{segment_names[i]} <-> {segment_names[j]}: {similarity_matrix[i, j]:.4f}")
```

### 3. 密度分析

```python
from sklearn.neighbors import NearestNeighbors

# 计算每个点的 k-NN 距离
nbrs = NearestNeighbors(n_neighbors=5).fit(avg_vectors)
distances, indices = nbrs.kneighbors(avg_vectors)

# 高密度区域（距离小）可能表示相似的片段
```

---

## 参数调优建议

| 参数 | 当前值 | 调整建议 |
|------|--------|---------|
| `n_neighbors` | 15 | 对于 46 个点，可以尝试 5-10（更强调局部结构） |
| `min_dist` | 0.1 | 可以尝试 0.05-0.2（控制点的紧凑程度） |
| `metric` | euclidean | 可以尝试 cosine（对于高维向量更合适） |
| `n_components` | 2/3 | 保持不变 |

---

## 总结

| 方面 | 当前方法 | 改进方法 |
|------|---------|---------|
| **数据粒度** | 2422 个时间步向量 | 46 个片段平均向量 |
| **可视化目标** | 时间步相似性 | 片段相似性 |
| **簇类观察** | 难以识别 | 清晰可见 |
| **可解释性** | 低（点太多） | 高（点少，易标注） |
| **适用场景** | 时间序列分析 | 片段分类/聚类 |

**建议**: 立即采用改进方法，以达成你的目标。

