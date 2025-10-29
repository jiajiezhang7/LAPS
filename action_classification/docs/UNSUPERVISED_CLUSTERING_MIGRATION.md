# 无监督聚类迁移完成报告

## 概述
已成功将整个聚类流程转换为**完全无监督模式**，所有算法现在仅基于数据自身的嵌入特征进行聚类，完全不依赖任何标签信息。

## 修改的文件

### 1. `/action_classification/clustering/fit_bayes_gmm.py`
**主要变更**:
- ✅ 使用 `load_embed_dir_unlabeled()` 替代 `load_embed_dir()`，仅加载 `embed.npy`
- ✅ 移除所有标签加载逻辑（`labels.npy`, `label_names.txt`）
- ✅ 移除 `merge_map` 标签合并功能
- ✅ 删除监督评估指标：
  - 准确率 (accuracy)
  - 归一化互信息 (NMI)
  - 调整兰德指数 (ARI)
  - F1 分数
  - 混淆矩阵
  - Hungarian 对齐算法
- ✅ 新增无监督评估指标：
  - Silhouette Score（轮廓系数）
  - Calinski-Harabasz Index
  - Davies-Bouldin Index
  - 簇大小统计
- ✅ 简化 UMAP 可视化，仅显示聚类结果（不再显示"真实标签"）
- ✅ 输出文件重命名：`train_assignments.jsonl` → `cluster_assignments.jsonl`
- ✅ 输出格式简化：移除 `true_label` 字段，仅保留 `cluster` 字段

### 2. `/action_classification/clustering/fit_hdbscan.py`
**主要变更**:
- ✅ 使用 `load_embed_dir_unlabeled()` 替代 `load_embed_dir()`
- ✅ 移除所有标签依赖和监督评估指标（同上）
- ✅ 保留噪声点检测功能（HDBSCAN 特有的 `-1` 标签）
- ✅ 新增无监督评估指标（同上）
- ✅ 简化 UMAP 可视化，支持噪声点显示
- ✅ 输出文件重命名和格式简化（同上）
- ✅ 输出格式：`cluster`, `prob`, `is_noise`

### 3. `/action_classification/configs/eval_config.yaml`
**主要变更**:
- ✅ 移除 `label_processing.merge_map` 配置块
- ✅ 添加说明注释，明确现在是完全无监督聚类

### 4. `/action_classification/scripts/infer_sequence_embed_lstm.py`
**验证结果**:
- ✅ **已经是无标签模式**！使用 `scan_unlabeled_samples()` 函数
- ✅ 仅输出 `embed.npy` 和 `paths.txt`
- ✅ **不再生成** `labels.npy` 和 `label_names.txt`
- ✅ 无需修改

## 新的工作流程

### 第1步：生成嵌入向量（无标签）
```bash
python -m action_classification.scripts.infer_sequence_embed_lstm \
  --model-pt /path/to/model_best.pt \
  --json-root /path/to/code_indices_json \
  --out-dir /path/to/embeddings \
  --l2-normalize
```

**输出文件**:
- `embed.npy`: 嵌入向量 (N, d_model)
- `paths.txt`: 原始 JSON 文件路径
- ❌ ~~labels.npy~~ (已移除)
- ❌ ~~label_names.txt~~ (已移除)

### 第2步：无监督聚类

#### 方法A: Bayesian GMM
```bash
python -m action_classification.clustering.fit_bayes_gmm \
  --embed-dir /path/to/embeddings \
  --config action_classification/configs/eval_config.yaml \
  --out-dir /path/to/bayes_gmm_output
```

#### 方法B: HDBSCAN（推荐用于异常检测）
```bash
python -m action_classification.clustering.fit_hdbscan \
  --embed-dir /path/to/embeddings \
  --config action_classification/configs/eval_config.yaml \
  --out-dir /path/to/hdbscan_output
```

**输出文件**:
- `model_*.pkl`: 聚类模型
- `cluster_meta.json`: 元数据（用于在线推理）
- `aggregate_results.json`: 无监督评估指标
- `cluster_assignments.jsonl`: 每个样本的聚类结果
- `umap/umap_clusters_lstm.png`: UMAP 可视化（仅显示聚类）

## 输出格式变化

### `cluster_assignments.jsonl` 格式

**Bayesian GMM**:
```json
{
  "index": 0,
  "path": "relative/path/to/sample.json",
  "cluster": 2
}
```

**HDBSCAN**:
```json
{
  "index": 0,
  "path": "relative/path/to/sample.json",
  "cluster": 2,
  "prob": 0.95,
  "is_noise": false
}
```

### `aggregate_results.json` 格式

```json
{
  "method": "hdbscan",
  "num_samples": 1000,
  "noise_rate": 0.05,
  "num_pred_clusters": 8,
  "cluster_sizes": {
    "-1": 50,
    "0": 120,
    "1": 150,
    ...
  },
  "metrics": {
    "silhouette": 0.45,
    "calinski_harabasz": 234.56,
    "davies_bouldin": 0.78
  },
  "umap": {
    "metric": "cosine",
    "embeddings_path": "/path/to/umap_clusters_lstm.png",
    "unique_cluster_ids": [-1, 0, 1, 2, 3, 4, 5, 6, 7]
  }
}
```

## 无监督评估指标说明

### Silhouette Score（轮廓系数）
- **范围**: [-1, 1]
- **越高越好**: 接近 1 表示簇内紧密、簇间分离良好
- **解释**: 衡量样本与其所属簇的相似度相对于其他簇的相似度

### Calinski-Harabasz Index
- **范围**: [0, ∞)
- **越高越好**: 值越大表示簇间分离度越好
- **解释**: 簇间方差与簇内方差的比值

### Davies-Bouldin Index
- **范围**: [0, ∞)
- **越低越好**: 接近 0 表示簇间分离良好
- **解释**: 簇内距离与簇间距离的平均比值

## 关键优势

1. ✅ **完全无监督**: 不依赖任何人工标注或文件夹结构
2. ✅ **数据驱动**: 纯粹基于嵌入向量的内在结构进行聚类
3. ✅ **适用于新数据**: 可直接应用于任意未标注的数据集
4. ✅ **保留噪声检测**: HDBSCAN 仍可识别异常样本
5. ✅ **可解释性**: 提供多种无监督评估指标
6. ✅ **向后兼容**: 保留所有模型保存和在线推理功能

## 注意事项

1. **LSTM 训练阶段**: 如果需要从头训练 LSTM 模型，仍需要标签（用于 NTP 损失）。但推理和聚类阶段完全无监督。
2. **评估指标**: 无监督指标仅能评估聚类质量，无法评估与"真实类别"的一致性（因为不再使用真实标签）。
3. **簇数量**: Bayesian GMM 会自动推断有效簇数；HDBSCAN 会自动发现簇并标记噪声点。

## 迁移完成 ✅

所有代码已按照"最小侵入原则"修改完成，现在可以直接使用无监督聚类流程。
