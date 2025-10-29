# HDBSCAN 无监督聚类调参与诊断记录（2025-10-29）

- **数据规模**: 15002 embeddings（d=256，已 L2 归一化）
- **当前结果**: 噪声率 95.45%，9 个簇（聚类样本约 683）
- **主要指标（非噪声子集）**: silhouette=0.3387，CH=294.31，DB=1.10
- **当前 HDBSCAN 配置**: min_cluster_size=20, min_samples=3, metric=euclidean, cluster_selection_method=leaf, cluster_selection_epsilon=0.0
- **UMAP 可视化**: show_noise=true 时报错（索引维度不匹配）

---

## 一、问题与原因分析

- **[极高噪声率]** 大量样本被判为噪声（-1），仅 4.55% 样本被聚类。
  - **高维稀疏密度**: 在 d=256 的 L2 空间，邻接半径很小才能形成“高密度”区域，HDBSCAN 对稀疏背景更倾向判噪声。
  - **cluster_selection_method=leaf**: 倾向于保留细小“叶子”簇，边界样本较易被丢为噪声。通常 `eom` 更稳健、更“合并”。
  - **未使用 PCA**: 高维距离集中化导致密度估计更困难；适度降维（如 32–64）可缓解。
  - **metric=euclidean**: 对已 L2 归一化向量，余弦/欧氏等价性在阈值上仍可能影响密度图形；实务中 `cosine` 往往更贴合语义相似性。
  - **min_cluster_size 与数据结构不匹配**: 目前能形成的稳定簇较少，许多样本处于过渡/长尾区，导致被判噪声。

- **[UMAP 报错]**
  - 错误: `boolean index did not match ... dimension is 683 but ... 15002`
  - 根因: `run_umap_scatter()` 在 `show_noise=true` 路径下，传入了已掩码的 `y_pred[mask]`，却用 `noise_mask (len=N)` 去索引，发生长度不一致。
  - 影响: 仅影响可视化与诊断，不影响聚类结果文件。

---

## 二、参数影响方向（如何“降低噪声、保留结构”）

- **[method] leaf → eom**
  - 方向: 合并叶子到更稳定的父簇，扩大簇体积、降低噪声；簇数通常减少、簇内更稳。
- **[metric] euclidean ↔ cosine**
  - 方向: 在 L2 归一化场景，`cosine` 常能形成更符合语义的邻域密度；预期对“边界点吸收”为正向。
- **[PCA] 启用 32–64 维**
  - 方向: 降低维度带来的距离集中效应，使密度估计更清晰；通常可显著降低噪声。
- **[min_samples] 小一些（3–10）**
  - 方向: 较小的 `min_samples` 使更多点成为核心点，倾向降低噪声；过小可能引入链式效应，建议 3/5/10 小范围试探。
- **[min_cluster_size] 20/40/80 小网格**
  - 方向: 过小会产生大量微簇（不一定降低噪声），过大则更保守；建议与 `eom+PCA` 联动试探。
- **[cluster_selection_epsilon] 轻微正值（吸收边界）**
  - 方向: 设置小正数可把靠近簇边界的点吸收进簇，降低噪声；过大可能错误合并。
  - 建议范围: cosine: 0–0.05；euclidean: 0–0.1（需以数据尺度为准分步增大）。

---

## 三、推荐两阶段调参策略

- **阶段A（结构性调整，期望先显著降噪）**
  - **启用 PCA**: 64 维（min_dims=16, explained_variance_threshold=0.99）
  - **method**: `eom`
  - **metric**: 先试 `cosine`（保留 `euclidean` 作为对照）
  - **min_samples**: 先用 5（在 3 和 10 之间居中）
  - **min_cluster_size**: 40（在 20/40/80 中居中）
  - **epsilon**: 0.02（cosine），或 0.05（euclidean）

- **阶段B（局部网格微调）**
  - 固定阶段A中表现较好的 `metric` 与 `method`，扫描：
    - `min_cluster_size`: [20, 40, 80]
    - `min_samples`: [3, 5, 10]
    - `cluster_selection_epsilon`（cosine）: [0.0, 0.02, 0.05]
    - `cluster_selection_epsilon`（euclidean）: [0.0, 0.05, 0.10]

- **评价准则（选择最优）**
  - 噪声率（越低越好）
  - silhouette（≥0.25 为宜；当前 0.339 基于较少样本，后续以“降噪不显著恶化 silhouette”为目标）
  - 簇数与簇大小分布（避免绝大多数是微簇）

---

## 四、对 eval_config.yaml 的建议修改片段（可直接替换相应段）

```yaml
preprocessing:
  lstm_feature:
    pca:
      max_dims: 64
      explained_variance_threshold: 0.99
      min_dims: 16
    l2_normalize: true

clustering:
  hdbscan:
    min_cluster_size: 40
    min_samples: 5
    metric: cosine           # 对照可试 euclidean
    cluster_selection_epsilon: 0.02
    cluster_selection_method: eom

visualization:
  umap:
    enabled: true
    show_noise: true         # 若仍报错，临时改为 false
```

> 说明：`hdb_min_cluster_size_grid` 是旧评估脚本用的网格参数，`fit_hdbscan.py` 并不读取该键；以 `clustering.hdbscan.*` 为准。

---

## 五、运行与记录（示例）

- **前置**: 确保已激活环境
  - `conda activate laps`
- **运行（示例）**:
  - `python -m action_classification.clustering.fit_hdbscan \
      --embed-dir action_classification/seq_embed_infer/20251029_143628 \
      --config action_classification/configs/eval_config.yaml \
      --out-dir action_classification/results/hdbscan_fits/`
- **记录模板**（建议每次运行填一行）

| date | pca_dims | metric | method | mcs | ms | eps | noise_rate | n_clusters | silhouette |
|------|----------|--------|--------|-----|----|-----|------------|------------|------------|
| 2025-10-29 | 0 | euclidean | leaf | 20 | 3 | 0.0 | 0.9545 | 9 | 0.3387 |
|            | 64 | cosine   | eom  | 40 | 5 | 0.02|        |            |        |

> mcs=min_cluster_size, ms=min_samples, eps=cluster_selection_epsilon

---

## 六、UMAP 可视化报错的临时规避与最小修复建议

- **临时规避**: 在 `visualization.umap.show_noise` 设为 `false`，仅画非噪声子集，避免维度不匹配。
- **最小修复建议（开发者参考）**:
  - 在 `fit_hdbscan.py` 的 `show_noise=true` 分支中，调用 `run_umap_scatter()` 时，应传入未掩码的 `y_pred` 或在函数内统一使用 `y_pred_full` 与 `noise_mask` 来构造掩码；避免用 `len=N` 的布尔掩码去索引 `len=assigned` 的数组。

---

## 七、预期与下一步

- 按“阶段A”修改后，预计噪声率明显下降（通常先从 95%→中等水平），簇数减少、单簇更稳定；若 silhouette 下降不明显，即可进入“阶段B”做小网格微调。
- 若始终存在高噪声：
  - 再提高 PCA 维度上限（如 96），或增大 `epsilon` 小幅度步进；
  - 结合 UMAP 低维可视化诊断异常与长尾分布；
  - 作为对照，可用 `bayes_gmm` 做固定上界的软聚类评估整体可分性。
