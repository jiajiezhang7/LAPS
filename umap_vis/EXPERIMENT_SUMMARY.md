# 动作片段聚类实验总结（UMAP + 聚类 + 多数据源对比）


## （更新）实验进展概览
- 已完成阶段：
  - HDBSCAN 密度聚类（双口径评估：核心点/全样本映射）——因无法同时达成“低噪声（≤10%）+ 高质量全样本指标”已暂停
  - 轻量级序列模型（Transformer encoder，冻结推理）段级 embedding 提取——已完成微网格搜索，并集成“最佳配置”到主流程
- 当前状态：
  - 主推荐为“序列模型 embedding（pooling=mean, d_model=256, n_layers=4, n_heads=4）+ KMeans(k=3)”
  - 保留 attn_norm 聚合作为回退/对照
  - HDBSCAN 保留为辅助模块（核心簇/噪声识别），暂不作为主聚类方案
- 下一步计划（按需）：
  - 若强制 5 类输出：建议“层级聚类”（先 k=3 得元类别，再在最大簇细分）优于直接 k=5
  - 可选进行短时自监督微调（1–3h，InfoNCE/时序遮挡等）以提升细粒度可分性
  - 可选增加谱聚类等算法对比

## （更新）关键指标汇总（以 6444 段、cosine 距离为准）

### 1) 基线（attn_norm 聚合 + KMeans）
| 最佳 k | Silhouette | DB | CH | Intra/Inter |
|---:|---:|---:|---:|---:|
| 4 | 0.4985 | 1.0406 | 3523.61 | 0.2815 |

数据来源：umap_vis/statistics/cluster_metrics_attn_norm_cosine.csv

### 2) HDBSCAN（已暂停）
- 初始网格（代表性最佳，口径A=核心点，仅 labels!=-1）
  - 配置：min_cluster_size=100, min_samples=8, mapping=centroid（度量：cosine→L2 归一化后用欧氏等价）
  - 指标（Core）：Sil=0.6733，DB=0.6780，CH=2297.59，Intra/Inter=0.1509，n_clusters=4，noise_ratio=0.6306
  - 指标（All mapped）：Sil=0.4531，DB=1.1111，CH=2845.46，Intra/Inter=0.3031
- 调优后网格（更包容的 min_cluster_size/min_samples，目标降低噪声）
  - 最佳（示例）：min_cluster_size=60~120, min_samples=10~15 范围内最优时 noise_ratio≈0.448（仍远高于 0.10 目标）
  - 指标（Core）：Sil≈0.190（不达 ≥0.60 目标）
  - 指标（All mapped）：Sil≈0.209（不达 ≥0.49 目标），n_clusters=2
- 结论：在 768 维空间上，EOM 策略的 HDBSCAN 难以同时满足“低噪声率（≤10%）+ 竞争性的全样本指标”，因此暂停

数据来源：
- 网格：umap_vis/statistics/cluster_metrics_hdbscan_grid.csv（按 sil_core 降序）
- 对比：umap_vis/statistics/cluster_metrics_hdbscan_best_comparison.csv

### 3) 序列模型（冻结推理）
- 阶段1（单配置）：pool=mean, d=256, L=2, H=4，最佳 k=3
  - Sil=0.5444，DB=1.3393，CH=3114.62，Intra/Inter=0.3370
- 网格搜索全局最佳：pool=mean, d=256, L=4, H=4，最佳 k=3
  - Sil=0.5998，DB=1.2026，CH=4015.65，Intra/Inter=0.2938
  - 相对基线（attn_norm, k=4）：Sil +0.1013；Intra/Inter 略升（0.2815 → 0.2938）
- k=3 vs k=5（在最佳配置上扩展 k ∈ [2,15]）
  - k=3：Sil=0.5848，DB=1.2347，CH=3690.90，Intra/Inter=0.3079
  - k=5：Sil=0.4909，DB=1.4024，CH=2857.00，Intra/Inter=0.3001
  - 结论：k=3 明显优于 k=5（Sil/DB/CH 均更好）；k 增大后边界更不清晰，属于“在同一流形上做细分”的效应

数据来源：
- 总对比：umap_vis/statistics/sequence_model_grid_best_vs_baseline.csv
- 网格：umap_vis/statistics/sequence_model_grid_search.csv
- k 曲线：umap_vis/statistics/best_config_k_analysis.csv

## （更新）脚本功能与用途说明

### 1) segment_umap_cluster_analysis.py（基线/回退方案）
- 功能：基于连续向量（quantized_windows）做聚合（attn_norm/mean 等）+ UMAP + KMeans（k∈[k_min,k_max]）评估
- 关键参数：--agg, --metric, --neighbors, --min-dist, --k-min, --k-max
- 输出：cluster_metrics_*.csv，umap_2d/3d_*.png/html
- 运行示例：
<augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python umap_vis/scripts/segment_umap_cluster_analysis.py \
  --data-dir /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector \
  --output-dir umap_vis/figure \
  --aggs attn_norm mean --k-min 2 --k-max 10 --neighbors 15 --min-dist 0.1 --metric cosine
````
</augment_code_snippet>

### 2) hdbscan_cluster_analysis.py（已暂停的辅助模块）
- 功能：HDBSCAN 网格搜索 + 双口径评估（核心点/全样本映射）+ 最佳可视化 + 报告
- 关键参数：--min-cluster-size, --min-samples, --metric, --cluster-selection-method（固定 eom）, --mapping-method（centroid）
- 输出：cluster_metrics_hdbscan_grid.csv、cluster_metrics_hdbscan_best_comparison.csv、umap_2d/3d_hdbscan_best.*、docs/HDBSCAN_EVALUATION.md
- 运行示例：
<augment_code_snippet mode="EXCERPT">
````bash
conda run -n laps python umap_vis/scripts/hdbscan_cluster_analysis.py \
  --data-dir /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector \
  --fig-dir umap_vis/figure --stats-dir umap_vis/statistics \
  --agg attn_norm --metric cosine --neighbors 15 --min-dist 0.1
````
</augment_code_snippet>

### 3) sequence_model_embedding.py（当前推荐主流程）
- 功能：轻量级 Transformer encoder（随机初始化，冻结推理）提取段级 embedding；支持网格搜索与最佳配置运行
- 三种模式：
  1) 默认（无开关）：仅输出 attn_norm 基线评估
  2) --use-best-grid-config：使用网格全局最佳（pool=mean, d=256, L=4, H=4），可被 --d-model/--n-layers/--n-heads/--pooling 覆盖
  3) --grid-search：执行 pooling×d_model×n_layers 微网格（n_heads 随 d_model 自动取整）
- 关键参数：--d-model, --n-layers, --n-heads, --pooling, --k-analysis-max, --device
- 输出：
  - 网格：sequence_model_grid_search.csv, sequence_model_grid_best_vs_baseline.csv
  - k 分析：best_config_k_analysis.csv, best_config_metrics_vs_k.png
  - 可视化：umap_2d/3d_best_config_k3/k5.*
- 运行示例：
<augment_code_snippet mode="EXCERPT">
````bash
# 最佳配置（可覆盖）
conda run -n laps python umap_vis/scripts/sequence_model_embedding.py \
  --data-dir /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector \
  --fig-dir umap_vis/figure --stats-dir umap_vis/statistics \
  --metric cosine --neighbors 15 --min-dist 0.1 \
  --k-min 2 --k-max 10 --k-analysis-max 15 --device cpu \
  --use-best-grid-config
````
</augment_code_snippet>

## （更新）数据与环境信息
- 数据路径：/media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector
- 样本数量：6444 个片段（quantized_windows，768 维连续向量）
- 运行环境：conda 环境 laps（本次均为 CPU 推理）
- 主要依赖：numpy, scikit-learn, matplotlib, plotly, umap-learn, torch（HDBSCAN 需 hdbscan）

## （更新）结论与建议
- 当前最优方案：序列模型（pool=mean, d=256, L=4, H=4）+ KMeans(k=3)
  - 相对基线：Silhouette 提升约 +0.10；Intra/Inter 略有上升（0.2815→0.2938）
- k=3 vs k=5：k=3 为数据驱动最优；若需 5 类输出，建议采用层级聚类（k=3 → 最大簇内细分）而非直接 k=5
- HDBSCAN：保留为辅助模块（高置信核心簇检测 + 噪声识别），不作为主聚类方案

---


> 研究目标：验证动作分割系统能否识别相似/相同的动作模式；系统性比较不同数据源（连续向量 vs 离散码）、不同聚合/距离度量/表征方式对聚类质量和可视化分离度的影响。

- 数据来源：FSQ 量化后的动作片段，约 4K+ 片段（代码侧载 JSON，路径模式：`{data-dir}/**/code_indices/*.codes.json`）
- 结论预览：`quantized_windows`（连续向量）+ `attn_norm/mean` + `cosine` 在聚类与可视化上明显优于欧氏、DTW 与离散码表征（BoC/TF-IDF/BigramHash）。

---

## 1. 项目概述
- 研究目标：验证“动作分割系统”产生的片段能否在嵌入空间上呈现稳定的簇结构；比较不同向量聚合与距离度量的效果；探索离散码作为数据源的可行性。
- 数据规模：约 4,000–4,500 片段，单片段内部含若干窗口；每窗口包含：
  - `quantized_windows`：768 维连续向量序列
  - `codes_windows`：FSQ 离散码索引序列（0–2047）

## 2. 实验方法论
- 数据源对比：
  - 连续向量：`quantized_windows`（768 维浮点数）
  - 离散码：`codes_windows`（0–2047 的整数索引）
- 向量聚合（连续向量）：`mean`, `mean_std`, `max`, `first_last`, `attn_norm`
- 离散码表征：`Bag-of-Codes`（直方图，L1）、`TF-IDF`（TfidfTransformer, L2）、`BigramHash`（2-gram 哈希计数，默认 16,384 维，L1）
- 距离度量：`Euclidean`（欧氏）、`Cosine`（余弦），以及 `DTW`（子集，1D L2 + PAA=64）
- 聚类算法：KMeans，k ∈ [5, 10]，按 Silhouette 选最佳 k
- 降维可视化：UMAP（2D/3D，n_neighbors=15，min_dist=0.1）
- 运行环境：conda env `laps`

## 3. 评估指标说明
- Silhouette Score：[-1, 1]，越高越好；衡量簇内紧凑度与簇间分离度之差异
- Davies-Bouldin Index（DB）：越低越好；度量簇间相似性
- Calinski-Harabasz（CH）：越高越好；簇间离散 / 簇内离散
- Intra/Inter Distance Ratio（类内/类间均值比）：越低越好（<1 表示类间分离优于类内扩散）

## 4. 实验结果汇总（保留 3 位小数）

### 表格 1：连续向量（quantized_windows, 768D）
| 聚合 | 度量 | 最佳 k | Sil | DB | CH | Intra/Inter |
|---|---|---:|---:|---:|---:|---:|
| mean | Euclidean | 5 | 0.273 | 1.218 | 1593.584 | 0.646 |
| mean_std | Euclidean | 5 | 0.200 | 1.582 | 906.697 | 0.862 |
| max | Euclidean | 5 | 0.163 | 1.766 | 732.218 | 0.980 |
| first_last | Euclidean | 6 | 0.165 | 1.760 | 630.153 | 0.980 |
| attn_norm | Euclidean | 5 | 0.279 | 1.197 | 1613.062 | 0.639 |
| mean | Cosine | 5 | 0.502 | 1.055 | 2228.101 | 0.249 |
| mean_std | Cosine | 6 | 0.337 | 1.663 | 860.929 | 0.408 |
| max | Cosine | 5 | 0.286 | 1.949 | 704.446 | 0.477 |
| first_last | Cosine | 5 | 0.284 | 1.688 | 728.719 | 0.458 |
| attn_norm | Cosine | 5 | 0.505 | 1.047 | 2238.037 | 0.247 |

说明：余弦度量下 `attn_norm ≈ mean` 显著领先；k=5 稳定最优。

### 表格 2：离散码（codes_windows）
| 表征 | 度量 | 最佳 k | Sil | DB | CH | Intra/Inter |
|---|---|---:|---:|---:|---:|---:|
| Bag-of-Codes | Euclidean | 7 | 0.000 | 4.412 | 29.078 | 0.935 |
| Bag-of-Codes | Cosine | 8 | 0.041 | 6.038 | 48.800 | 0.819 |
| TF-IDF | Euclidean | 6 | -0.009 | 5.883 | 29.955 | 1.825 |
| TF-IDF | Cosine | 8 | 0.040 | 6.136 | 46.763 | 0.817 |
| BigramHash(16k) | Euclidean | 5 | 0.065 | 7.865 | 3.134 | 0.966 |
| BigramHash(16k) | Cosine | 8 | 0.004 | 17.492 | 5.765 | 0.885 |

说明：离散码表征的聚类质量整体显著低于连续向量（Sil ≈ 0.00–0.07 vs 0.50；Intra/Inter ≈ 0.82–0.97 vs 0.25）。

### 表格 3：DTW（子集 N=1000，1D L2 → PAA=64）
| 方法 | 最佳 k | Sil | DB | CH | Intra/Inter |
|---|---:|---:|---:|---:|---:|
| DTW_1D_PAA | 9 | 0.162 | — | — | 0.776 |

说明：DTW 需预计算距离矩阵，DB/CH 不适用此设置。

### 直观对比（最高 Sil & 最低比值）
- Silhouette（越高越好）
  - 连续/余弦/attn_norm ≈ `0.505`，连续/余弦/mean ≈ `0.502`
  - 连续/欧氏/attn_norm ≈ `0.279`
  - DTW（子集）≈ `0.162`
  - 离散码最佳（BigramHash/欧氏）≈ `0.065`；BoC/TF-IDF/余弦 ≈ `0.040`
- Intra/Inter（越低越好）
  - 连续/余弦/attn_norm ≈ `0.247`，连续/余弦/mean ≈ `0.249`
  - 连续/欧氏/attn_norm ≈ `0.639`
  - DTW（子集）≈ `0.776`
  - 离散码普遍 `0.82–0.97`（甚至 TF-IDF/欧氏 > 1）

## 5. 关键发现与结论
- 最佳配置：`quantized_windows + attn_norm/mean + cosine + k=5`
  - Silhouette ≈ `0.50`（优秀），Intra/Inter ≈ `0.247`（类内紧凑、类间分离）
- 度量对比：`Cosine >> Euclidean`（Silhouette 提升约 ~80%）
- 数据源对比：`连续向量 >> 离散码`（Silhouette 0.50 vs 0.04；Intra/Inter 0.25 vs 0.82–0.97）
- DTW 局限：1D 压缩致信息损失，且未利用多维相关性；不如聚合连续向量稳健
- 聚类数验证：k=5 在多设定下稳定最优，符合“至少 5 种动作类型”的先验

## 6. 原因分析
- Cosine 优于 Euclidean：余弦关注方向相似性、弱化幅值差异；对高维嵌入的类间边界更敏感
- 连续向量优于离散码：
  - 连续向量保留几何/相关信息；离散码仅索引，语义粒度粗
  - BoC/TF-IDF 丢失时序结构（顺序、节律、转移模式），对动作类别区分力弱
  - 离散直方图稀疏且高维，对 KMeans 不友好；Bigram 维度膨胀且存在哈希冲突
- DTW 表现不佳：高维→1D 的投影损失关键信息；未使用多变量 DTW/Soft-DTW；对噪声/长度差较敏感

## 7. 文件清单

### 7.1 脚本文件
- `scripts/segment_umap_cluster_analysis.py`：主实验脚本（连续向量）；5 聚合 × 欧氏/余弦；UMAP 2D/3D；输出 CSV 至 `statistics/`，图至 `figure/`
- `scripts/segment_dtw_umap_cluster_analysis.py`：DTW 子集实验（1D L2 + PAA）；AgglomerativeClustering（precomputed）；输出 `cluster_metrics_dtw.csv`、`dtw_distance_matrix.npy`、UMAP 图
- `scripts/codes_umap_cluster_analysis.py`：离散码聚类（BoC/TF-IDF/BigramHash × 欧氏/余弦）；UMAP 2D/3D；指标 CSV
- `scripts/segment_level_umap_analysis.py`：片段级 UMAP baseline（早期样例）
- `scripts/analyze_codes_indices.py`：检查 JSON 字段与统计（codes/quantized 分布、维度、范围）
- `scripts/advanced_codes_analysis.py`：离散码分布与量化向量统计的可视化（直方图/UMAP）

### 7.2 指标 CSV（`umap_vis/statistics/`）
- 连续向量 + 欧氏：`cluster_metrics_mean.csv`, `..._mean_std.csv`, `..._max.csv`, `..._first_last.csv`, `..._attn_norm.csv`
- 连续向量 + 余弦：`cluster_metrics_mean_cosine.csv`, `..._mean_std_cosine.csv`, `..._max_cosine.csv`, `..._first_last_cosine.csv`, `..._attn_norm_cosine.csv`
- 离散码：`cluster_metrics_codes_bow_{euclidean|cosine}.csv`, `cluster_metrics_codes_tfidf_{euclidean|cosine}.csv`, `cluster_metrics_codes_bigramhash16384_{euclidean|cosine}.csv`
- DTW：`cluster_metrics_dtw.csv`（子集 N=1000）

### 7.3 可视化文件（`umap_vis/figure/`）
- 连续向量 + 欧氏：`umap_2d_{mean|max|first_last|mean_std|attn_norm}_k*.png` 与对应 `umap_3d_*.html`
- 连续向量 + 余弦：`umap_2d_{agg}_cosine_k*.png` 与对应 `umap_3d_{agg}_cosine_k*.html`
- 离散码：`umap_2d_codes_{bow|tfidf|bigramhash16384}_{euclidean|cosine}_k*.png` 与对应 3D HTML
- DTW：`umap_2d_dtw_k9.png`, `umap_3d_dtw_k9.html`
- 其他：`code_distribution.png`, `vector_statistics.png`

## 8. 推荐配置与使用指南
- 生产推荐：`quantized_windows + attn_norm + cosine + k=5`
- 复现命令示例（在仓库根目录下，conda env `laps`）：

```bash
# 连续向量 + 余弦（多聚合）
conda run -n laps python umap_vis/scripts/segment_umap_cluster_analysis.py \
  --data-dir /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector \
  --output-dir /home/johnny/action_ws/umap_vis/figure \
  --aggs mean mean_std max first_last attn_norm \
  --k-min 5 --k-max 10 --neighbors 15 --min-dist 0.1 --metric cosine

# 离散码（BoC/TF-IDF/BigramHash）
conda run -n laps python umap_vis/scripts/codes_umap_cluster_analysis.py \
  --data-dir /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector \
  --output-dir /home/johnny/action_ws/umap_vis/figure \
  --reprs bow tfidf bigramhash \
  --k-min 5 --k-max 10 --neighbors 15 --min-dist 0.1 --metric cosine

# DTW 子集（已安装 fastdtw）
conda run -n laps python umap_vis/scripts/segment_dtw_umap_cluster_analysis.py \
  --data-dir /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector \
  --output-dir /home/johnny/action_ws/umap_vis/figure \
  --sample-size 1000 --paa 64 --neighbors 15 --min-dist 0.1 \
  --k-min 5 --k-max 10
```

- 指标/可视化解读：
  - 优先看 Silhouette（>0.4 说明非常好的可分性）与 Intra/Inter（<0.3 极佳）
  - 2D/3D UMAP 可目视验证簇的紧凑与分离；注意随机种子一致性

## 9. 未来改进方向（可选）
- 离散码直方图的更合适距离：Hellinger、Jensen–Shannon；结合 K-medoids/HDBSCAN
- 序列结构建模：Top-K 频繁 n-gram（PMI/PPMI）、code2vec/doc2vec、HMM 转移矩阵特征
- DTW 变体：多变量 DTW、Soft-DTW、带带宽约束（Sakoe–Chiba）
- 更多聚类算法：HDBSCAN、谱聚类；或与连续向量融合的混合策略（re-ranking/子簇一致性）

---

### 附：字段与可行性确认（代码片段）
```python
# 连续向量读取（segment_umap_cluster_analysis.py）
with open(jf, 'r', encoding='utf-8') as f:
    data = json.load(f)
qwins = data.get('quantized_windows', None)
# 拼成时间序列 (T, 768)

# 离散码读取与 BoC（codes_umap_cluster_analysis.py）
seq = [int(code) for win in data['codes_windows'] for code in win]
X = np.zeros((N, 2048), dtype=np.float32)
# 统计频次后按 L1 归一化
```

