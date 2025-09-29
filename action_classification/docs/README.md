# Action Classification 模块总览

## 目录结构

```
action_classification/
├── README.md                     # 本说明文档
├── __init__.py
├── analysis/                     # 离线分析输出（UMAP、报告、统计等）
├── clustering/                   # 聚类模型训练脚本
│   ├── __init__.py
│   ├── fit_bayes_gmm.py          # 训练/导出 BayesianGaussianMixture 聚类模型
│   └── fit_hdbscan.py            # 训练/导出 HDBSCAN 聚类模型
├── configs/                      # 统一配置文件
│   ├── eval_config.yaml          # 离线聚类评估与聚类训练共享配置
│   └── sequence_embed.yaml       # LSTM 序列嵌入训练配置
├── data/                         # 数据加载与特征工程工具
│   ├── __init__.py
│   └── features.py               # 读取 Motion Tokenizer JSON、构建 BoW/Avg 特征
├── docs/                         # 方案与阶段性总结文档
├── embedding/                    # 序列嵌入模型（LSTM/GRU）
│   ├── __init__.py
│   ├── core.py                   # `SeqEncoder` 模型、数据集、训练/导出核心逻辑
│   └── train.py                  # CLI：训练 LSTM 序列嵌入并导出 `embed.npy`
├── evaluation/                   # 离线聚类评估脚本
│   ├── __init__.py
│   └── embed_cluster_eval.py     # 针对 `embed.npy` 进行聚类评估、输出指标与 UMAP
├── models/                       # 已训练好的序列嵌入模型（`model_best.pt` 等）
├── scripts/                      # 命令行入口脚本（便于记忆与调用）
│   ├── __init__.py
│   ├── infer_sequence_embed_lstm.py
│   └── online_cluster_infer.py
├── seq_embed/                    # 导出的嵌入结果（`embed.npy`、`labels.npy`、UMAP 等）
└── online_cluster_infer.py       # 在线推理：LSTM + 聚类模型 → 异常检测
```

## 核心流程概述

1. **序列嵌入训练 (`embedding/`)**
   - `embedding/train.py`：入口脚本。读取 `sequence_embed.yaml` 配置，对 Motion Tokenizer JSON 进行 stratified split，训练 `SeqEncoder`（LSTM/GRU）并导出模型与嵌入：
     - `models/`：保存 `model_best.pt`、训练日志。
     - `seq_embed/`：保存 `embed.npy`、`labels.npy`、`paths.txt`。
   - `embedding/core.py`：封装 `SeqEncoder`、`SeqDataset`、训练循环、嵌入导出等核心逻辑，提供 `export_embeddings()`、`train_ntp()` 等函数，供训练与推理脚本复用。

2. **离线聚类评估 (`evaluation/`)**
   - `evaluation/embed_cluster_eval.py`：对 `embed.npy` 进行 KMeans/GMM/BayesGMM/HDBSCAN 聚类评估；输出 ACC/NMI/ARI/F1、混淆矩阵、分类精召率，按需生成 UMAP 可视化。
   - 依赖 `configs/eval_config.yaml` 中的聚类与可视化参数；可通过 `--config` 指定自定义配置。

3. **聚类模型拟合 (`clustering/`)**
   - `clustering/fit_bayes_gmm.py`：针对 LSTM 嵌入拟合 `BayesianGaussianMixture`，导出模型与元信息（`cluster_meta.json`、`model_bayes_gmm.pkl`、可选 PCA `preprocessor_pca.pkl`），并记录训练数据集评估指标。
   - `clustering/fit_hdbscan.py`：拟合 `HDBSCAN` 并保存软簇概率、噪声比率、混淆矩阵等；生成可选 UMAP（含噪声/无噪声两版）。

4. **在线推理 (`online_cluster_infer.py`)**
   - 加载训练好的序列编码器与聚类模型（HDBSCAN 或 BayesGMM），对实时 JSON / JSON 目录 / 手动输入的 code 序列进行嵌入与聚类预测。
   - 依据 `prob_thr` 阈值输出 `cluster_id`、`prob`、`anomaly` 标记，并可写入 JSONL。

5. **特征工程 (`data/features.py`)**
   - 提供 `read_json_sample()` 与 `build_dataset()`：支持 BoW/token diff 平均特征、代码本体统计、TF-IDF 等，为传统聚类基线与数据探索服务。

6. **分析输出 (`analysis/`)**
   - 存放运行 `evaluation/embed_cluster_eval.py`、`clustering/fit_*` 等脚本产生的 UMAP 图片、JSON 结果、Markdown 报告等。

7. **文档 (`docs/`)**
   - `UNSUPERVISED_ONLINE_CLUSTERING_PLAN.md`：无监督在线聚类整体方案。
   - `step1_bow_avg_done.md` / `step2_lstm_done.md`：阶段性进展总结，已更新为新的脚本路径与调用方式。

## 常用命令速览

请先激活 `amplify_mt` Conda 环境。以下命令均在仓库根目录执行：

```bash
# 序列嵌入训练
python -m action_classification.embedding.train \
  --json-root /path/to/json_root \
  --config action_classification/configs/sequence_embed.yaml

# 使用已训练模型导出嵌入（离线批量推理）
python -m action_classification.scripts.infer_sequence_embed_lstm \
  --json-root /path/to/json_root \
  --model-pt /path/to/models/XXXX/model_best.pt \
  --out-dir action_classification/seq_embed \
  --l2-normalize

# 离线聚类评估（KMeans/GMM/HDBSCAN/BayesGMM 指标与 UMAP）
python -m action_classification.evaluation.embed_cluster_eval \
  --embed-dir action_classification/seq_embed/2025xxxx_xxxxxx \
  --config action_classification/configs/eval_config.yaml

# 拟合 BayesGMM 并导出模型
python -m action_classification.clustering.fit_bayes_gmm \
  --embed-dir action_classification/seq_embed/2025xxxx_xxxxxx

# 拟合 HDBSCAN 并导出模型
python -m action_classification.clustering.fit_hdbscan \
  --embed-dir action_classification/seq_embed/2025xxxx_xxxxxx

# 在线推理（单文件 / 目录 / 手动 code 序列）
python -m action_classification.scripts.online_cluster_infer \
  --encoder-model action_classification/models/XXXX/model_best.pt \
  --cluster-dir action_classification/seq_embed/2025xxxx_xxxxxx/hdbscan_fits/ \
  --json /path/to/sample.json \
  --prob-thr 0.2
```

## 依赖与注意事项

- 请确认 `requirements.txt` 已安装 `hdbscan`、`umap-learn`、`matplotlib`、`seaborn` 等依赖。
- 所有脚本默认读取 YAML 配置；可通过命令行参数覆盖关键超参（如 `--clusters`、`--umap` 等）。
- 输出目录默认带时间戳，便于追踪版本；谨慎清理以免丢失实验结果。
- `online_cluster_infer.py` 支持 HDBSCAN 和 BayesGMM，两者输出略有差异（HDBSCAN 可返回噪声点 `cluster_id=-1` 与近似概率）。

## 快速回溯

- 当你忘记某个脚本的作用，可对照上方目录表或查阅 `scripts/` 目录中的入口脚本，它们会指向对应模块的 `main()` 函数。
- 若需扩展新的聚类算法，建议在 `clustering/` 中新增脚本，并在 `scripts/` 中提供入口，同时更新本 README。
