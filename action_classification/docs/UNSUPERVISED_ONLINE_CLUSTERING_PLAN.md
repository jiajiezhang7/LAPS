# LSTM 嵌入 + 无需预设簇数的在线聚类与异常检测方案（HDBSCAN 主线，BayesianGMM 备选）

更新时间：2025-09-18 17:20 (UTC+8)
目录：`amplify_motion_tokenizer/action_classification/`

## 目标
- 输入：来自 Motion Tokenizer 的 Latent Action Sequences（code indices 序列）。
- 管线：`Latent Action Seq` → `LSTM 时序编码 (Temporal Embedding)` → `无监督聚类` → `簇类标签 / 异常警告`。
- 需求：
  - 不预设簇数；
  - 支持在线分配新样本并输出簇 ID 与置信度；
  - 能判断“不属于任何簇/低置信度”为异常。

## 推荐方案概览
- 主线：`LSTM 嵌入 + HDBSCAN`（无需预设簇数、原生噪声(-1)类、支持 approximate_predict 在线分配与 membership probability）。
- 备选：`LSTM 嵌入 + BayesianGaussianMixture (BGMM)`（近似 DPGMM，上限组件数，能自动稀疏无效簇；用最大后验概率阈值作异常告警）。
- 异常阈值：优先使用 membership/posterior probability 阈值（建议初值 0.15~0.20，后续用验证集标定，目标 FPR ≤ 5%）。

## 与现有仓库的衔接
- 已有：
  - `sequence_embed_lstm.py`：NTP 训练 LSTM/GRU 编码器并导出 `embed.npy / labels.npy / paths.txt`。
  - `embed_cluster_eval.py`：已扩展支持 `method=hdbscan | bayes_gmm | kmeans | gmm`，并保持 UMAP 可视化与指标输出。
  - `analysis/code_indices_analysis.py`：统计、BoW/TF-IDF、Hashed-Bigram、UMAP 与聚类（KMeans/GMM）。
- 新增：
  - `fit_hdbscan_over_embeddings.py`：在 LSTM 嵌入上拟合 HDBSCAN，保存模型与评估产物。
  - `online_cluster_infer.py`：加载 LSTM 编码器与聚类模型，对新序列在线输出簇与异常。
- 依赖：
  - 已在 `requirements.txt` 增加 `hdbscan>=0.8.33`。

## 离线训练与评估流程
1) 训练 LSTM 序列嵌入并导出嵌入
```
# 激活环境（示例，需按你的环境路径）
conda activate amplify_mt

# 训练 + 导出（示例参数）
python amplify_motion_tokenizer/action_classification/sequence_embed_lstm.py \
  --json-root <INFERENCE_JSON_ROOT> \
  --config amplify_motion_tokenizer/action_classification/configs/sequence_embed.yaml

# 输出目录包含：embed.npy, labels.npy, paths.txt, model_best.pt
```

2) 在导出的 `embed.npy` 上拟合 HDBSCAN 并保存模型
```
python amplify_motion_tokenizer/action_classification/fit_hdbscan_over_embeddings.py \
  --embed-dir <EMBED_DIR> \
  --config amplify_motion_tokenizer/action_classification/configs/eval_config.yaml \
  --out-dir amplify_motion_tokenizer/action_classification/out/hdbscan_fits

# 产物：
# - model_hdbscan.pkl / preprocessor_pca.pkl（如启用）
# - cluster_meta.json（含预处理信息与超参）
# - aggregate_results.json（含噪声率、NMI/ARI/F1 等；评估仅在非噪声样本上）
# - train_assignments.jsonl（逐样本的簇与membership prob）
# - 可选 UMAP 可视化（只对已分配样本绘制）
```

3) 聚类评估（可选：对比 HDBSCAN/BayesGMM/KMeans/GMM）
```
# 修改 eval_config.yaml 中 clustering.method = hdbscan | bayes_gmm | kmeans | gmm
python amplify_motion_tokenizer/action_classification/embed_cluster_eval.py \
  --embed-dir <EMBED_DIR> \
  --config amplify_motion_tokenizer/action_classification/configs/eval_config.yaml \
  --out-dir amplify_motion_tokenizer/action_classification/analysis/out

# 输出：aggregate_results.json，包含 ACC/NMI/ARI/F1、每类精确率/召回率、
# Hungarian 对齐后的混淆矩阵、UMAP 可视化路径等。
```

## 在线推理与异常告警
```
python amplify_motion_tokenizer/action_classification/online_cluster_infer.py \
  --encoder-model <EMBED_DIR>/model_best.pt \
  --cluster-dir amplify_motion_tokenizer/action_classification/out/hdbscan_fits/<TIMESTAMP> \
  --prob-thr 0.2 \
  --json <A_SINGLE_JSON>  # 或 --json-root <JSON_ROOT> / --codes "1,2,3,4"

# 输出示例：
{"source": ".../sample.json", "cluster_id": 3, "prob": 0.67, "anomaly": false, "method": "hdbscan"}
```
- 判定逻辑（HDBSCAN）：
  - `cluster_id == -1` 或 `prob < τ` → `anomaly=true`。
- 判定逻辑（BayesGMM）：
  - `max posterior < τ` → `anomaly=true`。
- 可选：在 `cluster_dir/cluster_id_to_semantic.json` 提供 `cluster_id → 业务标签` 的映射，脚本会自动附加 `semantic` 字段。

## 关键超参数建议
- HDBSCAN：
  - `min_cluster_size`：建议从 `max(10, 0.5%~2% * N)` 网格搜索起步；
  - `min_samples`：可与 `min_cluster_size` 相同或略小；
  - `metric`：对 L2 归一化向量，`euclidean` 常见；如经验更稳，可尝试 `cosine`；
  - `cluster_selection_method`: `eom`（默认）通常更稳。
- BayesGMM：
  - `n_components` 为上界（例如 8/16）；
  - `weight_concentration_prior_type='dirichlet_process'`；
  - 依据后验概率阈值做异常。
- 阈值标定：
  - 用留出验证集/近期在线数据，计算 FPR/TPR 曲线；
  - 目标 FPR ≤ 5%（可按业务代价调整）。

## 目录与产物
- 训练导出：`<EMBED_DIR>/embed.npy, labels.npy, paths.txt, model_best.pt`
- HDBSCAN 拟合：`out/hdbscan_fits/<TS>/model_hdbscan.pkl, cluster_meta.json, aggregate_results.json, train_assignments.jsonl, (umap/*.png)`
- 评估输出：`analysis/out/<TS>_lstm/aggregate_results.json`
- 在线推理：可将结果保存到 `--out-jsonl` 指定路径。

## 目前进度与待办
- [x] 在 `requirements.txt` 增加 `hdbscan>=0.8.33`
- [x] 扩展 `embed_cluster_eval.py` 支持 `hdbscan` 与 `bayes_gmm`
- [x] 新增 `fit_hdbscan_over_embeddings.py`（离线拟合）
- [x] 新增 `online_cluster_infer.py`（在线推理）
- [ ] 安装依赖并验证（需在 `conda activate amplify_mt` 后执行安装）
- [ ] 在验证集上标定 `prob-thr`（建议初值 0.15~0.20，目标 FPR ≤ 5%）
- [ ] 可选：将 `cluster_id` 映射为业务标签，并训练轻量监督分类器作为在线兜底

## 环境与安装
```
# 强烈建议：先激活 conda 环境
conda activate amplify_mt

# 安装依赖（如未安装）
pip install -r requirements.txt
# 或者使用 conda-forge（更稳定的 hdbscan 构建）：
# conda install -c conda-forge hdbscan -y
```

## 评估指标与可视化
- 指标：ACC（Hungarian 对齐）、NMI、ARI、Macro-F1、按类 Precision/Recall。
- HDBSCAN 指标说明：为公平起见，ACC/NMI/ARI/Macro-F1 在“已分配样本（非 -1）”上计算，并给出 `noise_rate`。
- 可视化：UMAP（默认启用），支持 `lstm_metric` 指定度量（建议 cosine 或 euclidean）。

## 备注
- LSTM 嵌入已在本数据上验证优于 BoW/Avg（详见 `analysis/report_code_indices_quality_plus_classifcation_quality.md` 中下游特征对比）。
- 若后续需要“流式增量学习”，HDBSCAN 不支持在线增量拟合；建议采用“定期离线重拟合 + 在线近邻分配”的策略。
