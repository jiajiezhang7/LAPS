下面给你一个“指标分层解释”，明确哪些指标更能代表 code indices 本身的“内在质量/可学习性”，哪些仅仅说明“直接把 code 当特征做聚类不合适”（但序列嵌入如 LSTM 仍有提升空间）。我会引用脚本中的具体产物与字段，便于你对照结果目录与代码。

一、能反映 code indices“内在质量/可学习性”的指标
这些指标更多描述“序列本身的信息含量与可建模性”，如果它们健康，通常意味着用 LSTM/序列嵌入再处理会更有前景。

- 数据集层面分布与容量
  - 代码簿覆盖率与全局熵/困惑度（`dataset_stats.code_coverage_ratio`, `global_entropy`, `global_perplexity`，由 [aggregate_stats()](cci:1://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsupervised_quality_report.py:311:0-370:18) 写入 [stats/stats_summary.json](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/stats/stats_summary.json:0:0-0:0)）
    - 覆盖率高（例如你这次约 0.939），说明跨样本总体使用了丰富的 code 空间，数据“容量”足够，不是“量化塌缩”。
    - 全局困惑度高（绝对值依赖K，但相对K不是极低），也支持“多样而非塌缩”的判断。
- 单样本的分布稳定性与可建模性（由 [compute_codes_metrics()](cci:1://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsupervised_quality_report.py:239:0-302:18) 和 [aggregate_stats()](cci:1://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsupervised_quality_report.py:311:0-370:18) 产出）
  - 单样本困惑度/相对困惑度（`codes_usage_summary.perplexity_ratio` 以及每条的 `perplexity`）
    - 这在短序列（长度16~32）中自然偏低，不必直接判“差”。如果极低到远小于“序列长度所能触及的码数”，才值得警惕量化退化。
  - Gini 与 top1_prop（`codes_usage_summary.gini`, `top1_prop`）
    - 高 Gini、较高 top1 在短序列下是正常现象（计数稀疏）；它们提示“BoW直方图很稀薄”，但不等同于“序列不可学”。
  - 运行长度统计与转移多样性（由 [aggregate_stats()](cci:1://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsupervised_quality_report.py:311:0-370:18) 写入 [stats/sample_stats.jsonl](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/stats/sample_stats.jsonl:0:0-0:0) 的 `run_length.{mean,p50,p90,max}`, `bigram_diversity`）
    - 这是判断“序列可学性”的关键。较高的 `bigram_diversity`、不过度的“超长run”（例如 `run_length.p90` 不靠近 L）通常意味着存在可辨的转移/节律模式，序列模型（LSTM）可更好捕捉。
    - 若大量序列接近“常数序列”（run 非常长、bigram_diversity 很低），则后续任何方法都很难学到区分性，这才是“内在质量”问题。

解读你的现有结果小结
- 覆盖率很高（0.939）、全局困惑度高（≈595），说明整体“量化空间”并未塌缩。
- 单样本相对困惑度 p50 ≈ 0.0078（≈16 个“有效码”），本质受“片段很短”限制，不等同于“内在质量差”。
- 真正关键的时序可学性指标（`bigram_diversity`、`run_length`）在汇总 JSON 里不做总体统计，但已写入 [stats/sample_stats.jsonl](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/stats/sample_stats.jsonl:0:0-0:0)。如需，我可补一个汇总站脚本给出它们的分位数，以更定量地判断“LSTM 是否有戏”。

二、更多是“特征选择不合适（BoW/二元组）”的指标
这些指标衡量的是“用当前特征做聚类的可分性”，它们很容易被特征形态（BoW/二元组、短序列稀疏）限制。低分不代表 code indices 本身无用，往往正是提示“需要序列级嵌入（LSTM/Transformer）或更长片段”。

- KMeans 内部指标（`kmeans.{feat}.grid/best`，由 [evaluate_kmeans_internal()](cci:1://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsupervised_quality_report.py:424:0-460:42) 输出）
  - silhouette、Calinski-Harabasz、Davies-Bouldin、inertia
  - 你的 TF‑IDF 与 bigram 的 silhouette 全部很低（最佳仅≈0.028），主要说明“短序列 + 稀疏直方图/有限二元组”难以直接分簇，不代表序列模型也不行。
  - 我已为“未知K”加入重复与稳定性（ARI）与候选K输出，这些仍是“在该特征空间下”的信号。
- HDBSCAN 指标（`hdbscan` 或 `hdbscan.best`，由 [evaluate_hdbscan()](cci:1://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsupervised_quality_report.py:561:0-634:5)/[evaluate_hdbscan_grid()](cci:1://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsupervised_quality_report.py:637:0-656:39) 输出）
  - `noise_ratio` 高、`silhouette_non_noise` 低、`cluster_persistence` 低
  - 这些受“特征空间密度结构”强影响。你这次 `noise≈0.98` 明确传递“当前特征空间极其难聚”，但并不直接否定“换成 LSTM 序列嵌入后能否好转”。

如何用这些指标指导“是否上 LSTM/是否延长片段”
- 当出现以下组合时，优先考虑“延长片段 + 序列嵌入”：
  - 数据集层面：`code_coverage_ratio` 不低、`global_perplexity` 不低（说明整体容量足）。
  - 单样本层面：`bigram_diversity` 不是极低、`run_length` 不普遍接近 L（存在转移/节律），即“序列里有模式可学”。
  - 聚类层面（BoW/二元组）：silhouette 很低、HDBSCAN 几乎全噪声。也就是“直方图特征不行，但序列仍可能可学”的典型信号。
- 何时怀疑“内在质量问题”（即使 LSTM 也未必救得动）：
  - `code_coverage_ratio` 极低（大量 code 从不被用到）或全局困惑度异常低（接近某个很小的值），提示量化严重塌缩。
  - 大多数样本 `run_length` 极高、`bigram_diversity` 极低，序列基本“无转移、无节律”，缺乏时序结构。

建议的下一步（可选）
- 我可以写一个小脚本对 [stats/sample_stats.jsonl](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/stats/sample_stats.jsonl:0:0-0:0) 做汇总（分位数/箱线），输出：
  - `bigram_diversity` 的 p10/p50/p90
  - `run_length.{mean,p50,p90,max}` 的分位数
- 若这些值显示“有可学的转移/节律”，则强烈建议：
  - 先“延长片段”（合并同一动作段所有重叠窗口为更长序列），再跑一次报告；
  - 再用 `sequence_embed_lstm.py` 训练序列嵌入，配合 `embed_cluster_eval.py` 评估 silhouette/CH/DB 与 HDBSCAN 噪声率。