# 当前可尝试的方向
- 1. 增加片段长度：使用与动作片段相关的所有重叠窗口 code indices，组合起来作为这一个片段的code indices sequences
  - [ 今天9.25就跑上 ] 需要修改 stream_inference.py，目前采用的是完全无重叠情况 （需要写一个开关，用于开启/关闭是否启用重叠窗口的code indices）

- 2.对当前 codes 训练 LSTM 序列级嵌入，再聚类对比

- 3. 降低codebook的规模： 2048, 1024, 512，再次推理，再次分析



# 目前的核心问题

- 样本统计仍然稀薄，无法形成稳定的聚类结构

下面是对结果目录 [action_classification/analysis/unsup_quality/20250925_192325/](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325:0:0-0:0) 的全面读取与结论汇总。若你需要进一步深挖子簇的内部结构（如每簇的轮廓系数分布、簇内主导 codes 等），我可以补一支小脚本基于 [X_tfidf.npy](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/X_tfidf.npy:0:0-0:0) 与 [hdbscan_labels_tfidf.npy](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/hdbscan_labels_tfidf.npy:0:0-0:0) 做深入分析；但就当前目标“判断 code indices 的质量与可分性”而言，现有指标已足够得出结论。

# 已读取的关键文件与指标
- 文件: [unsupervised_quality_report.json](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/unsupervised_quality_report.json:0:0-0:0)
  - info:
    - num_samples: 9647
    - codebook_size: 2048
  - dataset_stats（全局分布/覆盖）:
    - code_coverage_ratio: 0.939（使用到 1924/2048 个 code）
    - global_entropy (nats): 6.389
    - global_perplexity: 595.34（约等于相对代码簿 595/2048 ≈ 0.29）
  - codes_usage_summary（逐样本的代码簿利用度）:
    - perplexity_ratio（困惑度/2048）中位数 p50: 0.0078
      - 折算“有效码数”≈ 0.0078 × 2048 ≈ 16（与单段长度 16~32 相符）
    - gini p50: 0.992（分布高度集中，符合短序列稀疏计数特性）
    - top1_prop p50: 0.125（最常见的 code 占比约 1/8，亦与长度 16 相匹配）
  - KMeans（TF-IDF 特征）:
    - k=2…20 所有轮廓系数 silhouette 极低（最大仅 0.028，best@k=20）
    - k=3 时 silhouette=0.015
  - KMeans（哈希 bigram 特征）:
    - 轮廓系数几乎为 0 或负值（例如 best@k=4 时仅 0.0016），同样显示不可分
  - HDBSCAN（选用最有前景的特征 TF-IDF）:
    - n_clusters: 3（非噪声簇）
    - noise_ratio: 0.981（98.08% 的样本被视为噪声）
    - silhouette_non_noise: 0.103（仅在非噪声子集计算）
    - cluster_persistence 均值: 0.012（非常低）
    - soft_membership_strength 均值: 0.320（软隶属置信偏低）
    - 簇大小：93、45、47；其余 9462 为噪声
  - 自动结论（脚本规则输出）:
    - verdict: weak
    - reason: “KMeans silhouette 0.028, HDBSCAN clusters=3, noise=0.98.”
- 文件: [stats/stats_summary.json](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/stats/stats_summary.json:0:0-0:0)（与上面 dataset_stats 一致）
- 文件: [X_tfidf.npy](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/X_tfidf.npy:0:0-0:0)、[X_bigram.npy](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/X_bigram.npy:0:0-0:0)、[hdbscan_labels_tfidf.npy](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/hdbscan_labels_tfidf.npy:0:0-0:0)（如需进一步簇内分析可用）
- 可视化: [umap_tfidf.png](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/umap_tfidf.png:0:0-0:0)（按 HDBSCAN/或最佳 KMeans 簇着色，结合上述指标预期类间分离不明显）

# 结论（关于 code indices 的“质量”与“可分性”）
- 可分性（用于无监督分类）：结论为“弱（weak separability）”
  - 证据一：TF-IDF 特征的 KMeans 在 k=2~20 的 silhouette 全部远低于常用的“可分阈值”（0.2~0.3），最佳也仅约 0.028；
  - 证据二：哈希 bigram 特征同样 silhouette≈0（甚至为负），说明考虑了转移关系后依然难以分簇。
  - 证据三：HDBSCAN 高噪声比（≈98%）且非噪声子集的 silhouette≈0.10、簇持久性很低（≈0.012），均表明不形成稳定/密集的簇结构。非噪声小簇（93、45、47）只覆盖极少数样本。
- 分布与代码簿利用度（整体“多样性/稳定性”角度）：
  - 全数据层面 code 覆盖率很高（0.939）、全局困惑度 595，说明“跨样本总体”使用了相当广的 code 空间。
  - 但单样本层面困惑度相对代码簿很低（pr@p50 ≈ 0.0078，≈16 个有效码），Gini 很高（p50≈0.992），Top1 比例≈1/8。这与每条序列长度仅 16~32 十分一致：单条记录本就只能触及极少数 code，导致统计分布高度稀疏/集中。
  - 这并不直接说明“编码本身差”，而是指出“仅靠这些极短序列的代码直方图/二元组特征”很难形成可分的类结构。

综上：就“直接基于 code indices 序列（长度约 16~32、1~3 个窗口）做无监督分类”这一目标而言，当前的可分性很弱，不足以靠传统的 BoW/TF‑IDF/二元组 或 HDBSCAN/KMeans 直接发现稳定且覆盖面广的动作簇。数据在“跨样本整体”上确实很丰富（高覆盖），但“单样本表达”过短且过稀疏，导致类结构难以从这些简单特征中显现。

# 建议的后续动作（可提升可分性）
- 训练序列级嵌入再聚类
  - 用你仓库已有的 `action_classification/embedding/sequence_embed_lstm.py`（NTP 等自监督目标）学习“动作段”的序列嵌入，然后用 `embed_cluster_eval.py` 做 KMeans/HDBSCAN 评估与 UMAP 可视化。此前你在有标签实验中的经验表明 LSTM 嵌入能显著提升聚类可分性。
- 增强片段长度或上下文
  - 将 1~3 个窗口的 code 段拼接为更长的“动作片段”进行建模；短序列（16~32）在 BoW/二元组空间里非常稀疏，增加长度通常可提升统计稳定性与分离度。
- 特征增强与正则
  - 若仍坚持纯 code 特征，可尝试更高阶 n‑gram（例如哈希 trigram）与多尺度统计，但根据本次结果（bigram 几乎不改善），预期改善有限。
  - 降低代码簿规模或做 code 合并/平滑（减少超稀疏性），但这属于量化端设计变更，代价较大。

# 可选的深入分析（如你需要，我可立即补脚本）
- 对 [X_tfidf.npy](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/X_tfidf.npy:0:0-0:0) 与 [hdbscan_labels_tfidf.npy](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/hdbscan_labels_tfidf.npy:0:0-0:0)：
  - 统计各非噪声簇的样本内“单样本 silhouette 分布”、簇内/簇间均值距离、簇内主导 codes（从原 JSON 重建 BoW）。
  - 目的：验证是否存在“少量稳固的小类”（本次 HDBSCAN 的 93/45/47 个小簇），这些可作为后续“原型发现/规则写作”的种子。

如需我立刻生成上述“簇内剖析”脚本，请告知，我会将其写入 `action_classification/analysis/` 并给出一键运行的命令。