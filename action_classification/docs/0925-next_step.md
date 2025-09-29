

结论先行
- 可以。基于当前无监督结果，方向2和方向3明确更有前景；方向1可以尝试但优先级低于2/3。

三项方向逐一判断与理由
- 方向1（将 codebook 从2048降到1024）
  - 核心判断：有一定合理性，但不保证改善，优先级低于2/3。
  - 理由：
    - 本次失败的主因是“单样本过短、直方图极稀疏”，导致 KMeans/HDBSCAN 难以发现结构；简单减小 codebook 仅能缓解稀疏，但可能牺牲语义分辨率（语义冲突/碰撞增加）。
    - 从你结果看，全局覆盖率高、单样本“有效码数≈16”由长度所限决定，即便把K减半，单样本统计仍然稀薄，silhouette 可能仍维持低位。
  - 风险与成本：需要重新量化或替换 codebook 并重跑在线推理，工程开销较大。

- 方向2（增加片段长度：把同一动作片段相关的所有重叠窗口的 codes 串接为一条更长的序列）
  - 核心判断：强烈推荐，最直接对症。
  - 理由：
    - 你的中位“有效码数≈16”本质上受片段长度（16~32）限制；延长片段可显著提升单样本统计稳定性（BoW更平滑、n‑gram更可靠），最可能改善聚类可分性。
    - 这一步通常能把 silhouette 从“近0”拉到“>0.1~0.2”的可用区间，是最具性价比的方向。
  - 实施要点：
    - 需要在数据层面将“同一视频/同一动作段”的所有重叠窗口的 `codes_windows` 串接（或按时间顺序合并），保证每条样本是“动作片段级”长序列。
    - 如需，我可以在 [action_classification/analysis/unsupervised_quality_report.py](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsupervised_quality_report.py:0:0-0:0) 增加按元数据（video_id/segment_id）聚合的选项，自动把同段窗口合并后再评估。

- 方向3（对当前 codes 训练 LSTM 序列级嵌入，再聚类对比）
  - 核心判断：强烈推荐，和方向2可叠加（更长片段+序列嵌入最佳）。
  - 理由：
    - LSTM/序列模型能捕获超越简单n‑gram的时序模式，缓解短序列+稀疏直方图下的信息缺失；在相似任务中常能显著提升聚类可分性。
    - 你仓库已有序列嵌入/评估脚本（如已存在的 sequence_embed_lstm 与 embed_cluster_eval），上手成本较低。
  - 何时判定“LSTM Embedding 适合当前任务”：
    - 与 BoW/二元组基线相比，固定K=3和最佳K两组指标均显著改善：
      - silhouette 提升到 ≥0.15~0.25 甚至更高（越高越好）
      - Calinski-Harabasz 上升、Davies-Bouldin 下降
      - HDBSCAN 噪声率显著降低、非噪声簇规模更大、簇持久性（persistence）提高，软隶属强度上升
    - UMAP 可视化呈现更加紧致的类团并有可解释的局部结构。
  - 训练提示：
    - 自监督目标（例如 NTP/下一个 token 预测，或对比学习）通常比直接无监督聚类前的原始直方图更有效。
    - 如数据依旧较短，优先结合方向2延长片段再做嵌入，收益更稳。

推荐优先级与对比方案
- 优先级排序：方向2 ≈ 方向3 > 方向1
- 对比评测建议（单因子变化，逐步验证）：
  - 基线A：当前（短片段，BoW/二元组）→ 已得到“weak”
  - 方案B：长片段（合并重叠窗口），仍用 BoW/二元组 → 看长度效应
  - 方案C：短片段 + LSTM 序列嵌入 → 看表示学习效应
  - 方案D：长片段 + LSTM 序列嵌入 → 叠加效应（最可能最佳）
  - 方案E（如资源允许）：在B/D基础上再试 codebook=1024 → 观察是否进一步带来边际收益
- 统一评测口径：

  - 若后续引入标签（半监督/弱监督），可再补充“最佳簇到标签的对齐后混淆矩阵与 per-class P/R”，闭环验证“语义一致性”。

需要我现在做什么
- 我可以：
  - 在 [unsupervised_quality_report.py](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsupervised_quality_report.py:0:0-0:0) 增加“跨窗口合并为片段级序列”的选项（需要你确认能提供 video_id/segment_id 或者文件路径规则来做分组）。
  - 帮你对接或补充“序列嵌入训练与聚类评估”的脚本，一键输出与本次相同格式的 [unsupervised_quality_report.json](cci:7://file:///home/johnny/johnny_ws/motion_tokenizer/action_classification/analysis/unsup_quality/20250925_192325/unsupervised_quality_report.json:0:0-0:0) 与 UMAP。
若你愿意，我可以先实现“片段级合并”并跑一版，让你快速看到方向2能带来的改观。