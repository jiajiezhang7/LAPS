
我将先在代码库中定位“attn_norm”的实现或说明，确认它的全称与具体含义，以便给出论文中规范的写法。随后给出建议的英文表述句式。


## 简明结论
- “attn_norm”的全称建议写为：L2 范数注意力加权聚合（英文：L2-norm attention-weighted aggregation），也可简称 Norm-based attention pooling（L2）。
- 在论文中建议统一命名为：Attn-Norm Aggregation（L2）或 Attention-Norm (L2) Pooling，并在方法段落用一两句话说明其计算方式与“训练自由（training-free）”属性。

## 论文中推荐写法
- 表格行名（英文，适合 Table）：Attn-Norm Aggregation (Baseline)
- 方法段落的一句话定义（英文示例）：
  - We use a training-free baseline that assigns per-frame attention weights proportional to the L2 norm of the frame feature and computes a weighted average: w_t = ||x_t||_2 / Σ_s ||x_s||_2, z = Σ_t w_t x_t.
- 中文描述（方法段落）：
  - 我们使用一个训练自由的基线聚合方法：以帧特征的 L2 范数作为“注意力”权重，并进行加权均值聚合。具体为 w_t = ||x_t||_2 / Σ_s ||x_s||_2，z = Σ_t w_t x_t。

## 依据与一致性
- 代码中对“attn_norm”的实现/注释均指向“用向量范数作为注意力权重的加权均值聚合（无训练）”，与上述命名一致：
  - umap_vis/scripts/sequence_model_embedding.py：函数 agg_attn_norm 与 features_by_agg("attn_norm")
  - umap_vis/scripts/segment_umap_cluster_analysis.py：注释“使用向量范数作为注意力权重（无训练）”
  - scripts/supplement/k_search.py：说明“attn_norm：作为基线，使用注意力范数加权均值聚合”
  - paper_block/code2paper/quantized_vector_cluster.md：给出同样的权重与加权公式

这样写既与代码/实验命名对齐，又清楚表明其本质是“基于 L2 范数的注意力式加权均值”，避免与 Transformer 注意力机制混淆。
