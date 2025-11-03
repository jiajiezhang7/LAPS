## Method: Unsupervised Clustering of Action Segments from Quantized Continuous Sequences

### 2.1 Problem Formulation
Given a collection of video action segments, each segment is represented as a sequence of continuous feature vectors extracted by a quantization pipeline: \(X = [x_1, x_2, \ldots, x_T] \in \mathbb{R}^{T \times 768}\), where \(T\) varies across segments and each frame-level descriptor \(x_t \in \mathbb{R}^{768}\). The objective is unsupervised clustering of segments into semantically coherent groups without labels.

Challenges include: (i) high-dimensional, variable-length time series; (ii) unknown, data-driven number of clusters; (iii) potentially large intra-class variation and subtle inter-class boundaries; and (iv) robustness to noise and domain shift. The method aims to construct segment-level embeddings that preserve semantic similarities and enable stable clustering under cosine geometry, with UMAP used purely for visualization.

---

### 2.2 Baseline: Attention-Norm Aggregation
We first consider a training-free, per-segment aggregation based on vector norms. For a segment sequence \(X\), a scalar weight is assigned to each time step proportional to the L2 norm of its feature vector, followed by a weighted average:

\[
\tilde{w}_t = \lVert x_t \rVert_2, \quad w_t = \frac{\tilde{w}_t}{\sum_{s=1}^{T} \tilde{w}_s}, \quad z = \sum_{t=1}^{T} w_t \, x_t \in \mathbb{R}^{768}.
\]

Rationale. Frame-wise norms highlight frames with higher activation magnitudes, which often correlate with salient motion/appearance patterns, while down-weighting low-energy, potentially noisy frames. This simple attention surrogate is entirely label-free and computationally light.

Distance Metric. In high-dimensional spaces, relative orientation is often more informative than vector magnitude. Hence, cosine distance is adopted in the feature space. When cosine is requested, the aggregated features \(z\) are standardized and L2-normalized before clustering, making Euclidean operations consistent with cosine geometry via the identity
\(\lVert \hat{z}_i - \hat{z}_j \rVert_2^2 = 2(1 - \cos(\hat{z}_i, \hat{z}_j))\), where \(\hat{z} = z / \lVert z \rVert_2\).

Clustering and Visualization. KMeans is applied on standardized (and optionally L2-normalized) embeddings with multiple candidate \(k\) values; the best \(k\) is chosen by the Silhouette score. UMAP is used to produce 2D/3D visualizations under the same metric (cosine or Euclidean) for qualitative inspection; UMAP is not used to form clusters.

---

### 2.3 Proposed: Transformer-based Sequence Embedding (Frozen Inference)
To capture long-range temporal dependencies beyond simple aggregation, a lightweight Transformer encoder is employed in a purely inference (no training) regime. This approach maintains training-free simplicity while modeling temporal context.

Architecture. For each segment sequence \(X\):
- Linear projection: \(x_t \mapsto h_t^{(0)} = W x_t + b\), where \(W \in \mathbb{R}^{d \times 768}\) and \(d\) is the model dimension.
- Positional encoding: sinusoidal encoding \(p_t \in \mathbb{R}^d\) is added to each token, yielding \(\bar{h}_t^{(0)} = h_t^{(0)} + p_t\).
- Transformer Encoder: a stack of \(L\) layers with multi-head self-attention (\(H\) heads) and feed-forward sublayers operates on the sequence \(\{\bar{h}_t^{(0)}\}_{t=1}^T\). For a head with key/query dimension \(d_k\):
\[
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V, \quad
\text{with } Q=W_Q H,\ K=W_K H,\ V=W_V H.
\]
- Pooling: the sequence representation is collapsed to a single segment-level embedding \(z \in \mathbb{R}^d\) using one of:
  1) Mean pooling: \(z = \frac{1}{T} \sum_t h_t^{(L)}\);
  2) CLS token: prepend a learned \(\mathrm{[CLS]}\) token and take its final state;
  3) Attention pooling: learn a query vector \(q \in \mathbb{R}^d\), compute weights \(w_t = \mathrm{softmax}(q^\top h_t^{(L)})\), and set \(z = \sum_t w_t h_t^{(L)}\).

Frozen Inference. All parameters (projection, encoder, CLS, query) are used as randomly initialized, fixed weights; there is no gradient update. The design intentionally avoids training to preserve generality, reduce compute, and function without labels.

Hyperparameters and Selection. The model dimension, depth, and head count are selected via a small grid search over pooling \(\in\{\text{mean}, \text{cls}, \text{attn}\}\), \(d\in\{256, 512\}\), \(L\in\{2,4\}\), with \(H\) scaled to 4 or 8 accordingly. The globally best configuration is found to be mean pooling with \(d{=}256\), \(L{=}4\), \(H{=}4\). Empirically, mean pooling outperforms CLS and attention pooling in this frozen setting, likely due to the stability of uniform aggregation when attention parameters are not trained.

Standardization and Metric. As with the baseline, embeddings are standardized; when cosine distance is used, vectors are L2-normalized before KMeans. UMAP visualizations are computed under the same metric as used in feature space.

---

### 2.4 Evaluation Metrics
Let \(\{(z_i, y_i)\}_{i=1}^N\) be segment embeddings and predicted cluster labels.

1) Silhouette. For sample \(i\): \(a_i\) is the average distance to its own cluster, \(b_i\) the minimum average distance to other clusters. The per-sample score is \(s_i = \frac{b_i - a_i}{\max(a_i, b_i)}\), and the overall Silhouette is the mean \(\frac{1}{N}\sum_i s_i\). When cosine is used, distances are computed in cosine geometry; otherwise Euclidean.

2) Davies–Bouldin (DB). For clusters \(C_k\) with centroids \(\mu_k\) and within-cluster scatter \(S_k\), the DB index is
\[
\mathrm{DB} = \frac{1}{K} \sum_{k=1}^K \max_{j \neq k} \frac{S_k + S_j}{\delta(\mu_k, \mu_j)},
\]
where \(\delta\) is inter-centroid distance; lower is better.

3) Calinski–Harabasz (CH). With between-cluster dispersion \(B\) and within-cluster dispersion \(W\), CH is \(\mathrm{CH} = \frac{\mathrm{tr}(B)/(K-1)}{\mathrm{tr}(W)/(N-K)}\); higher is better.

4) Intra/Inter Ratio. A direct separation indicator tailored to this setting. For each cluster, compute the mean pairwise distance among its members; average across clusters to obtain \(D_\mathrm{intra}\). Compute the mean pairwise distance among all cluster centroids to obtain \(D_\mathrm{inter}\). The ratio is
\[
\mathrm{Intra/Inter} = \frac{D_\mathrm{intra}}{D_\mathrm{inter}}.
\]
Lower values indicate tighter clusters separated by larger centroid distances.

Implementation Notes. For cosine metrics, embeddings are L2-normalized prior to Euclidean computations so that Euclidean distances align with cosine distances; silhouettes are computed directly in cosine when available. KMeans uses multiple restarts (n_init=10). UMAP projections for visualization are computed under the same metric as the feature space.

---

### 2.5 End-to-End Pipeline Overview
The overall pipeline integrates aggregation/encoding, metric-consistent preprocessing, clustering, model selection over \(k\), and visualization. UMAP serves qualitative analysis only.

```mermaid
flowchart LR
    A[Segment sequence X ∈ R^{T×768}] --> B{Embedding}
    B -->|Baseline: attention-norm| C1[Segment vector z ∈ R^D]
    B -->|Proposed: Transformer (frozen)| C2[Segment vector z ∈ R^d]
    C1 --> D[Standardize]
    C2 --> D
    D --> E{Cosine?}
    E -->|Yes| F[L2-normalize]
    E -->|No| G[Identity]
    F --> H[KMeans over k∈[k_min,k_max]]
    G --> H
    H --> I[Select k by Silhouette]
    D --> J[UMAP 2D/3D (same metric)]
```

---

### 2.6 Design Choices and Ablations
- Training-free design. Both the baseline and the proposed encoder operate without labels or gradient updates. This ensures robustness to domain shift, low compute, and rapid iteration.
- Cosine geometry. High-dimensional embeddings benefit from cosine-based separation. By L2-normalizing features, Euclidean KMeans becomes consistent with cosine similarity, stabilizing clustering and metric evaluation.
- Pooling strategies. In a frozen encoder, mean pooling showed superior stability and aggregate discrimination compared to CLS and attention pooling, likely because the latter require learned parameters and task-specific adaptation.
- Model scale. A modest model (d=256, L=4, H=4) was preferred over larger alternatives (e.g., d=512) in grid search, balancing representation richness and noise amplification.

---

### 2.7 Practical Remarks
- Standardization precedes any normalization or clustering to harmonize feature scales.
- UMAP visualizations are produced for interpretability only and do not influence clustering.
- When using HDBSCAN as an auxiliary tool, core-only and noise-mapped evaluations together provide a nuanced view of density-based behavior; however, in this data regime, KMeans on transformer embeddings yields more favorable trade-offs.

---

### 2.7 Summary
A training-free transformer encoder is used to derive segment-level embeddings from variable-length 768-D sequences. With cosine-aware preprocessing and KMeans clustering, this approach consistently improves unsupervised separation quality over norm-based aggregation while preserving efficiency and label-free practicality. UMAP projections enable qualitative validation, and an HDBSCAN module remains available for high-confidence core detection when density structure is pronounced.

