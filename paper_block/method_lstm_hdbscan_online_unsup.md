# Method: Online Unsupervised Analysis of Primitives via LSTM Embedding + HDBSCAN

## Overview
We present a method for online, unsupervised analysis of sequences of discrete motion primitives. A self-supervised sequence encoder (LSTM/GRU) maps variable-length token sequences into fixed-dimensional embeddings. A density-based hierarchical clustering algorithm (HDBSCAN) discovers clusters without pre-specifying their number and provides per-sample membership strengths that serve as confidence scores for anomaly detection. The approach naturally treats sparse or isolated patterns as noise, supporting open-set behavior in realistic deployments.

## Notation and Data
- Discrete primitive sequence: for a video/action instance, let `x = (c1, c2, ..., cT)` with `ct ∈ {0, 1, ..., K-1}` produced by an upstream motion tokenizer.
- Padding and shift: to reserve `0` for PAD, we shift tokens by `+1` during training and inference, i.e., `xt = ct + 1`, and pad with `0`. Sequences are truncated or padded to a maximum length `Tmax`.

## Sequence Encoder (Self-supervised)
- Architecture:
  - Token embedding `E ∈ R^{(K+1)×d}`.
  - A bidirectional RNN (LSTM/GRU) with `L` layers and hidden size `d` produces per-step hidden states `ht ∈ R^{d'}` where `d' = 2d` if bidirectional.
  - A linear prediction head `W ∈ R^{d'×(K+1)}` outputs next-token logits.
- Training objective (next-token prediction, NTP):
  - Given inputs at positions `t = 1..T-1`, predict targets `x_{t+1}`; PAD targets are ignored.
  - Cross-entropy over valid positions: `L = Σ_t CE(W^T h_t, x_{t+1})`.
- Rationale: NTP requires no manual labels and encourages modeling of sequential dependencies and transition regularities, yielding embeddings amenable to clustering.

## Sequence-level Embedding for Clustering
- Temporal aggregation: with the encoder frozen, run a forward pass to obtain `h = (h1..hT)` and mean-pool over non-PAD positions: `z = mean_t(h_t)`.
- Normalization: L2-normalize `z` to stabilize distance computations and density estimation.
- Optional dimensionality reduction: apply PCA to `z`, selecting the smallest number of components that reaches a specified cumulative explained-variance threshold while enforcing a minimum dimension.

## Unsupervised Clustering (HDBSCAN)
- Principle: HDBSCAN performs hierarchical density-based clustering, does not require `K`, forms stable clusters across density levels, and marks unstable/sparse samples as noise (label `-1`).
- Key settings:
  - `min_cluster_size` controls the minimum persistent cluster size (a practical rule is `max(10, round(1%·N))`).
  - `min_samples` adjusts the conservativeness of the core density estimate (defaults to `min_cluster_size` if unset).
  - Metric: Euclidean is recommended with L2-normalized embeddings.
  - Cluster selection: `eom` or `leaf` (`eom` tends to be more conservative).
- Outputs: for each sample, a cluster label (including `-1` for noise) and a membership strength in `[0, 1]`, interpreted as a confidence score.

## Online Assignment and Anomaly Detection
- Procedure:
  1. Encode a new token sequence with the trained encoder to obtain `z` (mean-pooling + L2).
  2. If PCA was used offline, project `z` with the same PCA to keep feature spaces aligned.
  3. Use HDBSCAN `approximate_predict` on the fitted model to obtain a cluster id and membership probability `p`.
- Decision rule:
  - If the predicted cluster id is `-1` or `p < τ`, declare the sequence an anomaly/unknown primitive; otherwise, report the cluster id.
  - This yields open-set behavior: novel or rare patterns fall in low-density regions and are flagged as noise or low-confidence.

## Hyperparameters and Recommended Settings
- Encoder:
  - `d = 128`, `L = 2`, `dropout = 0.1`, bidirectional RNN.
  - Objective: NTP with cross-entropy on valid (non-PAD) targets.
- Clustering:
  - Use Euclidean distance after L2 normalization; start `min_cluster_size` around ~1% of `N`; set `min_samples` equal to `min_cluster_size` for conservative density estimates.
  - Choose `τ` based on validation statistics or application tolerance (e.g., `0.2` as a starting point).
- Inference consistency: preprocessing used online must match the offline fit (L2, PCA, metric).

## Computational Considerations
- Encoding: per-sequence forward complexity is approximately `O(T·d·L)`, where `T` is the effective length; mean-pooling overhead is negligible.
- Online assignment: `approximate_predict` runs with substantially lower latency than reclustering and is suitable for near real-time inference in practice.

## Properties and Applicability
- No `K` required: density-based clustering adapts to data structure without predefining the number of clusters.
- Robust anomaly handling: low-density points are labeled noise or assigned low confidence, suitable for open-set monitoring.
- Sequence-aware representation: NTP-trained LSTM/GRU captures order and context, distinguishing patterns that bag-of-words cannot.

## Limitations and Extensions
- Limitations:
  - Embedding quality depends on the upstream discretization; codebook drift or distortion degrades cluster boundaries.
  - The method clusters whole sequences and does not explicitly address finer-grained online segmentation.
- Extensions:
  - Replace LSTM with GRU or Transformer encoders.
  - Use attention-based pooling or multi-scale representations.
  - Incorporate semi-supervised/self-training to improve semantic interpretability.

## Summary
The method learns sequence-level representations with a self-supervised encoder and performs density-aware unsupervised clustering in the embedding space with HDBSCAN. At inference, `approximate_predict` provides low-cost online assignment, and membership thresholds enable anomaly detection, yielding an online unsupervised analysis pipeline for discrete primitives.
