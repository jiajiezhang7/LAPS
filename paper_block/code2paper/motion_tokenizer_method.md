# Method: Motion Tokenizer — A Transformer VAE with FSQ for Multi-Track Motion Discretization

## Problem Formulation
- Input: a multi-view sequence of 2D track points `X ∈ R^{B×V×T×N×D}` with `D = 2`, where `V` is the number of views, `T` the number of timesteps, and `N` the number of tracks per instance.
- Goal: learn a discrete representation of motion (tokenization) while preserving the ability to reconstruct per-step relative displacements and full trajectories.
- Outputs: discrete code indices `codebook_indices`, quantized latents `codes`, and, for each `(view, time, track)`, classification logits over a spatial grid of relative displacements.

## Model Overview
- We implement a temporal quantized auto-encoder: a Transformer encoder maps the velocity sequence to a latent `z`, `FSQ` (factorized vector quantization) discretizes `z` into tokens, and a Transformer decoder reconstructs a categorical distribution over relative displacements (grid-based) for every view–time–track. This closes the loop from continuous motion → discrete tokens → reconstructable motion. Visual conditioning is optionally supported at the encoder.

## Architecture
### Sequence Construction
- Convert point trajectories to velocities along time using `points_to_velocities`, reducing the temporal length to `T−1`.
- View aggregation modes:
  - `per_view = true`: each view forms its own temporal sequence; the sequence length is `V · (T−1)`.
  - `per_view = false`: features from all views at the same timestep are concatenated; the sequence length is `T−1`.
- Per-step feature dimension: `in_dim = num_tracks · point_dim` (multiplied by `V` when `per_view = false`).

### Encoder (`VAEEncoder`)
- A linear projection maps inputs to `hidden_dim`, followed by a Transformer:
  - Without conditioning (`cond_on_img = false`), we use a `TransformerEncoder`.
  - With conditioning (`cond_on_img = true`), we use a `TransformerDecoder` where the motion sequence provides queries and visual tokens provide keys/values.
- Attention masking:
  - Causal temporal mask when `causal_encoder = true`.
  - Optional diagonal conditional mask for restricted cross-attention (`is_causal = 'diag'`).
  - Otherwise, all positions are visible.

### Visual Conditioning (Optional)
- A `VisionEncoder` converts images into temporal patch tokens (optionally including a `cls` token). Tokens are aligned across views and time, and are used as conditioning for cross-attention in the encoder.

### Vector Quantization (`FSQ`)
- `FSQ` discretizes the continuous latent `z` into `codes` and `codebook_indices`.
- The codebook size `codebook_size = K` is factorized into mixed-radix digits with small per-digit ranges, improving stability and utilization.
- During training, small Gaussian noise (`z_noise_std`) is added to `z` before quantization to promote code exploration.

### Decoder and Reconstruction (`VAEDecoder`)
- A `TransformerDecoder` takes zero query tokens (same sequence length) and uses the quantized `codes` as keys/values to produce reconstruction features.
- Structural priors:
  - Learnable track-index and view embeddings are fused with decoder features via a shallow MLP.
  - A linear projection maps each `(V, T−1, N)` feature to `H×W` logits over relative displacement bins, where `H×W = rel_cls_img_size[0] × rel_cls_img_size[1]`.
- The utility `rel_cls_logits_to_diffs` converts logits to continuous displacements by taking an expectation over the fixed grid. A scaling factor `decoder_pos_scale` mitigates template bias from positional/view embeddings.

## Learning Objective
- Relative displacement classification (primary): for each `(view, time, track)`, compute cross-entropy between the predicted grid distribution and the bin corresponding to the ground-truth displacement from timestep `t` to `t+1` (derived from `gt_traj`). Aggregate across views using `loss.loss_weights[view]`.
- Imbalance mitigation (optional): focal loss (`focal_gamma`, `focal_alpha`) and radial class weighting (`class_weight_mode = 'radial'`, `class_weight_strength`, `class_weight_power`) counter the dominance of near-zero displacements.
- Codebook diversity regularization (optional): compute a normalized entropy of token usage by mapping FSQ digits to a mixed-radix integer `code_id`, accumulating empirical frequencies `p`, and adding `loss.codebook_diversity_weight · (1 − H(p)/log(K_eff))`, where `K_eff` is the effective support.

## Training Procedure
- Preprocessing: convert points to velocities with `points_to_velocities`; for visualization, convert reconstructed velocities back to points with `velocities_to_points`.
- Optimization: AdamW with mixed precision; optional cosine schedule; gradient accumulation with factor `ceil(batch_size / gpu_max_bs)`.
- Monitoring:
  - Normalized codebook perplexity.
  - Step- and epoch-level code utilization: unique codes and dead-code ratio.
  - Trajectory metrics and image overlays via `vis_pred`.

## Inference
- Encoding: `encode(x, cond)` produces `z`; `FSQ(z)` yields the discrete `codebook_indices` (motion tokens) and `codes`.
- Decoding: given `codes`, `decode(codes)` predicts relative displacement distributions, which are converted to displacements and integrated into trajectories.
- Applications: compact tokenization enables indexing, retrieval, memory construction, and conditioning for downstream policies.

## Typical Shapes and Hyperparameters
- Tracks: `num_tracks = 400`, `point_dim = 2`.
- Temporal: `true_horizon = 16`; training operates on velocity sequences of length `T−1 = 15`.
- Codebook: `codebook_size = 2048`; grid size `rel_cls_img_size = 15×15`.
- Model: `hidden_dim = 768`, `num_heads = 8`, `num_layers = 2` (decoder uses half as many layers).
- Views: default `per_view = false` (concatenate views per timestep); optional `cond_on_img` for visual conditioning.

## Design Rationale
- Relative-displacement classification is more stable than direct regression and naturally represents uncertainty (multi-modality) through distributions over bins.
- `FSQ` provides stable, scalable tokenization via mixed-radix digits; small pre-quantization noise encourages exploration and utilization.
- Structural embeddings inject inductive bias; `decoder_pos_scale` reduces template bias.
- Optional visual conditioning aligns motion tokens with scene context.

## Limitations and Extensions
- Grid resolution trades accuracy for compute; multi-scale heads or adaptive binning are natural extensions.
- The current instantiation models 2D velocities; extending to 3D or richer dynamics (e.g., acceleration, contact) is straightforward.
- Visual conditioning can leverage stronger spatiotemporal encoders and tighter cross-modal alignment.

## Implementation Notes
- Core components: `MotionTokenizer`, `VAEEncoder`, `VAEDecoder`, `FSQ`, `compute_relative_classification_loss`, `rel_cls_logits_to_diffs`.
- Training loop: `train_motion_tokenizer.py` logs codebook utilization (unique codes, dead-code ratio, perplexity) and trajectory metrics.
