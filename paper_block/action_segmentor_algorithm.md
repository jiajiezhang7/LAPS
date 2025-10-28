# Method: Online Action Segmentation Driven by Latent Energy

## Problem Setup
Let x(t) denote a long or streaming video sampled at a target frame rate. The objective is to infer a set of action intervals S = { [t_s^m, t_e^m) }_m and, for each interval, an associated latent action sequence (codes). Inference proceeds causally on fixed-length windows of T frames advanced by stride k.

## Windowed Motion Representation
For each window, dense point trajectories are estimated on a regular grid to obtain per-point velocities. Velocities are temporally normalized to yield a scale-invariant representation V in R^{(T-1) x N x 2}. Lightweight gating (based on pixel differences or velocity magnitude) optionally rejects windows with negligible motion.

## Motion Tokenization
A motion tokenizer maps V to a continuous latent sequence Z_pre via an encoder, then projects it onto a learned finite codebook via a quantizer to obtain quantized latents Z_q and discrete token indices c (FSQ codes). Each window therefore yields a short latent action sequence c that summarizes the local motion dynamics.

## Latent Energy Signal
We define a scalar per-window energy E from one of three sources:
- prequant: computed on Z_pre.
- quantized: computed on Z_q (default in our experiments).
- velocity: computed on V.
Two generic aggregation modes are considered:
- l2_mean: E = (1/L) sum_i ||z_i||_2.
- token_diff_l2_mean: E = (1/(L-1)) sum_i ||z_i - z_{i-1}||_2.
In practice, quantized + token_diff_l2_mean is robust to appearance variation and sensitive to latent dynamics. A causal smoother y_t = alpha * E_t + (1 - alpha) * y_{t-1} (or a causal moving average) stabilizes the decision signal without future leakage.

## Online Segmentation with Hysteresis and Debouncing
We run a two-state machine (OFF/ON) on the energy sequence {E_t}:
- Start: enter ON when E_t >= theta_on for u consecutive windows and a cooldown has elapsed.
- End: return to OFF when E_t < theta_off for d consecutive windows or when a maximum duration is reached.
We set theta_off = r * theta_on with 0 < r <= 1 to realize hysteresis, and use u/d as debouncing to suppress flicker. This yields stable, low-latency segmentation suitable for streaming. An optional boundary alignment policy (start/center/end) refines the initial frames written for each segment.

## Segment-Conditioned Latent Sequence Export
At segment termination, we select whole windows whose frame coverage overlaps the segment interval by at least rho. A greedy non-overlapping selection (or an overlapped variant) yields a compact set of windows whose FSQ codes are exported as the segment-level latent action sequence, preserving temporal coherence while avoiding boundary fragmentation.

## Threshold Selection (Offline)
A dataset-level threshold theta is selected offline by constructing energy curves, deriving motion-based pseudo-labels (e.g., from velocity via Otsu or fixed percentile), and sweeping theta on the target energy (typically quantized/token_diff_l2_mean) to maximize a criterion such as F1 or Youden's J. The selected threshold transfers to streaming via the hysteresis-and-debouncing controller.

## Complexity and Deployment
The method is causal and single-pass. Per window, the dominant cost arises from tracking and tokenization, and the total runtime scales linearly with video length. Decision latency is bounded by the stride (k / FPS) and the smoothing horizon, which supports real-time operation under moderate compute budgets.

