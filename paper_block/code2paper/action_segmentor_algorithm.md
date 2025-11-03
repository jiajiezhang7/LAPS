# Method: Online Action Segmentation Driven by Latent Energy

## Problem Setup
Let x(t) denote a long or streaming video sampled at a target frame rate. The objective is to infer a set of action intervals S = { [t_s^m, t_e^m) }_m and, for each interval, an associated latent action sequence (codes). Inference proceeds causally on fixed-length windows of T frames advanced by stride k.

## Windowed Motion Representation
For each window of T frames sampled at target_fps and advanced by stride k, we estimate dense point trajectories on a regular N-point grid and compute per-point velocities. Velocities are normalized by a fixed decoder window size to yield a scale-invariant representation V ∈ R^{(T-1) × N × 2}. Two lightweight gates reduce compute and false positives before tokenization:
- Pre-gate: a pixel-difference gate using MAD and active-pixel ratio to drop near-static windows at very low cost.
- Motion-gate: a velocity gate using mean L2 velocity and active-track ratio on V to skip tokenization for static windows.

## Motion Tokenization
A motion tokenizer maps V to a continuous latent sequence Z_pre via an encoder, then projects it onto a learned finite codebook (FSQ) via a quantizer to obtain quantized latents Z_q and discrete token indices c. Shapes (typical): input (1, 1, T−1, N, 2) → Z_pre ∈ R^{1 × L × D} → Z_q ∈ R^{1 × L × D}, c ∈ {0,…,K−1}^L. Each time step selects the nearest codebook prototype e_k, so Z_q[t] = e_{c[t]}.

## Latent Energy Signal
We define a scalar per-window energy E from one of three sources (config energy.source):
- prequant: computed on Z_pre.
- quantized: computed on Z_q — chosen by default in our system.
- velocity: computed directly on V.
Aggregation modes (config energy.mode):
- l2_mean: E = (1/L) Σ_i ||z_i||_2.
- token_diff_l2_mean: E = (1/(L−1)) Σ_i ||z_i − z_{i−1}||_2.
We select quantized + token_diff_l2_mean as the default: it is amplitude-stable across videos (due to quantization) and sensitive to latent dynamics. A causal smoother (EMA or causal MA) optionally produces E_smooth that can be used for decisions without future leakage.

## Online Segmentation with Hysteresis and Debouncing
We run a two-state controller (OFF/ON) on the energy stream {E_t} (or {E_smooth,t} if smoothing is used):
- Start: enter ON when E_t ≥ θ_on for u consecutive windows and the cooldown has elapsed.
- End: return to OFF when E_t < θ_off for d consecutive windows, or when a maximum duration is reached.
Hysteresis: θ_off = r · θ_on with 0 < r ≤ 1. Debouncing uses integers u (up_count) and d (down_count) to suppress flicker. A cooldown of c windows prevents immediate re-trigger. Boundary alignment policy (start/center/end) determines which frames of the first window are written.

Pseudocode (per window):
1) Compute E (and optionally E_smooth). 2) If OFF: decrement cooldown; if E ≥ θ_on then pos_run++ else pos_run=0; if cooldown==0 and pos_run≥u → START. 3) If ON: write frames; if E < θ_off then neg_run++ else neg_run=0; if neg_run≥d or len_windows≥max_duration → END and set cooldown=c.

## Segment-Conditioned Latent Sequence Export
At segment termination, we select full windows whose frame coverage overlaps the segment interval by at least ρ. Non-overlapping selection (greedy) is default; an overlapped variant is supported. We export per-window FSQ codes (latent action sequence) and, importantly, the corresponding quantized vectors for each selected window. Very short segments (below a minimum window count) are dropped to reduce noise.

## Threshold Selection (Offline)
A dataset-level threshold θ_on is selected offline by constructing energy curves and sweeping θ on the chosen energy (quantized/token_diff_l2_mean by default) to maximize a criterion such as F1 or Youden’s J. The resulting best threshold is then used in streaming together with r, u, d, and cooldown. θ_off is derived as r · θ_on.

## Complexity and Deployment
The method is causal and single-pass. Per window, the dominant cost arises from tracking and tokenization, and the total runtime scales linearly with video length. Decision latency is bounded by the stride (k / FPS) and the smoothing horizon, which supports real-time operation under moderate compute budgets.


## Implementation Details and Parameters (Quantized as Default)
- Data source: energy.source = "quantized"; energy.mode = "token_diff_l2_mean" (default choice); smoothing: EMA (alpha≈0.4) or causal MA.
- Gates: pre-gate (pixel MAD + active-pixel ratio) and motion-gate (mean L2 velocity + active-track ratio) drop near-static windows prior to tokenization.
- Windowing: fixed window length T and stride k at target_fps; windows per second ≈ target_fps / k.
- Tokenization: encoder produces Z_pre; FSQ quantizer yields Z_q and codes c; each Z_q[t] equals the selected prototype e_{c[t]} in the learned codebook.
- Segmentation: parameters θ_on, r (θ_off = r·θ_on), up_count u, down_count d, cooldown c, optional max_duration (seconds or windows), and boundary alignment {start|center|end}.
- Export policy: overlap ratio ρ over [t_s, t_e) to select full windows; enforce min_save_windows and min_codes_windows; export both codes_windows and quantized_windows for each segment.
- Optional artifacts: per-window prequant tensors (for analysis), energy JSONL for diagnostics, and per-segment video files.

