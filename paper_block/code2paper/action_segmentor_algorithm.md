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


## Formal Mathematical Description (Supplement)

- **1. Windowing and motion representation**
  - Let x(t) be sampled at target_fps. Define windows indexed by t = 0,1,2,..., each covering T frames and advanced by stride k frames.
  - Within window t, dense grid trajectories yield normalized per-point velocities V_t ∈ R^{(T−1)×N×2}. Normalization uses a fixed decoder window size W_dec to achieve scale invariance across temporal resolutions.
  - Notation: V_t[τ,n] ∈ R^2 is the velocity at intra-window time τ ∈ {1,…,T−1} for track n ∈ {1,…,N}.

- **2. Latent tokenization**
  - An encoder produces Z_pre,t = Enc(V_t) ∈ R^{L×D}. A quantizer maps each time step to the nearest codebook prototype e_k from E = {e_1,…,e_K}, yielding Z_q,t with codes c_i:
    Z_q,t[i] = e_{c_i},  where  c_i = argmin_k ||Z_pre,t[i] − e_k||_2.
  - By default, energy is computed on Z_q,t (quantized).

- **3. Per-window energy definitions** (let Z denote Z_pre,t or Z_q,t according to energy.source):
  - l2_mean:
    E_t = (1/L) Σ_{i=1}^L ||Z[i]||_2.
  - token_diff_l2_mean (default with quantized):
    E_t = (1/(L−1)) Σ_{i=2}^L ||Z[i] − Z[i−1]||_2.
  - Velocity alternative (source = velocity):
    E_t = (1/((T−1)N)) Σ_{τ=1}^{T−1} Σ_{n=1}^N ||V_t[τ,n]||_2.

- **4. Causal smoothing (no future leakage, optional)**
  - Exponential moving average (EMA):
    Ê_t = α·E_t + (1−α)·Ê_{t−1},  with Ê_0 = E_0 and α ∈ (0,1).
  - Causal moving average (window w):
    Ê_t = (1/m_t) Σ_{j=max(0,t−w+1)}^{t} E_j,  where m_t = t − max(0,t−w+1) + 1.
  - If enabled for decisions, use E*_t := Ê_t; otherwise E*_t := E_t.

- **5. Online segmentation with hysteresis and debouncing**
  - State s_t ∈ {0,1} denotes OFF(0)/ON(1). Cooldown counter cd_t ∈ Z_≥0 prevents immediate re-trigger after an END. Let u,d ∈ Z_≥1 be up/down debouncing lengths. Let θ_on > 0 be the start threshold and θ_off = r·θ_on with r ∈ (0,1] be the release threshold (hysteresis).
  - Run-length counters:
    pos_t = 1 + pos_{t−1} if E*_t ≥ θ_on, else 0;  neg_t = 1 + neg_{t−1} if E*_t < θ_off, else 0; with pos_{−1} = neg_{−1} = 0.
  - Transitions:
    - If s_{t−1} = 0 (OFF): cd_t := max(0, cd_{t−1} − 1). If cd_t = 0 and pos_t ≥ u ⇒ START: s_t := 1; reset pos_t := 0, neg_t := 0.
    - If s_{t−1} = 1 (ON): If neg_t ≥ d ⇒ END: s_t := 0; set cd_t := c (cooldown windows); reset pos_t := 0, neg_t := 0. Otherwise s_t := 1.
  - Optional maximum duration: If current ON run length reaches D_max windows, force END as above.
  - Boundary alignment a ∈ {start, center, end} selects which subrange of the first window is written; subsequent windows append only the last k frames to avoid duplication.

- **6. Threshold selection (offline)**
  - From recorded curves {E_t} (or {Ê_t}), sweep θ_on on a development set to optimize a criterion (e.g., F1 or Youden’s J). Deploy θ_on together with r,u,d,c tuned for stability and latency. θ_off is set as r·θ_on.

- **7. Not peak detection**
  - The detector is not a peak finder. It triggers on sustained threshold crossings with hysteresis and debouncing. Transient spikes that do not satisfy pos_t ≥ u do not START; brief dips that do not satisfy neg_t ≥ d do not END.

- **8. Latency bound**
  - Decision latency is bounded by k/target_fps plus any smoothing horizon, supporting causal real-time operation.
