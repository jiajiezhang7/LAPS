# Technical Documentation: LAPS Methodology

This document outlines the core technical implementation of the 3-phase pipeline described in Section III (System Methodology) of the paper.

## Phase 1: Latent Action Representation via Motion Tokenizer

This phase is responsible for converting raw video data into a high-dimensional, abstract representation of *motion dynamics*.

### 1.1. Core Component: Motion Tokenizer ($M_\theta$)
* **Objective:** To learn a model that can encode short video clips (represented by keypoint tracks) into sequences of latent vectors.
* **Architecture:** Based on the AMPLIFY paper. It's a Transformer-based autoencoder with a Finite Scalar Quantization (FSQ) layer.
    * **Encoder ($E_\theta$):** A Transformer that maps input motion to a latent sequence.
    * **Quantizer (FSQ):** A non-learned quantization layer that discretizes the latent sequence.
    * **Decoder ($D_\theta$):** A Transformer that reconstructs the motion.
* **Input Data:** Dense keypoint tracks extracted from short video clips ($\mathcal{D}_{\text{clips}}$).
    * An off-the-shelf tracker (e.g., `CoTracker`) is used to get $\kappa \in \mathbb{R}^{T \times N \times 2}$ (Time, Num_Points, Coords).
    * The model is trained on the **velocities** derived from these tracks, not the absolute positions.
* **Training Objective:**
    * This is **NOT** pixel-level reconstruction.
    * The decoder is trained on a **classification objective**: to predict the *relative displacement* for each keypoint track.
    * The loss function is a **cross-entropy loss** over a discrete spatial grid, effectively modeling motion dynamics as a categorical distribution.
* **Inference-Time Output (Critical Distinction):**
    * When $M_\theta$ processes a sliding window over a long video, it produces two distinct but related outputs:
    1.  **Continuous Quantized Vectors ($S_q$):** A sequence of vectors $\{z_{q,1}, \dots, z_{q,T}\}$ where each $z_{q,t} \in \mathbb{R}^{d_m}$ is the *actual vector prototype* from the FSQ codebook.
    2.  **Discrete Code Indices ($S_{discrete}$):** A sequence of integers $\{c_1, \dots, c_T\}$ representing the *indices* of those vectors in the codebook.
* **Data Flow:**
    * $S_q$ (the continuous vectors) is the **foundational signal** used for **Phase 2 (Segmentation)** and **Phase 3 (Clustering)**, as it preserves geometric information needed for metric calculations (e.g., L2 norm).
    * $S_{discrete}$ (the discrete indices) is the **final deliverable** for downstream **VLA pre-training**.

---

## Phase 2: Segmentation via Latent Action Energy

This phase takes the continuous latent vector stream ($S_q$) from Phase 1 and uses it to detect the temporal boundaries of meaningful action primitives.

### 2.1. Core Concept: Latent Action Energy ($E_{\text{action}}$)
* **Hypothesis:** Meaningful semantic shifts (e.g., "reach" $\rightarrow$ "grasp") cause distinct, sharp changes in the latent *motion dynamics* space, whereas simple physical motion (e.g., moving a grasped object) results in smooth changes.
* **Mathematical Definition:** The metric is defined as the **L2 norm of the temporal difference** of the **continuous quantized vectors ($z_q$)**:
    $$
    E_{\text{action}}(t) = \| z_{q,t} - z_{q,t-1} \|_2
    $$
* **Behavior:**
    * `Low $E_{\text{action}}$`: During a stable or smoothly evolving coherent action.
    * `Sharp Peak in $E_{\text{action}}$`: At the boundary between two semantically distinct actions.

### 2.2. Core Component: The Unsupervised Online Action Segmentor
* **Design:** A **causal, online state-machine** that operates on the streaming 1D signal $E_{\text{action}}(t)$. It is *not* an offline peak-finding algorithm.
* **Implementation Steps:**

    1.  **Causal Signal Smoothing:**
        * The raw $E_{\text{action}}(t)$ signal is noisy. It must be smoothed *without using future data*.
        * **Method:** Use an Exponential Moving Average (EMA).
        * **Formula:** $y_t = \alpha E_{\text{action}}(t) + (1 - \alpha) y_{t-1}$, where $y_t$ is the smoothed signal.

    2.  **Online Boundary Detection (Hysteresis):**
        * A robust two-state (ON/OFF) controller is implemented on the smoothed signal $y_t$.
        * **State 'ON'** means an action primitive is "in progress." **State 'OFF'** means it is "between actions."
        * **Start (OFF $\rightarrow$ ON):** Triggered if $y_t$ exceeds a high threshold $\theta_{\text{on}}$ and *stays* above it for a "debounce" duration of $u$ windows.
        * **End (ON $\rightarrow$ OFF):** Triggered if $y_t$ drops below a low threshold $\theta_{\text{off}}$ and *stays* below it for a "debounce" duration of $d$ windows. (Note: $\theta_{\text{off}} \le \theta_{\text{on}}$).

    3.  **Primitive and Sequence Extraction:**
        * When the state machine transitions from **ON $\rightarrow$ OFF**, the segment is finalized.
        * The system extracts two things for this segment $i$:
            1.  **Video Primitive ($A_i$):** The raw video clip $V[p_i : p_{i+1}]$.
            2.  **Discrete Code Sequence ($\mathcal{S}_i$):** The sequence of **discrete code indices** $\{c_t\}$ from all windows that overlap with the $[p_i, p_{i+1}]$ interval. This $\mathcal{S}_i$ is the structured data for VLA pre-training.

### 2.3. Unsupervised Threshold Calibration (Important Detail)
* The primary threshold $\theta_{\text{on}}$ is **not** set manually. It is determined via an **unsupervised offline optimization** process *before* online segmentation begins.
* **Procedure:**
    1.  **Generate Proxy Signal:** Compute a simple, low-level motion metric over a validation dataset (e.g., "velocity energy" from keypoint temporal differences).
    2.  **Generate Pseudo-Labels:** Apply a heuristic threshold (e.g., Otsu's method or a fixed quantile) to this *proxy signal* to create coarse-grained binary pseudo-labels $y_{\text{pseudo}}$ (distinguishing "motion" vs. "no motion").
    3.  **Optimize $\theta_{\text{on}}$:** Perform a parameter sweep for $\theta_{\text{on}}$ on the *real* **Latent Action Energy** signal ($E_{\text{action}}(t)$). Select the $\theta_{\text{on}}$ that maximizes a metric (e.g., F1-score) when comparing the thresholded $E_{\text{action}}$ against the noisy $y_{\text{pseudo}}$.
    4.  **Set Final Thresholds:** This optimized $\theta_{\text{on}}$ is now fixed. The lower threshold is set as a ratio: $\theta_{\text{off}} = r \cdot \theta_{\text{on}}$ (where $r$ is a hysteresis factor, e.g., 0.5).

---

## Phase 3: Unsupervised Discovery of the Action Vocabulary

This phase is a *validation* step. It takes all the segmented primitives from Phase 2 and checks if they form semantically coherent clusters, confirming the "countable" nature of the workstation tasks.

### 3.1. Input Representation
* For each segmented primitive $A_i$, we use its corresponding **continuous feature vectors** *before* the FSQ codebook (referred to as $X_i$ in the paper).
* **Format:** $X_i = [x_1, \dots, x_T] \in \mathbb{R}^{T \times d_m}$, where $T$ (length) varies for each primitive.
* **Goal:** Cluster these variable-length, high-dimensional time series.

### 3.2. Core Component: Training-Free Temporal Embedding
* **Objective:** To embed each variable-length sequence $X_i$ into a single, fixed-dimensional vector $z_i$.
* **Method:** A lightweight Transformer encoder.
* **Key Feature (Frozen):** This Transformer is **training-free**. Its parameters are randomly initialized and **never updated** (i.e., it's used in a "frozen" inference regime).
    * **Rationale:** Ensures generality, requires zero labels, and is computationally cheap.
* **Architecture:**
    1.  **Input:** The sequence $X_i$.
    2.  **Projection:** A linear layer projects $x_t$ into the model dimension $d$, and sinusoidal positional encodings are added.
    3.  **Encoder:** A stack of $L$ Transformer layers (with $H$ heads) processes the sequence. (Paper uses $L=4, H=4, d=256$).
    4.  **Pooling:** The final hidden states $H^{(L)} \in \mathbb{R}^{T \times d}$ are aggregated into a single vector $z_i \in \mathbb{R}^{d}$.
    5.  **Pooling Method:** **Mean pooling** ($z_i = \frac{1}{T} \sum_t h_t^{(L)}$) was found to be the most stable and effective method in this frozen setting.

### 3.3. Core Component: Action Clustering (Cosine K-Means)
* **Objective:** To cluster the fixed-size embeddings $\{z_i\}$ into $K$ groups.
* **Algorithm:** K-Means.
* **Distance Metric:** **Cosine similarity** is used, as vector orientation is more informative than magnitude in high-d spaces.
* **Implementation (Cosine K-Means Trick):** To use a standard K-Means algorithm (which optimizes Euclidean distance) for cosine similarity:
    1.  **Standardize:** Standardize all embeddings $z_i$ (zero mean, unit variance).
    2.  **L2-Normalize:** L2-normalize all standardized embeddings: $\hat{z}_i = z_i / \lVert z_i \rVert_2$.
    3.  **Run K-Means:** Run the standard K-Means algorithm on these normalized vectors $\hat{z}_i$.
* **Parameter $K$:** $K$ is **not** optimized. It is set as a fixed hyperparameter based on domain knowledge (the expected "countable" number of tasks at the workstation).

### 3.4. Core Component: Semantic Validation (VLM)
* **Objective:** To quantitatively *prove* that the $K$ clusters are semantically coherent.
* **Metric:** **Intra-Cluster Semantic Similarity (ICSS)**.
* **Procedure:**
    1.  For each discovered cluster $\mathcal{C}_k$:
    2.  Randomly sample $M$ pairs of **video primitives** $(A_i, A_j)$ from that cluster.
    3.  Use the **visual encoder** of a pre-trained VLM to get normalized embeddings for each video: $v_i = \text{VLM}(A_i)$ and $v_j = \text{VLM}(A_j)$.
    4.  Calculate the average cosine similarity for the cluster:
        $$
        \text{ICSS}_k = \frac{1}{M} \sum_{(i,j)} \cos(v_i, v_j)
        $$
* **Interpretation:** A high $\text{ICSS}_k$ score (compared to a random baseline) proves that the pipeline successfully grouped actions that are *semantically* similar, not just spatially close in the latent embedding space.