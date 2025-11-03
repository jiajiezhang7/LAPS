# Experiment Implementation Plan for LAPS Paper

## 1. Core Objectives (The Three Questions)

All experiments are designed to definitively answer three core research questions:

1.  **Q1 (Signal Quality):** Is our proposed **"Latent Action Energy" ($E_{\text{action}}$)** a more robust and semantically accurate signal for action segmentation than traditional physical motion metrics (i.e., Optical Flow)?
2.  **Q2 (Segmentation Performance):** Does our full, unsupervised **Action Segmentor** pipeline outperform state-of-the-art (SOTA) unsupervised temporal action detection (TAD) baselines on our real-world industrial dataset?
3.  **Q3 (Primitive Quality):** Are the segmented action primitives **semantically coherent** and do they form a **"countable" set**? (i.e., are they high-quality and suitable for VLA pre-training?)

## 2. Setup, Components, and Pre-requisites

This section outlines the "what" and "how" for setting up the experiments.

### 2.1. Dataset
* **Source:** Our new, real-world "Industrial Motor Assembly" dataset.
* **Key Characteristics:** Continuous (hours-long), multi-view (top-down + exocentric), fine-grained, repetitive tasks.
* **Ground Truth (GT):** We will need manually annotated action boundaries and class labels for the *test set* to calculate segmentation metrics (mAP, F1) and clustering metrics (NMI).

### 2.2. Core Pipeline Components (To Implement)

1.  **Motion Tokenizer ($M_{\theta}$):**
    * **Architecture:** Based on AMPLIFY (Transformer Encoder/Decoder + FSQ).
    * **Training Data:** A large corpus of *unlabeled* video clips ($\mathcal{D}_{\text{clips}}$) from the *training split* of our industrial dataset.
    * **Key Task:** Must be trained *specifically for this workstation* to learn its "action vocabulary."
    * **Output:** For a given window of keypoint tracks, it outputs the quantized latent sequence $S_q$.

2.  **Action Segmentor (LAPS):**
    * **Input:** The streaming sequence of $S_q$ from the $M_{\theta}$.
    * **Core Logic:**
        1.  Calculate our metric: $E_{\text{action}}(t) = ||z_{q,t} - z_{q,t-1}||_2$.
        2.  Implement a causal, **online state-machine** (not offline peak-finding).
        3.  This state-machine must use **hysteresis** (two thresholds, $\theta_{\text{on}}$ and $\theta_{\text{off}}$) and debounce windows ($u, d$) to robustly find boundaries.
    * **Output:** A set of segmented video primitives (start/end timestamps).

3.  **Clustering & Analysis Pipeline:**
    * **Input:** The set of segmented primitives from the Action Segmentor.
    * **Step 1 (Embedding):**
        * For each variable-length primitive, get its corresponding latent sequence $X_i \in \mathbb{R}^{T \times 768}$.
        * Pass this sequence through a **training-free (Frozen) Transformer** encoder ($L=4, H=4, d=256$).
        * Apply **mean pooling** to the output tokens to get a fixed-size embedding $z_i \in \mathbb{R}^{256}$.
    * **Step 2 (Clustering):**
        * Implement **K-Means** with **cosine geometry**.
        * *Implementation:* First, standardize all embeddings $z_i$ (zero mean, unit variance), then L2-normalize them. Finally, apply standard K-Means.
        * $K$ is a pre-defined hyperparameter (based on our "countable" hypothesis).

### 2.3. Baselines (To Implement / Port)

1.  **Optical Flow Baseline:**
    * **Goal:** To isolate signal quality (Q1).
    * **Implementation:**
        1.  Compute a frame-wise Optical Flow Magnitude signal for the entire video.
        2.  Feed this raw signal into the *exact same* online state-machine (hysteresis, etc.) built for our Action Segmentor.
    * This provides a perfect apples-to-apples comparison of $E_{\text{action}}$ vs. Optical Flow.

2.  **OTAS (SOTA Baseline):**
    * **Goal:** To compare against a SOTA unsupervised TAD method (Q2).
    * **Implementation:** We must obtain the official implementation of OTAS and run it on our dataset. This method uses its own complex feature fusion (global, object-interaction, etc.).

### 2.4. Metrics (To Implement)

1.  **Segmentation Metrics:**
    * Standard **mAP** at IoU thresholds (0.50, 0.75).
    * Strict boundary-level **F1-score @ 2-second tolerance (F1@2s)**.

2.  **Clustering & Quality Metrics:**
    * **Normalized Mutual Information (NMI):** Requires cluster assignments and GT labels.
    * **Silhouette Score:** Internal metric, requires no labels.
    * **Intra-Cluster Semantic Similarity (ICSS):**
        * Requires a pre-trained VLM (e.g., CLIP-ViT).
        * **Logic:**
            1.  For each cluster $C_k$, randomly sample $M$ pairs of video primitives $(A_i, A_j)$ from it.
            2.  Get VLM visual embeddings $v_i = \text{VLM}(A_i)$ and $v_j = \text{VLM}(A_j)$.
            3.  Calculate $ICSS_k = \text{mean}(\text{cosine\_similarity}(v_i, v_j))$ for all $M$ pairs.
        * **Baseline:** Implement a "Random Pairs" baseline by sampling $M$ pairs from the *entire dataset* (not from within a cluster) and computing their average similarity.

## 3. Experiment Execution Plan & To-Do List

### ✅ Exp 1: Efficacy of Latent Motion Energy (Q1)

* **Goal:** Prove our signal is more semantically meaningful than optical flow.
* **Action:** Generate the qualitative plot for Figure 3.
* **To-Do:**
    1.  [ ] Pick a representative 60-second clip from the test set.
    2.  [ ] Plot the GT action boundaries (vertical dashed lines).
    3.  [ ] Run $M_{\theta}$ and compute/plot our $E_{\text{action}}$ signal (blue line).
    4.  [ ] Compute/plot the Optical Flow Magnitude signal (red line).
    5.  [ ] Verify that $E_{\text{action}}$ peaks align with GT boundaries (semantic shifts) while Optical Flow peaks align with all physical motion (noise).

### ✅ Exp 2: Unsupervised Action Segmentation Performance (Q2)

* **Goal:** Quantitatively prove our segmentor (LAPS) is superior to baselines.
* **Action:** Generate the quantitative data for Table 1.
* **To-Do:**
    1.  [ ] Run our **full LAPS pipeline** (Motion Tokenizer + Action Segmentor) over the entire test set to get predicted boundaries.
    2.  [ ] Run the **Optical Flow Baseline** over the entire test set.
    3.  [ ] Run the **OTAS baseline** over the entire test set.
    4.  [ ] For all three sets of predictions, compute the segmentation metrics (mAP@0.50, mAP@0.75, F1@5s, F1@2s) against the GT boundaries.
    5.  [ ] Confirm our method (LAPS) has the highest scores, especially for F1@2s and mAP@0.75.

### ✅ Exp 3: Primitive Quality & Semantic Coherence (Q3)

This is a 3-part experiment to validate the *output* of our segmentor.

#### 3.1. Qualitative Validation (Clustering Visualization)

* **Goal:** Show that the "countable" action vocabulary emerges visually.
* **Action:** Generate the t-SNE visualization for Figure 5.
* **To-Do:**
    1.  [ ] Take all primitives segmented by LAPS from the *training set*.
    2.  [ ] Pass all of them through the **Clustering & Analysis Pipeline** (Frozen Transformer + Mean Pooling) to get the $\mathbb{R}^{256}$ embeddings.
    3.  [ ] Run K-Means on these embeddings to get cluster IDs for each primitive.
    4.  [ ] Run t-SNE (or UMAP) on the $\mathbb{R}^{256}$ embeddings for visualization.
    5.  [ ] Plot the resulting 2D points, colored by their K-Means cluster ID.
    6.  [ ] Manually inspect the clusters to confirm they are well-separated and correspond to tasks (e.g., 'Pick Screwdriver', 'Fasten Screw').

#### 3.2. Quantitative Validation (Clustering Metrics)

* **Goal:** Show our temporal embedding (Frozen Transformer) is crucial for distinguishing actions.
* **Action:** Generate the data for Table 2.
* **To-Do:**
    1.  [ ] **Implement a baseline:** "Simple Mean Pooling" (i.e., take the mean of the raw $X_i \in \mathbb{R}^{T \times 768}$ latents, skipping the Frozen Transformer).
    2.  [ ] Get cluster assignments for both methods:
        * `Ours (Frozen Transformer)` $\rightarrow$ (Embeddings from 3.1)
        * `Mean Pooling (Baseline)`
    3.  [ ] Calculate **NMI** (vs. GT labels) for both sets of cluster assignments.
    4.  [ ] Calculate **Silhouette Score** (internal) for both sets of embeddings.
    5.  [ ] Confirm our Frozen Transformer method is superior on both metrics.

#### 3.3. Semantic Validation (VLM-based)

* **Goal:** Provide the final, objective proof of semantic coherence.
* **Action:** Generate the ICSS data for Table 3.
* **To-Do:**
    1.  [ ] Implement the **ICSS metric** and its **"Random Pairs" baseline** as defined in Section 2.4.
    2.  [ ] Use the K-Means clusters from step 3.1.
    3.  [ ] For each cluster $K_i$, run the ICSS calculation (sample pairs, get VLM embeddings, compute avg. cosine similarity).
    4.  [ ] Run the "Random Pairs" baseline calculation.
    5.  [ ] Report all $ICSS_k$ scores and the Mean ICSS.
    6.  [ ] Confirm `Ours (K-Means) Mean ICSS` is significantly higher than the `Random Pairs (Baseline)`.

### ✅ Exp 4: Ablation Studies

* **Goal:** Justify our key pipeline design choices.
* **Action:** Generate the data for Table 4.
* **To-Do:**
    1.  [ ] **(Signal Source Ablation):**
        * `Full Pipeline (Ours)`: Use F1 score from Exp 2.
        * Modify the $E_{\text{action}}$ calculation to use `Pre-Quantized Latents` (the continuous vectors *before* FSQ). Re-run segmentation (Exp 2) and report F1.
        * Modify the $E_{\text{action}}$ calculation to use `Raw Velocities` (the input to $M_{\theta}$). Re-run segmentation and report F1.
    2.  [ ] **(Encoder Ablation):**
        * `w/o Transformer (Mean-pool)`: Use the NMI score from Exp 3.2.
    3.  [ ] **(Representation Ablation):**
        * `w/o $M_\theta$`: Implement a new baseline. Instead of $M_{\theta}$ latents, use generic off-the-shelf video features (e.g., CLIP, IDT).
        * Feed these generic features into a segmentation method (e.g., ABD or our state-machine) and report the **Seg. F1**.
        * Cluster these generic features and report the **Cluster NMI**.
        * Confirm that our specialized, domain-trained $M_\theta$ is superior.