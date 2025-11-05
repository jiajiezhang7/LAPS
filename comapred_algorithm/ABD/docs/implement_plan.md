```markdown
# ABD Algorithm Reproduction Documentation (Offline Version)

## Overview
This document provides a complete guide to reproduce the offline version of the Action Boundary Detection (ABD) algorithm as described in the paper "Fast and Unsupervised Action Boundary Detection for Action Segmentation" by Zexing Du et al. (CVPR 2022). The algorithm is unsupervised, requires no training, and segments untrimmed videos into actions by detecting boundaries based on frame similarities.

The method involves:
- Smoothing frame features to reduce noise.
- Computing similarities between adjacent smoothed frames.
- Detecting action boundaries using Non-Maximum Suppression (NMS) on the similarity curve to find change points.
- Refining the initial segments via bottom-up clustering to merge over-segmented parts until reaching K action classes.

Key advantages: No training stage, low-latency inference, and state-of-the-art performance on datasets like Breakfast, YouTube Instructional Videos, Hollywood Extended, and 50Salads.

This reproduction assumes you have pre-extracted frame-wise features (e.g., Fisher Vectors of IDT or I3D features). The algorithm operates on these features.

## Prerequisites
- **Programming Language**: Python (using libraries like NumPy for vector operations, SciPy for cosine similarity if needed).
- **Libraries**:
  - NumPy: For array operations, averaging, and cosine similarity.
  - SciPy (optional): For efficient cosine distance computation.
- **Input Data**:
  - Frame features: A NumPy array `X` of shape (N, D), where N is the number of frames, D is the feature dimension.
  - K: Number of action classes (prior knowledge; use average per dataset, e.g., 5 for Breakfast).
- **No Internet or Additional Installs**: All operations are local.

## Inputs
- `X`: NumPy array of shape (N, D) – Frame-wise features.
- `K`: Integer – Number of action classes (e.g., average number of action classes per video in the dataset).
- `alpha`: Float – Hyperparameter for filter and window sizes (default: 0.4 or 0.6 based on experiments; tune between 0.2-0.8).

## Outputs
- `partition`: List or array of frame-wise action labels (integers from 0 to K-1).
- `boundaries`: List of detected boundary indices (frame indices where actions change).

## Algorithm Steps
Follow these steps exactly. Pseudocode and explanations are provided.

### Step 1: Feature Smoothing
Smooth the original features to reduce noise from occlusion, viewpoint changes, etc. This creates "internal consistency within actions and external discrepancy across actions."

- Use an **average filter** (recommended for efficiency; Gaussian filter yields similar results).
- Kernel size: `(2k + 1) = alpha * (N / K)`, where `k = floor((alpha * N / K - 1) / 2)`.
- For each frame t (from 0 to N-1, 0-indexed):
  - `g_t = average of X[max(0, t-k):min(N, t+k+1)]` (mean along axis=0).
- Handle edges: For t near 0 or N, use available frames (no padding needed).
- Result: Smoothed features `G` as NumPy array (N, D).

**Pseudocode**:
```python
import numpy as np

N, D = X.shape
k = int(np.floor((alpha * N / K - 1) / 2))
G = np.zeros((N, D))
for t in range(N):
    start = max(0, t - k)
    end = min(N, t + k + 1)
    G[t] = np.mean(X[start:end], axis=0)

```

**Notes**:

- Equation from paper: \( g_t = \frac{\sum_{i=t-k}^{t+k} W(x_t, x_i) x_i}{\sum_{i=t-k}^{t+k} W(x_t, x_i)} \)
- For average filter, \( W = 1 \) (uniform weights).
- Smoothing makes similarity curves "⊓"-shaped per action (high within, low at boundaries).

### Step 2: Compute Frame-wise Similarities

Compute cosine similarity between adjacent smoothed frames.

- For t from 0 to N-2:
    - \( S_t = \frac{g_t \cdot g_{t+1}}{||g_t|| \cdot ||g_{t+1}||} \)
- Result: Array `S` of length N-1 (similarities).
- Use NumPy for efficiency: `cosine_sim = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))`.

**Pseudocode**:

```python
from numpy.linalg import norm

S = np.zeros(N - 1)
for t in range(N - 1):
    S[t] = np.dot(G[t], G[t+1]) / (norm(G[t]) * norm(G[t+1]) + 1e-10)  # Avoid division by zero

```

**Notes**:

- Similarities are high within actions, low at boundaries (change points).
- Visualization: As in Figure 2b, curves are smoother after filtering.

### Step 3: Action Boundary Detection with NMS

Detect boundaries by finding local minima (change points) in S using Non-Maximum Suppression (NMS) adapted for minima.

- Window length: `L = alpha * (N / K)` (same alpha as above).
- Perform NMS on S to suppress non-minima:
    - For each position t in S (0 to N-2):
        - In local window [max(0, t - L//2), min(N-1, t + L//2 + 1)], find if S[t] is the minimum.
        - If yes, mark t+1 as a boundary (since S_t is between frames t and t+1, low S_t indicates boundary at t+1).
- Always include frame 0 as start and N as end (boundaries are at 0, b1, b2, ..., N).
- Result: List of candidate boundaries `B = [0, b1, b2, ..., N]` (sorted indices).

**Pseudocode**:

```python
L = int(alpha * N / K)
candidates = []
for t in range(N - 1):
    start = max(0, t - L // 2)
    end = min(N - 1, t + L // 2 + 1)
    window = S[start:end]
    min_idx_in_window = np.argmin(window) + start
    if min_idx_in_window == t:
        candidates.append(t + 1)  # Boundary after frame t

B = sorted([0] + candidates + [N])

```

**Notes**:

- NMS suppresses ambiguous responses near true boundaries and pseudo-boundaries within actions.
- M (number of segments) = len(B) - 1, typically M > K due to over-segmentation.

### Step 4: Refinement via Bottom-Up Clustering

Refine by merging segments based on semantic similarity until exactly K segments remain.

- Compute segment features: For each segment m (from B[m-1] to B[m]-1):
    - \( \hat{x_m} = \) average of X[B[m-1]:B[m]]
- Result: `P` as array (M, D).
- Compute similarity matrix: \( S(i,j) = \cos(\hat{x_i}, \hat{x_j}) \) for i != j.
- Repeat until M == K:
    - Find max S(i,j) (most similar pair).
    - Merge segments i and j: New feature = average of \hat{x_i} and \hat{x_j}.
    - Update labels: Assign all frames in merged segments to the same label.
    - Reduce M by 1.
- Assign labels: Start with initial segment labels 0 to M-1, update during merges.
- Final: Frame-wise labels (0 to K-1).

**Pseudocode**:

```python
M = len(B) - 1
P = np.zeros((M, D))  # Segment features
segment_labels = np.arange(M)  # Initial labels

for m in range(M):
    start, end = B[m], B[m+1]
    P[m] = np.mean(X[start:end], axis=0)

while M > K:
    # Compute similarity matrix (only off-diagonal)
    sim_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(i+1, M):
            sim = np.dot(P[i], P[j]) / (norm(P[i]) * norm(P[j]) + 1e-10)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    # Find max similarity pair
    max_idx = np.unravel_index(np.argmax(sim_matrix), (M, M))
    i, j = min(max_idx), max(max_idx)  # Ensure i < j

    # Merge: average features
    new_feature = (P[i] + P[j]) / 2
    P[i] = new_feature
    # Update labels: map j's label to i's
    segment_labels[segment_labels == segment_labels[j]] = segment_labels[i]
    # Remove j
    P = np.delete(P, j, axis=0)
    segment_labels = np.delete(segment_labels, j)
    # Renormalize labels to 0..M-1
    unique_labels = np.unique(segment_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    segment_labels = np.array([label_map[l] for l in segment_labels])
    M -= 1

# Assign frame-wise labels
frame_labels = np.zeros(N, dtype=int)
for m in range(len(B) - 1):  # Original segments
    start, end = B[m], B[m+1]
    original_label = m
    new_label = label_map[original_label] if original_label in label_map else segment_labels[np.where(unique_labels == original_label)[0][0]]
    frame_labels[start:end] = new_label

```

**Notes**:

- Algorithm 1 in paper: Bottom-up merging based on max similarity.
- No temporal distance considered (unlike TW-FINCH), only semantic.
- Efficient since M << N.
- After merging, use Hungarian algorithm for evaluation (not part of algorithm; for metrics only).

## Hyperparameters

- `alpha`: Controls filter size and NMS window. From Table 6:
    - Test values: 0.2, 0.4, 0.6, 0.8.
    - Best: Often 0.4-0.6 (e.g., 0.4 for Breakfast F1=52.3).
- `K`: Dataset-specific average action classes (e.g., 5 for Breakfast, see Table 5 for sensitivity).
- Features: Use IDT Fisher Vectors or I3D for better performance.

## Evaluation (Optional, for Verification)

- Metrics: F1, MoF (Mean over Frames).
- Mapping: Use Hungarian algorithm to map predicted labels to ground-truth.
- Datasets: Breakfast (K~5), YouTube Instructions (background removed), etc.
- Compare to paper results (Tables 1-4).

## Potential Issues and Tips

- Numerical Stability: Add epsilon (1e-10) to norms to avoid division by zero.
- Edge Cases: Short videos (N small), set min k/L=1.
- Over-segmentation: NMS reduces it; refinement handles remaining.
- Performance: Runs in O(N) for smoothing/NMS, O(M^2 log M) for refinement (M small).
- Online Version: Not included; uses adaptive threshold instead of K.

This document is self-contained. Implement in code step-by-step.