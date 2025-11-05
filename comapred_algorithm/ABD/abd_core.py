from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

_EPS = 1e-10


def _ensure_int(v: int, lo: int = 1) -> int:
    try:
        v = int(v)
    except Exception:
        v = lo
    return max(int(lo), int(v))


def smooth_features(X: np.ndarray, K: int, alpha: float = 0.5) -> np.ndarray:
    """
    Average filter smoothing over temporal axis.

    Args:
        X: (N, D) feature array
        K: target number of segments/classes
        alpha: smoothing factor controlling window size ~ alpha * (N / K)
    Returns:
        G: (N, D) smoothed features
    """
    assert X.ndim == 2, "X must be (N, D)"
    N, D = X.shape
    K = max(1, int(K))
    L = max(1, int(np.floor(alpha * N / float(K))))
    k = max(0, (L - 1) // 2)
    if N == 0:
        return X.copy()

    # Cumulative sum for efficient sliding mean
    csum = np.concatenate([np.zeros((1, D), dtype=X.dtype), np.cumsum(X, axis=0)], axis=0)  # (N+1, D)
    G = np.empty_like(X)
    for t in range(N):
        s = max(0, t - k)
        e = min(N, t + k + 1)
        cnt = float(e - s)
        G[t] = (csum[e] - csum[s]) / max(cnt, 1.0)
    return G


def adjacent_cosine_similarity(G: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between adjacent rows of G (N-1,).
    """
    assert G.ndim == 2, "G must be (N, D)"
    N = G.shape[0]
    if N <= 1:
        return np.zeros((0,), dtype=np.float32)
    # Normalize rows to unit length to accelerate
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    norms = np.maximum(norms, _EPS)
    U = G / norms
    S = np.sum(U[:-1] * U[1:], axis=1)
    return S.astype(np.float32)


def nms_minima(S: np.ndarray, N: int, K: int, alpha: float = 0.5) -> List[int]:
    """NMS for minima on similarity curve S (length N-1).

    Returns boundaries including 0 and N (frame indices).
    """
    if N <= 0:
        return [0]
    if S.size == 0:
        return [0, N]
    L = max(1, int(np.floor(alpha * N / float(max(K, 1)))))
    half = max(0, L // 2)
    candidates: List[int] = []
    for t in range(S.shape[0]):
        s = max(0, t - half)
        e = min(S.shape[0], t + half + 1)
        w = S[s:e]
        if t == (np.argmin(w) + s):
            candidates.append(t + 1)  # boundary between t and t+1 -> index t+1
    # Always include start and end; de-duplicate and sort
    B = [0] + sorted(set(candidates)) + [N]
    # Ensure strictly increasing and within [0, N]
    B = [int(x) for x in B if 0 <= int(x) <= int(N)]
    B = sorted(set(B))
    if B[0] != 0:
        B = [0] + B
    if B[-1] != N:
        B = B + [N]
    return B


def bottom_up_merge(X: np.ndarray, B: List[int], K: int) -> Tuple[np.ndarray, List[int]]:
    """
    Bottom-up merge segments until exactly K remain.

    Args:
        X: (N, D)
        B: list of boundaries including 0 and N
        K: desired number of segments
    Returns:
        labels: (N,) int labels in [0, K-1]
        B_new: refined boundaries including 0 and N (len = K+1)
    """
    assert X.ndim == 2, "X must be (N, D)"
    N, D = X.shape
    B = sorted([int(b) for b in B])
    B = [0] + [b for b in B if 0 < b < N] + [N]
    # Initialize segment features
    seg_starts = [B[i] for i in range(len(B) - 1)]
    seg_ends = [B[i + 1] for i in range(len(B) - 1)]
    P = []
    for s, e in zip(seg_starts, seg_ends):
        if e <= s:
            P.append(np.zeros((D,), dtype=X.dtype))
        else:
            P.append(np.mean(X[s:e], axis=0))
    P = np.stack(P, axis=0) if len(P) > 0 else np.zeros((0, D), dtype=X.dtype)

    # Helper to compute cosine sim matrix
    def _cos_sim_mat(A: np.ndarray) -> np.ndarray:
        if A.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32)
        n = np.linalg.norm(A, axis=1, keepdims=True)
        n = np.maximum(n, _EPS)
        U = A / n
        return (U @ U.T).astype(np.float32)

    # Iteratively merge most similar adjacent or global pair
    # Paper merges most similar pair globally; allow any pair
    while P.shape[0] > max(1, int(K)):
        M = P.shape[0]
        S = _cos_sim_mat(P)
        np.fill_diagonal(S, -np.inf)
        # Find max similarity pair
        flat_idx = int(np.argmax(S))
        i, j = divmod(flat_idx, M)
        if i > j:
            i, j = j, i
        # Merge j into i
        new_feat = (P[i] + P[j]) / 2.0
        # Update arrays
        P = np.delete(P, j, axis=0)
        P[i] = new_feat
        # Merge boundaries: segments i and j become single segment spanning min start to max end
        seg_starts[i] = min(seg_starts[i], seg_starts[j])
        seg_ends[i] = max(seg_ends[i], seg_ends[j])
        del seg_starts[j]
        del seg_ends[j]

    # Build refined boundaries
    B_new = [0]
    for s, e in zip(seg_starts, seg_ends):
        if s != B_new[-1]:
            # if gap, push s (shouldn't happen, but keep robust)
            if s > B_new[-1]:
                B_new.append(s)
        B_new.append(e)
    # Deduplicate and clamp
    B_new = sorted(set([int(b) for b in B_new if 0 <= int(b) <= int(N)]))
    if B_new[0] != 0:
        B_new = [0] + B_new
    if B_new[-1] != N:
        B_new = B_new + [N]

    # Assign labels 0..K-1 to frames per refined segments
    labels = np.zeros((N,), dtype=np.int32)
    for idx in range(len(B_new) - 1):
        s, e = B_new[idx], B_new[idx + 1]
        labels[s:e] = idx
    # If segments exceed K (rare due to dedup), trim by merging extra at end
    if len(B_new) - 1 > max(1, int(K)):
        # collapse tail segments into last label
        k = int(K)
        cut = B_new[k]
        labels[cut:] = k - 1
        B_new = B_new[: k + 1]

    return labels, B_new


@dataclass
class ABDResult:
    labels: np.ndarray  # (N,)
    boundaries: List[int]  # including 0 and N


def abd_offline(X: np.ndarray, K: int, alpha: float = 0.5) -> ABDResult:
    """Run the offline ABD pipeline on features X.

    Returns frame labels and boundaries (indices in [0, N]).
    """
    if X.ndim != 2 or X.shape[0] == 0:
        return ABDResult(labels=np.zeros((0,), dtype=np.int32), boundaries=[0])
    N, _ = X.shape
    K = _ensure_int(K, lo=1)
    alpha = float(alpha)
    G = smooth_features(X, K=K, alpha=alpha)
    S = adjacent_cosine_similarity(G)
    B0 = nms_minima(S, N=N, K=K, alpha=alpha)
    labels, B = bottom_up_merge(X, B0, K=K)
    return ABDResult(labels=labels, boundaries=B)

