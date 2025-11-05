import numpy as np
from comapred_algorithm.ABD.abd_core import abd_offline


def _make_piecewise_features(N=300, D=8, K=3, seed=0):
    rng = np.random.default_rng(seed)
    # Random segment lengths that sum to N
    lengths = rng.integers(low=max(5, N // (K * 2)), high=max(6, N // K + 20), size=K)
    s = int(np.sum(lengths))
    # Normalize lengths to sum to N
    ratios = lengths / s
    lens = [max(5, int(round(r * N))) for r in ratios]
    # Fix sum
    diff = N - sum(lens)
    lens[-1] += diff
    means = rng.normal(0, 1.0, size=(K, D)).astype(np.float32)
    X = []
    for i, L in enumerate(lens):
        X.append(means[i] + 0.05 * rng.normal(0, 1.0, size=(L, D)))
    X = np.concatenate(X, axis=0)
    # True boundaries
    b = [0]
    c = 0
    for L in lens:
        c += L
        b.append(c)
    return X.astype(np.float32), b


def test_abd_recovers_k_segments_basic():
    X, true_b = _make_piecewise_features(N=240, D=8, K=3, seed=42)
    res = abd_offline(X, K=3, alpha=0.5)
    assert len(res.boundaries) - 1 == 3
    # Allow tolerance around boundaries
    tol = 10
    # Compare internal boundaries (exclude 0 and N)
    tb = true_b[1:-1]
    pb = res.boundaries[1:-1]
    assert len(tb) == len(pb) == 2
    ok = sum([min([abs(p - t) for t in tb]) <= tol for p in pb])
    assert ok >= 2  # both should be close

