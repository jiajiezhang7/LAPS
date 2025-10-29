# Bayes GMM Tuning Plan (bays_gmm_tunning.md)

This note records how to tune Bayesian Gaussian Mixture (bayes_gmm) to improve unsupervised cluster quality on LSTM embeddings.

## Baseline (current run)
- **result_dir**: action_classification/results/bayes_gmm_fits/20251029_143628/20251029_153423
- **num_samples**: 15002
- **num_pred_clusters**: 12
- **cluster_sizes (min–max)**: 858–1761 (fairly balanced)
- **metrics**:
  - **silhouette (euclidean)**: 0.0929 (low; weak separation)
  - **calinski_harabasz**: 1264.24 (use for relative comparisons)
  - **davies_bouldin**: 2.392 (high; lower is better)
- **preprocessing**: L2-normalize, PCA disabled (max_dims=0)
- **model**: BayesianGaussianMixture(full covariance, DP prior, n_components=12, init=kmeans, max_iter=1000)

## Tuning goals
- **Increase separation**: higher silhouette, higher CH, lower DB.
- **Avoid over-fragmentation**: reasonable effective clusters; avoid many tiny clusters.
- **Improve stability**: similar results across seeds/subsamples.

## Primary levers
- **Preprocessing**
  - Enable PCA to 32–50 dims with 0.99 explained variance cap.
  - Keep L2 normalization on (works well with euclidean metrics).
- **Model hyperparameters**
  - `n_components`: upper bound; DP prior will prune. Try 8, 12, 20.
  - `covariance_type`: full → diag (often more stable in high-dim); also try tied.
  - `init_params`: keep `kmeans`; sweep seeds for robustness.
  - `max_iter`: keep 1000 unless convergence issues appear.
- **Advanced (requires tiny code change)**
  - `weight_concentration_prior` (scalar): smaller → fewer effective clusters; larger → more clusters. Grid e.g. [0.1, 0.3, 0.5, 1.0, 2.0].

## Quick-win configurations (no code change)
Use one change at a time, then combine the best.

- **Config A (PCA-32 + diag)**
  - Expect tighter clusters, improved silhouette/DB.
- **Config B (n_components=20)**
  - Allows DP to “select” effective K < 20 instead of forcing 12.
- **Config C (n_components=8 + diag)**
  - Encourages fewer, tighter clusters if data is coarse-grained.

### YAML snippets (eval_config.yaml)
```yaml
preprocessing:
  lstm_feature:
    pca:
      max_dims: 50           # try 32 or 50
      explained_variance_threshold: 0.99
      min_dims: 8
    l2_normalize: true

clustering:
  bayes_gmm:
    n_components: 20         # try 8, 12, 20
    covariance_type: diag    # try full, diag, tied
    init_params: kmeans
    max_iter: 1000
```

Note: To tune the DP strength explicitly, add the following and update the script to pass it to the model (see “Optional small code change” below):
```yaml
clustering:
  bayes_gmm:
    weight_concentration_prior_type: dirichlet_process
    weight_concentration_prior: 0.5   # try 0.1, 0.3, 0.5, 1.0, 2.0
```

## How to run
```bash
# Always activate the project environment first
conda activate laps

# Fit Bayes-GMM on LSTM embeddings (edit --embed-dir to your embeddings folder)
python -m action_classification.clustering.fit_bayes_gmm \
  --embed-dir /path/to/embeddings \
  --config /home/johnny/action_ws/action_classification/configs/eval_config.yaml \
  --out-dir /home/johnny/action_ws/action_classification/results/bayes_gmm_fits
```

Outputs in the new timestamped folder:
- **aggregate_results.json**: silhouette/CH/DB, cluster counts, UMAP path
- **cluster_assignments.jsonl**: index/path → cluster
- **cluster_meta.json**, **model_bayes_gmm.pkl**

## Recommended experiment matrix (lightweight subset)
Run the following 6 combos first; pick the best by silhouette (tie-break by lower DB, then higher CH):

- **E1**: PCA=0, cov=full, n_comp=12, seed=0 (baseline repeat)
- **E2**: PCA=32, cov=diag, n_comp=12, seed=0
- **E3**: PCA=50, cov=diag, n_comp=12, seed=0
- **E4**: PCA=32, cov=diag, n_comp=20, seed=0
- **E5**: PCA=32, cov=tied, n_comp=12, seed=0
- **E6**: Best of E2–E5 with seed=42 to check stability

If improvements are modest, expand to:
- PCA ∈ {0, 32, 50}
- cov ∈ {full, diag}
- n_components ∈ {8, 12, 20}
- seeds ∈ {0, 42}

## Results logging template
Append one line per run after it finishes:
```csv
datetime,result_dir,PCA_dims,covariance,n_components,seed,silhouette,DB,CH,num_clusters,min_cluster,max_cluster
20251029_153423,action_classification/results/bayes_gmm_fits/20251029_143628/20251029_153423,0,full,12,0,0.0929,2.392,1264.241,12,858,1761
```

## Stability check across seeds (optional)
Compute ARI between two runs’ assignments (align by `path`):
```python
import json
from sklearn.metrics import adjusted_rand_score

def load_assignments(jsonl_path):
    d = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            d[rec['path']] = rec['cluster']
    return d

def ari_between(run_a, run_b):
    a = load_assignments(run_a)
    b = load_assignments(run_b)
    keys = sorted(set(a) & set(b))
    ya = [a[k] for k in keys]
    yb = [b[k] for k in keys]
    return adjusted_rand_score(ya, yb)

# Example
# print(ari_between('.../runA/cluster_assignments.jsonl', '.../runB/cluster_assignments.jsonl'))
```
- **Interpretation**: ARI closer to 1.0 indicates higher stability.

## Optional small code change (to expose DP strength)
To directly tune `weight_concentration_prior`, add reading and pass-through in `fit_bayes_gmm.py`:
- **Read from config**: `bgmm_cfg.get('weight_concentration_prior', None)`
- **Pass to model**: `BayesianGaussianMixture(..., weight_concentration_prior=that_value, ...)`

This allows explicit control over effective cluster count (smaller → fewer active components).

## Selection criteria
- **Primary**: maximize silhouette.
- **Secondary**: minimize DB; maximize CH.
- **Sanity**: avoid too many tiny clusters (check min/max cluster size; consider `min_cluster_size >= ~0.5% of N`).

## Next steps
- Start with E2/E3; if improvement ≥ 0.03 silhouette, lock PCA+cov and sweep n_components.
- If clusters still overlap, expose and tune `weight_concentration_prior`.
- Confirm stability with a second seed before adopting.
