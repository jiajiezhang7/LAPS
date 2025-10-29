import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Reuse loader from eval script
from action_classification.evaluation.embed_cluster_eval import load_embed_dir_unlabeled


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_umap_scatter(X: np.ndarray, y_pred: np.ndarray, out_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import umap
        import matplotlib.pyplot as plt
        import seaborn as sns

        vis_cfg = config.get('visualization', {}).get('umap', {})
        metric = vis_cfg.get('lstm_metric', vis_cfg.get('avg_metric', 'cosine'))
        reducer = umap.UMAP(
            n_neighbors=vis_cfg.get('n_neighbors', 15),
            min_dist=vis_cfg.get('min_dist', 0.1),
            n_components=2,
            metric=metric,
            random_state=config.get('seed', 0)
        )
        emb = reducer.fit_transform(X)

        out_dir.mkdir(parents=True, exist_ok=True)

        # Color mapping for cluster IDs
        unique_clusters = sorted(np.unique(y_pred).tolist())
        palette = sns.color_palette('tab20', n_colors=max(1, len(unique_clusters)))
        color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}
        colors = [color_map[int(c)] for c in y_pred]

        # Cluster visualization plot
        plt.figure(figsize=tuple(vis_cfg.get('plot', {}).get('figsize', (7, 6))))
        plt.scatter(emb[:, 0], emb[:, 1], c=colors, s=vis_cfg.get('plot', {}).get('marker_size', 10), linewidths=0.0)
        
        # Build legend
        from matplotlib.lines import Line2D
        legend_elems = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {cid}', 
                              markerfacecolor=color_map[cid], markersize=6)
                       for cid in unique_clusters]
        plt.legend(handles=legend_elems, title='Cluster ID', markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('UMAP - Predicted Clusters (Unsupervised)')
        pred_path = out_dir / 'umap_clusters_lstm.png'
        plt.tight_layout()
        plt.savefig(pred_path, dpi=vis_cfg.get('plot', {}).get('dpi', 200))
        plt.close()

        res['umap'] = {
            'metric': metric,
            'neighbors': vis_cfg.get('n_neighbors', 15),
            'min_dist': vis_cfg.get('min_dist', 0.1),
            'embeddings_path': str(pred_path),
            'unique_cluster_ids': unique_clusters,
        }
    except Exception as e:
        res['umap_error'] = str(e)
    return res


def main():
    ap = argparse.ArgumentParser(description='Fit BayesianGaussianMixture over LSTM embeddings (fully unsupervised) and save model for online inference')
    ap.add_argument('--embed-dir', type=str, required=True, help='Directory that contains embed.npy (unlabeled)')
    ap.add_argument('--config', type=str, default='amplify_motion_tokenizer/action_classification/configs/eval_config.yaml', help='Config with clustering/visualization keys')
    ap.add_argument('--out-dir', type=str, default=None, help='Base directory to save model and reports')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = int(config.get('seed', 0))
    X = load_embed_dir_unlabeled(Path(args.embed_dir).resolve())
    print(f"[Info] Loaded {X.shape[0]} samples with embedding dimension {X.shape[1]}")

    # Optional preprocessing for lstm features
    pre_cfg = config.get('preprocessing', {}).get('lstm_feature', {})
    l2_norm = bool(pre_cfg.get('l2_normalize', True))
    pca_cfg = pre_cfg.get('pca', None)

    X_proc = X.copy()
    preproc: Dict[str, Any] = {}
    if l2_norm:
        X_proc = normalize(X_proc, norm='l2')
        preproc['l2_normalize'] = True

    pca_obj = None
    if isinstance(pca_cfg, dict):
        max_dims = int(pca_cfg.get('max_dims', 0))
        if max_dims and max_dims > 0:
            ev_thr = float(pca_cfg.get('explained_variance_threshold', 0.99))
            min_dims = int(pca_cfg.get('min_dims', 8))
            from sklearn.decomposition import PCA as SKPCA
            pca_full = SKPCA(n_components=min(X_proc.shape[1], max_dims), random_state=seed)
            X_p = pca_full.fit_transform(X_proc)
            csum = np.cumsum(pca_full.explained_variance_ratio_)
            k = int(np.searchsorted(csum, ev_thr) + 1)
            k = max(min_dims, min(k, max_dims))
            pca_obj = SKPCA(n_components=k, random_state=seed)
            X_proc = pca_obj.fit_transform(X_proc)
            preproc['pca_dims'] = int(k)

    # Bayes GMM fit
    bgmm_cfg = config.get('clustering', {}).get('bayes_gmm', {})
    n_components = int(bgmm_cfg.get('n_components', 12))
    covariance_type = bgmm_cfg.get('covariance_type', 'full')
    weight_concentration_prior_type = bgmm_cfg.get('weight_concentration_prior_type', 'dirichlet_process')
    init_params = bgmm_cfg.get('init_params', 'kmeans')
    max_iter = int(bgmm_cfg.get('max_iter', 1000))

    model = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type=weight_concentration_prior_type,
        init_params=init_params,
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(X_proc)

    # Build output dir
    out_base = Path(args.out_dir).resolve() if args.out_dir else (Path(__file__).resolve().parent / 'out' / 'bayes_gmm_fits')
    # Use embed_dir folder name as subdirectory name
    embed_dir_name = Path(args.embed_dir).resolve().name
    out_dir = out_base / embed_dir_name
    ensure_dir(out_dir)

    # Save artifacts
    with open(out_dir / 'model_bayes_gmm.pkl', 'wb') as f:
        pickle.dump(model, f)
    if pca_obj is not None:
        with open(out_dir / 'preprocessor_pca.pkl', 'wb') as f:
            pickle.dump(pca_obj, f)
    with open(out_dir / 'config_used.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    meta = {
        'method': 'bayes_gmm',
        'preprocess': preproc,
        'bayes_gmm': {
            'n_components': int(n_components),
            'covariance_type': covariance_type,
            'weight_concentration_prior_type': weight_concentration_prior_type,
            'init_params': init_params,
            'max_iter': int(max_iter),
        },
    }
    with open(out_dir / 'cluster_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Unsupervised evaluation
    y_pred = model.predict(X_proc)
    pred_ids = np.unique(y_pred)
    n_clusters = len(pred_ids)
    
    # Compute cluster sizes
    cluster_sizes = {}
    for cid in pred_ids:
        cluster_sizes[str(int(cid))] = int((y_pred == cid).sum())
    
    # Compute unsupervised metrics (only if we have multiple clusters)
    metrics: Dict[str, Any] = {}
    if n_clusters >= 2:
        try:
            sil = float(silhouette_score(X_proc, y_pred, metric='euclidean'))
            metrics['silhouette'] = sil
        except Exception as e:
            metrics['silhouette_error'] = str(e)
        
        try:
            ch = float(calinski_harabasz_score(X_proc, y_pred))
            metrics['calinski_harabasz'] = ch
        except Exception as e:
            metrics['calinski_harabasz_error'] = str(e)
        
        try:
            db = float(davies_bouldin_score(X_proc, y_pred))
            metrics['davies_bouldin'] = db
        except Exception as e:
            metrics['davies_bouldin_error'] = str(e)
    
    results: Dict[str, Any] = {
        'method': 'bayes_gmm',
        'num_samples': int(X.shape[0]),
        'num_pred_clusters': int(n_clusters),
        'cluster_sizes': cluster_sizes,
        'metrics': metrics,
    }

    # Optional UMAP visualization
    if config.get('visualization', {}).get('umap', {}).get('enabled', True):
        vis_dir = out_dir / 'umap'
        umap_res = run_umap_scatter(X_proc, y_pred, vis_dir, config)
        results.update(umap_res)

    # Save per-sample cluster assignments
    # Load paths for traceability if available
    paths_txt = Path(args.embed_dir).resolve() / 'paths.txt'
    paths: List[str]
    if paths_txt.exists():
        paths = [line.strip() for line in paths_txt.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        paths = [str(i) for i in range(X.shape[0])]
    
    assign_path = out_dir / 'cluster_assignments.jsonl'
    with open(assign_path, 'w', encoding='utf-8') as f:
        for i in range(X.shape[0]):
            rec = {
                'index': int(i),
                'path': paths[i] if i < len(paths) else str(i),
                'cluster': int(y_pred[i]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    with open(out_dir / 'aggregate_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[Saved] BayesGMM model: {out_dir / 'model_bayes_gmm.pkl'}")
    print(f"[Saved] Meta: {out_dir / 'cluster_meta.json'}")
    print(f"[Saved] Aggregate results: {out_dir / 'aggregate_results.json'}")
    print(f"[Saved] Cluster assignments: {assign_path}")
    print(f"\n[Summary] Found {n_clusters} clusters from {X.shape[0]} samples")
    if 'silhouette' in metrics:
        print(f"  Silhouette score: {metrics['silhouette']:.4f}")


if __name__ == '__main__':
    main()
