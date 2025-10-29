import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from typing import Optional

import numpy as np
import yaml

try:
    import hdbscan  # type: ignore
except Exception as e:  # pragma: no cover
    hdbscan = None

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Reuse loader from eval script
from action_classification.evaluation.embed_cluster_eval import load_embed_dir_unlabeled


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_umap_scatter(
    X: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    config: Dict[str, Any],
    X_all: Optional[np.ndarray] = None,
    noise_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import umap
        import matplotlib.pyplot as plt
        import seaborn as sns

        vis_cfg = config.get('visualization', {}).get('umap', {})
        metric = vis_cfg.get('lstm_metric', vis_cfg.get('avg_metric', 'cosine'))
        show_noise = bool(vis_cfg.get('show_noise', False))
        use_all = show_noise and (X_all is not None) and (noise_mask is not None) and (X_all.shape[0] == noise_mask.shape[0])

        reducer = umap.UMAP(
            n_neighbors=vis_cfg.get('n_neighbors', 15),
            min_dist=vis_cfg.get('min_dist', 0.1),
            n_components=2,
            metric=metric,
            random_state=config.get('seed', 0)
        )
        X_src = X_all if use_all else X
        emb = reducer.fit_transform(X_src)

        out_dir.mkdir(parents=True, exist_ok=True)
        plot_size = vis_cfg.get('plot', {}).get('marker_size', 10)

        # Color mapping for cluster IDs
        unique_clusters = sorted(np.unique(y_pred).tolist())
        palette = sns.color_palette('tab20', n_colors=max(1, len(unique_clusters)))
        color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}
        
        # Prepare indices
        if use_all:
            assigned_idx = np.where(~noise_mask)[0]
            emb_assigned = emb[assigned_idx]
            emb_noise = emb[noise_mask]
            y_pred_assigned = y_pred[~noise_mask]
            noise_style = vis_cfg.get('noise', {})
            noise_color = noise_style.get('color', '#b0b0b0')
            noise_alpha = float(noise_style.get('alpha', 0.4))
            noise_size = int(noise_style.get('marker_size', max(1, int(0.8 * plot_size))))
        else:
            emb_assigned = emb
            y_pred_assigned = y_pred

        # Cluster visualization plot
        colors = [color_map[int(c)] for c in y_pred_assigned]
        plt.figure(figsize=tuple(vis_cfg.get('plot', {}).get('figsize', (7, 6))))
        if use_all:
            # draw noise first (background)
            if emb_noise.shape[0] > 0:
                plt.scatter(emb_noise[:, 0], emb_noise[:, 1], c=noise_color, s=noise_size, alpha=noise_alpha, linewidths=0.0, label='Noise')
        plt.scatter(emb_assigned[:, 0], emb_assigned[:, 1], c=colors, s=plot_size, linewidths=0.0)
        
        # Build legend
        from matplotlib.lines import Line2D
        legend_elems = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {cid}', 
                              markerfacecolor=color_map[cid], markersize=6)
                       for cid in unique_clusters if cid != -1]
        if use_all and emb_noise.shape[0] > 0:
            legend_elems.append(Line2D([0], [0], marker='o', color='w', label='Noise', 
                                      markerfacecolor=noise_color, markersize=6, alpha=noise_alpha))
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


def run_umap_scatter_clusters(X: np.ndarray, y_clusters: np.ndarray, out_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """UMAP visualization: color by raw cluster IDs.
    
    Note:
    - Uses only cluster IDs (including -1 for noise if present).
    - Uses same UMAP parameters as run_umap_scatter for consistency.
    """
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

        # Color mapping by cluster ID (including -1 for noise if present)
        unique_clusters = sorted(np.unique(y_clusters).tolist())
        palette = sns.color_palette('tab20', n_colors=max(1, len(unique_clusters)))
        color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}
        colors = [color_map[int(c)] for c in y_clusters]

        plt.figure(figsize=tuple(vis_cfg.get('plot', {}).get('figsize', (7, 6))))
        plt.scatter(emb[:, 0], emb[:, 1], c=colors, s=vis_cfg.get('plot', {}).get('marker_size', 10), linewidths=0.0)
        # Build legend
        from matplotlib.lines import Line2D
        legend_elems = [Line2D([0], [0], marker='o', color='w', 
                              label='Noise' if cid == -1 else f'Cluster {cid}', 
                              markerfacecolor=color_map[cid], markersize=6)
                        for cid in unique_clusters]
        plt.legend(handles=legend_elems, title='Cluster ID', markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('UMAP - All Clusters (Raw IDs)')
        pred_clusters_path = out_dir / 'umap_all_clusters_lstm.png'
        plt.tight_layout()
        plt.savefig(pred_clusters_path, dpi=vis_cfg.get('plot', {}).get('dpi', 200))
        plt.close()

        res['umap_all_clusters'] = {
            'metric': metric,
            'neighbors': vis_cfg.get('n_neighbors', 15),
            'min_dist': vis_cfg.get('min_dist', 0.1),
            'embeddings_path': str(pred_clusters_path),
            'unique_cluster_ids': unique_clusters,
        }
    except Exception as e:
        res['umap_all_clusters_error'] = str(e)
    return res


def main():
    if hdbscan is None:
        raise ImportError("hdbscan is not installed. Please install hdbscan>=0.8.33.")

    ap = argparse.ArgumentParser(description='Fit HDBSCAN over LSTM embeddings (fully unsupervised) and save model for online inference')
    ap.add_argument('--embed-dir', type=str, required=True, help='Directory that contains embed.npy (unlabeled)')
    ap.add_argument('--config', type=str, default='action_classification/configs/hdbscan_config.yaml', help='Config with clustering/visualization keys (hdbscan_config.yaml or eval_config.yaml)')
    ap.add_argument('--out-dir', type=str, default=None, help='Base directory to save model and reports')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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
            pca_full = PCA(n_components=min(X_proc.shape[1], max_dims), random_state=int(config.get('seed', 0)))
            X_p = pca_full.fit_transform(X_proc)
            # choose dims by cumulative explained variance
            csum = np.cumsum(pca_full.explained_variance_ratio_)
            k = int(np.searchsorted(csum, ev_thr) + 1)
            k = max(min_dims, min(k, max_dims))
            pca_obj = PCA(n_components=k, random_state=int(config.get('seed', 0)))
            X_proc = pca_obj.fit_transform(X_proc)
            preproc['pca_dims'] = int(k)

    # HDBSCAN fit
    hdb_cfg = config.get('clustering', {}).get('hdbscan', {})
    min_cluster_size = int(hdb_cfg.get('min_cluster_size', max(10, int(0.01 * max(1, X_proc.shape[0])))))
    min_samples = hdb_cfg.get('min_samples', None)
    if min_samples is not None:
        min_samples = int(min_samples)
    metric = hdb_cfg.get('metric', 'euclidean')
    cluster_selection_epsilon = float(hdb_cfg.get('cluster_selection_epsilon', 0.0))
    cluster_selection_method = hdb_cfg.get('cluster_selection_method', 'eom')

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )
    model.fit(X_proc)

    # Build output dir
    out_base = Path(args.out_dir).resolve() if args.out_dir else (Path(__file__).resolve().parent / 'out' / 'hdbscan_fits')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = out_base / timestamp
    ensure_dir(out_dir)

    # Save artifacts
    with open(out_dir / 'model_hdbscan.pkl', 'wb') as f:
        pickle.dump(model, f)
    if pca_obj is not None:
        with open(out_dir / 'preprocessor_pca.pkl', 'wb') as f:
            pickle.dump(pca_obj, f)
    with open(out_dir / 'config_used.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Save meta for online inference
    meta = {
        'method': 'hdbscan',
        'preprocess': preproc,
        'hdbscan': {
            'min_cluster_size': int(min_cluster_size),
            'min_samples': None if min_samples is None else int(min_samples),
            'metric': metric,
            'cluster_selection_epsilon': float(cluster_selection_epsilon),
            'cluster_selection_method': cluster_selection_method,
        },
    }
    with open(out_dir / 'cluster_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Unsupervised evaluation
    y_pred = model.labels_.astype(int)
    mask = (y_pred != -1)
    pred_ids = np.unique(y_pred[mask]) if np.any(mask) else np.array([], dtype=int)
    n_clusters = len(pred_ids)
    noise_rate = float(1.0 - (float(mask.sum()) / max(1, X.shape[0])))
    
    # Compute cluster sizes
    cluster_sizes = {}
    for cid in np.unique(y_pred):
        cluster_sizes[str(int(cid))] = int((y_pred == cid).sum())
    
    results: Dict[str, Any] = {
        'method': 'hdbscan',
        'num_samples': int(X.shape[0]),
        'noise_rate': noise_rate,
        'num_pred_clusters': int(n_clusters),
        'cluster_sizes': cluster_sizes,
    }

    # Compute unsupervised metrics on non-noise points (only if we have multiple clusters)
    metrics: Dict[str, Any] = {}
    if n_clusters >= 2 and mask.sum() >= 2:
        try:
            sil = float(silhouette_score(X_proc[mask], y_pred[mask], metric='euclidean'))
            metrics['silhouette'] = sil
        except Exception as e:
            metrics['silhouette_error'] = str(e)
        
        try:
            ch = float(calinski_harabasz_score(X_proc[mask], y_pred[mask]))
            metrics['calinski_harabasz'] = ch
        except Exception as e:
            metrics['calinski_harabasz_error'] = str(e)
        
        try:
            db = float(davies_bouldin_score(X_proc[mask], y_pred[mask]))
            metrics['davies_bouldin'] = db
        except Exception as e:
            metrics['davies_bouldin_error'] = str(e)
    
    results['metrics'] = metrics

    # Optional UMAP visualization
    if config.get('visualization', {}).get('umap', {}).get('enabled', True):
        vis_dir = out_dir / 'umap'
        vis_cfg = config.get('visualization', {}).get('umap', {})
        show_noise = bool(vis_cfg.get('show_noise', False))
        if show_noise:
            umap_res = run_umap_scatter(
                X_proc,
                y_pred,
                vis_dir,
                config,
                X_all=X_proc,
                noise_mask=(y_pred == -1)
            )
            results.update(umap_res)
            # Also create cluster-only visualization
            umap_clusters_res = run_umap_scatter_clusters(X_proc, y_pred, vis_dir, config)
            results.update(umap_clusters_res)
        else:
            umap_res = run_umap_scatter(X_proc[mask], y_pred[mask], vis_dir, config)
            results.update(umap_res)
            # Also create cluster-only visualization
            umap_clusters_res = run_umap_scatter_clusters(X_proc[mask], y_pred[mask], vis_dir, config)
            results.update(umap_clusters_res)

    # Save aggregate results
    with open(out_dir / 'aggregate_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save per-sample assignments with membership probability
    # For in-sample, we can compute soft memberships
    try:
        memb = hdbscan.all_points_membership_vectors(model)  # shape NxK
        # For noise points, rows will be all zeros
        max_prob = memb.max(axis=1)
        pred_prob = np.where(y_pred == -1, 0.0, max_prob)
    except Exception:
        pred_prob = np.where(y_pred == -1, 0.0, 1.0)

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
                'prob': float(pred_prob[i]),
                'is_noise': bool(y_pred[i] == -1),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"[Saved] HDBSCAN model: {out_dir / 'model_hdbscan.pkl'}")
    print(f"[Saved] Meta: {out_dir / 'cluster_meta.json'}")
    print(f"[Saved] Aggregate results: {out_dir / 'aggregate_results.json'}")
    print(f"[Saved] Cluster assignments: {assign_path}")
    print(f"\n[Summary] Found {n_clusters} clusters from {X.shape[0]} samples (noise rate: {noise_rate:.2%})")
    if 'silhouette' in metrics:
        print(f"  Silhouette score: {metrics['silhouette']:.4f}")


if __name__ == '__main__':
    main()
