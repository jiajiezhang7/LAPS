import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None


def _ensure_outdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _umap_scatter_unsup(
    X: np.ndarray,
    labels: np.ndarray,
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
        metric = vis_cfg.get('lstm_metric', 'cosine')
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

        plot_size = vis_cfg.get('plot', {}).get('marker_size', 10)
        _ensure_outdir(out_dir / 'dummy')

        plt.figure(figsize=vis_cfg.get('plot', {}).get('figsize', (7, 6)))
        if use_all and noise_mask is not None:
            emb_noise = emb[noise_mask]
            emb_assigned = emb[~noise_mask]
            noise_style = vis_cfg.get('noise', {})
            noise_color = noise_style.get('color', '#b0b0b0')
            noise_alpha = float(noise_style.get('alpha', 0.4))
            noise_size = int(noise_style.get('marker_size', max(1, int(0.8 * plot_size))))
            if emb_noise.shape[0] > 0:
                plt.scatter(emb_noise[:, 0], emb_noise[:, 1], c=noise_color, s=noise_size, alpha=noise_alpha, linewidths=0.0)
            sns.scatterplot(x=emb_assigned[:, 0], y=emb_assigned[:, 1], hue=[str(int(v)) for v in labels[~noise_mask]], palette='tab10', s=plot_size, linewidth=0.0)
        else:
            sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=[str(int(v)) for v in labels], palette='tab10', s=plot_size, linewidth=0.0)

        plt.title('UMAP (lstm) - Unsupervised clusters')
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        out_path = out_dir / 'umap_lstm_unsup.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=vis_cfg.get('plot', {}).get('dpi', 200))
        plt.close()

        res['umap'] = {
            'metric': metric,
            'neighbors': vis_cfg.get('n_neighbors', 15),
            'min_dist': vis_cfg.get('min_dist', 0.1),
            'embeddings_path': str(out_path),
        }
    except Exception as e:
        res['umap_error'] = str(e)
    return res


def load_embed_dir(embed_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load embeddings with labels from embed.npy, labels.npy, label_names.txt (labeled evaluation).
    
    If labels.npy is not available, extracts labels from paths.txt (top-level folder name).
    """
    X = np.load(embed_dir / 'embed.npy')
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise RuntimeError(f'embed.npy must be a 2D array, got shape={None if not isinstance(X, np.ndarray) else X.shape}')
    
    # Try to load labels.npy first
    labels_path = embed_dir / 'labels.npy'
    if labels_path.exists():
        y = np.load(labels_path)
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise RuntimeError(f'labels.npy must be a 1D array, got shape={None if not isinstance(y, np.ndarray) else y.shape}')
    else:
        # Fallback: extract labels from paths.txt (top-level folder name)
        paths_path = embed_dir / 'paths.txt'
        if paths_path.exists():
            paths = [line.strip() for line in paths_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            # Extract top-level folder name from each path
            label_strs = [str(p).split('/')[0] for p in paths]
            unique_labels = sorted(set(label_strs))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            y = np.asarray([label_to_idx[label_str] for label_str in label_strs], dtype=np.int64)
        else:
            raise RuntimeError(f'Neither labels.npy nor paths.txt found in {embed_dir}')
    
    label_names_path = embed_dir / 'label_names.txt'
    if label_names_path.exists():
        label_names = [line.strip() for line in label_names_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        label_names = [str(i) for i in range(int(np.max(y)) + 1)]
    
    return X, y, label_names


def load_embed_dir_unlabeled(embed_dir: Path) -> np.ndarray:
    """Load embeddings only from embed.npy (unlabeled evaluation)."""
    X = np.load(embed_dir / 'embed.npy')
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise RuntimeError(f'embed.npy must be a 2D array, got shape={None if not isinstance(X, np.ndarray) else X.shape}')
    return X


def _merge_labels(y_true: np.ndarray, label_names: List[str], merge_map: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
    """Merge label classes according to merge_map.
    
    Args:
        y_true: Label indices array
        label_names: List of label names
        merge_map: Dict mapping old label names to new label names
    
    Returns:
        Tuple of (merged_y_true, merged_label_names)
    """
    if not merge_map:
        return y_true, label_names
    
    # Build reverse mapping: old_label_idx -> new_label_name
    idx_to_old_name = {i: name for i, name in enumerate(label_names)}
    old_name_to_new_name = merge_map
    
    # Build new label space
    new_label_names = []
    old_idx_to_new_idx = {}
    
    for old_idx, old_name in idx_to_old_name.items():
        new_name = old_name_to_new_name.get(old_name, old_name)
        if new_name not in new_label_names:
            new_label_names.append(new_name)
        old_idx_to_new_idx[old_idx] = new_label_names.index(new_name)
    
    # Remap y_true
    y_merged = np.asarray([old_idx_to_new_idx[int(idx)] for idx in y_true], dtype=np.int64)
    
    return y_merged, new_label_names


def _k_grid_from_config(config: Dict[str, Any]) -> List[int]:
    kmin = int(config.get('kmin', 2))
    kmax = int(config.get('kmax', 10))
    if kmax < kmin:
        kmax = kmin
    return list(range(max(2, kmin), max(kmin, kmax) + 1))


def evaluate_kmeans_internal(
    X: np.ndarray,
    seed: int,
    k_grid: List[int],
    repeats: int = 1,
    select_by: str = 'silhouette',
    sil_sample_size: int = 0,
    metric: str = 'cosine',
) -> Dict[str, Any]:
    results = []
    for k in k_grid:
        if k < 2 or k > X.shape[0]:
            continue
        sil_list: List[float] = []
        ch_list: List[float] = []
        db_list: List[float] = []
        inertia_list: List[float] = []
        labels_list: List[np.ndarray] = []

        for r in range(max(1, int(repeats))):
            model = KMeans(n_clusters=int(k), n_init=20, random_state=seed + int(r))
            labels = model.fit_predict(X)
            inertia_list.append(float(model.inertia_))
            if len(set(labels.tolist())) >= 2:
                try:
                    sil = float(silhouette_score(
                        X,
                        labels,
                        metric=metric,
                        sample_size=None if sil_sample_size <= 0 else int(sil_sample_size),
                        random_state=seed + int(r),
                    ))
                    ch = float(calinski_harabasz_score(X, labels))
                    db = float(davies_bouldin_score(X, labels))
                except Exception:
                    sil, ch, db = float('nan'), float('nan'), float('nan')
            else:
                sil, ch, db = float('nan'), float('nan'), float('nan')
            sil_list.append(sil)
            ch_list.append(ch)
            db_list.append(db)
            labels_list.append(labels)

        # stability via ARI between repeats
        ari_vals: List[float] = []
        try:
            for i in range(len(labels_list)):
                for j in range(i + 1, len(labels_list)):
                    ari_vals.append(float(adjusted_rand_score(labels_list[i], labels_list[j])))
        except Exception:
            ari_vals = []

        def _mstats(arr: List[float]) -> Tuple[float, float]:
            x = np.asarray(arr, dtype=np.float64)
            return float(np.nanmean(x)), float(np.nanstd(x))

        sil_mu, sil_sd = _mstats(sil_list)
        ch_mu, ch_sd = _mstats(ch_list)
        db_mu, db_sd = _mstats(db_list)
        in_mu, in_sd = _mstats(inertia_list)
        ari_mu, ari_sd = _mstats(ari_vals) if ari_vals else (float('nan'), float('nan'))

        def _score_sign(name: str) -> int:
            return 1 if name in ('silhouette', 'calinski_harabasz') else -1

        sel_map = {
            'silhouette': sil_mu,
            'calinski_harabasz': ch_mu,
            'davies_bouldin': db_mu,
            'inertia': in_mu,
        }
        base = sel_map.get(select_by, sil_mu)
        score = float(base) if _score_sign(select_by) > 0 else float(-base)

        results.append({
            'k': int(k),
            'silhouette_mean': sil_mu,
            'silhouette_std': sil_sd,
            'calinski_harabasz_mean': ch_mu,
            'calinski_harabasz_std': ch_sd,
            'davies_bouldin_mean': db_mu,
            'davies_bouldin_std': db_sd,
            'inertia_mean': in_mu,
            'inertia_std': in_sd,
            'ari_mean': ari_mu,
            'ari_std': ari_sd,
            'selection_score': score,
        })

    best = None
    if results:
        best = sorted(results, key=lambda x: x['selection_score'], reverse=True)[0]

    candidates = []
    if results:
        candidates = sorted(results, key=lambda x: x['selection_score'], reverse=True)

    return {
        'grid': results,
        'best': best,
        'candidates': candidates,
    }


def evaluate_hdbscan(X: np.ndarray, seed: int, min_cluster_size: int, min_samples: Optional[int] = None) -> Dict[str, Any]:
    try:
        import hdbscan  # type: ignore
    except Exception as e:
        return {'error': f'hdbscan not available: {e}'}

    from sklearn.metrics import silhouette_score

    algo = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                           min_samples=None if min_samples is None else int(min_samples),
                           metric='euclidean',
                           core_dist_n_jobs=1,
                           prediction_data=True)
    labels = algo.fit_predict(X)
    # cluster labels: -1 = noise
    n_clusters = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
    noise_ratio = float((labels == -1).sum() / max(labels.size, 1))

    # non-noise silhouette
    mask = labels != -1
    sil = float('nan')
    if mask.sum() >= 2 and len(set(labels[mask].tolist())) >= 2:
        try:
            sil = float(silhouette_score(X[mask], labels[mask], metric='euclidean'))
        except Exception:
            sil = float('nan')

    # sizes per cluster
    sizes: Dict[str, int] = {}
    try:
        uniq, cnts = np.unique(labels, return_counts=True)
        for u, c in zip(uniq.tolist(), cnts.tolist()):
            sizes[str(int(u))] = int(c)
    except Exception:
        pass

    return {
        'n_clusters': int(n_clusters),
        'noise_ratio': float(noise_ratio),
        'silhouette_non_noise': sil,
        'cluster_sizes': sizes,
        'labels': labels.tolist(),
    }


def evaluate_hdbscan_grid(
    X: np.ndarray,
    seed: int,
    size_list: List[int],
    min_samples: Optional[int] = None,
) -> Dict[str, Any]:
    grid: List[Dict[str, Any]] = []
    best = None
    best_sil = -1.0
    for mcs in size_list:
        res = evaluate_hdbscan(X, seed=seed, min_cluster_size=int(mcs), min_samples=min_samples)
        res['min_cluster_size'] = int(mcs)
        grid.append(res)
        s = res.get('silhouette_non_noise', float('nan'))
        if s is not None and not math.isnan(s) and s > best_sil:
            best_sil = float(s)
            best = res
    return {'grid': grid, 'best': best}


def run_embed_eval(embed_dir: Path, config: Dict[str, Any], out_dir: Optional[Path] = None) -> Dict[str, Any]:
    X = load_embed_dir_unlabeled(embed_dir)
    seed = int(config.get('seed', 0))
    k_grid = _k_grid_from_config(config)
    kmeans_repeats = int(config.get('kmeans_repeats', 1))
    kmeans_select_by = str(config.get('kmeans_select_by', 'silhouette'))
    silhouette_sample_size = int(config.get('silhouette_sample_size', 0))

    # metric for silhouette: reuse umap.lstm_metric when available
    metric = ((config.get('visualization', {}) or {}).get('umap', {}) or {}).get('lstm_metric', 'cosine')

    # KMeans internal evaluation over grid
    km = evaluate_kmeans_internal(
        X,
        seed=seed,
        k_grid=k_grid,
        repeats=kmeans_repeats,
        select_by=kmeans_select_by,
        sil_sample_size=silhouette_sample_size,
        metric=metric,
    )

    # Determine best K and labels for UMAP overlay
    best_labels_for_plot: Optional[np.ndarray] = None
    kmeans_labels_path = None
    if km.get('best') and isinstance(km['best'], dict):
        k_best = km['best'].get('k', None)
        if isinstance(k_best, int) and k_best >= 2:
            model = KMeans(n_clusters=int(k_best), n_init=20, random_state=seed)
            best_labels_for_plot = model.fit_predict(X)
            if out_dir is not None:
                kmeans_labels_path = out_dir / 'kmeans_labels.npy'
                np.save(kmeans_labels_path, best_labels_for_plot)

    # HDBSCAN evaluation (single or grid if provided)
    cluster_cfg = config.get('clustering', {}) or {}
    hdb_cfg = cluster_cfg.get('hdbscan', {}) or {}
    min_cluster_size = int(hdb_cfg.get('min_cluster_size', max(10, int(0.01 * max(1, X.shape[0])))))
    min_samples = hdb_cfg.get('min_samples', None)
    if min_samples is not None:
        try:
            min_samples = int(min_samples)
        except Exception:
            min_samples = None

    grid_str = str(config.get('hdb_min_cluster_size_grid', '')).strip()
    hdbscan_result: Dict[str, Any] = {}
    if grid_str:
        try:
            size_list = [int(s) for s in grid_str.split(',') if s.strip()]
        except Exception:
            size_list = []
        if size_list:
            hdb_grid = evaluate_hdbscan_grid(X, seed=seed, size_list=size_list, min_samples=min_samples)
            hdbscan_result = {'best': (hdb_grid.get('best') or {}), 'grid': hdb_grid.get('grid')}
        else:
            hdbscan_result = evaluate_hdbscan(X, seed=seed, min_cluster_size=min_cluster_size, min_samples=min_samples)
    else:
        hdbscan_result = evaluate_hdbscan(X, seed=seed, min_cluster_size=min_cluster_size, min_samples=min_samples)

    # Save HDBSCAN labels if available
    noise_mask = None
    if isinstance(hdbscan_result, dict) and 'error' not in hdbscan_result:
        labels = None
        if 'best' in hdbscan_result and isinstance(hdbscan_result['best'], dict):
            labels = hdbscan_result['best'].get('labels', None)
        else:
            labels = hdbscan_result.get('labels', None)
        if labels is not None:
            labels_np = np.asarray(labels, dtype=int)
            if out_dir is not None:
                hdb_path = out_dir / 'hdbscan_labels.npy'
                np.save(hdb_path, labels_np)
                if 'best' in hdbscan_result and isinstance(hdbscan_result['best'], dict):
                    hdbscan_result['best']['labels_path'] = str(hdb_path)
                else:
                    hdbscan_result['labels_path'] = str(hdb_path)
            noise_mask = (labels_np == -1)
            # Prefer HDBSCAN labels for UMAP if available
            best_labels_for_plot = labels_np

    # UMAP plotting
    umap_res: Dict[str, Any] = {}
    if (config.get('visualization', {}) or {}).get('umap', {}).get('enabled', True) and (best_labels_for_plot is not None):
        od = out_dir if out_dir is not None else (Path(__file__).resolve().parent / 'analysis' / 'out' / 'umap' / 'lstm')
        umap_res = _umap_scatter_unsup(X, best_labels_for_plot, od, config, X_all=X, noise_mask=noise_mask)

    report: Dict[str, Any] = {
        'info': {
            'embed_dir': str(embed_dir),
            'num_samples': int(X.shape[0]),
            'embedding_dim': int(X.shape[1]),
        },
        'kmeans': km,
        'hdbscan': hdbscan_result,
        'umap': umap_res,
    }
    if kmeans_labels_path is not None:
        report['kmeans_labels_path'] = str(kmeans_labels_path)
    return report


def main():
    ap = argparse.ArgumentParser(description='Unsupervised evaluation on LSTM sequence embeddings (embed.npy only)')
    ap.add_argument('--embed-dir', type=str, required=True, help='Directory that contains embed.npy (and optionally paths.txt)')
    ap.add_argument('--config', type=str, default='action_classification/configs/eval_config.yaml', help='Evaluation config (seed, clustering.hdbscan, visualization.umap, optional kmin/kmax)')
    ap.add_argument('--out-dir', type=str, default=None, help='Base directory to save results')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    embed_dir = Path(args.embed_dir).resolve()
    out_dir_base = Path(args.out_dir).resolve() if args.out_dir else (Path(__file__).resolve().parent / 'analysis' / 'out')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = out_dir_base / f'{timestamp}_lstm_unsup'
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run_embed_eval(embed_dir, config=config, out_dir=out_dir)

    out_path = out_dir / 'aggregate_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'lstm_unsup': res}, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Unsupervised evaluation results: {out_path}")
    # brief summary
    km_best = (res.get('kmeans') or {}).get('best') or {}
    hdb = res.get('hdbscan') or {}
    if 'best' in hdb and isinstance(hdb['best'], dict):
        hdb = hdb['best']
    print(f"Best-K silhouette={km_best.get('silhouette_mean', float('nan'))} at K={km_best.get('k')}, HDBSCAN clusters={hdb.get('n_clusters')}, noise_ratio={hdb.get('noise_ratio')}")


if __name__ == '__main__':
    main()
