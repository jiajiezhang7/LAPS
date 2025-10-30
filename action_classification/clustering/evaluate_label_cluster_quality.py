import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score, calinski_harabasz_score,
                             classification_report, davies_bouldin_score,
                             homogeneity_completeness_v_measure,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from action_classification.embedding.common import ensure_dir

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - optional dependency
    linear_sum_assignment = None


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_labeled_embeddings(embed_root: Path, embed_subdir: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str], Dict[str, int]]:
    label_dirs = [p for p in sorted(embed_root.iterdir()) if p.is_dir()]
    entries: List[Tuple[str, np.ndarray, List[str]]] = []
    label_counts: Dict[str, int] = {}
    for label_dir in label_dirs:
        embed_dir = label_dir / embed_subdir if embed_subdir else label_dir
        embed_path = embed_dir / 'embed.npy'
        if not embed_path.exists():
            print(f"[Warn] Missing embed.npy under {embed_dir}, skipping label '{label_dir.name}'.")
            continue
        X_label = np.load(embed_path)
        if not isinstance(X_label, np.ndarray) or X_label.ndim != 2:
            raise RuntimeError(f"embed.npy for label '{label_dir.name}' must be 2D array, got shape={None if not isinstance(X_label, np.ndarray) else X_label.shape}")
        n_samples = X_label.shape[0]
        paths_file = embed_dir / 'paths.txt'
        if paths_file.exists():
            paths = [line.strip() for line in paths_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        else:
            paths = []
        if len(paths) < n_samples:
            start = len(paths)
            paths.extend([f"sample_{i:05d}.json" for i in range(start, n_samples)])
        else:
            paths = paths[:n_samples]
        entries.append((label_dir.name, X_label, paths))
        label_counts[label_dir.name] = int(n_samples)

    if not entries:
        raise RuntimeError(f'No embeddings found under {embed_root} (looked into subdir "{embed_subdir}").')

    label_names: List[str] = []
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    sample_paths: List[str] = []
    sample_label_names: List[str] = []

    for idx, (label_name, X_label, paths) in enumerate(entries):
        label_names.append(label_name)
        X_list.append(X_label)
        y_list.append(np.full((X_label.shape[0],), idx, dtype=np.int64))
        sample_paths.extend([f"{label_name}/{pth}" for pth in paths])
        sample_label_names.extend([label_name] * X_label.shape[0])

    X = np.concatenate(X_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)
    return X, y_true, label_names, sample_paths, sample_label_names, label_counts


def preprocess_embeddings(X: np.ndarray, config: Dict[str, Any], seed: int) -> Tuple[np.ndarray, Dict[str, Any], Optional[PCA]]:
    pre_cfg = config.get('preprocessing', {}).get('lstm_feature', {})
    X_proc = X.copy()
    preproc_info: Dict[str, Any] = {}

    if pre_cfg.get('l2_normalize', True):
        X_proc = normalize(X_proc, norm='l2')
        preproc_info['l2_normalize'] = True

    pca_cfg = pre_cfg.get('pca')
    pca_obj: Optional[PCA] = None
    if isinstance(pca_cfg, dict):
        max_dims = int(pca_cfg.get('max_dims', 0))
        if max_dims > 0:
            ev_thr = float(pca_cfg.get('explained_variance_threshold', 0.99))
            min_dims = int(pca_cfg.get('min_dims', 8))
            pca_full = PCA(n_components=min(X_proc.shape[1], max_dims), random_state=seed)
            X_full = pca_full.fit_transform(X_proc)
            csum = np.cumsum(pca_full.explained_variance_ratio_)
            k = int(np.searchsorted(csum, ev_thr) + 1)
            k = max(min_dims, min(k, max_dims))
            pca_obj = PCA(n_components=k, random_state=seed)
            X_proc = pca_obj.fit_transform(X_proc)
            preproc_info['pca_dims'] = int(k)
    return X_proc, preproc_info, pca_obj


def derive_cluster_label_mapping(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Tuple[Dict[int, int], float, np.ndarray, np.ndarray]:
    true_ids = np.unique(y_true)
    cluster_ids = np.unique(y_pred)
    contingency = contingency_matrix(y_true, y_pred).astype(np.int64)

    if contingency.size == 0:
        raise RuntimeError('Empty contingency matrix while computing cluster-label mapping.')

    if linear_sum_assignment is not None:
        cost = contingency.max() - contingency
        row_ind, col_ind = linear_sum_assignment(cost)
    else:  # fallback to greedy majority mapping
        row_ind = []
        col_ind = []
        used_rows: set[int] = set()
        for j, cid in enumerate(cluster_ids):
            row = int(np.argmax(contingency[:, j]))
            if row in used_rows:
                continue
            used_rows.add(row)
            row_ind.append(row)
            col_ind.append(j)
        row_ind = np.asarray(row_ind, dtype=int)
        col_ind = np.asarray(col_ind, dtype=int)

    mapping: Dict[int, int] = {}
    matched = 0
    for r, c in zip(row_ind, col_ind):
        cluster_label = int(cluster_ids[c])
        true_label = int(true_ids[r])
        mapping[cluster_label] = true_label
        matched += contingency[r, c]

    total = contingency.sum()
    accuracy = float(matched / total) if total else 0.0

    y_pred_mapped = np.asarray([mapping.get(int(cid), 0) for cid in y_pred], dtype=np.int64)
    return mapping, accuracy, true_ids, cluster_ids


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_mapped: np.ndarray, mapping: Dict[int, int],
                    label_names: List[str]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics['adjusted_rand'] = float(adjusted_rand_score(y_true, y_pred))
    metrics['adjusted_mutual_info'] = float(adjusted_mutual_info_score(y_true, y_pred))
    metrics['normalized_mutual_info'] = float(normalized_mutual_info_score(y_true, y_pred))
    hom, comp, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)
    metrics['homogeneity'] = float(hom)
    metrics['completeness'] = float(comp)
    metrics['v_measure'] = float(v_measure)
    metrics['accuracy_via_mapping'] = float(np.mean(y_true == y_pred_mapped))
    metrics['balanced_accuracy_via_mapping'] = float(balanced_accuracy_score(y_true, y_pred_mapped))
    metrics['classification_report'] = classification_report(
        y_true, y_pred_mapped, target_names=label_names, output_dict=True
    )
    metrics['cluster_to_label_mapping'] = {
        str(cluster): label_names[label_idx] for cluster, label_idx in mapping.items()
    }
    return metrics


def compute_unsupervised_metrics(X_proc: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    unique_clusters = np.unique(y_pred)
    if unique_clusters.shape[0] >= 2 and X_proc.shape[0] > unique_clusters.shape[0]:
        try:
            metrics['silhouette'] = float(silhouette_score(X_proc, y_pred, metric='euclidean'))
        except Exception as exc:  # pragma: no cover - defensive
            metrics['silhouette_error'] = str(exc)
        try:
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(X_proc, y_pred))
        except Exception as exc:
            metrics['calinski_harabasz_error'] = str(exc)
        try:
            metrics['davies_bouldin'] = float(davies_bouldin_score(X_proc, y_pred))
        except Exception as exc:
            metrics['davies_bouldin_error'] = str(exc)
    else:
        metrics['note'] = 'Not enough distinct clusters to compute silhouette/CH/DB metrics.'
    return metrics


def run_umap_visualization(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str],
                           mapping: Dict[int, int], out_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import umap

        vis_cfg = config.get('visualization', {}).get('umap', {})
        metric = vis_cfg.get('lstm_metric', vis_cfg.get('avg_metric', 'cosine'))
        reducer = umap.UMAP(
            n_neighbors=vis_cfg.get('n_neighbors', 15),
            min_dist=vis_cfg.get('min_dist', 0.1),
            n_components=2,
            metric=metric,
            random_state=config.get('seed', 0),
        )
        emb = reducer.fit_transform(X)

        ensure_dir(out_dir)

        # Palette for true labels
        palette_true = sns.color_palette('tab10', n_colors=max(1, len(label_names)))
        true_colors_map = {idx: palette_true[idx % len(palette_true)] for idx in range(len(label_names))}
        true_colors = [true_colors_map[int(lbl)] for lbl in y_true]

        cluster_ids = sorted({int(c) for c in y_pred})
        palette_cluster = sns.color_palette('tab20', n_colors=max(1, len(cluster_ids)))
        cluster_color_map = {cid: palette_cluster[i % len(palette_cluster)] for i, cid in enumerate(cluster_ids)}
        cluster_colors = [cluster_color_map[int(cid)] for cid in y_pred]

        mapped_labels = {cid: label_names[label_idx] for cid, label_idx in mapping.items()}

        figsize = tuple(vis_cfg.get('plot', {}).get('figsize', (7, 6)))
        marker_size = vis_cfg.get('plot', {}).get('marker_size', 14)
        dpi = vis_cfg.get('plot', {}).get('dpi', 200)

        # True labels figure
        fig_true, ax_true = plt.subplots(figsize=figsize)
        ax_true.scatter(emb[:, 0], emb[:, 1], c=true_colors, s=marker_size, linewidths=0)
        ax_true.set_title('UMAP - True Labels')
        legend_elems_true = [
            matplotlib.lines.Line2D([0], [0], marker='o', color='w', label=label_names[i],
                                    markerfacecolor=true_colors_map[i], markersize=6)
            for i in range(len(label_names))
        ]
        ax_true.legend(handles=legend_elems_true, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax_true.set_xticks([])
        ax_true.set_yticks([])
        fig_true.tight_layout()
        fig_true_path = out_dir / 'umap_true_labels.png'
        fig_true.savefig(fig_true_path, dpi=dpi)
        plt.close(fig_true)

        # Predicted clusters figure
        fig_cluster, ax_cluster = plt.subplots(figsize=figsize)
        ax_cluster.scatter(emb[:, 0], emb[:, 1], c=cluster_colors, s=marker_size, linewidths=0)
        ax_cluster.set_title('UMAP - Predicted Clusters')
        legend_elems_cluster = []
        for cid in cluster_ids:
            label_str = mapped_labels.get(cid, f'Cluster {cid}')
            legend_elems_cluster.append(
                matplotlib.lines.Line2D([0], [0], marker='o', color='w',
                                        label=f'Cluster {cid} → {label_str}',
                                        markerfacecolor=cluster_color_map[cid], markersize=6)
            )
        ax_cluster.legend(handles=legend_elems_cluster, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax_cluster.set_xticks([])
        ax_cluster.set_yticks([])
        fig_cluster.tight_layout()
        fig_cluster_path = out_dir / 'umap_predicted_clusters.png'
        fig_cluster.savefig(fig_cluster_path, dpi=dpi)
        plt.close(fig_cluster)

        res['umap_true'] = {
            'metric': metric,
            'neighbors': vis_cfg.get('n_neighbors', 15),
            'min_dist': vis_cfg.get('min_dist', 0.1),
            'figure_path': str(fig_true_path),
        }
        res['umap_clusters'] = {
            'metric': metric,
            'neighbors': vis_cfg.get('n_neighbors', 15),
            'min_dist': vis_cfg.get('min_dist', 0.1),
            'figure_path': str(fig_cluster_path),
        }

        # Confusion matrix heatmap
        contingency = contingency_matrix(y_true, y_pred)
        plt.figure(figsize=(max(6, len(label_names) * 1.8), max(4, len(cluster_ids) * 1.2)))
        sns.heatmap(
            contingency,
            annot=True,
            fmt='g',
            cmap='Blues',
            xticklabels=[f'Cluster {cid}' for cid in cluster_ids],
            yticklabels=label_names,
        )
        plt.xlabel('Predicted Cluster')
        plt.ylabel('True Label')
        plt.title('True vs Predicted Cluster Counts')
        heatmap_path = out_dir / 'confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=vis_cfg.get('plot', {}).get('dpi', 200))
        plt.close()

        res['confusion_matrix_figure'] = str(heatmap_path)
    except Exception as exc:  # pragma: no cover - optional visualization stack
        res['umap_error'] = str(exc)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate clustering quality over labeled embeddings using Gaussian Mixture Model.')
    ap.add_argument('--embed-root', type=str, required=True, help='Root directory whose sub-folders (labels) each contain embeddings/')
    ap.add_argument('--embed-subdir', type=str, default='embeddings', help='Sub-folder under each label that stores embed.npy/paths.txt')
    ap.add_argument('--config', type=str, default=None, help='Optional YAML config for preprocessing/visualization')
    ap.add_argument('--n-components', type=int, default=None, help='Number of clusters (defaults to number of label folders)')
    ap.add_argument('--covariance-type', type=str, default='full', choices=['full', 'tied', 'diag', 'spherical'], help='Covariance type for GaussianMixture')
    ap.add_argument('--max-iter', type=int, default=500, help='Maximum EM iterations for GaussianMixture')
    ap.add_argument('--init-params', type=str, default='kmeans', choices=['kmeans', 'random'], help='Initialization method for GaussianMixture')
    ap.add_argument('--seed', type=int, default=0, help='Random seed')
    ap.add_argument('--out-dir', type=str, default=None, help='Directory to store evaluation artifacts')
    args = ap.parse_args()

    embed_root = Path(args.embed_root).resolve()
    if not embed_root.is_dir():
        raise FileNotFoundError(f'embed_root not found: {embed_root}')

    config = load_config(args.config)
    seed = int(config.get('seed', args.seed))
    np.random.seed(seed)

    X, y_true, label_names, sample_paths, sample_label_names, label_counts = load_labeled_embeddings(embed_root, args.embed_subdir)
    print(f"[Info] Loaded {X.shape[0]} embeddings across {len(label_names)} labels from {embed_root}")

    X_proc, preproc_info, pca_obj = preprocess_embeddings(X, config, seed)

    n_components = int(args.n_components) if args.n_components else len(label_names)
    print(f"[Info] Fitting GaussianMixture with n_components={n_components}")

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=args.covariance_type,
        init_params=args.init_params,
        max_iter=int(args.max_iter),
        random_state=seed,
    )
    model.fit(X_proc)
    y_pred = model.predict(X_proc)
    prob = model.predict_proba(X_proc)

    mapping, acc_mapping, true_ids, cluster_ids = derive_cluster_label_mapping(y_true, y_pred, label_names)
    y_pred_mapped = np.asarray([mapping.get(int(cid), 0) for cid in y_pred], dtype=np.int64)

    supervised_metrics = compute_metrics(y_true, y_pred, y_pred_mapped, mapping, label_names)
    supervised_metrics['assignment_accuracy'] = acc_mapping

    unsupervised_metrics = compute_unsupervised_metrics(X_proc, y_pred)

    cluster_sizes = {str(int(cid)): int(np.sum(y_pred == int(cid))) for cid in cluster_ids}
    label_distribution = {
        label_names[int(true_id)]: {
            str(int(cluster_ids[col_idx])): int(contingency_matrix(y_true, y_pred)[row_idx, col_idx])
            for col_idx in range(len(cluster_ids))
        }
        for row_idx, true_id in enumerate(true_ids)
    }

    out_base = Path(args.out_dir).resolve() if args.out_dir else (Path(__file__).resolve().parent / 'out' / 'label_cluster_eval')
    out_dir = out_base / embed_root.name
    ensure_dir(out_dir)

    with open(out_dir / 'model_gmm.pkl', 'wb') as f:
        pickle.dump(model, f)
    if pca_obj is not None:
        with open(out_dir / 'preprocessor_pca.pkl', 'wb') as f:
            pickle.dump(pca_obj, f)

    assignments_path = out_dir / 'cluster_assignments.jsonl'
    with assignments_path.open('w', encoding='utf-8') as f:
        for idx, (path, lbl_name) in enumerate(zip(sample_paths, sample_label_names)):
            rec = {
                'index': int(idx),
                'path': path,
                'true_label_name': lbl_name,
                'true_label_id': int(y_true[idx]),
                'cluster_id': int(y_pred[idx]),
                'mapped_label_name': supervised_metrics['cluster_to_label_mapping'].get(str(int(y_pred[idx])), lbl_name),
                'mapped_label_id': int(y_pred_mapped[idx]),
                'cluster_confidence': float(prob[idx].max()),
                'cluster_probabilities': [float(v) for v in prob[idx]],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    metrics_payload = {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'seed': seed,
        'n_components': n_components,
        'label_counts': label_counts,
        'cluster_sizes': cluster_sizes,
        'preprocessing': preproc_info,
        'supervised_metrics': supervised_metrics,
        'unsupervised_metrics': unsupervised_metrics,
        'cluster_to_label_mapping': supervised_metrics['cluster_to_label_mapping'],
        'label_cluster_distribution': label_distribution,
        'config_path': str(Path(args.config).resolve()) if args.config else None,
        'arguments': vars(args),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    vis_res = run_umap_visualization(X_proc, y_true, y_pred, label_names, mapping, out_dir, config)
    if vis_res:
        metrics_payload.update(vis_res)
        (out_dir / 'metrics.json').write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    run_meta = {
        'embed_root': str(embed_root),
        'embed_subdir': args.embed_subdir,
        'out_dir': str(out_dir),
        'generated_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    }
    (out_dir / 'run_meta.json').write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"[Saved] Model: {out_dir / 'model_gmm.pkl'}")
    print(f"[Saved] Metrics: {out_dir / 'metrics.json'}")
    print(f"[Saved] Assignments: {assignments_path}")
    if 'umap' in vis_res:
        print(f"[Saved] UMAP figure: {vis_res['umap']['figure_path']}")
    if 'confusion_matrix_figure' in vis_res:
        print(f"[Saved] Confusion matrix: {vis_res['confusion_matrix_figure']}")
    print("[Done] Cluster evaluation complete.")


if __name__ == '__main__':
    main()
