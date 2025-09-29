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

# Reuse loader from eval script
from action_classification.evaluation.embed_cluster_eval import load_embed_dir, _merge_labels


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_umap_scatter(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred_aligned: np.ndarray,
    label_names: List[str],
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
        # Consistent color mapping across plots
        hue_order = sorted(label_names)
        palette = sns.color_palette('tab10', n_colors=len(hue_order))
        color_map = dict(zip(hue_order, palette))
        plot_size = vis_cfg.get('plot', {}).get('marker_size', 10)

        # Prepare indices
        if use_all:
            assigned_idx = np.where(~noise_mask)[0]
            emb_assigned = emb[assigned_idx]
            emb_noise = emb[noise_mask]
            noise_style = vis_cfg.get('noise', {})
            noise_color = noise_style.get('color', '#b0b0b0')
            noise_alpha = float(noise_style.get('alpha', 0.4))
            noise_size = int(noise_style.get('marker_size', max(1, int(0.8 * plot_size))))
        else:
            emb_assigned = emb

        # True labels plot
        plt.figure(figsize=tuple(vis_cfg.get('plot', {}).get('figsize', (7, 6))))
        if use_all:
            # draw noise first (background)
            if emb_noise.shape[0] > 0:
                plt.scatter(emb_noise[:, 0], emb_noise[:, 1], c=noise_color, s=noise_size, alpha=noise_alpha, linewidths=0.0)
        sns.scatterplot(
            x=emb_assigned[:, 0],
            y=emb_assigned[:, 1],
            hue=[label_names[i] for i in y_true],
            palette=color_map,
            hue_order=hue_order,
            s=plot_size,
            linewidth=0.0
        )
        plt.title('UMAP (lstm) - True Labels')
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        true_path = out_dir / 'umap_true_lstm.png'
        plt.tight_layout()
        plt.savefig(true_path, dpi=vis_cfg.get('plot', {}).get('dpi', 200))
        plt.close()

        # Predicted labels plot
        plt.figure(figsize=tuple(vis_cfg.get('plot', {}).get('figsize', (7, 6))))
        if use_all:
            if emb_noise.shape[0] > 0:
                plt.scatter(emb_noise[:, 0], emb_noise[:, 1], c=noise_color, s=noise_size, alpha=noise_alpha, linewidths=0.0)
        sns.scatterplot(
            x=emb_assigned[:, 0],
            y=emb_assigned[:, 1],
            hue=[label_names[i] if 0 <= i < len(label_names) else 'UNK' for i in y_pred_aligned],
            palette=color_map,
            hue_order=hue_order,
            s=plot_size,
            linewidth=0.0
        )
        plt.title('UMAP (lstm) - Predicted (Aligned)')
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        pred_path = out_dir / 'umap_pred_lstm.png'
        plt.tight_layout()
        plt.savefig(pred_path, dpi=vis_cfg.get('plot', {}).get('dpi', 200))
        plt.close()

        res['umap'] = {
            'metric': metric,
            'neighbors': vis_cfg.get('n_neighbors', 15),
            'min_dist': vis_cfg.get('min_dist', 0.1),
            'embeddings_path_true': str(true_path),
            'embeddings_path_pred': str(pred_path),
        }
    except Exception as e:
        res['umap_error'] = str(e)
    return res


def run_umap_scatter_clusters(X: np.ndarray, y_clusters: np.ndarray, out_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """UMAP 可视化（未对齐）：按簇ID着色。

    说明：
    - 仅使用簇ID（包含 -1 表示噪声时可自行选择是否传入）。
    - 与 run_umap_scatter 保持相同的 UMAP 参数，便于对比。
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

        # 颜色映射：按簇ID（包含-1时也一并显示）
        unique_clusters = sorted(np.unique(y_clusters).tolist())
        # 为了可靠可读，最多取到 unique 数量的调色板颜色
        palette = sns.color_palette('tab20', n_colors=max(1, len(unique_clusters)))
        color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}
        colors = [color_map[int(c)] for c in y_clusters]

        plt.figure(figsize=tuple(vis_cfg.get('plot', {}).get('figsize', (7, 6))))
        plt.scatter(emb[:, 0], emb[:, 1], c=colors, s=vis_cfg.get('plot', {}).get('marker_size', 10), linewidths=0.0)
        # 构造图例
        from matplotlib.lines import Line2D
        legend_elems = [Line2D([0], [0], marker='o', color='w', label=str(cid), markerfacecolor=color_map[cid], markersize=6)
                        for cid in unique_clusters]
        plt.legend(handles=legend_elems, title='Cluster ID', markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('UMAP (lstm) - Predicted Clusters (raw IDs)')
        pred_clusters_path = out_dir / 'umap_pred_clusters_lstm.png'
        plt.tight_layout()
        plt.savefig(pred_clusters_path, dpi=vis_cfg.get('plot', {}).get('dpi', 200))
        plt.close()

        res['umap_clusters'] = {
            'metric': metric,
            'neighbors': vis_cfg.get('n_neighbors', 15),
            'min_dist': vis_cfg.get('min_dist', 0.1),
            'embeddings_path_pred_clusters': str(pred_clusters_path),
            'unique_cluster_ids': unique_clusters,
        }
    except Exception as e:
        res['umap_clusters_error'] = str(e)
    return res


def hungarian_alignment(cm: np.ndarray) -> Tuple[Dict[int, int], float]:
    from scipy.optimize import linear_sum_assignment
    r_ind, c_ind = linear_sum_assignment(-cm)
    mapping = {int(c): int(r) for r, c in zip(r_ind, c_ind)}
    acc = cm[r_ind, c_ind].sum() / cm.sum() if cm.sum() > 0 else 0.0
    return mapping, float(acc)


def per_class_precision_recall(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)
    true_pos = cm.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    return precision, recall


def main():
    if hdbscan is None:
        raise ImportError("hdbscan is not installed. Please install hdbscan>=0.8.33.")

    ap = argparse.ArgumentParser(description='Fit HDBSCAN over LSTM embeddings and save model for online inference')
    ap.add_argument('--embed-dir', type=str, required=True, help='Directory that contains embed.npy, labels.npy, label_names.txt')
    ap.add_argument('--config', type=str, default='amplify_motion_tokenizer/action_classification/configs/eval_config.yaml', help='Config with clustering/visualization keys')
    ap.add_argument('--out-dir', type=str, default=None, help='Base directory to save model and reports')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    X, y_true, label_names = load_embed_dir(Path(args.embed_dir).resolve())
    # Optional label merging for reporting (e.g., merge nok_* into nok)
    label_proc_cfg = config.get('label_processing', {}) if isinstance(config, dict) else {}
    merge_map = label_proc_cfg.get('merge_map', {}) or {}
    if len(merge_map) > 0:
        y_true, label_names = _merge_labels(y_true, label_names, merge_map)

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
        'label_names': label_names,
    }
    with open(out_dir / 'cluster_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Train-set assignment analysis (exclude noise)
    y_pred = model.labels_.astype(int)
    mask = (y_pred != -1)
    C_true = int(np.max(y_true)) + 1
    pred_ids = np.unique(y_pred[mask]) if np.any(mask) else np.array([], dtype=int)

    results: Dict[str, Any] = {
        'method': 'hdbscan',
        'num_samples': int(X.shape[0]),
        'noise_rate': float(1.0 - (float(mask.sum()) / max(1, X.shape[0]))),
        'num_pred_clusters': int(pred_ids.size),
    }

    if pred_ids.size > 0:
        # Rectangular confusion (true x pred)
        pred_id_to_col = {int(cid): i for i, cid in enumerate(pred_ids.tolist())}
        from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, f1_score
        cm_rect = np.zeros((C_true, pred_ids.size), dtype=np.int64)
        for t, p in zip(y_true[mask].tolist(), y_pred[mask].tolist()):
            cm_rect[int(t), int(pred_id_to_col[int(p)])] += 1
        mapping_rect, acc = hungarian_alignment(cm_rect)
        # Complete mapping for columns not covered by Hungarian (when K_pred > C_true)
        mapping_full = dict(mapping_rect)
        K_pred = cm_rect.shape[1]
        for col in range(K_pred):
            if col not in mapping_full:
                col_sum = cm_rect[:, col].sum()
                fallback_row = int(np.argmax(cm_rect[:, col])) if col_sum > 0 else 0
                mapping_full[col] = fallback_row

        # Compute aligned metrics on assigned subset using full mapping
        y_pred_aligned_masked = np.asarray([mapping_full[int(pred_id_to_col[int(p)])] for p in y_pred[mask]], dtype=np.int64)
        nmi = float(normalized_mutual_info_score(y_true[mask], y_pred[mask]))
        ari = float(adjusted_rand_score(y_true[mask], y_pred[mask]))
        cm_aligned = confusion_matrix(y_true[mask], y_pred_aligned_masked, labels=list(range(C_true)))
        prec, rec = per_class_precision_recall(cm_aligned)
        f1_macro = float(f1_score(y_true[mask], y_pred_aligned_masked, average='macro'))

        results.update({
            'metrics': {
                'acc': float(acc),
                'nmi': float(nmi),
                'ari': float(ari),
                'f1_macro': float(f1_macro),
                'precision_per_class': [float(x) for x in prec.tolist()],
                'recall_per_class': [float(x) for x in rec.tolist()],
                'label_names': label_names,
            },
            'confusion_matrix': cm_aligned.astype(int).tolist(),
            'mapping_predCluster_to_trueLabel': {str(int(cid)): int(mapping_full[col]) for cid, col in pred_id_to_col.items()},
        })

        # Optional UMAP on assigned points
        if config.get('visualization', {}).get('umap', {}).get('enabled', True):
            vis_dir = out_dir / 'umap'
            vis_cfg = config.get('visualization', {}).get('umap', {})
            show_noise = bool(vis_cfg.get('show_noise', False))
            if show_noise:
                umap_res = run_umap_scatter(
                    X_proc[mask],
                    y_true[mask],
                    y_pred_aligned_masked,
                    label_names,
                    vis_dir,
                    config,
                    X_all=X_proc,
                    noise_mask=(y_pred == -1)
                )
                results.update(umap_res)
                # 未对齐、按簇ID着色的 UMAP（包含噪声 -1）
                umap_clusters_res = run_umap_scatter_clusters(X_proc, y_pred, vis_dir, config)
                results.update(umap_clusters_res)
            else:
                umap_res = run_umap_scatter(X_proc[mask], y_true[mask], y_pred_aligned_masked, label_names, vis_dir, config)
                results.update(umap_res)
                # 未对齐、按簇ID着色的 UMAP（仅对已分配的样本）
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

    assign_path = out_dir / 'train_assignments.jsonl'
    with open(assign_path, 'w', encoding='utf-8') as f:
        for i in range(X.shape[0]):
            rec = {
                'index': int(i),
                'path': paths[i] if i < len(paths) else str(i),
                'true_label': int(y_true[i]) if i < y_true.shape[0] else -1,
                'pred_cluster': int(y_pred[i]),
                'pred_prob': float(pred_prob[i]),
                'is_noise': bool(y_pred[i] == -1),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"[Saved] HDBSCAN model: {out_dir / 'model_hdbscan.pkl'}")
    print(f"[Saved] Meta: {out_dir / 'cluster_meta.json'}")
    print(f"[Saved] Aggregate results: {out_dir / 'aggregate_results.json'}")
    print(f"[Saved] Train assignments: {assign_path}")


if __name__ == '__main__':
    main()
