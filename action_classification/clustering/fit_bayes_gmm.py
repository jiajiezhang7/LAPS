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
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, f1_score

# Reuse loader from eval script
from action_classification.evaluation.embed_cluster_eval import load_embed_dir, _merge_labels


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def run_umap_scatter(X: np.ndarray, y_true: np.ndarray, y_pred_aligned: np.ndarray, label_names: List[str], out_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
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

        # Consistent color mapping across plots
        hue_order = sorted(label_names)
        palette = sns.color_palette('tab10', n_colors=len(hue_order))
        color_map = dict(zip(hue_order, palette))

        # True labels plot
        plt.figure(figsize=tuple(vis_cfg.get('plot', {}).get('figsize', (7, 6))))
        sns.scatterplot(
            x=emb[:, 0],
            y=emb[:, 1],
            hue=[label_names[i] for i in y_true],
            palette=color_map,
            hue_order=hue_order,
            s=vis_cfg.get('plot', {}).get('marker_size', 10),
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
        sns.scatterplot(
            x=emb[:, 0],
            y=emb[:, 1],
            hue=[label_names[i] if 0 <= i < len(label_names) else 'UNK' for i in y_pred_aligned],
            palette=color_map,
            hue_order=hue_order,
            s=vis_cfg.get('plot', {}).get('marker_size', 10),
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


def main():
    ap = argparse.ArgumentParser(description='Fit BayesianGaussianMixture over LSTM embeddings and save model for online inference')
    ap.add_argument('--embed-dir', type=str, required=True, help='Directory that contains embed.npy, labels.npy, label_names.txt')
    ap.add_argument('--config', type=str, default='amplify_motion_tokenizer/action_classification/configs/eval_config.yaml', help='Config with clustering/visualization keys')
    ap.add_argument('--out-dir', type=str, default=None, help='Base directory to save model and reports')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = int(config.get('seed', 0))
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
    n_components = int(bgmm_cfg.get('n_components', max(8, len(label_names))))
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = out_base / timestamp
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
        'label_names': label_names,
    }
    with open(out_dir / 'cluster_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Train-set evaluation
    y_pred = model.predict(X_proc)
    pred_ids = np.unique(y_pred)
    pred_id_to_col = {int(cid): i for i, cid in enumerate(pred_ids.tolist())}
    C_true = int(np.max(y_true)) + 1

    cm_rect = np.zeros((C_true, pred_ids.size), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm_rect[int(t), int(pred_id_to_col[int(p)])] += 1
    mapping_rect, acc = hungarian_alignment(cm_rect)
    # Complete mapping for columns not covered by Hungarian (when K_pred > C_true)
    # mapping_rect: column (pred-col index) -> row (true label)
    mapping_full = dict(mapping_rect)
    K_pred = cm_rect.shape[1]
    for col in range(K_pred):
        if col not in mapping_full:
            # Fallback to the dominant true class for this predicted column
            col_sum = cm_rect[:, col].sum()
            fallback_row = int(np.argmax(cm_rect[:, col])) if col_sum > 0 else 0
            mapping_full[col] = fallback_row

    y_pred_aligned = np.asarray([mapping_full[int(pred_id_to_col[int(p)])] for p in y_pred], dtype=np.int64)
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    ari = float(adjusted_rand_score(y_true, y_pred))
    cm_aligned = confusion_matrix(y_true, y_pred_aligned, labels=list(range(C_true)))
    prec, rec = per_class_precision_recall(cm_aligned)
    f1_macro = float(f1_score(y_true, y_pred_aligned, average='macro'))

    results: Dict[str, Any] = {
        'method': 'bayes_gmm',
        'num_samples': int(X.shape[0]),
        'num_pred_clusters': int(pred_ids.size),
        'metrics': {
            'acc': float(acc),
            'nmi': float(nmi),
            'ari': float(ari),
            'f1_macro': float(f1_macro),
            'precision_per_class': [float(x) for x in prec.tolist()],
            'recall_per_class': [float(x) for x in rec.tolist()],
            'label_names': meta['label_names'],
        },
        'confusion_matrix': cm_aligned.astype(int).tolist(),
        # Map original predicted component id -> true label id using the completed mapping
        'mapping_predCluster_to_trueLabel': {str(int(cid)): int(mapping_full[col]) for cid, col in pred_id_to_col.items()},
    }

    # Optional UMAP visualization with consistent colors
    if config.get('visualization', {}).get('umap', {}).get('enabled', True):
        vis_dir = out_dir / 'umap'
        umap_res = run_umap_scatter(X_proc, y_true, y_pred_aligned, label_names, vis_dir, config)
        results.update(umap_res)

    with open(out_dir / 'aggregate_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[Saved] BayesGMM model: {out_dir / 'model_bayes_gmm.pkl'}")
    print(f"[Saved] Meta: {out_dir / 'cluster_meta.json'}")
    print(f"[Saved] Aggregate results: {out_dir / 'aggregate_results.json'}")


if __name__ == '__main__':
    main()
