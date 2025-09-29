import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

from action_classification.data.features import build_dataset


@dataclass
class EvalResult:
    acc: float
    nmi: float
    ari: float
    f1_macro: float
    confusion: List[List[int]]
    mapping: Dict[int, int]  # pred_cluster -> true_label
    precision_per_class: List[float]
    recall_per_class: List[float]
    label_names: List[str]


def _hungarian_alignment(cm: np.ndarray) -> Tuple[Dict[int, int], float]:
    # cm: rows = true label, cols = predicted cluster
    r_ind, c_ind = linear_sum_assignment(-cm)
    mapping = {int(c): int(r) for r, c in zip(r_ind, c_ind)}  # pred_cluster -> true_label
    acc = cm[r_ind, c_ind].sum() / cm.sum() if cm.sum() > 0 else 0.0
    return mapping, float(acc)


def _apply_mapping(y_pred: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    out = np.zeros_like(y_pred)
    for i, v in enumerate(y_pred):
        out[i] = mapping.get(int(v), -1)
    return out


def _per_class_precision_recall(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # cm: rows=true, cols=pred (after alignment)
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)
    true_pos = cm.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    return precision, recall


def _ensure_outdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def _bow_transform(X: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Apply BoW-specific transforms based on config: sqrt, log1p, tfidf, or none."""
    tcfg = config['features']['bow']
    transform = tcfg.get('transform', 'none')
    X_out = X.copy()
    if transform == 'sqrt':
        X_out = np.sqrt(np.clip(X_out, 0.0, None))
    elif transform == 'log1p':
        X_out = np.log1p(np.clip(X_out, 0.0, None))
    elif transform == 'tfidf':
        # Compute DF across documents (samples)
        N = X_out.shape[0]
        df = (X_out > 0).sum(axis=0)
        idf = np.log((N + 1.0) / (df + 1.0)) + 1.0
        # Term-frequency (L1)
        denom = X_out.sum(axis=1, keepdims=True) + 1e-12
        tf = X_out / denom
        X_out = tf * idf
    # Optional L2 normalization for spherical clustering
    bow_pcfg = config.get('preprocessing', {}).get('bow_feature', {})
    if bow_pcfg.get('l2_normalize', False):
        X_out = _l2_normalize(X_out)
    return X_out


def _umap_scatter(X: np.ndarray, y_true: np.ndarray, y_pred_aligned: np.ndarray, label_names: List[str], out_dir: Path, feature: str, config: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    try:
        import matplotlib
        matplotlib.use('Agg')  # ensure headless plotting
        import umap
        import matplotlib.pyplot as plt
        import seaborn as sns

        vis_cfg = config['visualization']['umap']
        metric = vis_cfg[f'{feature}_metric']

        reducer = umap.UMAP(
            n_neighbors=vis_cfg['n_neighbors'],
            min_dist=vis_cfg['min_dist'],
            n_components=2,
            metric=metric,
            random_state=config['seed']
        )
        emb = reducer.fit_transform(X)

        _ensure_outdir(out_dir / 'dummy')
        # True labels
        plt.figure(figsize=vis_cfg['plot']['figsize'])
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=[label_names[i] for i in y_true], palette='tab10', s=vis_cfg['plot']['marker_size'], linewidth=0.0)
        plt.title(f'UMAP ({feature}) - True Labels')
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        true_path = out_dir / f'umap_true_{feature}.png'
        plt.tight_layout()
        plt.savefig(true_path, dpi=vis_cfg['plot']['dpi'])
        plt.close()

        # Predicted (aligned)
        plt.figure(figsize=vis_cfg['plot']['figsize'])
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=[label_names[i] if i >= 0 and i < len(label_names) else 'UNK' for i in y_pred_aligned], palette='tab10', s=vis_cfg['plot']['marker_size'], linewidth=0.0)
        plt.title(f'UMAP ({feature}) - Predicted (Aligned)')
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        pred_path = out_dir / f'umap_pred_{feature}.png'
        plt.tight_layout()
        plt.savefig(pred_path, dpi=vis_cfg['plot']['dpi'])
        plt.close()

        res['umap'] = {
            'metric': metric,
            'neighbors': vis_cfg['n_neighbors'],
            'min_dist': vis_cfg['min_dist'],
            'embeddings_path_true': str(true_path),
            'embeddings_path_pred': str(pred_path),
        }
    except Exception as e:
        res['umap_error'] = str(e)
    return res


def run_baseline(json_root: Path, feature: str, config: Dict[str, Any], out_dir: Optional[Path] = None) -> Dict[str, Any]:
    n_clusters = config['n_clusters']
    X, y_true, paths, label_names = build_dataset(json_root, feature=feature, config=config, expected_classes=n_clusters)

    # Preprocess
    X_proc = X
    pca_info: Dict[str, Any] = {}
    if feature == 'avg':
        pca_cfg = config['preprocessing']['avg_feature']['pca']
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        # PCA 保留 95% 方差，限制到不超过 128 维，避免过拟合噪声
        pca = PCA(n_components=min(Xs.shape[1], pca_cfg['max_dims']), svd_solver='full', random_state=config['seed'])
        Xp = pca.fit_transform(Xs)
        # 可选：解释方差比计算保留维度的累计值
        explained = np.cumsum(pca.explained_variance_ratio_)
        keep = np.searchsorted(explained, pca_cfg['explained_variance_threshold']) + 1
        X_proc = Xp[:, :max(keep, pca_cfg['min_dims'])]  # 至少 2 维
        # Optional L2 normalization to emphasize direction over magnitude
        if config['preprocessing']['avg_feature'].get('l2_normalize', True):
            X_proc = _l2_normalize(X_proc)
        pca_info = {
            'orig_dim': int(X.shape[1]),
            'scaled_dim': int(Xs.shape[1]),
            'pca_dim_used': int(X_proc.shape[1]),
            'explained_var_cum_0.95_index': int(keep),
        }
    elif feature == 'bow':
        # 直方图是 L1 归一，直接用或可做轻微 sqrt 压缩。
        # X_proc = np.sqrt(np.clip(X, 0, None))  # 可选
        X_proc = _bow_transform(X, config)

    # Clustering
    cluster_cfg = config['clustering']
    # Get method from feature-specific config
    method_by_feature = cluster_cfg.get('method_by_feature', {})
    method = method_by_feature.get(feature)
    if not method:
        raise ValueError(f"Clustering method for '{feature}' must be specified in config under 'clustering.method_by_feature.{feature}'")
    if method == 'kmeans':
        cluster_model = KMeans(n_clusters=n_clusters, n_init=cluster_cfg['kmeans']['n_init'], random_state=config['seed'])
    elif method == 'gmm':
        gmm_cfg = cluster_cfg.get('gmm', {})
        cluster_model = GaussianMixture(n_components=n_clusters, covariance_type=gmm_cfg.get('covariance_type', 'full'), random_state=config['seed'])
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    fit_info: Dict[str, Any] = {}
    if cluster_cfg['balance_fit']:
        # 依据文件夹标签统计每类样本数，取最小值作为 min_count，并从每类均匀采样用于拟合
        C = len(label_names)
        class_indices = {c: np.where(y_true == c)[0] for c in range(C)}
        counts = [int(class_indices[c].size) for c in range(C)]
        pos_counts = [cnt for cnt in counts if cnt > 0]
        if len(pos_counts) < C:
            print("[Warn] Some classes have zero samples; fallback to unbalanced fit.")
            if method == 'kmeans':
                y_pred = cluster_model.fit_predict(X_proc)
            else:
                cluster_model.fit(X_proc)
                y_pred = cluster_model.predict(X_proc)
            X_eval = X_proc
            y_true_eval = y_true
        else:
            min_count = int(min(pos_counts))
            rng = np.random.RandomState(config['seed'])
            fit_idx_parts = []
            for c in range(C):
                idx = class_indices[c]
                sel = rng.choice(idx, size=min_count, replace=False) if idx.size > min_count else idx
                fit_idx_parts.append(sel)
            fit_idx = np.concatenate(fit_idx_parts, axis=0)
            cluster_model.fit(X_proc[fit_idx])

            if cluster_cfg.get('evaluate_on_fit_subset', False):
                # Evaluate only on the balanced subset
                X_eval = X_proc[fit_idx]
                y_true_eval = y_true[fit_idx]
                y_pred = cluster_model.predict(X_eval)
                print(f"[Info] Evaluating on the balanced subset of size {len(y_true_eval)}.")
            else:
                # Evaluate on the full dataset
                X_eval = X_proc
                y_true_eval = y_true
                y_pred = cluster_model.predict(X_eval)
            fit_info = {
                'balance_fit': True,
                'min_count': int(min_count),
                'counts_per_class': counts,
                'num_fit_samples': int(fit_idx.size),
                'evaluate_on_fit_subset': cluster_cfg.get('evaluate_on_fit_subset', False),
            }
            print(f"[Info] Balance-fit subset used: min_count={min_count}, num_fit={fit_idx.size}, counts={counts}")
    else:
        if method == 'kmeans':
            y_pred = cluster_model.fit_predict(X_proc)
        else:
            cluster_model.fit(X_proc)
            y_pred = cluster_model.predict(X_proc)
        X_eval = X_proc
        y_true_eval = y_true

    # Metrics (pre-alignment)
    nmi = float(normalized_mutual_info_score(y_true_eval, y_pred))
    ari = float(adjusted_rand_score(y_true_eval, y_pred))

    # Alignment
    cm = confusion_matrix(y_true_eval, y_pred, labels=list(range(n_clusters)))
    mapping, acc = _hungarian_alignment(cm)
    y_pred_aligned = _apply_mapping(y_pred, mapping)

    # Confusion matrix after alignment
    cm_aligned = confusion_matrix(y_true_eval, y_pred_aligned, labels=list(range(n_clusters)))

    # Per-class metrics
    prec_per_class, rec_per_class = _per_class_precision_recall(cm_aligned)
    f1_macro = float(f1_score(y_true_eval, y_pred_aligned, average='macro'))

    result = EvalResult(
        acc=float(acc),
        nmi=float(nmi),
        ari=float(ari),
        f1_macro=float(f1_macro),
        confusion=cm_aligned.astype(int).tolist(),
        mapping=mapping,
        precision_per_class=[float(x) for x in prec_per_class.tolist()],
        recall_per_class=[float(x) for x in rec_per_class.tolist()],
        label_names=label_names,
    )

    out: Dict[str, Any] = {
        'feature': feature,
        'method': method,
        'n_clusters': int(n_clusters),
        'metrics': {
            'acc': result.acc,
            'nmi': result.nmi,
            'ari': result.ari,
            'f1_macro': result.f1_macro,
            'precision_per_class': result.precision_per_class,
            'recall_per_class': result.recall_per_class,
            'label_names': label_names,
        },
        'confusion_matrix': result.confusion,
        'mapping_predCluster_to_trueLabel': {str(k): int(v) for k, v in result.mapping.items()},
        'pca_info': pca_info,
    }
    if fit_info:
        out['fit_subset'] = fit_info

    if config['visualization']['umap']['enabled']:
        od = out_dir if out_dir is not None else (Path(__file__).resolve().parent / 'out' / 'umap')
        od = od / feature
        # UMAP visualization should also respect the evaluation subset
        umap_res = _umap_scatter(X_eval, y_true_eval, y_pred_aligned, label_names, od, feature=feature, config=config)
        out.update(umap_res)

    # Save JSON
    if out_dir is not None:
        out_path = out_dir / f'result_{feature}.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        out['result_path'] = str(out_path)

    # Console brief
    print(f"=== Baseline: {feature} ===")
    print(f"ACC={out['metrics']['acc']:.4f} NMI={out['metrics']['nmi']:.4f} ARI={out['metrics']['ari']:.4f} F1(macro)={out['metrics']['f1_macro']:.4f}")
    print("Confusion (aligned):")
    for row in out['confusion_matrix']:
        print('  ', row)
    print("Precision per class:", [f"{x:.3f}" for x in out['metrics']['precision_per_class']])
    print("Recall per class:", [f"{x:.3f}" for x in out['metrics']['recall_per_class']])

    return out


def main():
    ap = argparse.ArgumentParser(description='Run unsupervised baselines on Motion Tokenizer JSON outputs')
    # Primary arguments
    ap.add_argument('--json-root', type=str, required=True, help='Root directory of inference JSON outputs')
    ap.add_argument('--config', type=str, default='action_classification/configs/eval_config.yaml', help='Path to the evaluation config YAML file')
    ap.add_argument('--out-dir', type=str, default=None, help='Base directory to save results (overrides config)')
    ap.add_argument('--feature', type=str, choices=['bow', 'avg', 'both'], help='Which feature to run (overrides config)')

    # Override arguments (for values in config.yaml)
    ap.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    ap.add_argument('--clusters', type=int, default=None, help='Number of clusters (overrides config)')
    ap.add_argument('--balance-fit', dest='balance_fit', action='store_true', default=None, help='Enable balance-fit (overrides config)')
    ap.add_argument('--no-balance-fit', dest='balance_fit', action='store_false', help='Disable balance-fit (overrides config)')
    ap.add_argument('--umap', dest='umap', action='store_true', default=None, help='Enable UMAP plotting (overrides config)')
    ap.add_argument('--no-umap', dest='umap', action='store_false', help='Disable UMAP plotting (overrides config)')

    args = ap.parse_args()

    # --- Config Loading and Merging ---
    # 1. Load base config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Override with command-line arguments if provided
    if args.seed is not None:
        config['seed'] = args.seed
    if args.clusters is not None:
        config['n_clusters'] = args.clusters
    if args.balance_fit is not None:
        config['clustering']['balance_fit'] = args.balance_fit
    if args.umap is not None:
        config['visualization']['umap']['enabled'] = args.umap
    if args.feature:
        config['feature_to_run'] = args.feature
    else:
        config['feature_to_run'] = 'both' # Default if not in yaml or args
    # Determine output directory: command line > config > default
    if args.out_dir:
        out_dir_path = args.out_dir
    elif 'out_dir' in config:
        out_dir_path = config['out_dir']
    else:
        out_dir_path = 'action_classification/out'

    json_root = Path(args.json_root).resolve()
    base_out_dir = Path(out_dir_path).resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = base_out_dir / timestamp
    print(f"[Info] Saving results to: {out_dir}")

    features = ['bow', 'avg'] if config['feature_to_run'] == 'both' else [config['feature_to_run']]

    agg: Dict[str, Any] = {}
    for ft in features:
        try:
            res = run_baseline(
                json_root,
                feature=ft,
                config=config,
                out_dir=out_dir,
            )
            agg[ft] = res
        except Exception as e:
            print(f"[ERR] Baseline failed for feature={ft}: {e}")

    # Save aggregate result
    if len(agg) > 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        agg_path = out_dir / 'aggregate_results.json'
        with open(agg_path, 'w', encoding='utf-8') as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)
        print(f"[Saved] Aggregate results: {agg_path}")


if __name__ == '__main__':
    main()
