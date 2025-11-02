#!/usr/bin/env python3
"""
片段级别 UMAP 聚类可视化与定量评估
- 读取每个片段的 codes sidecar JSON（包含 quantized_windows）
- 按片段聚合时序向量为固定维特征（多种聚合方式）
- 使用 UMAP 降维到 2D/3D 并可视化
- 对聚类质量进行无监督定量评估（KMeans 上的多指标）

用法示例：
    python umap_vis/scripts/segment_umap_cluster_analysis.py \
        --data-dir /path/to/output_root \
        --output-dir umap_vis/figure \
        --aggs mean mean_std max first_last attn_norm \
        --k-min 2 --k-max 10 --neighbors 15 --min-dist 0.1

注意：
- 默认读取 {data-dir}/**/code_indices/*.codes.json
- 需要已安装: numpy, scikit-learn, matplotlib, plotly, umap-learn
"""

import os
import csv
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import matplotlib.pyplot as plt


def load_segments(data_dir: str) -> Tuple[List[np.ndarray], List[str], List[Dict]]:
    """读取所有片段 JSON，返回：
    - series_list: 每个片段的时间序列向量 (Ti, D)
    - names: 片段名（文件名去后缀）
    - metas: 原始元信息
    """
    json_files = sorted(glob.glob(str(Path(data_dir) / "**/code_indices/*.codes.json"), recursive=True))
    print(f"找到 {len(json_files)} 个 JSON 文件")
    series_list: List[np.ndarray] = []
    names: List[str] = []
    metas: List[Dict] = []

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            qwins = data.get("quantized_windows", None)
            if not isinstance(qwins, list) or len(qwins) == 0:
                continue
            # 拼成时间序列 (T_total, D)
            seq: List[List[float]] = []
            for win in qwins:
                if isinstance(win, list):
                    for vec in win:
                        if isinstance(vec, list):
                            seq.append(vec)
            if len(seq) == 0:
                continue
            arr = np.asarray(seq, dtype=np.float32)
            if arr.ndim != 2:
                continue
            series_list.append(arr)
            names.append(Path(jf).stem)
            metas.append({
                "json_path": jf,
                "num_windows": len(qwins),
                "num_vectors": int(arr.shape[0]),
                "vector_dim": int(arr.shape[1]),
                "video_path": data.get("video_segment_path", "unknown"),
            })
        except Exception as e:
            print(f"[WARN] 读取失败 {jf}: {e}")
    return series_list, names, metas


# ------------- 聚合函数 -------------

def agg_mean(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=0)

def agg_max(x: np.ndarray) -> np.ndarray:
    return np.max(x, axis=0)

def agg_mean_std(x: np.ndarray) -> np.ndarray:
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0)
    return np.concatenate([m, s], axis=0)

def agg_first_last(x: np.ndarray) -> np.ndarray:
    first = x[0]
    last = x[-1]
    return np.concatenate([first, last], axis=0)

def agg_attn_norm(x: np.ndarray) -> np.ndarray:
    # 使用向量范数作为注意力权重（无训练）
    # w_t = ||x_t||_2；归一化为和为1
    norms = np.linalg.norm(x, ord=2, axis=1)
    if np.allclose(norms.sum(), 0.0):
        return np.mean(x, axis=0)
    w = norms / (norms.sum() + 1e-8)
    return (x * w[:, None]).sum(axis=0)

AGG_FUNCS = {
    "mean": agg_mean,
    "max": agg_max,
    "mean_std": agg_mean_std,
    "first_last": agg_first_last,
    "attn_norm": agg_attn_norm,
}


def features_by_agg(series_list: List[np.ndarray], agg: str) -> np.ndarray:
    func = AGG_FUNCS.get(agg)
    if func is None:
        raise ValueError(f"未知聚合方式: {agg}")
    feats = [func(s) for s in series_list]
    return np.asarray(feats, dtype=np.float32)


# ------------- 降维与可视化 -------------

def umap_embed(X: np.ndarray, n_components: int, n_neighbors: int, min_dist: float, metric: str = "euclidean") -> Optional[np.ndarray]:
    try:
        import umap  # type: ignore
    except Exception as e:
        print("[WARN] 需要安装 umap-learn: pip install umap-learn | 错误:", e)
        return None
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )
    return reducer.fit_transform(X)


def plot_umap_2d(emb: np.ndarray, labels: Optional[np.ndarray], title: str, out_path: Path):
    plt.figure(figsize=(10, 8))
    if labels is None:
        c = np.arange(len(emb))
    else:
        c = labels
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=c, cmap="tab20", s=36, alpha=0.75, edgecolors="none")
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(sc, label=("Cluster" if labels is not None else "Index"))
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"保存 2D 图: {out_path}")


def plot_umap_3d(emb: np.ndarray, labels: Optional[np.ndarray], title: str, out_path: Path):
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as e:
        print("[WARN] 需要安装 plotly: pip install plotly | 错误:", e)
        return
    c = np.arange(len(emb)) if labels is None else labels
    fig = go.Figure(data=[go.Scatter3d(
        x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
        mode='markers',
        marker=dict(size=5, color=c, colorscale='Viridis', showscale=True,
                    line=dict(color='rgba(0,0,0,0)', width=0))
    )])
    fig.update_layout(title=title, scene=dict(xaxis_title='UMAP-1', yaxis_title='UMAP-2', zaxis_title='UMAP-3'))
    fig.write_html(str(out_path))
    print(f"保存 3D 图: {out_path}")


# ------------- 聚类与指标 -------------

def pairwise_dists_metric(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    if metric == 'euclidean':
        XX = np.sum(X * X, axis=1, keepdims=True)
        d2 = XX + XX.T - 2.0 * (X @ X.T)
        d2 = np.maximum(d2, 0.0)
        return np.sqrt(d2, dtype=np.float32)
    elif metric == 'cosine':
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = np.clip(Xn @ Xn.T, -1.0, 1.0)
        return (1.0 - S).astype(np.float32)
    else:
        raise ValueError(f'unsupported metric: {metric}')


def cluster_and_scores(X: np.ndarray, k_min: int, k_max: int, random_state: int = 42, metric: str = 'euclidean') -> Tuple[Dict[int, Dict[str, float]], int, np.ndarray]:
    n = X.shape[0]
    if n < 3:
        print("[WARN] 样本过少，跳过聚类评估")
        return {}, 0, np.zeros(n, dtype=int)
    k_max = max(2, min(k_max, n - 1))
    k_min = max(2, min(k_min, k_max))

    results: Dict[int, Dict[str, float]] = {}
    best_k = None
    best_score = -1.0
    best_labels = None

    # KMeans 本质是欧式度量；若使用 cosine，我们先进行 L2 行归一化，使欧式与余弦几何一致
    X_km = X
    if metric == 'cosine':
        X_km = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_km)
        if len(np.unique(labels)) < 2:
            continue
        try:
            sil_metric = 'cosine' if metric == 'cosine' else 'euclidean'
            sil = silhouette_score(X, labels, metric=sil_metric)
        except Exception:
            sil = float("nan")
        try:
            db = davies_bouldin_score(X_km, labels)
        except Exception:
            db = float("nan")
        try:
            ch = calinski_harabasz_score(X_km, labels)
        except Exception:
            ch = float("nan")
        w_intra, b_inter, ratio = intra_inter_stats(X, labels, metric=metric)
        results[k] = {
            "silhouette": float(sil),
            "davies_bouldin": float(db),
            "calinski_harabasz": float(ch),
            "intra_dist": float(w_intra),
            "inter_centroid_dist": float(b_inter),
            "intra_over_inter": float(ratio),
        }
        if not np.isnan(sil) and sil > best_score:
            best_score = sil
            best_k = k
            best_labels = labels

    if best_k is None:
        best_k = k_min
        best_labels = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit_predict(X_km)
    return results, int(best_k), best_labels


def intra_inter_stats(X: np.ndarray, labels: np.ndarray, metric: str = 'euclidean') -> Tuple[float, float, float]:
    # 类内平均两两距离 与 类间质心距离（使用指定度量；cosine 下质心先行归一化）
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan"), float("nan"), float("nan")
    intra_sum = 0.0
    intra_cnt = 0
    centroids = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        Xi = X[idx]
        # 类内平均两两距离
        if len(idx) > 1:
            dists = pairwise_dists_metric(Xi, metric=metric)
            iu = np.triu_indices_from(dists, k=1)
            vals = dists[iu]
            intra_sum += float(vals.mean())
            intra_cnt += 1
        # 质心
        mu = Xi.mean(axis=0)
        if metric == 'cosine':
            mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
    w_intra = intra_sum / max(1, intra_cnt)
    # 类间质心距离
    C = np.vstack(centroids)
    cd = pairwise_dists_metric(C, metric=metric)
    iu = np.triu_indices_from(cd, k=1)
    b_inter = float(cd[iu].mean()) if iu[0].size > 0 else float("nan")
    ratio = w_intra / (b_inter + 1e-8) if not np.isnan(w_intra) and not np.isnan(b_inter) else float("nan")
    return w_intra, b_inter, ratio


def save_metrics_csv(path: Path, agg: str, results: Dict[int, Dict[str, float]], n_samples: int, dim: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agg", "k", "n_samples", "feature_dim", "silhouette", "davies_bouldin", "calinski_harabasz", "intra_dist", "inter_centroid_dist", "intra_over_inter"])
        for k, m in sorted(results.items()):
            writer.writerow([
                agg, k, n_samples, dim,
                m.get("silhouette", np.nan), m.get("davies_bouldin", np.nan), m.get("calinski_harabasz", np.nan),
                m.get("intra_dist", np.nan), m.get("inter_centroid_dist", np.nan), m.get("intra_over_inter", np.nan),
            ])
    print(f"保存指标: {path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True, help="包含各视频 code_indices 的根目录")
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).parent.parent / "figure"))
    p.add_argument("--aggs", type=str, nargs="+", default=["mean"], choices=list(AGG_FUNCS.keys()))
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"], help="特征空间度量，影响UMAP/轮廓系数/类内类间距离")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读数据
    series_list, names, metas = load_segments(args.data_dir)
    if len(series_list) == 0:
        print("[ERR] 未找到任何 quantized_windows 数据，请检查路径 --data-dir")
        return
    d = series_list[0].shape[1]
    print(f"片段数: {len(series_list)}, 向量维度: {d}")

    for agg in args.aggs:
        print("\n" + "=" * 80)
        print(f"聚合方式: {agg}")
        # 2) 聚合
        X = features_by_agg(series_list, agg)
        print(f"特征形状: {X.shape}")
        # 3) 标准化 +（可选）L2 归一化
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        if args.metric == 'cosine':
            Xs = normalize(Xs, norm='l2')
        # 4) UMAP 2D/3D（保持与特征空间度量一致）
        emb2 = umap_embed(Xs, n_components=2, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)
        emb3 = umap_embed(Xs, n_components=3, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)
        # 5) 聚类与指标
        results, best_k, labels = cluster_and_scores(Xs, args.k_min, args.k_max, random_state=args.seed, metric=args.metric)
        print(f"最佳 k(按 silhouette): {best_k}")
        # 6) 可视化（文件名加入度量后缀以便对比）
        if emb2 is not None:
            plot_umap_2d(emb2, labels, title=f"UMAP 2D - agg={agg} metric={args.metric} (k={best_k})", out_path=out_dir / f"umap_2d_{agg}_{args.metric}_k{best_k}.png")
        if emb3 is not None:
            plot_umap_3d(emb3, labels, title=f"UMAP 3D - agg={agg} metric={args.metric} (k={best_k})", out_path=out_dir / f"umap_3d_{agg}_{args.metric}_k{best_k}.html")
        # 7) 保存指标
        save_metrics_csv(out_dir / f"cluster_metrics_{agg}_{args.metric}.csv", agg, results, n_samples=X.shape[0], dim=X.shape[1])

    print("\n完成。输出目录:", out_dir)


if __name__ == "__main__":
    main()

