#!/usr/bin/env python3
"""
基于离散 code indices 的片段级 UMAP 聚类分析
- 从每个片段 JSON 的 codes_windows 读取离散码序列（FSQ 码簇）
- 构造片段级离散序列表征：
  * bow: 2048 维词袋直方图（默认 vocab_size=2048，可覆盖）
  * tfidf: 基于 bow 的 TF-IDF 表示
  * bigramhash: 通过哈希技巧的 bigram 计数（默认 16384 维）
- 使用与连续向量实验一致的流程：标准化/归一化 → UMAP(2D/3D) → KMeans 评估（k∈[5,10]）
- 输出指标 CSV 与 UMAP 可视化

用法示例：
  python umap_vis/scripts/codes_umap_cluster_analysis.py \
      --data-dir /path/to/output_root \
      --output-dir umap_vis/figure \
      --reprs bow tfidf bigramhash \
      --metric cosine --k-min 5 --k-max 10
"""

from __future__ import annotations

import os
import csv
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt


# ---------------- 数据读取 ----------------

def load_codes_sequences(data_dir: str) -> Tuple[List[List[int]], List[str]]:
    json_files = sorted(glob.glob(str(Path(data_dir) / "**/code_indices/*.codes.json"), recursive=True))
    print(f"找到 {len(json_files)} 个 JSON 文件")
    seqs: List[List[int]] = []
    names: List[str] = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            cwins = data.get("codes_windows", None)
            if not isinstance(cwins, list) or len(cwins) == 0:
                continue
            seq: List[int] = []
            for win in cwins:
                if isinstance(win, list):
                    for code in win:
                        if isinstance(code, int):
                            seq.append(code)
            if len(seq) == 0:
                continue
            seqs.append(seq)
            names.append(Path(jf).stem)
        except Exception as e:
            print(f"[WARN] 读取失败 {jf}: {e}")
    return seqs, names


# ---------------- 表征构造 ----------------

def build_bow(seqs: List[List[int]], vocab_size: int = 2048, l1_normalize: bool = True) -> np.ndarray:
    X = np.zeros((len(seqs), vocab_size), dtype=np.float32)
    for i, seq in enumerate(seqs):
        if not seq:
            continue
        idxs, cnts = np.unique(np.clip(np.asarray(seq, dtype=np.int32), 0, vocab_size - 1), return_counts=True)
        X[i, idxs] = cnts.astype(np.float32)
    if l1_normalize:
        s = X.sum(axis=1, keepdims=True)
        X = X / (s + 1e-8)
    return X


def build_tfidf_from_counts(counts: np.ndarray, norm: str = "l2") -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer(norm=norm, use_idf=True, smooth_idf=True, sublinear_tf=False)
    return tfidf.fit_transform(counts).astype(np.float32).toarray()


def build_bigram_hash(seqs: List[List[int]], dim: int = 16384, l1_normalize: bool = True) -> np.ndarray:
    # 简单 2-gram 哈希：h = (a*x + b*y) % dim
    A = 1315423911
    B = 2654435761
    X = np.zeros((len(seqs), dim), dtype=np.float32)
    for i, s in enumerate(seqs):
        if len(s) < 2:
            continue
        arr = np.asarray(s, dtype=np.int64)
        x = arr[:-1]
        y = arr[1:]
        h = (A * x + B * y) % dim
        # 统计每个哈希桶出现次数
        # 用 np.add.at 累加
        np.add.at(X[i], h.astype(np.int64), 1.0)
    if l1_normalize:
        s = X.sum(axis=1, keepdims=True)
        X = X / (s + 1e-8)
    return X


# ---------------- 降维与可视化 ----------------

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
    c = np.arange(len(emb)) if labels is None else labels
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


# ---------------- 聚类与指标 ----------------

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


def intra_inter_stats(X: np.ndarray, labels: np.ndarray, metric: str = 'euclidean') -> Tuple[float, float, float]:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan"), float("nan"), float("nan")
    intra_vals = []
    centroids = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        Xi = X[idx]
        if len(idx) > 1:
            dists = pairwise_dists_metric(Xi, metric=metric)
            iu = np.triu_indices_from(dists, k=1)
            if iu[0].size > 0:
                intra_vals.append(float(dists[iu].mean()))
        mu = Xi.mean(axis=0)
        if metric == 'cosine':
            mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
    w_intra = float(np.mean(intra_vals)) if len(intra_vals) > 0 else float("nan")
    C = np.vstack(centroids)
    cd = pairwise_dists_metric(C, metric=metric)
    iu = np.triu_indices_from(cd, k=1)
    b_inter = float(cd[iu].mean()) if iu[0].size > 0 else float("nan")
    ratio = w_intra / (b_inter + 1e-8) if not (np.isnan(w_intra) or np.isnan(b_inter)) else float("nan")
    return w_intra, b_inter, ratio


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
            sil = float('nan')
        try:
            db = davies_bouldin_score(X_km, labels)
        except Exception:
            db = float('nan')
        try:
            ch = calinski_harabasz_score(X_km, labels)
        except Exception:
            ch = float('nan')
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


# ---------------- 主流程 ----------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).parent.parent / "figure"))
    p.add_argument("--reprs", type=str, nargs="+", default=["bow", "tfidf"], choices=["bow", "tfidf", "bigramhash"])
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--bigram-dim", type=int, default=16384)
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载离散码序列
    seqs, names = load_codes_sequences(args.data_dir)
    if len(seqs) == 0:
        print("[ERR] 未找到任何 codes_windows 数据，请检查路径 --data-dir")
        return
    print(f"片段数: {len(seqs)}")

    # 2) 针对每种表征构建特征并评估
    for rep in args.reprs:
        print("\n" + "=" * 80)
        print(f"表征: {rep}")
        if rep == "bow":
            X = build_bow(seqs, vocab_size=args.vocab_size, l1_normalize=True)
            agg_name = f"codes_bow"
        elif rep == "tfidf":
            counts = build_bow(seqs, vocab_size=args.vocab_size, l1_normalize=False)
            X = build_tfidf_from_counts(counts, norm="l2")
            agg_name = f"codes_tfidf"
        elif rep == "bigramhash":
            X = build_bigram_hash(seqs, dim=args.bigram_dim, l1_normalize=True)
            agg_name = f"codes_bigramhash{args.bigram_dim}"
        else:
            raise ValueError(rep)
        print(f"特征形状: {X.shape}")

        # 3) 标准化 + （可选）L2 归一化（cosine）
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        if args.metric == 'cosine':
            Xs = normalize(Xs, norm='l2')

        # 4) UMAP 可视化
        emb2 = umap_embed(Xs, n_components=2, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)
        emb3 = umap_embed(Xs, n_components=3, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)

        # 5) KMeans 聚类与无监督指标
        results, best_k, labels = cluster_and_scores(Xs, args.k_min, args.k_max, random_state=args.seed, metric=args.metric)
        print(f"最佳 k(按 silhouette): {best_k}")

        # 6) 可视化与指标保存
        metric_suffix = args.metric
        if emb2 is not None:
            plot_umap_2d(emb2, labels, title=f"UMAP 2D - {agg_name} metric={args.metric} (k={best_k})",
                         out_path=out_dir / f"umap_2d_{agg_name}_{metric_suffix}_k{best_k}.png")
        if emb3 is not None:
            plot_umap_3d(emb3, labels, title=f"UMAP 3D - {agg_name} metric={args.metric} (k={best_k})",
                         out_path=out_dir / f"umap_3d_{agg_name}_{metric_suffix}_k{best_k}.html")
        save_metrics_csv(out_dir / f"cluster_metrics_{agg_name}_{metric_suffix}.csv", agg_name, results, n_samples=X.shape[0], dim=X.shape[1])

    print("\n完成。输出目录:", out_dir)


if __name__ == "__main__":
    main()

