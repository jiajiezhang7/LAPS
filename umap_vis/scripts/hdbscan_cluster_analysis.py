#!/usr/bin/env python3
"""
HDBSCAN 密度聚类实验（段级连续向量）
- 读取每个片段的 quantized_windows，按聚合得到段级向量（默认 attn_norm）
- 对标准化+L2 归一化后的特征使用 HDBSCAN 做网格实验（cosine 度量）
- 双口径评估：A) 仅核心点（labels!=-1），B) 噪声点映射到最近簇后的全样本
- 生成 2D/3D UMAP 可视化（最佳配置），并保存统计 CSV 与分析报告

依赖：numpy, scikit-learn, matplotlib, plotly, umap-learn, hdbscan
运行环境：conda env "laps"

示例：
  conda run -n laps python umap_vis/scripts/hdbscan_cluster_analysis.py \
    --data-dir /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector \
    --fig-dir umap_vis/figure --stats-dir umap_vis/statistics \
    --agg attn_norm --metric cosine --neighbors 15 --min-dist 0.1
"""

from __future__ import annotations
import os, csv, json, glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# ---------------- 数据读取与聚合（复用现有风格） ----------------

def load_segments(data_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    json_files = sorted(glob.glob(str(Path(data_dir) / "**/code_indices/*.codes.json"), recursive=True))
    print(f"找到 {len(json_files)} 个 JSON 文件")
    series_list: List[np.ndarray] = []
    names: List[str] = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            qwins = data.get("quantized_windows", None)
            if not isinstance(qwins, list) or len(qwins) == 0:
                continue
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
        except Exception as e:
            print(f"[WARN] 读取失败 {jf}: {e}")
    return series_list, names


def agg_mean(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=0)

def agg_attn_norm(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, ord=2, axis=1)
    if np.allclose(norms.sum(), 0.0):
        return np.mean(x, axis=0)
    w = norms / (norms.sum() + 1e-8)
    return (x * w[:, None]).sum(axis=0)

AGG_FUNCS = {
    "mean": agg_mean,
    "attn_norm": agg_attn_norm,
}

def features_by_agg(series_list: List[np.ndarray], agg: str) -> np.ndarray:
    func = AGG_FUNCS.get(agg)
    if func is None:
        raise ValueError(f"未知聚合方式: {agg}")
    feats = [func(s) for s in series_list]
    return np.asarray(feats, dtype=np.float32)

# ---------------- UMAP 降维与绘图 ----------------

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


def plot_umap_2d_hdbscan(emb: np.ndarray, labels: np.ndarray, probs: np.ndarray, out_path: Path, title: str, core_prob_thresh: float = 0.6):
    plt.figure(figsize=(10, 8))
    uniq = sorted([u for u in np.unique(labels) if u != -1])
    # 噪声
    noise = labels == -1
    plt.scatter(emb[noise, 0], emb[noise, 1], c="#A0A0A0", s=20, alpha=0.5, label="noise(-1)", edgecolors="none")
    # 每簇：区分核心与边界
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(uniq):
        idx = labels == c
        core = idx & (probs >= core_prob_thresh)
        border = idx & (probs < core_prob_thresh)
        color = cmap(i % 20)
        if np.any(border):
            plt.scatter(emb[border, 0], emb[border, 1], c=[color], s=18, alpha=0.5, marker="x", label=f"cluster {c} (border)")
        if np.any(core):
            plt.scatter(emb[core, 0], emb[core, 1], c=[color], s=32, alpha=0.85, marker="o", label=f"cluster {c} (core)")
    plt.title(title)
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    # 图例可能过长，限制条目
    handles, labels_ = plt.gca().get_legend_handles_labels()
    if len(handles) > 15:
        handles = handles[:14] + [handles[-1]]
        labels_ = labels_[:14] + [labels_[-1]]
    plt.legend(handles, labels_, loc="best", fontsize=8)
    plt.tight_layout(); plt.savefig(str(out_path), dpi=150); plt.close()
    print(f"保存 2D 图: {out_path}")


def plot_umap_3d_hdbscan(emb: np.ndarray, labels: np.ndarray, probs: np.ndarray, out_path: Path, title: str, core_prob_thresh: float = 0.6):
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as e:
        print("[WARN] 需要安装 plotly: pip install plotly | 错误:", e)
        return
    uniq = sorted([u for u in np.unique(labels) if u != -1])
    fig = go.Figure()
    # 噪声
    noise = labels == -1
    if np.any(noise):
        fig.add_trace(go.Scatter3d(x=emb[noise,0], y=emb[noise,1], z=emb[noise,2], mode='markers',
            name='noise(-1)', marker=dict(size=3, color='gray', opacity=0.5)))
    # 簇
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    for i, c in enumerate(uniq):
        idx = labels == c
        core = idx & (probs >= core_prob_thresh)
        border = idx & (probs < core_prob_thresh)
        color = palette[i % len(palette)]
        if np.any(border):
            fig.add_trace(go.Scatter3d(x=emb[border,0], y=emb[border,1], z=emb[border,2], mode='markers',
                name=f"cluster {c} (border)", marker=dict(size=2.5, color=color, opacity=0.5, symbol='circle')))
        if np.any(core):
            fig.add_trace(go.Scatter3d(x=emb[core,0], y=emb[core,1], z=emb[core,2], mode='markers',
                name=f"cluster {c} (core)", marker=dict(size=4, color=color, opacity=0.9, symbol='circle')))
    fig.update_layout(title=title, scene=dict(xaxis_title='UMAP-1', yaxis_title='UMAP-2', zaxis_title='UMAP-3'))
    fig.write_html(str(out_path))
    print(f"保存 3D 图: {out_path}")

# ---------------- 指标与工具 ----------------

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
    intra_sum = 0.0; intra_cnt = 0; centroids = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        Xi = X[idx]
        if len(idx) > 1:
            dists = pairwise_dists_metric(Xi, metric=metric)
            iu = np.triu_indices_from(dists, k=1)
            vals = dists[iu]
            intra_sum += float(vals.mean()); intra_cnt += 1
        mu = Xi.mean(axis=0)
        if metric == 'cosine':
            mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
    w_intra = intra_sum / max(1, intra_cnt)
    C = np.vstack(centroids)
    cd = pairwise_dists_metric(C, metric=metric)
    iu = np.triu_indices_from(cd, k=1)
    b_inter = float(cd[iu].mean()) if iu[0].size > 0 else float("nan")
    ratio = w_intra / (b_inter + 1e-8) if not np.isnan(w_intra) and not np.isnan(b_inter) else float("nan")
    return w_intra, b_inter, ratio


def kmeans_baseline_scores(X: np.ndarray, metric: str = 'cosine', k: int = 5, seed: int = 42) -> Tuple[np.ndarray, Dict[str, float]]:
    X_km = X if metric != 'cosine' else (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X_km)
    sil_metric = 'cosine' if metric == 'cosine' else 'euclidean'
    try:
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
    _, _, ratio = intra_inter_stats(X, labels, metric=metric)
    return labels, {
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "intra_over_inter": float(ratio),
        "n_clusters": int(len(np.unique(labels))),
        "noise_ratio": 0.0,
    }


def assign_noise_by_centroid(X: np.ndarray, labels: np.ndarray, metric: str = 'cosine') -> np.ndarray:
    # 将噪声点映射到最近质心
    uniq = sorted([u for u in np.unique(labels) if u != -1])
    if len(uniq) == 0:
        return labels.copy()
    mapped = labels.copy()
    centroids = []
    for c in uniq:
        idx = labels == c
        mu = X[idx].mean(axis=0)
        if metric == 'cosine':
            mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
    C = np.vstack(centroids)
    # 选择最近质心（cosine 下使用欧氏等价）
    for i in np.where(labels == -1)[0]:
        x = X[i]
        if metric == 'cosine':
            x = x / (np.linalg.norm(x) + 1e-12)
        d = np.sum((C - x[None, :]) ** 2, axis=1)
        mapped[i] = uniq[int(np.argmin(d))]
    return mapped


# ---------------- 主逻辑 ----------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--fig-dir", type=str, default=str(Path(__file__).parent.parent / "figure"))
    p.add_argument("--stats-dir", type=str, default=str(Path(__file__).parent.parent / "statistics"))
    p.add_argument("--agg", type=str, default="attn_norm", choices=list(AGG_FUNCS.keys()))
    p.add_argument("--metric", type=str, default="cosine", choices=["euclidean", "cosine"])  # 建议使用 cosine
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--min-cluster-size", type=int, nargs="+", default=[120,180,240,300])
    p.add_argument("--min-samples", type=int, nargs="+", default=[5,10,20,40])
    p.add_argument("--mapping-method", type=str, default="centroid", choices=["centroid"])  # 目前实现最近质心
    p.add_argument("--core-prob-thresh", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = Path(args.stats_dir); stats_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读数据 + 聚合
    series_list, names = load_segments(args.data_dir)
    if len(series_list) == 0:
        print("[ERR] 未找到任何 quantized_windows 数据，请检查 --data-dir")
        return
    X = features_by_agg(series_list, args.agg)
    print(f"特征形状: {X.shape} (agg={args.agg})")

    # 2) 标准化 +（可选）L2 归一化
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    if args.metric == 'cosine':
        Xs = normalize(Xs, norm='l2')

    # 3) KMeans(k=5) 基线（同口径，仅供对比）
    km_labels, km_metrics = kmeans_baseline_scores(Xs, metric=args.metric, k=5, seed=args.seed)

    # 4) HDBSCAN 网格
    try:
        import hdbscan  # type: ignore
    except Exception as e:
        print("[ERR] 需要安装 hdbscan 库：conda run -n laps pip install hdbscan | 错误:", e)
        return

    grid_rows = []
    best = {"silhouette_core": -np.inf}
    best_artifacts = {}

    for mcs in args.min_cluster_size:
        for ms in args.min_samples:
            print(f"\n=== HDBSCAN: min_cluster_size={mcs}, min_samples={ms} ===")
            # HDBSCAN: sklearn 的 BallTree 不支持 'cosine'，对 L2 归一化后的特征使用 'euclidean' 等价于余弦
            hdb_metric = 'euclidean' if args.metric == 'cosine' else args.metric
            clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric=hdb_metric,
                                        cluster_selection_method='eom', prediction_data=True)
            labels = clusterer.fit_predict(Xs)
            probs = getattr(clusterer, 'probabilities_', np.ones_like(labels, dtype=float))
            core_mask = labels != -1
            uniq_core = [u for u in np.unique(labels) if u != -1]
            n_clusters = len(uniq_core)
            noise_ratio = float(np.mean(labels == -1))

            # 口径A：仅核心点
            if n_clusters >= 2 and np.sum(core_mask) >= 2:
                sil_core = silhouette_score(Xs[core_mask], labels[core_mask], metric=('cosine' if args.metric=='cosine' else 'euclidean'))
                db_core = davies_bouldin_score(Xs[core_mask], labels[core_mask])
                ch_core = calinski_harabasz_score(Xs[core_mask], labels[core_mask])
                _, _, ratio_core = intra_inter_stats(Xs[core_mask], labels[core_mask], metric=args.metric)
            else:
                sil_core = float('nan'); db_core = float('nan'); ch_core = float('nan'); ratio_core = float('nan')

            # 口径B：映射噪声到最近质心
            labels_map = assign_noise_by_centroid(Xs, labels, metric=args.metric) if args.mapping_method=='centroid' else labels
            if len(np.unique(labels_map)) >= 2:
                sil_all = silhouette_score(Xs, labels_map, metric=('cosine' if args.metric=='cosine' else 'euclidean'))
                db_all = davies_bouldin_score(Xs, labels_map)
                ch_all = calinski_harabasz_score(Xs, labels_map)
                _, _, ratio_all = intra_inter_stats(Xs, labels_map, metric=args.metric)
            else:
                sil_all = float('nan'); db_all = float('nan'); ch_all = float('nan'); ratio_all = float('nan')

            row = {
                "agg": args.agg,
                "metric": args.metric,
                "min_cluster_size": mcs,
                "min_samples": ms,
                "silhouette_core": float(sil_core),
                "davies_bouldin_core": float(db_core),
                "calinski_harabasz_core": float(ch_core),
                "intra_over_inter_core": float(ratio_core),
                "noise_ratio": float(noise_ratio),
                "n_clusters_core": int(n_clusters),
                "silhouette_all": float(sil_all),
                "davies_bouldin_all": float(db_all),
                "calinski_harabasz_all": float(ch_all),
                "intra_over_inter_all": float(ratio_all),
                "mapping_method": args.mapping_method,
                "n_samples": int(Xs.shape[0]),
                "feature_dim": int(Xs.shape[1]),
            }
            grid_rows.append(row)

            # 选最佳（按口径A silhouette）
            score_for_best = sil_core if not np.isnan(sil_core) else -np.inf
            if score_for_best > best["silhouette_core"]:
                best = {k: row[k] for k in row}
                best_artifacts = {"labels": labels, "probs": probs}

    # 保存网格 CSV
    grid_csv = stats_dir / "cluster_metrics_hdbscan_grid.csv"
    with open(grid_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader(); writer.writerows(sorted(grid_rows, key=lambda r: (-(r['silhouette_core'] if not np.isnan(r['silhouette_core']) else -1e9))))
    print(f"保存网格指标: {grid_csv}")

    # 最佳配置可视化（UMAP）
    labels_best = best_artifacts["labels"]
    probs_best = best_artifacts["probs"]
    emb2 = umap_embed(Xs, n_components=2, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)
    emb3 = umap_embed(Xs, n_components=3, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)
    if emb2 is not None:
        plot_umap_2d_hdbscan(emb2, labels_best, probs_best, out_path=fig_dir/"umap_2d_hdbscan_best.png",
                             title=f"UMAP 2D - HDBSCAN best (mcs={best['min_cluster_size']}, ms={best['min_samples']})",
                             core_prob_thresh=args.core_prob_thresh)
    if emb3 is not None:
        plot_umap_3d_hdbscan(emb3, labels_best, probs_best, out_path=fig_dir/"umap_3d_hdbscan_best.html",
                             title=f"UMAP 3D - HDBSCAN best (mcs={best['min_cluster_size']}, ms={best['min_samples']})",
                             core_prob_thresh=args.core_prob_thresh)

    # 基线 vs 最佳对比表
    labels_map_best = assign_noise_by_centroid(Xs, labels_best, metric=args.metric)
    _, _, ratio_map = intra_inter_stats(Xs, labels_map_best, metric=args.metric)
    sil_map = silhouette_score(Xs, labels_map_best, metric=('cosine' if args.metric=='cosine' else 'euclidean')) if len(np.unique(labels_map_best))>=2 else float('nan')
    db_map = davies_bouldin_score(Xs, labels_map_best) if len(np.unique(labels_map_best))>=2 else float('nan')
    ch_map = calinski_harabasz_score(Xs, labels_map_best) if len(np.unique(labels_map_best))>=2 else float('nan')

    comp_csv = stats_dir / "cluster_metrics_hdbscan_best_comparison.csv"
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "silhouette", "davies_bouldin", "calinski_harabasz", "intra_over_inter", "n_clusters", "noise_ratio"])
        writer.writerow(["KMeans_k5", km_metrics.get("silhouette"), km_metrics.get("davies_bouldin"), km_metrics.get("calinski_harabasz"), km_metrics.get("intra_over_inter"), km_metrics.get("n_clusters"), km_metrics.get("noise_ratio")])
        writer.writerow(["HDBSCAN_core", best.get("silhouette_core"), best.get("davies_bouldin_core"), best.get("calinski_harabasz_core"), best.get("intra_over_inter_core"), best.get("n_clusters_core"), best.get("noise_ratio")])
        writer.writerow(["HDBSCAN_all_mapped", sil_map, db_map, ch_map, ratio_map, best.get("n_clusters_core"), 0.0])
    print(f"保存对比指标: {comp_csv}")

    # 报告
    doc = Path(__file__).parent.parent / "docs" / "HDBSCAN_EVALUATION.md"
    (Path(__file__).parent.parent / "docs").mkdir(exist_ok=True)
    with open(doc, "w", encoding="utf-8") as f:
        f.write("# HDBSCAN 密度聚类评估\n\n")
        f.write(f"- 数据源聚合: {args.agg}，度量: {args.metric}\n")
        f.write(f"- 网格搜索: min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}\n")
        f.write(f"- 结果 CSV: {grid_csv}\n")
        f.write(f"- 最佳可视化: {fig_dir/'umap_2d_hdbscan_best.png'}, {fig_dir/'umap_3d_hdbscan_best.html'}\n\n")
        f.write("## 表格1：HDBSCAN 网格（口径A - 仅核心点，按 Silhouette 降序）\n\n")
        # 写前若干行预览
        top = sorted(grid_rows, key=lambda r: (-(r['silhouette_core'] if not np.isnan(r['silhouette_core']) else -1e9)))[:10]
        f.write("| min_cluster_size | min_samples | sil_core | db_core | ch_core | intra/inter_core | noise_ratio | n_clusters |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in top:
            f.write(f"| {r['min_cluster_size']} | {r['min_samples']} | {r['silhouette_core']:.3f} | {r['davies_bouldin_core']:.3f} | {r['calinski_harabasz_core']:.1f} | {r['intra_over_inter_core']:.3f} | {r['noise_ratio']:.3f} | {r['n_clusters_core']} |\n")
        f.write("\n## 表格2：最佳 HDBSCAN vs KMeans(k=5)\n\n")
        f.write("| 方法 | Silhouette | DB | CH | Intra/Inter | 簇数量 | 噪声比例 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        f.write(f"| KMeans 基线 | {km_metrics['silhouette']:.3f} | {km_metrics['davies_bouldin']:.3f} | {km_metrics['calinski_harabasz']:.1f} | {km_metrics['intra_over_inter']:.3f} | {km_metrics['n_clusters']} | 0.000 |\n")
        f.write(f"| HDBSCAN 核心 | {best['silhouette_core']:.3f} | {best['davies_bouldin_core']:.3f} | {best['calinski_harabasz_core']:.1f} | {best['intra_over_inter_core']:.3f} | {best['n_clusters_core']} | {best['noise_ratio']:.3f} |\n")
        f.write(f"| HDBSCAN 映射 | {sil_map:.3f} | {db_map:.3f} | {ch_map:.1f} | {ratio_map:.3f} | {best['n_clusters_core']} | 0.000 |\n")
        # 结论部分由运行后人工/自动增补
        f.write("\n> 注：口径A为核心点，仅在 labels!=-1 子集上计算；口径B将噪声映射到最近质心后在全样本上计算。\n")
    print(f"保存报告: {doc}")

    print("\n完成。输出目录:\n  - 图: ", fig_dir, "\n  - 指标: ", stats_dir)


if __name__ == "__main__":
    main()

