#!/usr/bin/env python3
"""
基于 DTW 的片段级 UMAP 对比分析
- 从 JSON 读取每个片段的 quantized_windows 时序向量
- 将每段序列降到 1D（默认按时间步 L2 范数）并做 PAA 降采样
- 计算子集样本两两 DTW 距离矩阵（优先 fastdtw，加速）
- 基于预计算距离做 UMAP 可视化（metric='precomputed'）
- 用层次聚类（Agglomerative，average linkage，precomputed）在多 k 上打标签
- 计算 silhouette（基于预计算距离）与 类内/类间 平均距离比值

注意：
- 完整 4000+ 样本做 DTW 距离矩阵非常耗时，默认随机采样子集
- 如需全量，请谨慎设置 --sample-size，并准备长时间运行
"""

from __future__ import annotations

import os
import csv
import json
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


# ---------- 数据读取（复用聚类脚本逻辑） ----------

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


# ---------- 序列降维与下采样 ----------

def series_to_1d(seq2d: np.ndarray, method: str = "l2") -> np.ndarray:
    """将 (T, D) 降到 1D 时间序列。
    method:
      - "l2": 每步向量的 L2 范数
      - "mean": 每步向量的均值
    """
    if method == "l2":
        return np.linalg.norm(seq2d, ord=2, axis=1)
    elif method == "mean":
        return seq2d.mean(axis=1)
    else:
        raise ValueError(f"未知 method: {method}")


def paa_reduce(ts: np.ndarray, L: int) -> np.ndarray:
    """Piecewise Aggregate Approximation: 将时间序列长度压缩到 L。
    采用等宽分段求均值。
    """
    n = len(ts)
    if n == 0:
        return ts
    if L >= n:
        return ts.astype(np.float32)
    # 将索引均匀映射到 L 个段
    idx = (np.arange(n) * L) // n
    out = np.zeros(L, dtype=np.float32)
    cnt = np.zeros(L, dtype=np.int32)
    for i in range(n):
        j = idx[i]
        out[j] += ts[i]
        cnt[j] += 1
    cnt = np.maximum(cnt, 1)
    out = out / cnt
    return out


# ---------- 距离矩阵（DTW） ----------

def dtw_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """纯 Python DTW（无窗口约束）。如安装 fastdtw 将优先使用。"""
    try:
        from fastdtw import fastdtw  # type: ignore
        dist, _ = fastdtw(ts1, ts2)  # 基于 L1 距离，足够用于相对比较
        return float(dist)
    except Exception:
        # 退化到经典 DP（O(n*m)），仅在小 L 和小样本数时可接受
        n, m = len(ts1), len(ts2)
        dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(ts1[i - 1] - ts2[j - 1])
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(dp[n, m])


def pairwise_dtw_matrix(series: List[np.ndarray]) -> np.ndarray:
    N = len(series)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        D[i, i] = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = dtw_distance(series[i], series[j])
            D[i, j] = D[j, i] = d
        if (i + 1) % 20 == 0:
            print(f"  DTW 进度: {i+1}/{N}")
    return D


# ---------- 指标 ----------

def intra_inter_from_distmat(D: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan"), float("nan"), float("nan")
    # 类内（均值的均值）
    intra_vals = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        if len(idx) <= 1:
            continue
        sub = D[np.ix_(idx, idx)]
        iu = np.triu_indices_from(sub, k=1)
        if iu[0].size > 0:
            intra_vals.append(float(sub[iu].mean()))
    w_intra = float(np.mean(intra_vals)) if len(intra_vals) > 0 else float("nan")
    # 类间（所有簇对的均值）
    inter_vals = []
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            ia = np.where(labels == uniq[i])[0]
            ib = np.where(labels == uniq[j])[0]
            sub = D[np.ix_(ia, ib)]
            if sub.size > 0:
                inter_vals.append(float(sub.mean()))
    b_inter = float(np.mean(inter_vals)) if len(inter_vals) > 0 else float("nan")
    ratio = w_intra / (b_inter + 1e-8) if not (np.isnan(w_intra) or np.isnan(b_inter)) else float("nan")
    return w_intra, b_inter, ratio


# ---------- UMAP 可视化（预计算距离） ----------

def umap_embed_precomputed(D: np.ndarray, n_components: int, n_neighbors: int, min_dist: float) -> Optional[np.ndarray]:
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
        metric="precomputed",
    )
    return reducer.fit_transform(D)


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


# ---------- 主流程 ----------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).parent.parent / "figure"))
    p.add_argument("--sample-size", type=int, default=250, help="DTW 对比的随机子集样本数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paa", type=int, default=64, help="PAA 目标长度（每段序列压缩到该长度）")
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--k-max", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series_list, names = load_segments(args.data_dir)
    if len(series_list) == 0:
        print("[ERR] 未找到任何 quantized_windows 数据")
        return
    print(f"总片段数: {len(series_list)}，将进行随机采样: {args.sample_size}")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(series_list))
    if args.sample_size < len(series_list):
        idx = rng.choice(idx, size=args.sample_size, replace=False)
    # 1D + PAA
    seq_1d = []
    sel_names = []
    for i in idx:
        ts = series_to_1d(series_list[i], method="l2")
        ts_paa = paa_reduce(ts, L=args.paa)
        # 标准化到零均值单位方差，避免幅值差异主导 DTW
        ts_paa = (ts_paa - ts_paa.mean()) / (ts_paa.std() + 1e-8)
        seq_1d.append(ts_paa.astype(np.float32))
        sel_names.append(names[i])

    print("开始计算 DTW 距离矩阵（可能耗时）...")
    D = pairwise_dtw_matrix(seq_1d)
    np.save(out_dir / "dtw_distance_matrix.npy", D)
    print("保存 DTW 距离矩阵: ", out_dir / "dtw_distance_matrix.npy")

    # UMAP
    emb2 = umap_embed_precomputed(D, n_components=2, n_neighbors=args.neighbors, min_dist=args.min_dist)
    emb3 = umap_embed_precomputed(D, n_components=3, n_neighbors=args.neighbors, min_dist=args.min_dist)

    # 聚类与指标
    results: Dict[int, Dict[str, float]] = {}
    best_k = None
    best_score = -1.0
    best_labels = None

    for k in range(args.k_min, args.k_max + 1):
        try:
            try:
                clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
            except Exception:
                clustering = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
            labels = clustering.fit_predict(D)
        except Exception as e:
            print(f"[WARN] 聚类失败 k={k}: {e}")
            continue
        # silhouette（基于预计算距离）
        try:
            sil = silhouette_score(D, labels, metric='precomputed')
        except Exception:
            sil = float('nan')
        w_intra, b_inter, ratio = intra_inter_from_distmat(D, labels)
        results[k] = {
            "silhouette": float(sil),
            "davies_bouldin": float('nan'),  # 对纯距离矩阵不直接适用
            "calinski_harabasz": float('nan'),
            "intra_dist": float(w_intra),
            "inter_centroid_dist": float(b_inter),  # 这里是类间平均距离
            "intra_over_inter": float(ratio),
        }
        if not np.isnan(sil) and sil > best_score:
            best_score = sil
            best_k = k
            best_labels = labels

    # 可视化
    if emb2 is not None and best_labels is not None:
        plot_umap_2d(emb2, best_labels, title=f"UMAP(Precomputed DTW) 2D - k={best_k}", out_path=out_dir / f"umap_2d_dtw_k{best_k}.png")
    if emb3 is not None and best_labels is not None:
        plot_umap_3d(emb3, best_labels, title=f"UMAP(Precomputed DTW) 3D - k={best_k}", out_path=out_dir / f"umap_3d_dtw_k{best_k}.html")

    # 保存指标
    csv_path = out_dir / "cluster_metrics_dtw.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["agg", "k", "n_samples", "feature_dim", "silhouette", "davies_bouldin", "calinski_harabasz", "intra_dist", "inter_centroid_dist", "intra_over_inter"])
        for k, m in sorted(results.items()):
            writer.writerow(["dtw_1d_paa", k, len(seq_1d), args.paa, m.get("silhouette", np.nan), m.get("davies_bouldin", np.nan), m.get("calinski_harabasz", np.nan), m.get("intra_dist", np.nan), m.get("inter_centroid_dist", np.nan), m.get("intra_over_inter", np.nan)])
    print(f"保存指标: {csv_path}")

    print("完成。输出目录:", out_dir)


if __name__ == "__main__":
    main()

