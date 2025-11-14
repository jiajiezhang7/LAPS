#!/usr/bin/env python3
"""
从已缓存的特征与标签生成 UMAP 可视化（2D PNG + 3D HTML）。
建议在 base 环境运行（安装了 umap-learn 与 plotly）。

输入：
  --features <.npy>    通过序列模型提取并缓存的特征 (N, D)
  --labels <.npy>      KMeans 聚类标签 (N,)
  --out-dir <dir>      输出目录，对齐 figures/<label>/
  --neighbors <int>    UMAP 邻居数（默认 15）
  --min-dist <float>   UMAP min_dist（默认 0.1）
  --metric <str>       特征度量（默认 cosine，可选 euclidean/cosine）
  --title <str>        标题（可选）

输出：
  <out-dir>/2d_figure/umap_2d_seq_model_best.png
  <out-dir>/umap_3d_seq_model_best.html
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def umap_embed(X: np.ndarray, n_components: int, n_neighbors: int, min_dist: float, metric: str = "euclidean"):
    umap_mod = None
    try:
        import umap as _umap
        umap_mod = _umap
    except Exception as e_import:
        print(f"[WARN] 需要安装 umap-learn: pip install umap-learn | 错误: {e_import}")
        umap_mod = None

    if umap_mod is not None:
        try:
            reducer = umap_mod.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
            )
            return reducer.fit_transform(X)
        except Exception as e1:
            print(f"[WARN] UMAP(metric={metric}) 失败: {e1}")

    try:
        from sklearn.manifold import TSNE
        perplexity = max(5, min(50, n_neighbors * 2, X.shape[0] // 3 if X.shape[0] > 3 else 5))
        return TSNE(n_components=n_components, random_state=42, perplexity=perplexity, init="pca").fit_transform(X)
    except Exception as e3:
        print(f"[WARN] TSNE 失败: {e3}")

    try:
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=42).fit_transform(X)
    except Exception as e4:
        print(f"[ERROR] UMAP/TSNE/PCA 全部失败: {e4}")
        return None


def plot_umap_2d(emb: np.ndarray, labels: np.ndarray, title: str, out_path: Path):
    plt.figure(figsize=(10, 8))
    c = np.arange(len(emb)) if labels is None else labels
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=c, cmap="tab20", s=36, alpha=0.75, edgecolors="none")
    plt.title(title, fontsize=12)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(sc, label=("Cluster" if labels is not None else "Index"))
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"[WARN] 紧凑布局失败: {e}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"[SAVE] 2D 图: {out_path}")


def plot_umap_3d(emb: np.ndarray, labels: np.ndarray, title: str, out_path: Path):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print(f"[WARN] 需要安装 plotly: pip install plotly | 错误: {e}")
        return
    c = np.arange(len(emb)) if labels is None else labels
    fig = go.Figure(data=[go.Scatter3d(
        x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
        mode='markers', marker=dict(size=5, color=c, colorscale='Viridis', showscale=True,
                                    line=dict(color='rgba(0,0,0,0)', width=0))
    )])
    fig.update_layout(title=title, scene=dict(xaxis_title='UMAP-1', yaxis_title='UMAP-2', zaxis_title='UMAP-3'))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"[SAVE] 3D 图: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--labels", type=str, required=False, default=None)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--metric", type=str, default="cosine", choices=["euclidean", "cosine"])
    ap.add_argument("--title", type=str, default="UMAP - seq_model")
    args = ap.parse_args()

    X = np.load(args.features)
    y = None
    if args.labels:
        try:
            y = np.load(args.labels)
        except Exception as e:
            print(f"[WARN] 读取标签失败，将不着色: {e}")
            y = None

    out_dir = Path(args.out_dir)
    print(f"[INFO] X shape={X.shape}, labels={None if y is None else y.shape}")

    emb2 = umap_embed(X, n_components=2, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)
    if emb2 is not None:
        plot_umap_2d(emb2, y, title=args.title, out_path=out_dir / "2d_figure/umap_2d_seq_model_best.png")

    emb3 = umap_embed(X, n_components=3, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)
    if emb3 is not None:
        plot_umap_3d(emb3, y, title=args.title, out_path=out_dir / "umap_3d_seq_model_best.html")


if __name__ == "__main__":
    main()

