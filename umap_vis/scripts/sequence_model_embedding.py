#!/usr/bin/env python3
"""
轻量级序列模型提取段级 embedding 并评估聚类质量
- 读取 quantized_windows (T, 768)
- 轻量 Transformer Encoder（随机初始化，冻结推理）提取段级向量
- 与 attn_norm 基线对比（UMAP+KMeans 指标）

运行示例：
    conda run -n laps python umap_vis/scripts/sequence_model_embedding.py \
        --data-dir /path/to/output_root \
        --fig-dir umap_vis/figure --stats-dir umap_vis/statistics \
        --metric cosine --neighbors 15 --min-dist 0.1 \
        --d-model 256 --n-layers 2 --n-heads 4 --pooling mean

依赖：numpy, scikit-learn, matplotlib, plotly, umap-learn, torch
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


# ---------------- 数据加载 ----------------

def load_segments(data_dir: str) -> Tuple[List[np.ndarray], List[str], List[Dict]]:
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
                "num_vectors": int(arr.shape[0]),
                "vector_dim": int(arr.shape[1]),
            })
        except Exception as e:
            print(f"[WARN] 读取失败 {jf}: {e}")
    return series_list, names, metas


# ---------------- 基线聚合 ----------------

def agg_attn_norm(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, ord=2, axis=1)
    if np.allclose(norms.sum(), 0.0):
        return np.mean(x, axis=0)
    w = norms / (norms.sum() + 1e-8)
    return (x * w[:, None]).sum(axis=0)


def features_by_agg(series_list: List[np.ndarray], agg: str) -> np.ndarray:
    if agg == "attn_norm":
        feats = [agg_attn_norm(s) for s in series_list]
    elif agg == "mean":
        feats = [np.mean(s, axis=0) for s in series_list]
    else:
        raise ValueError(f"未知聚合: {agg}")
    return np.asarray(feats, dtype=np.float32)


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
    """
    生成论文级别的 UMAP 2D 可视化
    - 高分辨率输出（300 DPI）
    - 专业配色方案
    - 优化的字体和布局
    - 清晰的簇边界
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    
    c = np.arange(len(emb)) if labels is None else labels
    
    # 使用高质量配色方案（适合论文）
    if labels is not None and len(np.unique(labels)) <= 10:
        # 离散簇：使用专业的离散色板
        cmap = plt.cm.get_cmap('tab10')
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(emb[mask, 0], emb[mask, 1], 
                      c=[cmap(label % 10)], 
                      s=80, alpha=0.7, 
                      edgecolors='black', linewidth=0.5,
                      label=f'Cluster {int(label)}',
                      rasterized=True)
        ax.legend(loc='best', fontsize=28, framealpha=0.95, edgecolor='black', 
                 title_fontsize=28, frameon=True, fancybox=False)
    else:
        # 连续标签：使用连续色板
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap="viridis", 
                       s=80, alpha=0.7, edgecolors='black', linewidth=0.5,
                       rasterized=True)
        cbar = plt.colorbar(sc, ax=ax, label=("Cluster ID" if labels is not None else "Sample Index"),
                           pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=12)
    
    # 轴标签（不显示标题）
    ax.set_xlabel("UMAP Dimension 1", fontsize=16, fontweight='bold')
    ax.set_ylabel("UMAP Dimension 2", fontsize=16, fontweight='bold')
    
    # 优化网格和脊
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # 优化刻度（增大字体）
    ax.tick_params(axis='both', which='major', labelsize=13, width=1.5, length=6)
    
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"保存 2D 图 (300 DPI): {out_path}")


def plot_umap_3d(emb: np.ndarray, labels: Optional[np.ndarray], title: str, out_path: Path):
    """
    生成论文级别的 UMAP 3D 交互式可视化
    - 专业的配色方案
    - 优化的标记大小和透明度
    - 清晰的轴标签和标题
    - 高质量的 HTML 输出
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as e:
        print("[WARN] 需要安装 plotly: pip install plotly | 错误:", e)
        return
    
    c = np.arange(len(emb)) if labels is None else labels
    
    # 定义专业的离散色板（适合论文）
    discrete_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # 如果是离散簇标签，使用离散色板
    if labels is not None and len(np.unique(labels)) <= 10:
        unique_labels = sorted(np.unique(labels))
        traces = []
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            color = discrete_colors[idx % len(discrete_colors)]
            traces.append(go.Scatter3d(
                x=emb[mask, 0], y=emb[mask, 1], z=emb[mask, 2],
                mode='markers',
                name=f'Cluster {int(label)}',
                marker=dict(
                    size=6,
                    color=color,
                    opacity=0.8,
                    line=dict(color='rgba(0,0,0,0.3)', width=0.5)
                ),
                text=[f'Cluster {int(label)}'] * mask.sum(),
                hovertemplate='<b>Cluster %{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            ))
        fig = go.Figure(data=traces)
    else:
        # 连续标签：使用连续色板
        fig = go.Figure(data=[go.Scatter3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=c,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Cluster ID" if labels is not None else "Sample Index",
                    thickness=15,
                    len=0.7,
                    x=1.02
                ),
                opacity=0.8,
                line=dict(color='rgba(0,0,0,0.3)', width=0.5)
            ),
            hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
        )])
    
    # 优化布局（删除标题，增大字体）
    fig.update_layout(
        title=None,
        scene=dict(
            xaxis=dict(
                title=dict(text='UMAP Dimension 1', font=dict(size=16, family='Arial, sans-serif', color='#000000')),
                showgrid=True,
                gridwidth=1,
                gridcolor='#e0e0e0',
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor='#000000',
                tickfont=dict(size=13, family='Arial, sans-serif')
            ),
            yaxis=dict(
                title=dict(text='UMAP Dimension 2', font=dict(size=16, family='Arial, sans-serif', color='#000000')),
                showgrid=True,
                gridwidth=1,
                gridcolor='#e0e0e0',
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor='#000000',
                tickfont=dict(size=13, family='Arial, sans-serif')
            ),
            zaxis=dict(
                title=dict(text='UMAP Dimension 3', font=dict(size=16, family='Arial, sans-serif', color='#000000')),
                showgrid=True,
                gridwidth=1,
                gridcolor='#e0e0e0',
                zeroline=False,
                showline=True,
                linewidth=2,
                linecolor='#000000',
                tickfont=dict(size=13, family='Arial, sans-serif')
            ),
            bgcolor='rgba(240, 240, 240, 0.9)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1000,
        height=900,
        font=dict(family='Arial, sans-serif', size=13, color='#000000'),
        paper_bgcolor='white',
        plot_bgcolor='rgba(240, 240, 240, 0.9)',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#000000',
            borderwidth=1,
            font=dict(size=26, family='Arial, sans-serif')
        )
    )
    
    fig.write_html(str(out_path))
    print(f"保存 3D 图 (高质量 HTML): {out_path}")


# ---------------- 距离/指标 ----------------

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


def cluster_and_scores(X: np.ndarray, k_min: int, k_max: int, random_state: int = 42, metric: str = 'euclidean') -> Tuple[Dict[int, Dict[str, float]], int, np.ndarray]:
    n = X.shape[0]
    if n < 3:
        print("[WARN] 样本过少，跳过聚类评估")
        return {}, 0, np.zeros(n, dtype=int)
    k_max = max(2, min(k_max, n - 1)); k_min = max(2, min(k_min, k_max))
    results: Dict[int, Dict[str, float]] = {}; best_k = None; best_score = -1.0; best_labels = None
    X_km = X if metric != 'cosine' else X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
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
            best_score = sil; best_k = k; best_labels = labels
    if best_k is None:
        best_k = k_min
        best_labels = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit_predict(X_km)
    return results, int(best_k), best_labels


# ---------------- 轻量 Transformer ----------------

def ensure_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except Exception as e:
        raise RuntimeError("未检测到 PyTorch，请先在 laps 环境安装：conda install pytorch -c pytorch | 错误: %s" % e)


def build_positional_encoding(d_model: int, max_len: int = 4096) -> np.ndarray:
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe  # (L, d)


class TinyTransformer:
    def __init__(self, in_dim: int = 768, d_model: int = 256, n_heads: int = 4, n_layers: int = 2, dim_ff: int = 512, dropout: float = 0.1, pooling: str = "mean", device: str = "cpu"):
        ensure_torch()
        import torch
        import torch.nn as nn
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.torch = torch
        self.nn = nn
        self.device = torch.device(device)
        self.pooling = pooling
        self.use_cls = (pooling == "cls")
        self.proj = nn.Linear(in_dim, d_model)
        self.pe = build_positional_encoding(d_model)
        enc_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, dropout=dropout, batch_first=False, activation="gelu")
        self.encoder = TransformerEncoder(enc_layer, num_layers=n_layers)
        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls = None
        if self.pooling == "attn":
            self.attn_query = self.nn.Parameter(torch.randn(d_model))
        else:
            self.attn_query = None
        # 转设备
        for m in [self.proj, self.encoder]:
            m.to(self.device)
        if self.cls is not None:
            self.cls = self.cls.to(self.device)
        if self.attn_query is not None:
            self.attn_query = self.attn_query.to(self.device)
        # eval & no grad
        for p in self.proj.parameters(): p.requires_grad_(False)
        for p in self.encoder.parameters(): p.requires_grad_(False)
        if self.cls is not None: self.cls.requires_grad_(False)
        if self.attn_query is not None: self.attn_query.requires_grad_(False)

    def _add_pos(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (T, 1, d)
        T = x.size(0)
        pe = self.torch.as_tensor(self.pe[:T], dtype=x.dtype, device=x.device).unsqueeze(1)  # (T,1,d)
        return x + pe

    def encode_one(self, seq: np.ndarray) -> np.ndarray:
        torch = self.torch; nn = self.nn
        x = torch.from_numpy(seq).to(self.device)  # (T,in)
        with torch.no_grad():
            z = self.proj(x)  # (T,d)
            z = z.unsqueeze(1)  # (T,1,d)
            if self.use_cls:
                cls = self.cls.expand(1, 1, -1)  # (1,1,d)
                z = torch.cat([cls, z], dim=0)  # (T+1,1,d)
            z = self._add_pos(z)
            z = self.encoder(z)  # (S,1,d)
            if self.pooling == "mean":
                pooled = z.mean(dim=0).squeeze(0)  # (d,)
            elif self.pooling == "cls":
                pooled = z[0, 0, :]
            elif self.pooling == "attn":
                seqz = z[1:, 0, :] if self.use_cls else z[:, 0, :]
                w = self.torch.softmax(seqz @ self.attn_query, dim=0)  # (T,)
                pooled = (w.unsqueeze(1) * seqz).sum(dim=0)
            else:
                pooled = z.mean(dim=0).squeeze(0)
            return pooled.detach().cpu().numpy().astype(np.float32)


def extract_seq_embeddings(series_list: List[np.ndarray], d_model: int, n_layers: int, n_heads: int, pooling: str, device: str = "cpu") -> np.ndarray:
    model = TinyTransformer(in_dim=series_list[0].shape[1], d_model=d_model, n_heads=n_heads, n_layers=n_layers, pooling=pooling, device=device)
    embs = [model.encode_one(seq) for seq in series_list]
    return np.asarray(embs, dtype=np.float32)


# ---------------- 视频采样导出（验证聚类结果） ----------------
from typing import Iterable
import shutil

ALLOWED_VIDEO_EXTS = [".mp4", ".avi", ".mkv", ".mov"]


def _stem_without_codes(p: Path) -> str:
    base = p.stem  # e.g., foo.codes
    if base.endswith(".codes"):
        base = base[: -len(".codes")]
    return base


def _resolve_video_for_json(json_path: Path) -> Optional[Path]:
    """根据 JSON 路径，在相邻目录 segmented_videos 中查找同名视频（支持多种扩展名）。"""
    try:
        vdir = json_path.parent.parent / "segmented_videos"
        base = _stem_without_codes(json_path)
        # 先按常见扩展名精确匹配
        for ext in ALLOWED_VIDEO_EXTS:
            cand = vdir / f"{base}{ext}"
            if cand.exists():
                return cand
        # 兜底：glob 同名的任意扩展名
        for p in vdir.glob(f"{base}.*"):
            if p.is_file():
                return p
    except Exception as e:
        print(f"[WARN] 解析视频路径失败: {json_path} -> {e}")
    return None


def _unique_target(dst_dir: Path, filename: str) -> Path:
    """若存在重名文件，在文件名后追加 _1/_2 序号。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = dst_dir / f"{base}{ext}"
    if not candidate.exists():
        return candidate
    n = 1
    while True:
        candidate = dst_dir / f"{base}_{n}{ext}"
        if not candidate.exists():
            return candidate
        n += 1


def export_video_samples(labels: np.ndarray, metas: List[Dict], out_root: Path, max_per_cluster: int = 100, seed: int = 42) -> List[Dict]:
    """
    将聚类标签映射回原始 JSON，查找相邻 segmented_videos 下的同名视频，并按簇采样导出。
    返回每个簇的摘要行：cluster_id/total_samples/exported_samples/export_dir。
    """
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    summary: List[Dict] = []
    uniq = sorted([int(u) for u in np.unique(labels)])
    for cid in uniq:
        idxs = np.where(labels == cid)[0]
        idxs = idxs.tolist()
        total = len(idxs)
        if total == 0:
            continue
        # 采样索引
        if total > max_per_cluster:
            sampled = rng.choice(idxs, size=max_per_cluster, replace=False).tolist()
        else:
            sampled = idxs
        dst_dir = out_root / f"cluster_{cid}"
        exported = 0
        for i in sampled:
            jpath = Path(metas[i]["json_path"]) if isinstance(metas[i].get("json_path"), str) else None
            if jpath is None:
                print(f"[WARN] 缺少 json_path 元数据，索引={i}")
                continue
            vpath = _resolve_video_for_json(jpath)
            if vpath is None:
                print(f"[WARN] 未找到对应视频：{jpath}")
                continue
            try:
                target = _unique_target(dst_dir, vpath.name)
                shutil.copy2(str(vpath), str(target))
                exported += 1
            except Exception as e:
                print(f"[WARN] 复制失败 {vpath} -> {dst_dir}: {e}")
        print(f"[EXPORT] cluster {cid}: 总样本={total}, 采样={len(sampled)}, 成功导出={exported}, 目录={dst_dir}")
        summary.append({
            "cluster_id": int(cid),
            "total_samples": int(total),
            "exported_samples": int(exported),
            "export_dir": str(dst_dir),
        })
    return summary


# ---------------- 主流程 ----------------

def save_metrics_csv(path: Path, method: str, results: Dict[int, Dict[str, float]], n_samples: int, dim: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "k", "n_samples", "feature_dim", "silhouette", "davies_bouldin", "calinski_harabasz", "intra_dist", "inter_centroid_dist", "intra_over_inter"])
        for k, m in sorted(results.items()):
            writer.writerow([method, k, n_samples, dim, m.get("silhouette", np.nan), m.get("davies_bouldin", np.nan), m.get("calinski_harabasz", np.nan), m.get("intra_dist", np.nan), m.get("inter_centroid_dist", np.nan), m.get("intra_over_inter", np.nan)])
    print(f"保存指标: {path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--fig-dir", type=str, default=str(Path(__file__).parent.parent / "figure"))
    p.add_argument("--stats-dir", type=str, default=str(Path(__file__).parent.parent / "statistics"))
    p.add_argument("--metric", type=str, default="cosine", choices=["euclidean", "cosine"])
    p.add_argument("--neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    # 模式开关
    p.add_argument("--grid-search", action="store_true", help="运行冻结推理微网格搜索")
    p.add_argument("--use-best-grid-config", action="store_true", help="使用网格全局最佳配置 (mean/256/4/4) 作为默认序列模型配置")
    p.add_argument("--k-analysis-max", type=int, default=15, help="最佳配置下的 k 扩展上限，用于曲线分析")
    # 序列模型参数（允许覆盖默认最佳配置）
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls", "attn"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--export-video-samples", action="store_true", help="在 --use-best-grid-config 模式下，完成 k=3 聚类与可视化后，将每簇最多100个视频片段复制到 /home/johnny/action_ws/classify_res")

    args = p.parse_args()

    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = Path(args.stats_dir); stats_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载数据
    series_list, names, metas = load_segments(args.data_dir)
    if len(series_list) == 0:
        print("[ERR] 未找到任何 quantized_windows 数据")
        return
    print(f"片段数: {len(series_list)}, 向量维度: {series_list[0].shape[1]}")

    # 2) 基线特征（attn_norm）与评估（始终输出，便于对照）
    X_base = features_by_agg(series_list, "attn_norm")
    scaler_b = StandardScaler()
    Xb = scaler_b.fit_transform(X_base)
    if args.metric == 'cosine':
        Xb = normalize(Xb, norm='l2')
    res_base, kb_base, labels_base = cluster_and_scores(Xb, args.k_min, args.k_max, metric=args.metric)
    save_metrics_csv(stats_dir / f"cluster_metrics_attn_norm_{args.metric}.csv", "attn_norm", res_base, n_samples=Xb.shape[0], dim=Xb.shape[1])

    # ========== 分支A：网格搜索（按需） ==========
    if args.grid_search:
        grid_pooling = ["mean", "cls", "attn"]
        grid_d_model = [256, 512]
        grid_n_layers = [2, 4]
        grid_rows: List[List] = []
        best = None  # (metrics, cfg_dict, best_k, labels, Xs)
        cfg_id = 0
        def cfg_better(m_new: Dict[str, float], cfg_new: Dict, m_old: Optional[Dict[str, float]], cfg_old: Optional[Dict]):
            if m_old is None:
                return True
            s_new, s_old = m_new.get("silhouette", float("nan")), m_old.get("silhouette", float("nan"))
            if np.isnan(s_new):
                return False
            if np.isnan(s_old):
                return True
            if s_new > s_old + 1e-12:
                return True
            if abs(s_new - s_old) < 0.005:
                r_new, r_old = m_new.get("intra_over_inter", float("inf")), m_old.get("intra_over_inter", float("inf"))
                if not np.isnan(r_new) and not np.isnan(r_old):
                    if r_new < r_old - 1e-12:
                        return True
                    if abs(r_new - r_old) < 1e-6:
                        if cfg_new["d_model"] < cfg_old["d_model"]:
                            return True
                        if cfg_new["d_model"] == cfg_old["d_model"] and cfg_new["n_layers"] < cfg_old["n_layers"]:
                            return True
            return False
        for pooling in grid_pooling:
            for d_model in grid_d_model:
                n_heads = 4 if d_model == 256 else 8
                for n_layers in grid_n_layers:
                    cfg_id += 1
                    print(f"[GRID] cfg#{cfg_id}: pool={pooling}, d={d_model}, L={n_layers}, H={n_heads}")
                    try:
                        X_seq = extract_seq_embeddings(series_list, d_model=d_model, n_layers=n_layers, n_heads=n_heads, pooling=pooling, device=args.device)
                    except RuntimeError as e:
                        print("[ERR] ", e); return
                    Xs = StandardScaler().fit_transform(X_seq)
                    if args.metric == 'cosine':
                        Xs = normalize(Xs, norm='l2')
                    res_seq, k_seq, labels_seq = cluster_and_scores(Xs, args.k_min, args.k_max, metric=args.metric)
                    m = res_seq.get(k_seq, {})
                    grid_rows.append([cfg_id, pooling, d_model, n_layers, n_heads, k_seq, m.get("silhouette", np.nan), m.get("davies_bouldin", np.nan), m.get("calinski_harabasz", np.nan), m.get("intra_over_inter", np.nan)])
                    cfg = {"pooling": pooling, "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads}
                    if cfg_better(m, cfg, (best[0] if best else None), (best[1] if best else None)):
                        best = (m, cfg, k_seq, labels_seq, Xs)
        grid_csv = stats_dir / "sequence_model_grid_search.csv"
        with open(grid_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["config_id", "pooling", "d_model", "n_layers", "n_heads", "best_k", "silhouette", "davies_bouldin", "calinski_harabasz", "intra_over_inter"])
            for row in grid_rows: w.writerow(row)
        print(f"保存网格汇总: {grid_csv}")
        # 阶段1基准
        phase1 = None
        for row in grid_rows:
            _, p0, d0, L0, H0, bk0, sil0, db0, ch0, r0 = row
            if p0 == "mean" and d0 == 256 and L0 == 2 and H0 == 4:
                phase1 = {"k": bk0, "silhouette": sil0, "davies_bouldin": db0, "calinski_harabasz": ch0, "intra_over_inter": r0}
                break
        kb2 = max(res_base.keys(), key=lambda k: (res_base[k].get("silhouette", float('-inf')) if not np.isnan(res_base[k].get("silhouette", np.nan)) else float('-inf')))
        mb = res_base[kb2]
        assert best is not None, "缺少最佳配置结果"
        m_best, cfg_best, k_best, labels_best, Xs_best = best
        best_vs_csv = stats_dir / "sequence_model_grid_best_vs_baseline.csv"
        with open(best_vs_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "config", "k", "silhouette", "davies_bouldin", "calinski_harabasz", "intra_over_inter"])
            w.writerow(["attn_norm", "attn_norm", kb2, mb.get("silhouette"), mb.get("davies_bouldin"), mb.get("calinski_harabasz"), mb.get("intra_over_inter")])
            if phase1 is not None:
                w.writerow(["seq_model_phase1", "pool=mean,d=256,L=2,H=4", phase1["k"], phase1["silhouette"], phase1["davies_bouldin"], phase1["calinski_harabasz"], phase1["intra_over_inter"]])
            w.writerow(["seq_model_grid_best", f"pool={cfg_best['pooling']},d={cfg_best['d_model']},L={cfg_best['n_layers']},H={cfg_best['n_heads']}", k_best, m_best.get("silhouette"), m_best.get("davies_bouldin"), m_best.get("calinski_harabasz"), m_best.get("intra_over_inter")])
        print(f"保存全局最佳对比: {best_vs_csv}")
        emb2 = umap_embed(Xs_best, n_components=2, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)
        emb3 = umap_embed(Xs_best, n_components=3, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)
        if emb2 is not None:
            plot_umap_2d(emb2, labels_best, title=f"UMAP 2D - GRID BEST pool={cfg_best['pooling']} d={cfg_best['d_model']} L={cfg_best['n_layers']} H={cfg_best['n_heads']} (k={k_best})", out_path=fig_dir / "umap_2d_seq_model_grid_best.png")
        if emb3 is not None:
            plot_umap_3d(emb3, labels_best, title=f"UMAP 3D - GRID BEST pool={cfg_best['pooling']} d={cfg_best['d_model']} L={cfg_best['n_layers']} H={cfg_best['n_heads']} (k={k_best})", out_path=fig_dir / "umap_3d_seq_model_grid_best.html")
        print("\n完成网格搜索。输出目录:\n  - 图: ", fig_dir, "\n  - 指标: ", stats_dir)
        return

    # ========== 分支B：主流程（默认） ==========
    if args.use_best_grid_config:
        # 使用全局最佳配置作为默认，可被显式传入的 d_model/n_layers/n_heads/pooling 覆盖
        d_model, n_layers, n_heads, pooling = args.d_model, args.n_layers, args.n_heads, args.pooling
        print(f"[BEST-CONFIG] 使用序列模型: pool={pooling}, d={d_model}, L={n_layers}, H={n_heads}")
        try:
            X_seq = extract_seq_embeddings(series_list, d_model=d_model, n_layers=n_layers, n_heads=n_heads, pooling=pooling, device=args.device)
        except RuntimeError as e:
            print("[ERR] ", e); return
        Xs = StandardScaler().fit_transform(X_seq)
        if args.metric == 'cosine':
            Xs = normalize(Xs, norm='l2')
        # 当前范围 [k_min, k_max]
        res_seq, k_seq, labels_seq = cluster_and_scores(Xs, args.k_min, args.k_max, metric=args.metric)
        save_metrics_csv(stats_dir / f"cluster_metrics_seq_model_{args.metric}.csv", "seq_model", res_seq, n_samples=Xs.shape[0], dim=Xs.shape[1])

        # 任务2.1：扩展 k 到 [2, k_analysis_max]，输出曲线 CSV + 可视化
        k_lo, k_hi = 2, max(args.k_analysis_max, max(3, args.k_max))
        res_full, k_best_full, labels_best_full = cluster_and_scores(Xs, k_lo, k_hi, metric=args.metric)
        ka_csv = stats_dir / "best_config_k_analysis.csv"
        with open(ka_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["k", "silhouette", "davies_bouldin", "calinski_harabasz", "intra_over_inter"])
            for k in sorted(res_full.keys()):
                m = res_full[k]
                w.writerow([k, m.get("silhouette", np.nan), m.get("davies_bouldin", np.nan), m.get("calinski_harabasz", np.nan), m.get("intra_over_inter", np.nan)])
        print(f"保存 k 曲线: {ka_csv}")
        # 画图
        ks = sorted(res_full.keys())
        sil = [res_full[k].get("silhouette", np.nan) for k in ks]
        dbi = [res_full[k].get("davies_bouldin", np.nan) for k in ks]
        ch = [res_full[k].get("calinski_harabasz", np.nan) for k in ks]
        ratio = [res_full[k].get("intra_over_inter", np.nan) for k in ks]
        plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(2,2,1); ax1.plot(ks, sil, marker='o'); ax1.set_title('Silhouette vs k'); ax1.set_xlabel('k'); ax1.set_ylabel('Silhouette')
        ax2 = plt.subplot(2,2,2); ax2.plot(ks, dbi, marker='o'); ax2.set_title('Davies-Bouldin vs k'); ax2.set_xlabel('k'); ax2.set_ylabel('DB (lower better)')
        ax3 = plt.subplot(2,2,3); ax3.plot(ks, ch, marker='o'); ax3.set_title('Calinski-Harabasz vs k'); ax3.set_xlabel('k'); ax3.set_ylabel('CH (higher better)')
        ax4 = plt.subplot(2,2,4); ax4.plot(ks, ratio, marker='o'); ax4.set_title('Intra/Inter vs k'); ax4.set_xlabel('k'); ax4.set_ylabel('Intra/Inter (lower better)')
        plt.tight_layout(); plt.savefig(str(fig_dir / "best_config_metrics_vs_k.png"), dpi=150); plt.close()

        # 任务2.2：k=3 与 k=5 的差异
        def eval_at_k(X: np.ndarray, k: int, metric: str):
            X_km = X if metric != 'cosine' else X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_km)
            sil = silhouette_score(X, labels, metric=('cosine' if metric=='cosine' else 'euclidean'))
            db = davies_bouldin_score(X_km, labels)
            chv = calinski_harabasz_score(X_km, labels)
            w_intra, b_inter, ratio = intra_inter_stats(X, labels, metric=metric)
            sizes = [int((labels==c).sum()) for c in sorted(np.unique(labels))]
            return labels, {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": chv, "intra_over_inter": ratio}, sizes
        labels_k3, m_k3, sizes_k3 = eval_at_k(Xs, 3, args.metric)
        labels_k5, m_k5, sizes_k5 = eval_at_k(Xs, 5, args.metric)
        # UMAP 可视化
        emb2 = umap_embed(Xs, n_components=2, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)
        emb3 = umap_embed(Xs, n_components=3, n_neighbors=args.neighbors, min_dist=args.min_dist, metric=args.metric)
        if emb2 is not None:
            plot_umap_2d(emb2, labels_k3, title=f"UMAP 2D - best cfg k=3 (pool={pooling},d={d_model},L={n_layers},H={n_heads})", out_path=fig_dir / "umap_2d_best_config_k3.png")
            plot_umap_2d(emb2, labels_k5, title=f"UMAP 2D - best cfg k=5 (pool={pooling},d={d_model},L={n_layers},H={n_heads})", out_path=fig_dir / "umap_2d_best_config_k5.png")
        if emb3 is not None:
            plot_umap_3d(emb3, labels_k3, title=f"UMAP 3D - best cfg k=3 (pool={pooling},d={d_model},L={n_layers},H={n_heads})", out_path=fig_dir / "umap_3d_best_config_k3.html")
            plot_umap_3d(emb3, labels_k5, title=f"UMAP 3D - best cfg k=5 (pool={pooling},d={d_model},L={n_layers},H={n_heads})", out_path=fig_dir / "umap_3d_best_config_k5.html")
        # 视频采样导出（可选，需显式 --export-video-samples）
        if args.export_video_samples:
            out_root = Path("/home/johnny/action_ws/classify_res")
            summary = export_video_samples(labels_k3, metas, out_root, max_per_cluster=100, seed=42)
            # 写CSV汇总
            try:
                sum_csv = out_root / "cluster_summary.csv"
                with open(sum_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["cluster_id", "total_samples", "exported_samples", "export_dir"])
                    for r in sorted(summary, key=lambda x: x["cluster_id"]):
                        w.writerow([r["cluster_id"], r["total_samples"], r["exported_samples"], r["export_dir"]])
                print(f"[EXPORT] 写入汇总: {sum_csv}")
            except Exception as e:
                print(f"[WARN] 写入汇总CSV失败: {e}")

        # 控制台摘要（便于外部收集）
        print("[K3] sil=%.4f DB=%.4f CH=%.2f Intra/Inter=%.4f sizes=%s" % (m_k3['silhouette'], m_k3['davies_bouldin'], m_k3['calinski_harabasz'], m_k3['intra_over_inter'], sizes_k3))
        print("[K5] sil=%.4f DB=%.4f CH=%.2f Intra/Inter=%.4f sizes=%s" % (m_k5['silhouette'], m_k5['davies_bouldin'], m_k5['calinski_harabasz'], m_k5['intra_over_inter'], sizes_k5))
        print("完成主流程（最佳配置）。输出目录:\n  - 图:", fig_dir, "\n  - 指标:", stats_dir)
        return

    # 默认回退：仅输出 attn_norm 基线（保持向后兼容）
    print("[FALLBACK] 使用 attn_norm 基线完成评估（未启用 --use-best-grid-config）。")
    print("完成主流程（基线）。输出目录:\n  - 图:", fig_dir, "\n  - 指标:", stats_dir)


if __name__ == "__main__":
    main()

