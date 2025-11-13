#!/usr/bin/env python3
"""
K 值搜索（Table S2/S3 — 扩展版）：基于 Frozen Transformer（或基线聚合）在段级特征上执行 KMeans，并输出多项聚类质量指标：
- Silhouette（越大越好）
- Calinski-Harabasz（越大越好）
- Davies-Bouldin（越小越好）
- Intra/Inter 统计：簇内平均距离、簇间中心平均距离、二者比值（越小越好）

- 数据来源：在 data_roots 下递归查找 **/code_indices/*.codes.json，读取其中的 quantized_windows（若缺失则回退 quantized_vectors）。
- 模式：
  * seq_model（默认）：调用 umap_vis/scripts/sequence_model_embedding.py 中的 TinyTransformer 提取段级 embedding（冻结推理）
  * attn_norm：作为基线，使用注意力范数加权均值聚合
- 环境：
  * 计算（提取+聚类）：laps 环境（需要 torch 与 scikit-learn）
  * 绘图：base 环境

用法示例：
  # 1) 计算（laps）
  conda run -n laps python scripts/supplement/k_search.py \
    --data_roots /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector \
    --mode seq_model --pooling mean --d_model 256 --n_layers 4 --n_heads 4 \
    --k_min 2 --k_max 10 --metric cosine --stage compute \
    --out_csv supplement_output/clusters/table_S2_k_search_extended_metrics.csv \
    --out_pdf supplement_output/clusters/fig_k_search_extended.pdf

  # 2) 绘图（base）
  conda run -n base python scripts/supplement/k_search.py \
    --stage plot \
    --out_csv supplement_output/clusters/table_S2_k_search_extended_metrics.csv \
    --out_pdf supplement_output/clusters/fig_k_search_extended.pdf
"""

import os
import sys
import csv
import glob
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# sklearn 在绘图阶段不强制导入，避免 base 环境缺失时报错

def _safe_import_sklearn():
    try:
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # type: ignore
        from sklearn.preprocessing import StandardScaler, normalize  # type: ignore
        return KMeans, silhouette_score, calinski_harabasz_score, davies_bouldin_score, StandardScaler, normalize
    except Exception as e:
        raise RuntimeError("需要在 laps 环境执行计算阶段：缺少 scikit-learn。原始错误：%r" % e)


def find_code_jsons(root: str) -> List[str]:
    return sorted(glob.glob(str(Path(root) / "**/code_indices/*.codes.json"), recursive=True))


def load_segments_multi(roots: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """加载所有 roots 下的 quantized_windows，返回 (series_list, names)。"""
    # 优先尝试复用 umap_vis 的加载器（逻辑一致）
    series_list: List[np.ndarray] = []
    names: List[str] = []
    # 动态导入 sequence_model_embedding.py
    sme = None
    sme_path = Path("umap_vis/scripts").resolve()
    if sme_path.exists():
        sys.path.append(str(sme_path))
        try:
            import sequence_model_embedding as sme  # type: ignore
        except Exception:
            sme = None
    if sme is not None and hasattr(sme, "load_segments"):
        for r in roots:
            sl, nm, _mt = sme.load_segments(r)
            series_list.extend(sl)
            names.extend(nm)
        return series_list, names
    # 兜底：直接解析 JSON
    import json
    for r in roots:
        jfs = find_code_jsons(r)
        for jf in jfs:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                qwins = data.get("quantized_windows", None)
                if qwins is None:
                    qwins = data.get("quantized_vectors", None)
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
            except Exception:
                continue
    return series_list, names

# --------- 距离/统计（参考 umap_vis/scripts/sequence_model_embedding.py） ---------

def pairwise_dists_metric(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if metric == "euclidean":
        XX = np.sum(X * X, axis=1, keepdims=True)
        d2 = XX + XX.T - 2.0 * (X @ X.T)
        d2 = np.maximum(d2, 0.0)
        return np.sqrt(d2, dtype=np.float32)
    elif metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = np.clip(Xn @ Xn.T, -1.0, 1.0)
        return (1.0 - S).astype(np.float32)
    else:
        raise ValueError(f"unsupported metric: {metric}")


def intra_inter_stats(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> Tuple[float, float, float]:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan"), float("nan"), float("nan")
    intra_sum = 0.0
    intra_cnt = 0
    centroids = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        Xi = X[idx]
        if len(idx) > 1:
            dists = pairwise_dists_metric(Xi, metric=metric)
            iu = np.triu_indices_from(dists, k=1)
            vals = dists[iu]
            intra_sum += float(vals.mean())
            intra_cnt += 1
        mu = Xi.mean(axis=0)
        if metric == "cosine":
            mu = mu / (np.linalg.norm(mu) + 1e-12)
        centroids.append(mu)
    w_intra = intra_sum / max(1, intra_cnt)
    C = np.vstack(centroids)
    cd = pairwise_dists_metric(C, metric=metric)
    iu = np.triu_indices_from(cd, k=1)
    b_inter = float(cd[iu].mean()) if iu[0].size > 0 else float("nan")
    ratio = w_intra / (b_inter + 1e-8) if not np.isnan(w_intra) and not np.isnan(b_inter) else float("nan")
    return w_intra, b_inter, ratio


def features_attn_norm(series_list: List[np.ndarray]) -> np.ndarray:
    feats = []
    for x in series_list:
        norms = np.linalg.norm(x, axis=1)
        if np.allclose(norms.sum(), 0.0):
            feats.append(x.mean(axis=0))
        else:
            w = norms / (norms.sum() + 1e-8)
            feats.append((x * w[:, None]).sum(axis=0))
    return np.asarray(feats, dtype=np.float32)


def features_seq_model(series_list: List[np.ndarray], d_model: int, n_layers: int, n_heads: int, pooling: str, device: str = "cpu") -> np.ndarray:
    # 通过 umap_vis 的 TinyTransformer 提取，避免重复实现
    sme_path = Path("umap_vis/scripts").resolve()
    if not sme_path.exists():
        raise RuntimeError("未找到 umap_vis/scripts 目录，无法使用 Frozen Transformer 提取特征")
    if str(sme_path) not in sys.path:
        sys.path.append(str(sme_path))
    try:
        import sequence_model_embedding as sme  # type: ignore
    except Exception as e:
        raise RuntimeError("导入 sequence_model_embedding 失败：%r" % e)
    try:
        X = sme.extract_seq_embeddings(series_list, d_model=d_model, n_layers=n_layers, n_heads=n_heads, pooling=pooling, device=device)
        return np.asarray(X, dtype=np.float32)
    except Exception as e:
        raise RuntimeError("Frozen Transformer 特征提取失败：%r" % e)


def compute_metrics(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    metric: str = "cosine",
    random_state: int = 42,
) -> List[Dict]:
    KMeans, silhouette_score, calinski_harabasz_score, davies_bouldin_score, StandardScaler, normalize = _safe_import_sklearn()
    Xs = StandardScaler().fit_transform(X)
    if metric == "cosine":
        Xs = normalize(Xs, norm="l2")
    results: List[Dict] = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(Xs)
        # 指标计算
        try:
            sil = float(silhouette_score(Xs, labels, metric=("cosine" if metric == "cosine" else "euclidean")))
        except Exception:
            sil = float("nan")
        try:
            db = float(davies_bouldin_score(Xs, labels))
        except Exception:
            db = float("nan")
        try:
            ch = float(calinski_harabasz_score(Xs, labels))
        except Exception:
            ch = float("nan")
        try:
            w_intra, b_inter, ratio = intra_inter_stats(Xs, labels, metric=metric)
        except Exception:
            w_intra = b_inter = ratio = float("nan")
        results.append({
            "k": k,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "intra_dist": w_intra,
            "inter_centroid_dist": b_inter,
            "intra_over_inter": ratio,
        })
    return results


def save_metrics_csv(path: Path, results: List[Dict], n_samples: int, feat_dim: int, mode: str, pooling: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "pooling", "k", "n_samples", "feature_dim", "silhouette", "davies_bouldin", "calinski_harabasz", "intra_dist", "inter_centroid_dist", "intra_over_inter"])
        for r in results:
            w.writerow([mode, pooling, r["k"], n_samples, feat_dim, r.get("silhouette", np.nan), r.get("davies_bouldin", np.nan), r.get("calinski_harabasz", np.nan), r.get("intra_dist", np.nan), r.get("inter_centroid_dist", np.nan), r.get("intra_over_inter", np.nan)])


def plot_from_csv(csv_path: Path, out_pdf: Path, title: str):
    import matplotlib.pyplot as plt
    import csv as _csv
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = _csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        print("[WARN] CSV 为空：", csv_path)
        return
    ks = np.array([int(r["k"]) for r in rows])
    order = np.argsort(ks)
    ks = ks[order]
    def col(name):
        return np.array([float(rows[i].get(name, "nan")) for i in order])
    sil = col("silhouette"); ch = col("calinski_harabasz"); db = col("davies_bouldin")
    intra = col("intra_dist"); inter = col("inter_centroid_dist"); ratio = col("intra_over_inter")

    fig = plt.figure(figsize=(10.5, 7.5))
    ax1 = plt.subplot(2,3,1); ax1.plot(ks, sil, marker='o'); ax1.set_title('Silhouette vs K'); ax1.set_xlabel('K'); ax1.set_ylabel('Silhouette')
    ax2 = plt.subplot(2,3,2); ax2.plot(ks, db, marker='o'); ax2.set_title('Davies-Bouldin vs K (lower better)'); ax2.set_xlabel('K'); ax2.set_ylabel('DB Index')
    ax3 = plt.subplot(2,3,3); ax3.plot(ks, ch, marker='o'); ax3.set_title('Calinski-Harabasz vs K'); ax3.set_xlabel('K'); ax3.set_ylabel('CH Index')
    ax4 = plt.subplot(2,3,4); ax4.plot(ks, intra, marker='o'); ax4.set_title('Intra-cluster distance'); ax4.set_xlabel('K'); ax4.set_ylabel('Avg intra')
    ax5 = plt.subplot(2,3,5); ax5.plot(ks, inter, marker='o'); ax5.set_title('Inter-centroid distance'); ax5.set_xlabel('K'); ax5.set_ylabel('Avg inter')
    ax6 = plt.subplot(2,3,6); ax6.plot(ks, ratio, marker='o'); ax6.set_title('Intra/Inter ratio (lower better)'); ax6.set_xlabel('K'); ax6.set_ylabel('Ratio')
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_roots", type=str, nargs="*", default=[], help="根目录，递归查找 **/code_indices/*.codes.json")
    p.add_argument("--mode", type=str, default="seq_model", choices=["seq_model", "attn_norm"], help="特征提取模式")
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls", "attn"], help="Frozen Transformer 池化策略（仅 seq_model 生效）")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--metric", type=str, default="cosine", choices=["euclidean", "cosine"])
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=10)
    p.add_argument("--stage", type=str, default="both", choices=["compute", "plot", "both"], help="执行阶段：计算/绘图/两者")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--out_pdf", type=str, required=True)
    args = p.parse_args()

    out_csv = Path(args.out_csv)
    out_pdf = Path(args.out_pdf)

    if args.stage in ("compute", "both"):
        if len(args.data_roots) == 0:
            # 默认使用 LAPS 输出目录（D01/D02）
            args.data_roots = [
                "datasets/output/segmentation_outputs/D01_LAPS",
                "datasets/output/segmentation_outputs/D02_LAPS",
            ]
        print("[INFO] 加载数据...", args.data_roots)
        series_list, names = load_segments_multi(args.data_roots)
        if len(series_list) == 0:
            raise RuntimeError("未找到任何 quantized_windows（检查 data_roots 是否指向包含 code_indices 的目录）")
        print(f"[INFO] 片段数={len(series_list)}, 向量维度={series_list[0].shape[1]}")
        # 特征
        if args.mode == "attn_norm":
            X = features_attn_norm(series_list)
            pooling = "attn_norm"
        else:
            X = features_seq_model(series_list, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, pooling=args.pooling, device="cpu")
            pooling = args.pooling
        # 指标
        results = compute_metrics(X, k_min=args.k_min, k_max=args.k_max, metric=args.metric)
        save_metrics_csv(out_csv, results, n_samples=X.shape[0], feat_dim=X.shape[1], mode=args.mode, pooling=pooling)
        print("[OK] 指标已写入:", out_csv)

    if args.stage in ("plot", "both"):
        title = f"K-search ({args.mode}, pool={args.pooling})"
        plot_from_csv(out_csv, out_pdf, title=title)
        print("[OK] 图像已保存:", out_pdf)


if __name__ == "__main__":
    main()

