#!/usr/bin/env python3
"""
Pooling & Distance metric comparison for K-search (Table S2):
- Pooling: mean / cls / attn
- Distance metric: euclidean / cosine
- For each combo, run KMeans for K in [k_min, k_max], compute 6 metrics:
  Silhouette, Davies-Bouldin, Calinski-Harabasz, Intra, Inter, Intra/Inter

Compute stage: laps env
Plot stage: base env

Defaults are set to the new online inference data source used in the supplement experiments.

Usage examples:
  # Compute (laps)
  conda run -n laps python scripts/supplement/pooling_metric_compare.py \
    --data_roots /media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector \
    --k_min 2 --k_max 10 --stage compute \
    --out_csv supplement_output/clusters/table_S2_pooling_metric_comparison.csv \
    --out_pdf supplement_output/clusters/fig_S2_pooling_metric_comparison.pdf

  # Plot (base)
  conda run -n base python scripts/supplement/pooling_metric_compare.py \
    --stage plot \
    --out_csv supplement_output/clusters/table_S2_pooling_metric_comparison.csv \
    --out_pdf supplement_output/clusters/fig_S2_pooling_metric_comparison.pdf
"""
from __future__ import annotations
import os
import sys
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# Dynamically import functions from k_search.py by file path
_DEF_K_SEARCH_PATH = Path("scripts/supplement/k_search.py").resolve()

def _import_k_search():
    import importlib.util
    spec = importlib.util.spec_from_file_location("k_search_mod", str(_DEF_K_SEARCH_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {_DEF_K_SEARCH_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def compute_all(
    data_roots: List[str],
    poolings: List[str],  # ["mean","cls","attn"]
    distance_metrics: List[str],  # ["euclidean","cosine"]
    d_model: int,
    n_layers: int,
    n_heads: int,
    k_min: int,
    k_max: int,
) -> List[Dict]:
    ks = _import_k_search()
    # Load sequences once
    print("[INFO] 加载数据...", data_roots)
    series_list, names = ks.load_segments_multi(data_roots)
    if len(series_list) == 0:
        raise RuntimeError("未找到任何 quantized_windows/quantized_vectors 片段")
    print(f"[INFO] 片段数={len(series_list)}, 向量维度={series_list[0].shape[1]}")

    all_rows: List[Dict] = []
    for pooling in poolings:
        print(f"[INFO] 提取特征：pooling={pooling}")
        X = ks.features_seq_model(series_list, d_model=d_model, n_layers=n_layers, n_heads=n_heads, pooling=pooling, device="cpu")
        n_samples, feat_dim = X.shape
        for dist in distance_metrics:
            print(f"[INFO] 指标计算：pooling={pooling}, metric={dist}")
            results = ks.compute_metrics(X, k_min=k_min, k_max=k_max, metric=dist)
            for r in results:
                row = {
                    "mode": "seq_model",
                    "pooling": pooling,
                    "metric": dist,
                    "k": r["k"],
                    "n_samples": n_samples,
                    "feature_dim": feat_dim,
                    "silhouette": r.get("silhouette", np.nan),
                    "davies_bouldin": r.get("davies_bouldin", np.nan),
                    "calinski_harabasz": r.get("calinski_harabasz", np.nan),
                    "intra_dist": r.get("intra_dist", np.nan),
                    "inter_centroid_dist": r.get("inter_centroid_dist", np.nan),
                    "intra_over_inter": r.get("intra_over_inter", np.nan),
                }
                all_rows.append(row)
    return all_rows


def save_rows_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "mode", "pooling", "metric", "k", "n_samples", "feature_dim",
        "silhouette", "davies_bouldin", "calinski_harabasz",
        "intra_dist", "inter_centroid_dist", "intra_over_inter",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    print("[OK] 汇总CSV已写入:", path)


def plot_multi(csv_path: Path, out_pdf: Path, title: str = "Pooling & Metric Comparison"):
    import csv as _csv
    import matplotlib.pyplot as plt
    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = _csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        print("[WARN] CSV 为空：", csv_path)
        return
    # Collect combos
    combos = {}
    for row in rows:
        key = (row["pooling"], row.get("metric", "cosine"))
        combos.setdefault(key, []).append(row)
    # Sort by k within each combo
    for key in combos:
        combos[key] = sorted(combos[key], key=lambda d: int(d["k"]))
    # Style mapping: color by pooling, linestyle by metric
    color_map = {"mean": "tab:blue", "cls": "tab:orange", "attn": "tab:green"}
    linestyle_map = {"cosine": "-", "euclidean": "--"}

    metrics = [
        ("silhouette", "Silhouette (higher better)"),
        ("davies_bouldin", "Davies-Bouldin (lower better)"),
        ("calinski_harabasz", "Calinski-Harabasz (higher better)"),
        ("intra_dist", "Intra-cluster distance"),
        ("inter_centroid_dist", "Inter-centroid distance"),
        ("intra_over_inter", "Intra/Inter ratio (lower better)"),
    ]

    fig = plt.figure(figsize=(12, 8))
    axes = [plt.subplot(2, 3, i + 1) for i in range(6)]

    for ax, (col, ylab) in zip(axes, metrics):
        for (pooling, dist), rows_k in combos.items():
            ks = np.array([int(r["k"]) for r in rows_k])
            ys = np.array([float(r.get(col, "nan")) for r in rows_k])
            ax.plot(
                ks, ys,
                label=f"{pooling}-{dist}",
                color=color_map.get(pooling, None),
                linestyle=linestyle_map.get(dist, "-"),
                marker="o", markersize=3,
            )
        ax.set_xlabel("K")
        ax.set_ylabel(ylab)
        ax.set_title(f"{col} vs K")
        ax.grid(True, alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9)
    # fig.suptitle(title)  # 禁用总标题，避免与论文 caption 冲突
    fig.tight_layout(rect=[0, 0, 1, 0.93])  # 为顶部图例预留 4% 空间
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("[OK] 对比图已保存:", out_pdf)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_roots", type=str, nargs="*", default=[
        "/media/johnny/48FF-AA60/online_inference_output/epochs5_complete500_d02_m10_cb2048_stride4_vector"
    ])
    p.add_argument("--poolings", type=str, nargs="*", default=["mean", "cls", "attn"], choices=["mean", "cls", "attn"])
    p.add_argument("--distance_metrics", type=str, nargs="*", default=["euclidean", "cosine"], choices=["euclidean", "cosine"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=10)
    p.add_argument("--stage", type=str, default="both", choices=["compute", "plot", "both"])
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--out_pdf", type=str, required=True)
    args = p.parse_args()

    out_csv = Path(args.out_csv)
    out_pdf = Path(args.out_pdf)

    if args.stage in ("compute", "both"):
        rows = compute_all(
            data_roots=args.data_roots,
            poolings=args.poolings,
            distance_metrics=args.distance_metrics,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            k_min=args.k_min,
            k_max=args.k_max,
        )
        save_rows_csv(out_csv, rows)

    if args.stage in ("plot", "both"):
        plot_multi(out_csv, out_pdf, title="Pooling & Distance Metric Comparison (K=2..10)")


if __name__ == "__main__":
    main()

