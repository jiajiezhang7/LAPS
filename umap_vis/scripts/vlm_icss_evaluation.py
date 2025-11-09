#!/usr/bin/env python3
"""
VLM-based ICSS evaluation on clustered action segments.
- Reuse TinyTransformer + KMeans (from umap_vis/scripts/sequence_model_embedding.py) to get cluster labels
- Extract CLIP ViT-B/32 image embeddings on sampled frames per segment video, then aggregate to segment-level vectors
- Compute Intra-Cluster Semantic Similarity (ICSS) and Random baseline
- Save CSV/JSON stats and visualizations to datasets/output/vlm_icss_exp/

Config per user:
- Multi-frame sampling with default N=8; long segments cap to 12~16 frames
- Frame-level L2-norm weighting when aggregating frames
- CLIP local model dir: /home/johnny/action_ws/clip-vit-base-patch32
- K selection by silhouette, k in [3, 10], metric='cosine'
- Random baseline repeats R=5
- Per-cluster pair cap: 100k
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAVE_SEABORN = True
except Exception:
    sns = None
    _HAVE_SEABORN = False

# Optional deps
import importlib.util
try:
    from scipy.stats import gaussian_kde, bootstrap
    _HAVE_SCIPY = True
except Exception:
    gaussian_kde = None
    bootstrap = None
    _HAVE_SCIPY = False

# sklearn preprocessing
from sklearn.preprocessing import StandardScaler, normalize as sk_normalize

# torch / transformers / cv2
import torch
from PIL import Image
import cv2

from transformers import CLIPModel
try:
    from transformers import CLIPProcessor
    _PROCESSOR_CLASS = 'CLIPProcessor'
except Exception:
    from transformers import AutoProcessor as CLIPProcessor  # fallback
    _PROCESSOR_CLASS = 'AutoProcessor'

ROOT = Path('/home/johnny/action_ws')
DEFAULT_OUT = ROOT / 'datasets/output/vlm_icss_exp'
CLIP_DIR = ROOT / 'clip-vit-base-patch32'
SEQ_SCRIPT = ROOT / 'umap_vis/scripts/sequence_model_embedding.py'
D01 = ROOT / 'datasets/output/segmentation_outputs/D01'
D02 = ROOT / 'datasets/output/segmentation_outputs/D02'


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_sequence_module(path: Path):
    spec = importlib.util.spec_from_file_location('sequence_model_embedding', str(path))
    module = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    return module


def codes_json_to_mp4_path(json_path: Path) -> Path:
    # json: .../<sample_dir>/code_indices/<stem>.codes.json
    # video: .../<sample_dir>/segmented_videos/<stem>.mp4
    stem = json_path.stem
    if stem.endswith('.codes'):
        stem = stem[:-len('.codes')]
    mp4 = json_path.parent.parent / 'segmented_videos' / f'{stem}.mp4'
    return mp4


def load_all_series_and_entries(data_roots: List[Path], seq_mod) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    series_all: List[np.ndarray] = []
    entries: List[Dict[str, Any]] = []
    for root in data_roots:
        s_list, names, metas = seq_mod.load_segments(str(root))  # type: ignore
        for idx, meta in enumerate(metas):
            jp = Path(meta['json_path'])
            mp4 = codes_json_to_mp4_path(jp)
            if not mp4.exists():
                # skip if no video
                continue
            series_all.append(s_list[idx])
            entries.append({
                'json_path': str(jp),
                'mp4_path': str(mp4),
                'dataset': jp.as_posix().split('/segmentation_outputs/')[-1].split('/')[0],
                'name': jp.stem,
            })
    return series_all, entries


def build_clip(clip_dir: Path, device: str):
    model = CLIPModel.from_pretrained(str(clip_dir), local_files_only=True)
    processor = CLIPProcessor.from_pretrained(str(clip_dir), local_files_only=True)
    model.to(device)
    model.eval()
    return model, processor


def sample_frame_indices(n_frames: int, n_default: int = 8, mid_cap: int = 12, max_cap: int = 16,
                          mode: str = 'cap') -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=int)
    if mode == 'fixed':
        k = min(n_frames, n_default)
    else:
        if n_frames >= max_cap:
            k = max_cap
        elif n_frames >= mid_cap:
            k = mid_cap
        elif n_frames >= n_default:
            k = n_default
        else:
            k = n_frames
    idxs = np.linspace(0, n_frames - 1, num=k).round().astype(int)
    return np.unique(idxs)


def read_segment_frames(mp4_path: Path, idxs: np.ndarray) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(mp4_path))
    imgs: List[Image.Image] = []
    try:
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(Image.fromarray(rgb))
    finally:
        cap.release()
    return imgs


def encode_segment_clip(mp4_path: Path, model: CLIPModel, processor, device: str,
                        n_default: int = 8, mid_cap: int = 12, max_cap: int = 16,
                        weight_by_norm: bool = True, sampling_mode: str = 'cap') -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    idxs = sample_frame_indices(n_frames, n_default, mid_cap, max_cap, mode=sampling_mode)
    if idxs.size == 0:
        # fallback: decode first frame
        idxs = np.array([0], dtype=int)
    imgs = read_segment_frames(mp4_path, idxs)
    if len(imgs) == 0:
        # generate a black image to avoid crash
        imgs = [Image.new('RGB', (224, 224), color=(0, 0, 0))]

    # 手工预处理，避免老版本 transformers 与 numpy/torch 的 dtype 兼容问题
    # 目标：生成 (m,3,224,224) 的 float32 张量，按 CLIP 规范归一化
    def _preprocess_clip_images(img_list: List[Image.Image]) -> torch.Tensor:
        try:
            import numpy as _np
            import torch as _torch
        except Exception:
            raise
        mean = _np.array([0.48145466, 0.4578275, 0.40821073], dtype=_np.float32)
        std  = _np.array([0.26862954, 0.26130258, 0.27577711], dtype=_np.float32)
        arrs = []
        for im in img_list:
            if not isinstance(im, Image.Image):
                # 尝试从 ndarray 恢复
                try:
                    if isinstance(im, _np.ndarray):
                        if im.ndim == 3 and im.shape[2] == 3:
                            im = Image.fromarray(im.astype(_np.uint8))
                        else:
                            # 灰度或其他情况，转RGB
                            im = Image.fromarray(_np.stack([im]*3, axis=-1).astype(_np.uint8))
                except Exception:
                    pass
            if not isinstance(im, Image.Image):
                # 兜底：黑图
                im = Image.new('RGB', (224, 224), color=(0, 0, 0))
            im = im.convert('RGB')
            # resize 最短边到 224，并中心裁剪到 224x224
            w, h = im.size
            scale = 224.0 / min(w, h) if min(w, h) > 0 else 1.0
            new_w, new_h = int(round(w*scale)), int(round(h*scale))
            im_resized = im.resize((max(1, new_w), max(1, new_h)), resample=Image.BICUBIC)
            # 中心裁剪
            left = max(0, (im_resized.width - 224)//2)
            top = max(0, (im_resized.height - 224)//2)
            im_cropped = im_resized.crop((left, top, left+224, top+224))
            x = _np.asarray(im_cropped, dtype=_np.float32) / 255.0  # (224,224,3)
            x = (x - mean) / std
            arrs.append(x)
        X = _np.stack(arrs, axis=0)  # (m,224,224,3)
        X = _np.transpose(X, (0, 3, 1, 2))  # (m,3,224,224)
        return _torch.tensor(X, dtype=_torch.float32)

    with torch.no_grad():
        pixel_values = _preprocess_clip_images(imgs).to(device)
        feats_t = model.get_image_features(pixel_values=pixel_values)  # (m, d) torch.Tensor
        feats_t = feats_t.detach().float().cpu()
    # 全部在 torch 中完成，避免 numpy C-API 兼容性问题
    norms_t = torch.linalg.norm(feats_t, dim=1) + 1e-9  # (m,)
    normed_t = feats_t / norms_t.unsqueeze(1)           # (m,d)
    if weight_by_norm:
        w_t = norms_t / norms_t.sum()
    else:
        w_t = torch.ones_like(norms_t) / float(norms_t.numel())
    z_t = (w_t.unsqueeze(1) * normed_t).sum(dim=0)      # (d,)
    z_t = z_t / (torch.linalg.norm(z_t) + 1e-9)
    return z_t.cpu().numpy().astype(np.float32)


def extract_seq_labels(series_list: List[np.ndarray], seq_mod, metric: str = 'cosine', k_min: int = 3, k_max: int = 10,
                       d_model: int = 256, n_layers: int = 2, n_heads: int = 4, pooling: str = 'mean', device: str = 'cpu') -> Tuple[np.ndarray, Dict[int, Dict[str, float]], int]:
    X_seq = seq_mod.extract_seq_embeddings(series_list, d_model=d_model, n_layers=n_layers, n_heads=n_heads, pooling=pooling, device=device)  # type: ignore
    Xs = StandardScaler().fit_transform(X_seq)
    if metric == 'cosine':
        Xs = sk_normalize(Xs, norm='l2')
    results, best_k, labels = seq_mod.cluster_and_scores(Xs, k_min, k_max, metric=metric)  # type: ignore
    return labels, results, int(best_k)


def cluster_pair_stats(labels: np.ndarray, Z: np.ndarray, max_pairs_per_cluster: int = 100_000,
                       rng: np.random.RandomState | None = None) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[int, np.ndarray]]:
    if rng is None:
        rng = np.random.RandomState(42)
    stats: List[Dict[str, Any]] = []
    all_sims: List[float] = []
    per_cluster: Dict[int, np.ndarray] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = len(idx)
        if n < 2:
            continue
        total_pairs = n * (n - 1) // 2
        m = min(total_pairs, max_pairs_per_cluster)
        # sample m pairs (approx uniform, allow duplicates negligible effect when m << total_pairs)
        ii = rng.randint(0, n, size=m)
        jj = (ii + rng.randint(1, n, size=m)) % n  # ensure jj != ii
        sims = np.sum(Z[idx[ii]] * Z[idx[jj]], axis=1)
        all_sims.extend(sims.tolist())
        per_cluster[int(c)] = sims.astype(np.float32)
        stats.append({
            'cluster': int(c),
            'n_segments': int(n),
            'num_pairs_total': int(total_pairs),
            'num_pairs_used': int(m),
            'mean': float(np.mean(sims)),
            'std': float(np.std(sims)),
        })
    return stats, np.asarray(all_sims, dtype=np.float32), per_cluster


def random_baseline(Z: np.ndarray, M: int, R: int = 5, rng: np.random.RandomState | None = None) -> Dict[str, Any]:
    if rng is None:
        rng = np.random.RandomState(123)
    n = Z.shape[0]
    rep_means, rep_stds = [], []
    for _ in range(R):
        i = rng.randint(0, n, size=M)
        j = (i + rng.randint(1, n, size=M)) % n
        sims = np.sum(Z[i] * Z[j], axis=1)
        rep_means.append(float(np.mean(sims)))
        rep_stds.append(float(np.std(sims)))
    return {
        'R': int(R),
        'M_per_rep': int(M),
        'mean_of_means': float(np.mean(rep_means)),
        'std_of_means': float(np.std(rep_means)),
        'mean_of_stds': float(np.mean(rep_stds)),
        'std_of_stds': float(np.std(rep_stds)),
        'per_rep_means': rep_means,
        'per_rep_stds': rep_stds,
    }

# --- Publication-quality figures helpers ---

def compute_bootstrap_ci(data: np.ndarray, n_boot: int = 2000, ci: float = 95.0, rng: np.random.RandomState | None = None) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using bootstrap for the mean."""
    if rng is None:
        rng = np.random.RandomState(123)
    data = np.asarray(data, dtype=np.float64)
    mean = float(np.mean(data))
    alpha = 100.0 - ci
    if _HAVE_SCIPY and bootstrap is not None:
        try:
            res = bootstrap((data,), np.mean, vectorized=False, n_resamples=int(n_boot), method='basic', random_state=rng)
            low = float(res.confidence_interval.low)
            high = float(res.confidence_interval.high)
            return mean, low, high
        except Exception:
            pass
    # manual bootstrap
    n = data.shape[0]
    bs_means = []
    for _ in range(int(n_boot)):
        idx = rng.randint(0, n, size=n)
        bs_means.append(float(np.mean(data[idx])))
    low, high = np.percentile(bs_means, [alpha / 2.0, 100.0 - alpha / 2.0])
    return mean, float(low), float(high)


def sample_random_pair_sims(Z: np.ndarray, M: int, rng: np.random.RandomState | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(321)
    n = Z.shape[0]
    i = rng.randint(0, n, size=int(M))
    j = (i + rng.randint(1, n, size=int(M))) % n
    sims = np.sum(Z[i] * Z[j], axis=1)
    return sims.astype(np.float32)


def _apply_pub_style():
    if _HAVE_SEABORN and sns is not None:
        sns.set_theme(context='paper', style='whitegrid', palette='colorblind')
    else:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'font.size': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 1.2,
    })



def _plot_kde(ax, data: np.ndarray, color, label: str):
    arr = np.asarray(data, dtype=np.float64)
    if _HAVE_SEABORN and sns is not None:
        sns.kdeplot(arr, ax=ax, label=label, color=color, linewidth=1.8, fill=False, clip=(0, 1))
    elif _HAVE_SCIPY and gaussian_kde is not None and arr.size > 1:
        xs = np.linspace(0.0, 1.0, 512)
        try:
            kde = gaussian_kde(arr)
            ax.plot(xs, kde(xs), color=color, lw=1.8, label=label)
        except Exception:
            ax.hist(arr, bins=50, density=True, histtype='step', color=color, lw=1.8, label=label)
    else:
        ax.hist(arr, bins=50, density=True, histtype='step', color=color, lw=1.8, label=label)


def visualize_publication_quality(per_cluster_sims: Dict[int, np.ndarray], baseline_sims: np.ndarray, out_dir: Path,
                                  n_boot: int = 2000):
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_pub_style()

    clusters_sorted = sorted(per_cluster_sims.keys())
    palette = (sns.color_palette('colorblind', n_colors=len(clusters_sorted) + 1)
               if (_HAVE_SEABORN and sns is not None) else list(plt.cm.tab10.colors))
    cluster_colors = {c: palette[i % len(palette)] for i, c in enumerate(clusters_sorted)}
    baseline_color = (0.4, 0.4, 0.4)

    # --- KDE Figure ---
    fig_kde, ax_kde = plt.subplots(figsize=(7.2, 4.8))
    for c in clusters_sorted:
        vals = per_cluster_sims[c]
        if len(vals) < 50:
            # small sample warning in title/legend may be added if needed
            pass
        _plot_kde(ax_kde, vals, cluster_colors[c], label=f'Cluster {c}')
        ax_kde.axvline(float(np.mean(vals)), color=cluster_colors[c], linestyle='--', linewidth=1.5, alpha=0.9)
    _plot_kde(ax_kde, baseline_sims, baseline_color, label='Random Baseline')
    ax_kde.axvline(float(np.mean(baseline_sims)), color=baseline_color, linestyle='--', linewidth=1.5, alpha=0.9)
    ax_kde.set_title('ICSS: KDE per Cluster vs Random')
    ax_kde.set_xlabel('Cosine Similarity'); ax_kde.set_ylabel('Density')
    ax_kde.set_xlim(0.0, 1.0)
    ax_kde.grid(alpha=0.2)
    ax_kde.legend(frameon=False)
    fig_kde.tight_layout()
    fig_kde.savefig(out_dir / 'icss_kde_distribution.pdf', bbox_inches='tight')
    fig_kde.savefig(out_dir / 'icss_kde_distribution.png', dpi=600, bbox_inches='tight')
    plt.close(fig_kde)

    # --- Confidence Intervals Figure (error bars) ---
    labels = [f'Cluster {c}' for c in clusters_sorted] + ['Random']
    dists = [per_cluster_sims[c] for c in clusters_sorted] + [baseline_sims]
    colors = [cluster_colors[c] for c in clusters_sorted] + [baseline_color]

    means, lows, highs, ns = [], [], [], []
    rng = np.random.RandomState(777)
    for arr in dists:
        m, lo, hi = compute_bootstrap_ci(np.asarray(arr), n_boot=n_boot, ci=95.0, rng=rng)
        means.append(m); lows.append(lo); highs.append(hi); ns.append(len(arr))

    x = np.arange(len(labels))
    fig_ci, ax_ci = plt.subplots(figsize=(7.2, 4.8))
    for i in range(len(labels)):
        ax_ci.errorbar([x[i]], [means[i]], yerr=[[means[i]-lows[i]], [highs[i]-means[i]]], fmt='o', color=colors[i],
                       ecolor=colors[i], elinewidth=1.8, capsize=4, markersize=6, markeredgewidth=1.2)
        ax_ci.text(x[i], means[i], f'n={ns[i]}', ha='center', va='bottom', fontsize=9, color=colors[i])
    ax_ci.set_xticks(x); ax_ci.set_xticklabels(labels, rotation=0)
    ax_ci.set_ylabel('Mean Cosine Similarity')
    ax_ci.set_title('ICSS: Mean with 95% Bootstrap CI')
    ax_ci.set_ylim(0.0, 1.0)
    ax_ci.grid(alpha=0.2)
    fig_ci.tight_layout()
    fig_ci.savefig(out_dir / 'icss_confidence_intervals.pdf', bbox_inches='tight')
    fig_ci.savefig(out_dir / 'icss_confidence_intervals.png', dpi=600, bbox_inches='tight')
    plt.close(fig_ci)

    # --- Combined Figure ---
    fig_c, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    # Left KDE
    for c in clusters_sorted:
        vals = per_cluster_sims[c]
        _plot_kde(ax_left, vals, cluster_colors[c], label=f'Cluster {c}')
        ax_left.axvline(float(np.mean(vals)), color=cluster_colors[c], linestyle='--', linewidth=1.5, alpha=0.9)
    _plot_kde(ax_left, baseline_sims, baseline_color, label='Random Baseline')
    ax_left.axvline(float(np.mean(baseline_sims)), color=baseline_color, linestyle='--', linewidth=1.5, alpha=0.9)
    ax_left.set_title('KDE per Cluster vs Random')
    ax_left.set_xlabel('Cosine Similarity'); ax_left.set_ylabel('Density')
    ax_left.set_xlim(0.0, 1.0)
    ax_left.grid(alpha=0.2); ax_left.legend(frameon=False)

    # Right CI errorbars
    for i in range(len(labels)):
        ax_right.errorbar([i], [means[i]], yerr=[[means[i]-lows[i]], [highs[i]-means[i]]], fmt='o', color=colors[i],
                          ecolor=colors[i], elinewidth=1.8, capsize=4, markersize=6, markeredgewidth=1.2)
        ax_right.text(i, means[i], f'n={ns[i]}', ha='center', va='bottom', fontsize=9, color=colors[i])
    ax_right.set_xticks(np.arange(len(labels))); ax_right.set_xticklabels(labels, rotation=0)
    ax_right.set_ylabel('Mean Cosine Similarity'); ax_right.set_title('Mean with 95% CI')
    ax_right.set_ylim(0.0, 1.0)
    ax_right.grid(alpha=0.2)

    fig_c.tight_layout()
    fig_c.savefig(out_dir / 'icss_combined_analysis.pdf', bbox_inches='tight')
    fig_c.savefig(out_dir / 'icss_combined_analysis.png', dpi=600, bbox_inches='tight')
    plt.close(fig_c)



def visualize(all_cluster_sims: np.ndarray, per_cluster_sims: Dict[int, np.ndarray], baseline_means: List[float] | None, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Histogram of all cluster pairs
    plt.figure(figsize=(8, 5))
    plt.hist(all_cluster_sims, bins=50, alpha=0.6, density=True, label='Clusters (Pairs)')
    if baseline_means is not None:
        plt.axvline(np.mean(all_cluster_sims), color='C0', linestyle='--', label=f'Clusters mean={np.mean(all_cluster_sims):.3f}')
    plt.title('Similarity Distribution: Cluster Pairs')
    plt.xlabel('Cosine similarity'); plt.ylabel('Density')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / 'similarity_hist_cluster_pairs.png', dpi=200)
    plt.close()

    # Boxplot per cluster (sampled pairs)
    if per_cluster_sims:
        clusters_sorted = sorted(per_cluster_sims.keys())
        data = [per_cluster_sims[c] for c in clusters_sorted]
        plt.figure(figsize=(max(6, len(data) * 1.2), 5))
        plt.boxplot(data, labels=[str(c) for c in clusters_sorted], showfliers=False)
        plt.xlabel('Cluster ID'); plt.ylabel('Cosine similarity')
        plt.title('Per-Cluster Similarity (Boxplot)')
        plt.tight_layout(); plt.savefig(out_dir / 'similarity_boxplot_per_cluster.png', dpi=200)
        plt.close()


def save_csv(path: Path, rows: List[Dict[str, Any]]):
    import csv
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            f.write('')
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-roots', nargs='+', default=[str(D01), str(D02)])
    p.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT))
    p.add_argument('--clip-model-dir', type=str, default=str(CLIP_DIR))
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--k-min', type=int, default=3)
    p.add_argument('--k-max', type=int, default=10)
    p.add_argument('--metric', type=str, default='cosine')
    p.add_argument('--seq-d-model', type=int, default=256)
    p.add_argument('--seq-n-layers', type=int, default=2)
    p.add_argument('--seq-n-heads', type=int, default=4)
    p.add_argument('--seq-pooling', type=str, default='mean')
    p.add_argument('--sample-n', type=int, default=8)
    p.add_argument('--mid-cap', type=int, default=12)
    p.add_argument('--max-cap', type=int, default=16)
    p.add_argument('--sampling-mode', type=str, choices=['cap', 'fixed'], default='cap')
    # Aggregation weighting toggle: default True, allow explicit disabling via --no-weight-by-norm
    p.add_argument('--weight-by-norm', dest='weight_by_norm', action='store_true', default=True)
    p.add_argument('--no-weight-by-norm', dest='weight_by_norm', action='store_false')
    p.add_argument('--cluster-pairs-cap', type=int, default=100_000)
    p.add_argument('--baseline-R', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--skip-figs', action='store_true', help='Skip matplotlib visualizations to avoid env issues')
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); (out_dir / 'figs').mkdir(parents=True, exist_ok=True)

    # 1) Import sequence module & load series
    seq_mod = import_sequence_module(SEQ_SCRIPT)
    series_list, entries = load_all_series_and_entries([Path(p) for p in args.data_roots], seq_mod)
    if len(series_list) == 0:
        print('[ERR] No segments found.')
        return
    print(f'[INFO] segments: {len(series_list)}')

    # 2) CLIP model
    model, processor = build_clip(Path(args.clip_model_dir), args.device)

    # 3) Segment-level CLIP embedding
    Z = []
    for e in entries:


        z = encode_segment_clip(Path(e['mp4_path']), model, processor, args.device,
                                n_default=args.sample_n, mid_cap=args.mid_cap, max_cap=args.max_cap,
                                weight_by_norm=args.weight_by_norm, sampling_mode=args.sampling_mode)
        Z.append(z)
    Z = np.asarray(Z, dtype=np.float32)
    assert Z.shape[0] == len(series_list)

    # 4) Cluster labels via TinyTransformer + KMeans
    labels, results, best_k = extract_seq_labels(series_list, seq_mod, metric=args.metric, k_min=args.k_min, k_max=args.k_max,
                                                 d_model=args.seq_d_model, n_layers=args.seq_n_layers, n_heads=args.seq_n_heads,
                                                 pooling=args.seq_pooling, device=args.device)
    print(f'[INFO] best_k={best_k}, unique labels={np.unique(labels).tolist()}')

    # 5) ICSS by clusters (sampled pairs per cluster)
    cluster_stats, all_sims, per_cluster_sims = cluster_pair_stats(
        labels, Z, max_pairs_per_cluster=args.cluster_pairs_cap, rng=np.random.RandomState(args.seed)
    )
    overall = {
        'overall_pairs_used': int(all_sims.size),
        'overall_mean': float(np.mean(all_sims)) if all_sims.size > 0 else float('nan'),
        'overall_std': float(np.std(all_sims)) if all_sims.size > 0 else float('nan'),
    }

    # 6) Random baseline (match number of pairs)
    baseline = random_baseline(Z, M=int(all_sims.size), R=args.baseline_R, rng=np.random.RandomState(args.seed + 1))

    # 7) Visualizations (hist + per-cluster boxplot)
    if not args.skip_figs:
        visualize(all_sims, per_cluster_sims, baseline_means=baseline['per_rep_means'], out_dir=out_dir / 'figs')

    # 8) Save outputs
    # segments mapping
    seg_rows = [{**e, 'cluster': int(l)} for e, l in zip(entries, labels.tolist())]
    save_csv(out_dir / 'segments_with_clusters.csv', seg_rows)
    np.save(out_dir / 'labels.npy', labels)
    np.save(out_dir / 'clip_features.npy', Z)

    # 7b) Publication-quality visualizations (KDE + 95% CI + combined)
    if not args.skip_figs:
        baseline_sims = sample_random_pair_sims(Z, M=int(all_sims.size), rng=np.random.RandomState(args.seed + 2))
        visualize_publication_quality(per_cluster_sims=per_cluster_sims,
                                     baseline_sims=baseline_sims,
                                     out_dir=out_dir / 'figs',
                                     n_boot=2000)


    # per-cluster stats
    save_csv(out_dir / 'cluster_icss_stats.csv', cluster_stats)

    # overall + baseline summary
    summary = {
        'config': {
            'clip_model_dir': str(args.clip_model_dir),
            'sample_frames': {'default': args.sample_n, 'mid_cap': args.mid_cap, 'max_cap': args.max_cap},
            'weight_by_norm': bool(args.weight_by_norm),
            'k_range': [args.k_min, args.k_max],
            'metric': args.metric,
            'baseline_R': args.baseline_R,
            'cluster_pairs_cap': args.cluster_pairs_cap,
            'data_roots': args.data_roots,
            'processor_class': _PROCESSOR_CLASS,
        },
        'clustering': {'best_k': int(best_k), 'metrics': results},
        'icss_overall': overall,
        'baseline': baseline,
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('[DONE] Outputs saved to:', out_dir)


if __name__ == '__main__':
    main()

