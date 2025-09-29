#!/usr/bin/env python3
"""Analyze energy_*.jsonl files and generate plots + metrics report.
Run:
  python -m video_action_segmenter.analyze_energy_jsonl \
    --input-dir video_action_segmenter/energy_sweep_out \
    --output-dir video_action_segmenter/energy_sweep_report
"""
import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.stats import ks_2samp, pearsonr, spearmanr
import csv

JSONL_RE = re.compile(r"stream_energy_(?P<src>[^_]+)_(?P<mode>[^.]+)\.jsonl$")


def load_energy_jsonl(path: Path):
    src = mode = None
    win, val = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                win.append(int(obj.get('window')))
                val.append(float(obj.get('energy')))
                src = src or obj.get('source')
                mode = mode or obj.get('mode')
            except Exception:
                continue
    if not src or not mode:
        m = JSONL_RE.search(path.name)
        if m:
            src = src or m.group('src')
            mode = mode or m.group('mode')
    idx = np.argsort(np.asarray(win))
    win = np.asarray(win, dtype=np.int64)[idx]
    val = np.asarray(val, dtype=np.float32)[idx]
    return {'source': str(src), 'mode': str(mode), 'windows': win, 'values': val, 'path': path}


def find_jsonl_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob('stream_energy_*.jsonl') if p.is_file()])


def align_common_windows(series: List[dict]) -> List[dict]:
    if not series:
        return []
    common = set(series[0]['windows'].tolist())
    for s in series[1:]:
        common &= set(s['windows'].tolist())
    if not common:
        return series
    common = np.array(sorted(list(common)), dtype=np.int64)
    out = []
    for s in series:
        idx = np.searchsorted(s['windows'], common)
        ok = (idx < len(s['windows'])) & (s['windows'][idx] == common)
        out.append({**s, 'windows': common[ok], 'values': s['values'][idx[ok]]})
    return out


def compute_descriptive_stats(series_map: Dict[Tuple[str, str], dict]) -> Dict[Tuple[str, str], Dict[str, float]]:
    stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for k, s in series_map.items():
        x = s['values']
        x_f = x[np.isfinite(x)]
        if x_f.size == 0:
            stats[k] = {
                'n': 0, 'mean': float('nan'), 'std': float('nan'), 'min': float('nan'), 'max': float('nan'),
                'q10': float('nan'), 'q25': float('nan'), 'q50': float('nan'), 'q75': float('nan'), 'q90': float('nan')
            }
            continue
        stats[k] = {
            'n': int(x_f.size),
            'mean': float(np.mean(x_f)),
            'std': float(np.std(x_f)),
            'min': float(np.min(x_f)),
            'max': float(np.max(x_f)),
            'q10': float(np.quantile(x_f, 0.10)),
            'q25': float(np.quantile(x_f, 0.25)),
            'q50': float(np.quantile(x_f, 0.50)),
            'q75': float(np.quantile(x_f, 0.75)),
            'q90': float(np.quantile(x_f, 0.90)),
        }
    return stats


def write_metrics_csv(metrics: Dict[Tuple[str, str], Dict[str, float]], out_path: Path):
    headers = ['source', 'mode', 'auc', 'cohens_d', 'ks', 'qgap_med', 'qgap_90_10', 'pearson_r', 'spearman_r', 'n_pos', 'n_neg', 'n_used']
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for (src, mode), m in sorted(metrics.items()):
            w.writerow([
                src, mode,
                m.get('auc', ''), m.get('cohens_d', ''), m.get('ks', ''), m.get('qgap_med', ''), m.get('qgap_90_10', ''),
                m.get('pearson_r', ''), m.get('spearman_r', ''), m.get('n_pos', ''), m.get('n_neg', ''), m.get('n_used', ''),
            ])


def write_descriptive_csv(stats: Dict[Tuple[str, str], Dict[str, float]], out_path: Path):
    headers = ['source', 'mode', 'n', 'mean', 'std', 'min', 'max', 'q10', 'q25', 'q50', 'q75', 'q90']
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for (src, mode), s in sorted(stats.items()):
            w.writerow([
                src, mode, s.get('n', ''), s.get('mean', ''), s.get('std', ''), s.get('min', ''), s.get('max', ''),
                s.get('q10', ''), s.get('q25', ''), s.get('q50', ''), s.get('q75', ''), s.get('q90', ''),
            ])


def write_correlation_csv(series_list: List[dict], out_path: Path):
    labels = [f"{s['source']}/{s['mode']}" for s in series_list]
    n = len(series_list)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            a, b = series_list[i]['values'], series_list[j]['values']
            r = pearsonr(a, b)[0] if len(a) == len(b) and len(a) > 1 else np.nan
            mat[i, j] = r
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([''] + labels)
        for i, lab in enumerate(labels):
            row = [lab] + [mat[i, j] for j in range(n)]
            w.writerow(row)


def otsu_threshold(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float('nan')
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return vmin
    hist, edges = np.histogram(x, bins=256, range=(vmin, vmax))
    prob = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(prob)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b2))
    return float(centers[k])


def make_velocity_labels(series_map: Dict[Tuple[str, str], dict], vel_mode: str, thr_spec: str):
    key = ('velocity', vel_mode)
    if key not in series_map:
        vkeys = [k for k in series_map.keys() if k[0] == 'velocity']
        if not vkeys:
            return np.array([]), np.array([])
        key = sorted(vkeys)[0]
    v = series_map[key]['values']
    if thr_spec == 'auto':
        thr = otsu_threshold(v)
    elif thr_spec.startswith('quantile:'):
        thr = float(np.quantile(v, float(thr_spec.split(':', 1)[1])))
    elif thr_spec.startswith('value:'):
        thr = float(thr_spec.split(':', 1)[1])
    else:
        thr = otsu_threshold(v)
    y = (v > thr).astype(np.int64)
    return y, v


def compute_metrics(y: np.ndarray, ref: np.ndarray, series_map: Dict[Tuple[str, str], dict]):
    out = {}
    valid = np.unique(y).size == 2
    for k, s in series_map.items():
        x = s['values']
        n = min(len(y), len(x))
        y_n, x_n, r_n = y[:n], x[:n], ref[:n]
        pos, neg = x_n[y_n == 1], x_n[y_n == 0]
        try:
            auc_v = float(roc_auc_score(y_n, x_n)) if valid else float('nan')
        except Exception:
            auc_v = float('nan')
        try:
            mu1, mu0 = float(np.mean(pos)), float(np.mean(neg))
            s1 = float(np.std(pos, ddof=1) + 1e-12)
            s0 = float(np.std(neg, ddof=1) + 1e-12)
            sp = math.sqrt(((len(pos) - 1) * s1**2 + (len(neg) - 1) * s0**2) / max(1, (len(pos) + len(neg) - 2)))
            d_eff = (mu1 - mu0) / (sp + 1e-12)
        except Exception:
            d_eff = float('nan')
        try:
            ks_v = float(ks_2samp(pos, neg).statistic) if len(pos) > 0 and len(neg) > 0 else float('nan')
        except Exception:
            ks_v = float('nan')
        qpos = np.quantile(pos, [0.1, 0.5, 0.9]) if len(pos) > 0 else [np.nan, np.nan, np.nan]
        qneg = np.quantile(neg, [0.1, 0.5, 0.9]) if len(neg) > 0 else [np.nan, np.nan, np.nan]
        qgap_med = float(qpos[1] - qneg[1]) if np.isfinite(qpos[1]) and np.isfinite(qneg[1]) else float('nan')
        qgap_90_10 = float(qpos[2] - qneg[0]) if np.isfinite(qpos[2]) and np.isfinite(qneg[0]) else float('nan')
        r_p = pearsonr(r_n, x_n)[0] if len(r_n) == len(x_n) and len(r_n) > 1 else np.nan
        r_s = spearmanr(r_n, x_n)[0] if len(r_n) == len(x_n) and len(r_n) > 1 else np.nan
        out[k] = {
            'auc': auc_v,
            'cohens_d': float(d_eff),
            'ks': ks_v,
            'qgap_med': qgap_med,
            'qgap_90_10': qgap_90_10,
            'pearson_r': float(r_p) if np.isfinite(r_p) else float('nan'),
            'spearman_r': float(r_s) if np.isfinite(r_s) else float('nan'),
            'n_pos': int((y_n == 1).sum()),
            'n_neg': int((y_n == 0).sum()),
            'n_used': int(n),
        }
    return out


def plot_time_series(series_list: List[dict], out: Path, title: str):
    plt.figure(figsize=(14, 4))
    for s in series_list:
        plt.plot(s['windows'], s['values'], label=f"{s['source']}/{s['mode']}", lw=1.0)
    plt.xlabel('window')
    plt.ylabel('energy')
    plt.title(title)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out, dpi=140); plt.close()


def plot_hist(series_list: List[dict], out: Path, title: str):
    plt.figure(figsize=(10, 6))
    for s in series_list:
        plt.hist(s['values'], bins=40, alpha=0.35, density=True, label=f"{s['source']}/{s['mode']}")
    plt.xlabel('energy'); plt.ylabel('density'); plt.title(title); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


def plot_box(series_list: List[dict], out: Path, title: str):
    plt.figure(figsize=(12, 5))
    data = [s['values'] for s in series_list]
    labels = [f"{s['source']}/{s['mode']}" for s in series_list]
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel('energy'); plt.title(title); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


def plot_corr(series_list: List[dict], out: Path, title: str):
    labels = [f"{s['source']}/{s['mode']}" for s in series_list]
    n = len(series_list)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            a, b = series_list[i]['values'], series_list[j]['values']
            r = pearsonr(a, b)[0] if len(a) == len(b) and len(a) > 1 else np.nan
            mat[i, j] = r
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title); plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


def plot_rocs(y: np.ndarray, series_map: Dict[Tuple[str, str], dict], out: Path, title: str):
    if np.unique(y).size < 2:
        return
    plt.figure(figsize=(6, 6))
    for (src, mode), s in series_map.items():
        try:
            fpr, tpr, _ = roc_curve(y, s['values'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, label=f"{src}/{mode} (AUC={roc_auc:.3f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(title); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out, dpi=140); plt.close()


def write_html(out_dir: Path, images: Dict[str, str], metrics: Dict[Tuple[str, str], dict], label_info: str, files: Dict[str, str]):
    html = out_dir / 'report.html'
    def itag(k):
        p = images.get(k, '')
        return f'<h3>{k}</h3>\n<img src="{p}" style="max-width:100%">' if p else ''
    rows = ['<tr><th>source</th><th>mode</th><th>auc</th><th>d</th><th>ks</th><th>qgap_med</th><th>qgap_90_10</th><th>pearson_r</th><th>spearman_r</th><th>n_pos</th><th>n_neg</th><th>n_used</th></tr>']
    for (src, mode), m in sorted(metrics.items()):
        def fmt(v):
            try:
                return f"{float(v):.4f}"
            except Exception:
                return 'nan'
        rows.append('<tr>' + ''.join([
            f'<td>{src}</td>', f'<td>{mode}</td>',
            f'<td>{fmt(m.get("auc"))}</td>', f'<td>{fmt(m.get("cohens_d"))}</td>', f'<td>{fmt(m.get("ks"))}</td>',
            f'<td>{fmt(m.get("qgap_med"))}</td>', f'<td>{fmt(m.get("qgap_90_10"))}</td>', f'<td>{fmt(m.get("pearson_r"))}</td>', f'<td>{fmt(m.get("spearman_r"))}</td>',
            f'<td>{m.get("n_pos","")}</td>', f'<td>{m.get("n_neg","")}</td>', f'<td>{m.get("n_used","")}</td>',
        ]) + '</tr>')
    table = '<table border="1" cellspacing="0" cellpadding="4">' + '\n'.join(rows) + '</table>'
    html_s = f"""
<!DOCTYPE html><html><head><meta charset='utf-8'><title>Energy Report</title>
<style>body{{font-family:Arial;margin:20px}} img{{border:1px solid #ccc;margin-bottom:16px}}</style></head>
<body>
<h1>Energy Analysis Report</h1>
<div>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
<p>Labeling: {label_info}</p>
{itag('time_series')}
{itag('histograms')}
{itag('boxplot')}
{itag('correlation')}
{itag('roc_curves')}
<h2>Metrics</h2>
{table}
<h2>Downloads</h2>
<ul>
  <li><a href="{files.get('metrics_json','')}">metrics.json</a></li>
  <li><a href="{files.get('metrics_csv','')}">metrics.csv</a></li>
  <li><a href="{files.get('descriptive_csv','')}">descriptive_stats.csv</a></li>
  <li><a href="{files.get('correlation_csv','')}">correlation.csv</a></li>
</ul>
</body></html>
"""
    with open(html, 'w', encoding='utf-8') as f:
        f.write(html_s)
    return html


def main():
    ap = argparse.ArgumentParser(description='Analyze energy JSONL and generate report')
    ap.add_argument('--input-dir', type=str, default=str(Path(__file__).with_name('energy_sweep_out')))
    ap.add_argument('--output-dir', type=str, default=None)
    ap.add_argument('--velocity-mode', type=str, default='l2_mean')
    ap.add_argument('--vel-threshold', type=str, default='auto', help="auto | quantile:0.7 | value:12.0")
    ap.add_argument('--title', type=str, default='Energy Analysis')
    args = ap.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (in_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_jsonl_files(in_dir)
    if not files:
        print(f"[ERR] No stream_energy_*.jsonl under {in_dir}"); return
    series = [load_energy_jsonl(p) for p in files]
    series = align_common_windows(series)

    # map
    smap: Dict[Tuple[str, str], dict] = {}
    for s in series:
        smap[(s['source'], s['mode'])] = s

    # labels
    y, ref = make_velocity_labels(smap, args.velocity_mode, args.vel_threshold)
    label_info = f"velocity/{args.velocity_mode} with thr={args.vel_threshold}"
    if y.size == 0:
        # fallback: use first series quantiles
        s0 = series[0]
        q25, q75 = np.quantile(s0['values'], [0.25, 0.75])
        y = (s0['values'] > q75).astype(np.int64)
        ref = s0['values'].copy()
        label_info = f"self-quantile labeling on {s0['source']}/{s0['mode']} (Q75)"

    # metrics
    metrics = compute_metrics(y, ref, smap)
    metrics_json = out_dir / 'metrics.json'
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump({f"{k[0]}/{k[1]}": v for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)
    # descriptive stats
    desc = compute_descriptive_stats(smap)
    metrics_csv = out_dir / 'metrics.csv'
    desc_csv = out_dir / 'descriptive_stats.csv'
    write_metrics_csv(metrics, metrics_csv)
    write_descriptive_csv(desc, desc_csv)

    # plots
    imgs = {}
    plot_time_series(list(smap.values()), out_dir / 'time_series.png', args.title + ' - Time Series'); imgs['time_series'] = 'time_series.png'
    plot_hist(list(smap.values()), out_dir / 'histograms.png', args.title + ' - Distributions'); imgs['histograms'] = 'histograms.png'
    plot_box(list(smap.values()), out_dir / 'boxplot.png', args.title + ' - Boxplot'); imgs['boxplot'] = 'boxplot.png'
    series_list = list(smap.values())
    plot_corr(series_list, out_dir / 'correlation.png', args.title + ' - Pearson Correlation'); imgs['correlation'] = 'correlation.png'
    # correlation csv
    corr_csv = out_dir / 'correlation.csv'
    write_correlation_csv(series_list, corr_csv)
    if np.unique(y).size == 2:
        plot_rocs(y, smap, out_dir / 'roc_curves.png', 'ROC vs velocity labels'); imgs['roc_curves'] = 'roc_curves.png'

    files = {
        'metrics_json': 'metrics.json',
        'metrics_csv': 'metrics.csv',
        'descriptive_csv': 'descriptive_stats.csv',
        'correlation_csv': 'correlation.csv',
    }
    html = write_html(out_dir, imgs, metrics, label_info, files)
    print(f"Report written to: {html}")


if __name__ == '__main__':
    main()
