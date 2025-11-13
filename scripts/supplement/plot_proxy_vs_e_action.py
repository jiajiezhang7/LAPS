#!/usr/bin/env python3
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_energy_jsonl(path):
    xs, ys = [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            xs.append(int(d.get('window', len(xs))))
            ys.append(float(d['energy']))
    return np.asarray(xs, dtype=int), np.asarray(ys, dtype=float)


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


def resolve_threshold(values: np.ndarray, spec: str) -> float:
    spec = (spec or 'auto').strip()
    if spec == 'auto':
        return otsu_threshold(values)
    if spec.startswith('quantile:'):
        q = float(spec.split(':', 1)[1])
        return float(np.quantile(values, q))
    if spec.startswith('value:'):
        return float(spec.split(':', 1)[1])
    return otsu_threshold(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_dir', required=True, help='Directory that contains stream_energy_*.jsonl for one sample')
    ap.add_argument('--outpdf', required=True)
    ap.add_argument('--pseudo_thr', default='auto', help='auto | quantile:0.9 | value:0.02')
    # Allow overriding file names if needed
    ap.add_argument('--proxy_file', default=None, help='Path to velocity proxy JSONL; defaults to sample_dir/stream_energy_velocity_token_diff_l2_mean.jsonl')
    ap.add_argument('--eaction_file', default=None, help='Path to E_action JSONL; defaults to sample_dir/stream_energy_quantized_token_diff_l2_mean.jsonl')
    args = ap.parse_args()

    proxy_path = args.proxy_file or os.path.join(args.sample_dir, 'stream_energy_velocity_token_diff_l2_mean.jsonl')
    eact_path = args.eaction_file or os.path.join(args.sample_dir, 'stream_energy_quantized_token_diff_l2_mean.jsonl')

    if not os.path.isfile(proxy_path) or not os.path.isfile(eact_path):
        raise SystemExit(f"Missing file(s). proxy={proxy_path} e_action={eact_path}")

    x_proxy, y_proxy = load_energy_jsonl(proxy_path)
    x_eact, y_eact = load_energy_jsonl(eact_path)

    # align by window index
    common = np.intersect1d(x_proxy, x_eact)
    if common.size == 0:
        raise SystemExit('No overlapping window indices between proxy and E_action')
    i1 = np.searchsorted(x_proxy, common)
    i2 = np.searchsorted(x_eact, common)
    xs = common
    v_proxy = y_proxy[i1]
    v_eact = y_eact[i2]

    thr = resolve_threshold(v_proxy, args.pseudo_thr)
    y_pseudo = (v_proxy >= thr).astype(int)

    # Plot
    plt.figure(figsize=(7.5, 4.2), dpi=200)
    plt.plot(xs, v_proxy, label='Proxy: velocity token diff L2-mean', color='#1f77b4', lw=1.2)
    plt.plot(xs, v_eact, label='E_action: quantized token diff L2-mean', color='#d62728', lw=1.2, alpha=0.9)
    plt.axhline(thr, color='gray', linestyle='--', linewidth=1.1, label=f'Pseudo thr ({args.pseudo_thr}) = {thr:.4f}')

    # Visualize pseudo-labels (shaded positives + thin step)
    ymin, ymax = float(np.min([v_proxy.min(), v_eact.min()])), float(np.max([v_proxy.max(), v_eact.max()]))
    for i in range(len(xs)):
        if y_pseudo[i] == 1:
            plt.axvspan(xs[i]-0.5, xs[i]+0.5, color='#1f77b4', alpha=0.08)
    y_step = ymin + 0.06 * (ymax - ymin) * y_pseudo
    plt.step(xs, y_step, where='post', color='k', lw=0.8, alpha=0.6, label='y_pseudo (from velocity, unsup)')

    plt.xlabel('Window index')
    plt.ylabel('Energy')
    # Title disabled per request
    # plt.title('Proxy (velocity) vs y_pseudo vs E_action (Fig S3)')
    plt.legend(fontsize=8, loc='upper right')
    os.makedirs(os.path.dirname(args.outpdf), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.outpdf)

if __name__ == '__main__':
    main()
