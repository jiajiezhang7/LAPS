#!/usr/bin/env python3
import argparse, csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_series(csv_path):
    by_run = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row['run']
            e = int(float(row['epoch']))
            tl = float(row['train_loss'])
            vl = float(row['val_loss'])
            by_run.setdefault(run, {'epochs': [], 'train': [], 'val': []})
            by_run[run]['epochs'].append(e)
            by_run[run]['train'].append(tl)
            by_run[run]['val'].append(vl)
    # sort by epoch per run
    for run, d in by_run.items():
        zipped = sorted(zip(d['epochs'], d['train'], d['val']))
        e, t, v = zip(*zipped)
        d['epochs'], d['train'], d['val'] = list(e), list(t), list(v)
    return by_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outpdf', required=True)
    args = ap.parse_args()

    series = load_series(args.csv)

    plt.figure(figsize=(6.5, 4.0), dpi=200)
    for run, d in series.items():
        plt.plot(d['epochs'], d['train'], marker='o', label=f"{run} Train")
        plt.plot(d['epochs'], d['val'], marker='s', linestyle='--', label=f"{run} Val")
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Motion Tokenizer Training Loss (Fig S2)')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(args.outpdf), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.outpdf)

if __name__ == '__main__':
    main()

