#!/usr/bin/env python3
import argparse, re, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCH_LINE = re.compile(r"Epoch\s+(\d+)\s*\|\s*Train Loss:\s*([0-9.eE+-]+)\s*\|\s*Val Loss:\s*([0-9.eE+-]+)")

def parse_output_log(log_path):
    epochs, train_losses, val_losses = [], [], []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = EPOCH_LINE.search(line)
            if m:
                e = int(m.group(1))
                tl = float(m.group(2))
                vl = float(m.group(3))
                epochs.append(e)
                train_losses.append(tl)
                val_losses.append(vl)
    return epochs, train_losses, val_losses

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='Paths to wandb run dirs (e.g., wandb/run-...-... )')
    ap.add_argument('--outpdf', required=True)
    ap.add_argument('--outcsv', required=True)
    args = ap.parse_args()

    series = []
    for rd in args.runs:
        name = os.path.basename(rd)
        log_path = os.path.join(rd, 'files', 'output.log')
        if not os.path.exists(log_path):
            raise FileNotFoundError(f'output.log not found in {rd}')
        epochs, train_losses, val_losses = parse_output_log(log_path)
        series.append({'run': name, 'epochs': epochs, 'train': train_losses, 'val': val_losses})

    # Save CSV-like TSV
    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    with open(args.outcsv, 'w') as f:
        f.write('run,epoch,train_loss,val_loss\n')
        for s in series:
            for e, tl, vl in zip(s['epochs'], s['train'], s['val']):
                f.write(f"{s['run']},{e},{tl},{vl}\n")

    # Plot
    plt.figure(figsize=(6.5, 4.0), dpi=200)
    for s in series:
        plt.plot(s['epochs'], s['train'], marker='o', label=f"{s['run']} Train")
        plt.plot(s['epochs'], s['val'], marker='s', linestyle='--', label=f"{s['run']} Val")
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

