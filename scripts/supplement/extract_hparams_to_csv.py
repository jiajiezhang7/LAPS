#!/usr/bin/env python3
import argparse, os, yaml, csv

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def infer_dataset(conf):
    ckpt = conf.get('checkpoint_dir', {}).get('value') if isinstance(conf.get('checkpoint_dir'), dict) else conf.get('checkpoint_dir')
    root = conf.get('root_dir', {}).get('value') if isinstance(conf.get('root_dir'), dict) else conf.get('root_dir')
    text = (ckpt or '') + ' ' + (root or '')
    if 'd01' in text.lower():
        return 'D01'
    if 'd02' in text.lower():
        return 'D02'
    return ''

def flatten_hparams(conf):
    def getv(key, default=''):
        v = conf.get(key)
        if isinstance(v, dict) and 'value' in v:
            return v['value']
        return v if v is not None else default
    loss = getv('loss', {}) or {}
    return {
        'run_name': getv('run_name',''),
        'dataset': infer_dataset({k:getv(k) for k in conf}),
        'num_layers': getv('num_layers',''),
        'num_heads': getv('num_heads',''),
        'hidden_dim': getv('hidden_dim',''),
        'codebook_size': getv('codebook_size',''),
        'batch_size': getv('batch_size',''),
        'lr': getv('lr',''),
        'num_epochs': getv('num_epochs',''),
        'weight_decay': getv('weight_decay',''),
        'adam_betas': getv('adam_betas',''),
        'attn_pdrop': getv('attn_pdrop',''),
        'amp': getv('amp',''),
        'z_noise_std': getv('z_noise_std',''),
        'track_method': getv('track_method',''),
        'true_horizon': getv('true_horizon',''),
        'track_pred_horizon': getv('track_pred_horizon',''),
        'num_tracks': getv('num_tracks',''),
        'train_datasets': getv('train_datasets',''),
        'val_datasets': getv('val_datasets',''),
        'loss_fn': loss.get('loss_fn',''),
        'focal_alpha': loss.get('focal_alpha',''),
        'focal_gamma': loss.get('focal_gamma',''),
        'codebook_diversity_weight': loss.get('codebook_diversity_weight',''),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='Paths to wandb run dirs')
    ap.add_argument('--out', required=True, help='Output CSV for Table S1 values')
    args = ap.parse_args()

    rows = []
    for rd in args.runs:
        cfg_path = os.path.join(rd, 'files', 'config.yaml')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(cfg_path)
        conf = load_config(cfg_path)
        rows.append(flatten_hparams(conf))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

if __name__ == '__main__':
    main()

