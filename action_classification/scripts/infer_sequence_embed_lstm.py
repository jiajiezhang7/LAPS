import argparse
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from action_classification.embedding.common import SeqDataset, SeqEncoder, ensure_dir, Sample, flatten_codes
from action_classification.data.features import read_json_sample


# ------------------------- Export -------------------------

def export_embeddings(model: SeqEncoder, dataset: SeqDataset, device: torch.device, l2_normalize: bool) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    outs: List[np.ndarray] = []
    paths: List[str] = []
    with torch.no_grad():
        for x, L, y, p in loader:
            x = x.to(device)
            _, h = model(x)
            # mean-pool over valid positions (exclude pad)
            # mask shape [B, T, 1]
            B, T, H = h.shape
            mask = (x != model.pad_id).float().unsqueeze(-1)
            masked = h * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = masked.sum(dim=1) / denom  # [B, H]
            if l2_normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outs.append(pooled.cpu().numpy())
            paths.extend([str(pi) for pi in p])
    X = np.concatenate(outs, axis=0)
    return X, paths


# ------------------------- Data Scan (Unlabeled) -------------------------

def scan_unlabeled_samples(json_root: Path) -> List[Sample]:
    """Scan all JSON files under json_root and build unlabeled Samples.

    - Does NOT infer labels from folder names.
    - Uses flatten_codes() to obtain a single list of codes per file; skips empty.
    - Sets label_name='unknown', label=0 for compatibility with SeqDataset.
    """
    files = sorted(Path(json_root).rglob('*.json'))
    if not files:
        raise RuntimeError(f'No JSON files found under: {json_root}')
    samples: List[Sample] = []
    n_empty = 0
    n_err = 0
    for fp in files:
        try:
            obj = read_json_sample(fp)
        except Exception:
            n_err += 1
            continue
        codes = flatten_codes(obj)
        if not codes:
            n_empty += 1
            continue
        samples.append(Sample(path=fp, label_name='unknown', label=0, codes=codes))
    if not samples:
        raise RuntimeError(
            'All samples are empty or invalid sequences '
            f'(files={len(files)}, parse_error={n_err}, empty_codes={n_empty})'
        )
    return samples


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description='Export sequence embeddings using a pre-trained LSTM/GRU model.')
    ap.add_argument('--json-root', type=str, required=True, help='Root directory of inference JSON outputs to embed')
    ap.add_argument('--model-pt', type=str, required=True, help='Path to the trained model_best.pt file')
    ap.add_argument('--out-dir', type=str, required=True, help='Directory to save the exported embeddings')
    ap.add_argument('--l2-normalize', action='store_true', help='Apply L2 normalization to the output embeddings')
    ap.add_argument('--device', type=str, default=None, help='cuda | cpu (auto if None)')
    args = ap.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Using device: {device}")

    # Load model checkpoint
    ckpt = torch.load(args.model_pt, map_location=device)
    cfg = ckpt['config']
    print(f"[Info] Loaded model trained with config:\n{yaml.dump(cfg, sort_keys=False)}")

    # Load data for inference (unlabeled)
    json_root = Path(args.json_root).resolve()
    samples = scan_unlabeled_samples(json_root)
    print(f"[Info] Loaded {len(samples)} unlabeled samples for inference from {json_root}")

    # Build dataset
    seq_cfg = cfg.get('sequence', {}) or {}
    max_len_cfg = seq_cfg.get('max_len_resolved') or seq_cfg.get('max_len')
    if max_len_cfg is None:
        raise RuntimeError('sequence.max_len_resolved or sequence.max_len missing in model config')
    max_len = int(max_len_cfg)
    pad_id = int(cfg['sequence'].get('pad_id', 0))
    ds_all = SeqDataset(samples, max_len=max_len, pad_id=pad_id)

    # Re-create model from config
    d_model = int(cfg['model']['d_model'])
    rnn_type = str(cfg['model'].get('rnn_type', 'lstm'))
    num_layers = int(cfg['model'].get('num_layers', 2))
    dropout = float(cfg['model'].get('dropout', 0.1))
    bidirectional = bool(cfg['model'].get('bidirectional', True))
    
    # Determine vocabulary size from loaded config/model state
    num_tokens = ckpt['state_dict']['embed.weight'].shape[0]
    print(f"[Info] Inferred num_tokens={num_tokens} from model state dict.")

    model = SeqEncoder(num_tokens=num_tokens, d_model=d_model, rnn_type=rnn_type, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, pad_id=pad_id)
    model.load_state_dict(ckpt['state_dict'])

    # Export embeddings
    out_base = Path(args.out_dir).resolve()
    # Use json_root folder name as subdirectory name
    json_root_name = json_root.name
    out_dir = (out_base / json_root_name).resolve()
    ensure_dir(out_dir)

    X, paths = export_embeddings(model, ds_all, device=device, l2_normalize=args.l2_normalize)
    
    np.save(out_dir / 'embed.npy', X)
    
    # Paths as relative to json_root if possible
    with open(out_dir / 'paths.txt', 'w', encoding='utf-8') as f:
        for p in paths:
            try:
                rp = str(Path(p).resolve().relative_to(json_root))
            except Exception:
                rp = p
            f.write(rp + '\n')

    print(f"\n[Success] Export complete. Results saved to:")
    print(out_dir)
    print(f"[Saved] Embeddings: {out_dir / 'embed.npy'}")
    print(f"[Saved] Paths: {out_dir / 'paths.txt'}")


if __name__ == '__main__':
    main()
