import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from action_classification.embedding.common import SeqDataset, SeqEncoder, Sample, ensure_dir, flatten_codes
from action_classification.data.features import read_json_sample


def export_embeddings(model: SeqEncoder, dataset: SeqDataset, device: torch.device, l2_normalize: bool) -> Tuple[np.ndarray, List[str]]:
    """Run the encoder to obtain pooled embeddings for the provided dataset."""
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    outs: List[np.ndarray] = []
    paths: List[str] = []
    with torch.no_grad():
        for x, _, _, p in loader:
            x = x.to(device)
            _, h = model(x)
            mask = (x != model.pad_id).float().unsqueeze(-1)
            masked = h * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = masked.sum(dim=1) / denom
            if l2_normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outs.append(pooled.cpu().numpy())
            paths.extend([str(pi) for pi in p])
    X = np.concatenate(outs, axis=0) if outs else np.zeros((0, model.embed.embedding_dim), dtype=np.float32)
    return X, paths


def scan_label_samples(label_dir: Path, codes_subdir: str) -> Tuple[List[Sample], Path, Dict[str, int]]:
    """Collect samples under a label directory.

    Returns samples list, the directory used as scan root, and basic stats.
    """
    if codes_subdir:
        candidate = label_dir / codes_subdir
        search_root = candidate if candidate.is_dir() else label_dir
    else:
        search_root = label_dir

    files = sorted(search_root.rglob('*.json'))
    stats = {
        'total_files': len(files),
        'parse_error': 0,
        'empty_codes': 0,
    }
    samples: List[Sample] = []
    for fp in files:
        try:
            obj = read_json_sample(fp)
        except Exception:
            stats['parse_error'] += 1
            continue
        codes = flatten_codes(obj)
        if not codes:
            stats['empty_codes'] += 1
            continue
        samples.append(
            Sample(
                path=fp,
                label_name=label_dir.name,
                label=0,
                codes=codes,
            )
        )
    stats['used_samples'] = len(samples)
    return samples, search_root, stats


def main():
    ap = argparse.ArgumentParser(description='Export LSTM embeddings per labeled folder.')
    ap.add_argument('--json-root', type=str, required=True, help='Root directory containing labeled sub-folders')
    ap.add_argument('--model-pt', type=str, required=True, help='Path to the trained model_best.pt file')
    ap.add_argument('--out-subdir', type=str, default='embeddings', help='Sub-directory (under each label folder) to store outputs')
    ap.add_argument('--codes-subdir', type=str, default='codes', help='Sub-folder that holds JSON codes files (fallback to label folder if missing)')
    ap.add_argument('--l2-normalize', action='store_true', help='Apply L2 normalization to the output embeddings')
    ap.add_argument('--device', type=str, default=None, help='cuda | cpu (auto if None)')
    args = ap.parse_args()

    json_root = Path(args.json_root).resolve()
    if not json_root.is_dir():
        raise FileNotFoundError(f'json_root not found: {json_root}')

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Using device: {device}")

    ckpt = torch.load(args.model_pt, map_location=device)
    cfg = ckpt['config']
    print(f"[Info] Loaded model trained with config:\n{yaml.dump(cfg, sort_keys=False)}")

    # Dataset params from checkpoint
    seq_cfg = cfg.get('sequence', {}) or {}
    max_len_cfg = seq_cfg.get('max_len_resolved') or seq_cfg.get('max_len')
    if max_len_cfg is None:
        raise RuntimeError('sequence.max_len_resolved or sequence.max_len missing in model config')
    max_len = int(max_len_cfg)
    pad_id = int(cfg['sequence'].get('pad_id', 0))

    d_model = int(cfg['model']['d_model'])
    rnn_type = str(cfg['model'].get('rnn_type', 'lstm'))
    num_layers = int(cfg['model'].get('num_layers', 2))
    dropout = float(cfg['model'].get('dropout', 0.1))
    bidirectional = bool(cfg['model'].get('bidirectional', True))

    num_tokens = ckpt['state_dict']['embed.weight'].shape[0]
    print(f"[Info] Inferred num_tokens={num_tokens} from model state dict.")

    model = SeqEncoder(
        num_tokens=num_tokens,
        d_model=d_model,
        rnn_type=rnn_type,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        pad_id=pad_id,
    )
    model.load_state_dict(ckpt['state_dict'])

    label_dirs = sorted([p for p in json_root.iterdir() if p.is_dir()])
    if not label_dirs:
        raise RuntimeError(f'No label sub-directories found under {json_root}')

    summary = []
    for label_dir in label_dirs:
        print(f"\n[Info] Processing label folder: {label_dir.name}")
        samples, search_root, stats = scan_label_samples(label_dir, args.codes_subdir)
        if stats['total_files'] == 0:
            print(f"  [Warn] No JSON files found under {search_root}. Skipping.")
            continue
        if not samples:
            print(
                f"  [Warn] All samples skipped (parse_error={stats['parse_error']}, empty_codes={stats['empty_codes']}). Skipping."
            )
            continue

        ds = SeqDataset(samples, max_len=max_len, pad_id=pad_id)
        X, paths = export_embeddings(model, ds, device=device, l2_normalize=args.l2_normalize)

        out_dir = label_dir / args.out_subdir
        ensure_dir(out_dir)

        np.save(out_dir / 'embed.npy', X)

        paths_txt = out_dir / 'paths.txt'
        with paths_txt.open('w', encoding='utf-8') as f:
            for p in paths:
                try:
                    rel = str(Path(p).resolve().relative_to(search_root))
                except Exception:
                    rel = p
                f.write(rel + '\n')

        meta = {
            'label_name': label_dir.name,
            'num_embeddings': int(X.shape[0]),
            'json_source': str(search_root),
            'model_checkpoint': str(Path(args.model_pt).resolve()),
            'stats': stats,
            'l2_normalize': bool(args.l2_normalize),
            'generated_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        }
        meta_path = out_dir / 'meta.json'
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

        print(f"  [Saved] Embeddings: {out_dir / 'embed.npy'}")
        print(f"  [Saved] Paths: {paths_txt}")
        print(f"  [Saved] Meta: {meta_path}")

        summary.append({
            'label': label_dir.name,
            'num_embeddings': int(X.shape[0]),
            'out_dir': str(out_dir),
        })

    if summary:
        summary_path = json_root / 'embedding_export_summary.json'
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\n[Done] Summary saved to {summary_path}")
    else:
        print("\n[Warn] No embeddings were exported. Check input data.")


if __name__ == '__main__':
    main()
