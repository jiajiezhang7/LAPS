import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from .common import SeqDataset, SeqEncoder, ensure_dir, scan_samples, set_seed


# ------------------------- Train/Eval -------------------------

def stratified_split(indices: np.ndarray, labels: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split indices into train and validation sets using stratified sampling.
    """
    rng = np.random.RandomState(seed)
    unique = np.unique(labels)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for c in unique:
        cls_idx = indices[labels == c]
        if cls_idx.size == 0:
            continue
        n_val = max(1, int(round(val_ratio * cls_idx.size)))
        rng.shuffle(cls_idx)
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def build_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    # inverse frequency weights
    classes, counts = np.unique(labels, return_counts=True)
    count_map = {int(c): float(cnt) for c, cnt in zip(classes, counts)}
    weights = np.asarray([1.0 / count_map[int(y)] for y in labels], dtype=np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(labels), replacement=True)
    return sampler


def compute_ntp_loss(logits: torch.Tensor, x: torch.Tensor, pad_id: int = 0) -> Tuple[torch.Tensor, int]:
    # logits: [B, T, V], x: [B, T]
    # Predict next token: input positions 0..T-2 predict target positions 1..T-1
    B, T, V = logits.shape
    if T <= 1:
        return logits.new_tensor(0.0), 0
    logits_next = logits[:, :-1, :].contiguous().view(-1, V)
    targets = x[:, 1:].contiguous().view(-1)  # int64
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    loss = loss_fn(logits_next, targets)
    # Count valid targets (non-pad)
    valid = (targets != pad_id).sum().item()
    return loss, int(valid)


def evaluate(model: SeqEncoder, loader: DataLoader, device: torch.device, pad_id: int = 0) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, L, y, _ in loader:
            x = x.to(device)
            logits, _ = model(x)
            loss, valid = compute_ntp_loss(logits, x, pad_id=pad_id)
            total_loss += loss.item() * max(valid, 1)
            total_tokens += valid
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(max(avg_loss, 1e-9), 20.0))  # clamp to avoid overflow
    return {'loss': avg_loss, 'ppl': ppl, 'tokens': int(total_tokens)}


def train_ntp(
    model: SeqEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    clip_grad_norm: float,
    epochs: int,
    pad_id: int,
    log_interval: int = 50,
    patience: int = 6,
    min_delta: float = 0.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float('inf')
    best_state = None
    best_epoch = 0
    history = {'train': [], 'val': []}
    steps = 0
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        run_tokens = 0
        for x, L, y, _ in train_loader:
            x = x.to(device)
            logits, _ = model(x)
            loss, valid = compute_ntp_loss(logits, x, pad_id=pad_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad_norm and clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            run_loss += loss.item() * max(valid, 1)
            run_tokens += valid
            steps += 1
            if steps % log_interval == 0:
                avg = run_loss / max(run_tokens, 1)
                print(f"[Train] epoch={ep} step={steps} ntp_loss={avg:.4f}")
                run_loss = 0.0
                run_tokens = 0

        # epoch end: eval
        tr = evaluate(model, train_loader, device, pad_id)
        va = evaluate(model, val_loader, device, pad_id)
        history['train'].append(tr)
        history['val'].append(va)
        print(f"[Epoch {ep}] train_loss={tr['loss']:.4f} ppl={tr['ppl']:.2f} | val_loss={va['loss']:.4f} ppl={va['ppl']:.2f}")

        if va['loss'] < best_val - float(min_delta):
            print(f"[Improve] val_loss {best_val:.6f} -> {va['loss']:.6f}")
            best_val = va['loss']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1
            print(f"[EarlyStop] no_improve={no_improve}/{patience} (min_delta={min_delta})")
            if patience and no_improve >= patience:
                print(f"[EarlyStop] Stop at epoch {ep}. Best epoch={best_epoch} val_loss={best_val:.6f}")
                break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, {'best_val_loss': float(best_val), 'best_epoch': int(best_epoch), 'epochs_ran': int(ep)}


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser(description='Train LSTM/GRU sequence embedding over code indices (NTP)')
    ap.add_argument('--json-root', type=str, required=True, help='Root directory of inference JSON outputs for training')
    ap.add_argument('--config', type=str, default='action_classification/configs/sequence_embed.yaml', help='YAML config path')
    ap.add_argument('--out-dir', type=str, default=None, help='Override export.out_dir in config')
    ap.add_argument('--device', type=str, default=None, help='cuda | cpu (auto if None)')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get('seed', 0))
    set_seed(seed)

    json_root = Path(args.json_root).resolve()
    samples, label_names, max_code_seen = scan_samples(json_root)
    print(f"[Info] Loaded {len(samples)} samples across {len(label_names)} classes: {label_names}")

    lengths = np.asarray([len(s.codes) for s in samples], dtype=np.int64)
    if lengths.size == 0:
        raise RuntimeError('No valid sequences found for training')

    seq_cfg = cfg.get('sequence', {}) or {}

    def _quantile_safe(values: np.ndarray, q: float) -> float:
        q = max(0.0, min(1.0, float(q)))
        return float(np.quantile(values, q))

    length_stats = {
        'count': int(lengths.size),
        'min': int(lengths.min()),
        'max': int(lengths.max()),
        'mean': float(lengths.mean()),
        'std': float(lengths.std()),
        'p10': _quantile_safe(lengths, 0.10),
        'p25': _quantile_safe(lengths, 0.25),
        'p50': _quantile_safe(lengths, 0.50),
        'p75': _quantile_safe(lengths, 0.75),
        'p90': _quantile_safe(lengths, 0.90),
        'p95': _quantile_safe(lengths, 0.95),
        'p98': _quantile_safe(lengths, 0.98),
        'p99': _quantile_safe(lengths, 0.99),
    }

    unique_lengths, unique_counts = np.unique(lengths, return_counts=True)
    histogram_discrete = [
        {'length': int(L), 'count': int(C)}
        for L, C in zip(unique_lengths.tolist(), unique_counts.tolist())
    ]
    max_hist_entries = int(seq_cfg.get('auto_histogram_max_entries', 60))
    if len(histogram_discrete) > max_hist_entries:
        # keep lower and upper segments to avoid huge dumps
        head = histogram_discrete[: max_hist_entries // 2]
        tail = histogram_discrete[-max_hist_entries // 2 :]
        histogram_discrete = head + [{'length': None, 'count': None}] + tail

    hist_bins = int(seq_cfg.get('auto_histogram_bins', 20))
    hist_bins = max(1, hist_bins)
    bin_counts, bin_edges = np.histogram(lengths, bins=hist_bins)
    histogram_binned = {
        'bin_edges': [float(x) for x in bin_edges.tolist()],
        'counts': [int(x) for x in bin_counts.tolist()],
    }

    max_len_cfg = seq_cfg.get('max_len', 'auto')
    auto_used = False
    auto_info = {}
    if (isinstance(max_len_cfg, str) and max_len_cfg.lower() == 'auto') or (
        isinstance(max_len_cfg, (int, float)) and max_len_cfg <= 0
    ):
        auto_used = True
        auto_quantile = float(seq_cfg.get('auto_quantile', 0.98))
        auto_multiple = int(seq_cfg.get('auto_multiple', 16))
        auto_cap = seq_cfg.get('auto_cap', None)
        quantile_value = _quantile_safe(lengths, auto_quantile)
        resolved = quantile_value
        if auto_multiple and auto_multiple > 0:
            resolved = math.ceil(resolved / float(auto_multiple)) * auto_multiple
        resolved = max(resolved, length_stats['p90'])  # avoid too small max_len
        if auto_cap is not None:
            try:
                resolved = min(resolved, float(auto_cap))
            except Exception:
                pass
        resolved = min(resolved, float(length_stats['max']))
        max_len = int(max(1, round(resolved)))
        auto_info = {
            'auto_quantile': auto_quantile,
            'quantile_value': float(quantile_value),
            'auto_multiple': auto_multiple,
            'auto_cap': auto_cap,
        }
    else:
        try:
            max_len = int(max_len_cfg)
        except Exception:
            raise ValueError(f"Invalid sequence.max_len value: {max_len_cfg}")

    num_truncated = int(np.sum(lengths > max_len))
    trunc_ratio = float(num_truncated / max(lengths.size, 1))

    print("[Seq] Length statistics:")
    for k in ['count', 'min', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p98', 'p99', 'max', 'mean', 'std']:
        v = length_stats.get(k)
        if isinstance(v, float):
            print(f"  - {k}: {v:.2f}")
        else:
            print(f"  - {k}: {v}")
    print(f"[Seq] Using max_len={max_len} ({'auto' if auto_used else 'config'}) | truncated={num_truncated} ({trunc_ratio*100:.2f}%)")

    if auto_used:
        print("[Seq] Auto max_len details:")
        print(f"  - auto_quantile={auto_info['auto_quantile']}")
        print(f"  - quantile_value={auto_info['quantile_value']:.2f}")
        print(f"  - auto_multiple={auto_info['auto_multiple']}")
        if auto_info['auto_cap'] is not None:
            print(f"  - auto_cap={auto_info['auto_cap']}")

    cfg.setdefault('sequence', {})['max_len_resolved'] = int(max_len)
    cfg['sequence']['truncation_ratio'] = trunc_ratio

    length_analysis = {
        'stats': length_stats,
        'auto_used': auto_used,
        'auto_info': auto_info,
        'max_len_resolved': int(max_len),
        'num_truncated': num_truncated,
        'truncation_ratio': trunc_ratio,
        'histogram_discrete': histogram_discrete,
        'histogram_binned': histogram_binned,
    }

    # Determine vocabulary size (num_tokens) = codebook_size + 1 (for PAD)
    cfg_k = int(cfg.get('codebook_size', max_code_seen + 1))
    inferred_k = max(cfg_k, max_code_seen + 1)
    num_tokens = int(inferred_k + 1)  # +1 offset due to PAD=0
    print(f"[Info] codebook_size(inferred)={inferred_k}, num_tokens(emb)={num_tokens}")

    # Build datasets
    pad_id = int(cfg['sequence'].get('pad_id', 0))

    # indices for split
    all_idx = np.arange(len(samples), dtype=np.int64)
    labels_np = np.asarray([s.label for s in samples], dtype=np.int64)
    val_split = float(cfg['train'].get('val_split', 0.1))
    train_idx, val_idx = stratified_split(all_idx, labels_np, val_ratio=val_split, seed=seed)
    if train_idx.size == 0 or val_idx.size == 0:
        raise RuntimeError('Stratified split failed (empty train/val). Check class distribution and val_split.')
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    ds_train = SeqDataset(train_samples, max_len=max_len, pad_id=pad_id)
    ds_val = SeqDataset(val_samples, max_len=max_len, pad_id=pad_id)

    print(f"[Info] Split: train={len(ds_train)} val={len(ds_val)}")

    # Dataloaders
    batch_size = int(cfg['train']['batch_size'])
    num_workers = int(cfg['train'].get('num_workers', 2))
    pin_memory = bool(cfg['train'].get('pin_memory', True))

    if bool(cfg['train'].get('balance_sampler', True)):
        sampler = build_sampler(ds_train.labels)
        train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Using device: {device}")

    # Model
    d_model = int(cfg['model']['d_model'])
    rnn_type = str(cfg['model'].get('rnn_type', 'lstm'))
    num_layers = int(cfg['model'].get('num_layers', 2))
    dropout = float(cfg['model'].get('dropout', 0.1))
    bidirectional = bool(cfg['model'].get('bidirectional', True))

    model = SeqEncoder(num_tokens=num_tokens, d_model=d_model, rnn_type=rnn_type, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, pad_id=pad_id)

    # Train
    lr = float(cfg['train']['lr'])
    weight_decay = float(cfg['train']['weight_decay'])
    clip_grad_norm = float(cfg['train'].get('clip_grad_norm', 1.0))
    epochs = int(cfg['train']['epochs'])
    patience = int(cfg['train'].get('patience', 6))
    min_delta = float(cfg['train'].get('min_delta', 0.0))

    history, summary = train_ntp(
        model,
        train_loader,
        val_loader,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        epochs=epochs,
        pad_id=pad_id,
        log_interval=50,
        patience=patience,
        min_delta=min_delta,
    )

    # Export
    out_base = Path(args.out_dir) if args.out_dir else Path(cfg['export']['out_dir'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = (out_base / timestamp).resolve()
    ensure_dir(out_dir)

    # Save artifacts
    torch.save({'state_dict': model.state_dict(), 'config': cfg, 'label_names': label_names}, out_dir / 'model_best.pt')
    with open(out_dir / 'train_log.json', 'w', encoding='utf-8') as f:
        json.dump({'history': history, 'summary': summary, 'length_analysis': length_analysis}, f, ensure_ascii=False, indent=2)
    with open(out_dir / 'length_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(length_analysis, f, ensure_ascii=False, indent=2)
    with open(out_dir / 'config_used.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(out_dir / 'label_names.txt', 'w', encoding='utf-8') as f:
        for name in label_names:
            f.write(name + '\n')

    print(f"\n[Success] Training complete. Model and logs saved to:")
    print(out_dir)


if __name__ == '__main__':
    main()
