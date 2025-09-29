import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ..data.features import read_json_sample


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class Sample:
    path: Path
    label_name: str
    label: int
    codes: List[int]


def flatten_codes(obj: Dict[str, Any]) -> List[int]:
    codes_windows = obj.get('codes_windows')
    if codes_windows is not None:
        flattened: List[int] = []
        for window in codes_windows:
            if not isinstance(window, (list, tuple)):
                continue
            for v in window:
                try:
                    flattened.append(int(v))
                except Exception:
                    continue
        if flattened:
            return flattened

    codes_flat = obj.get('codes_flat') or obj.get('code_sequences_flat')
    if codes_flat is not None:
        try:
            return [int(v) for v in codes_flat]
        except Exception:
            pass
    codes_2d = obj.get('codes_2d') or obj.get('code_sequences')
    if codes_2d is not None:
        try:
            return [int(v) for row in codes_2d for v in row]
        except Exception:
            pass
    return []


def scan_samples(json_root: Path) -> Tuple[List[Sample], List[str], int]:
    files = sorted(Path(json_root).rglob('*.json'))
    if not files:
        raise RuntimeError(f'No JSON files under {json_root}')

    label_names = sorted({f.relative_to(json_root).parts[0] for f in files})
    label_map = {name: i for i, name in enumerate(label_names)}

    samples: List[Sample] = []
    max_code_seen = -1
    n_parse_error = 0
    n_empty = 0
    n_other = 0
    for fp in files:
        try:
            label_name = fp.relative_to(json_root).parts[0]
        except Exception:
            label_name = 'unknown'
        try:
            obj = read_json_sample(fp)
        except Exception:
            n_parse_error += 1
            continue
        codes = flatten_codes(obj)
        if not codes:
            n_empty += 1
            continue
        max_code_seen = max(max_code_seen, max(codes))
        samples.append(
            Sample(
                path=fp,
                label_name=label_name,
                label=label_map.get(label_name, 0),
                codes=codes,
            )
        )

    if not samples:
        raise RuntimeError(
            'All samples are empty or invalid sequences '
            f'(files={len(files)}, parse_error={n_parse_error}, empty_codes={n_empty}, other={n_other})'
        )

    return samples, label_names, int(max_code_seen)


class SeqDataset(Dataset):
    def __init__(self, samples: List[Sample], max_len: int, pad_id: int = 0):
        self.samples = samples
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self._seqs: List[np.ndarray] = []
        self._lengths: List[int] = []
        self._labels: List[int] = []
        self._paths: List[Path] = []

        for s in samples:
            seq = [int(x) + 1 for x in s.codes]
            if len(seq) >= self.max_len:
                seq = seq[: self.max_len]
                L = self.max_len
            else:
                L = len(seq)
                seq = seq + [self.pad_id] * (self.max_len - len(seq))
            self._seqs.append(np.asarray(seq, dtype=np.int64))
            self._lengths.append(L)
            self._labels.append(int(s.label))
            self._paths.append(s.path)

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._seqs[idx])
        L = int(self._lengths[idx])
        y = int(self._labels[idx])
        return x, L, y, str(self._paths[idx])

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self._labels, dtype=np.int64)

    @property
    def paths(self) -> List[str]:
        return [str(p) for p in self._paths]


class SeqEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        d_model: int = 128,
        rnn_type: str = 'lstm',
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(num_tokens, d_model, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.rnn_type = rnn_type.lower()
        rnn_cls = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = d_model * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, num_tokens)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(x)
        emb = self.dropout(emb)
        h, _ = self.rnn(emb)
        logits = self.head(self.dropout(h))
        return logits, h


__all__ = [
    'set_seed',
    'ensure_dir',
    'Sample',
    'flatten_codes',
    'scan_samples',
    'SeqDataset',
    'SeqEncoder',
]
