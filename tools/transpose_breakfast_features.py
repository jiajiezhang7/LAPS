#!/usr/bin/env python3
"""
Transpose Breakfast precomputed features to time-major (T, D) for ABD.
- Source: ./online_datasets/breakfast/breakfast/features/{stem}.npy (shape typically (2048, T))
- Target: ./online_datasets/breakfast/breakfast/features_t/{stem}.npy (shape (T, 2048))
- Optionally restrict to stems listed in a split bundle file.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def read_split_stems(bundle_path: str | None) -> set[str] | None:
    if not bundle_path:
        return None
    stems: set[str] = set()
    with open(bundle_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().rstrip('\r')
            if not line:
                continue
            stem = line[:-4] if line.endswith('.txt') else line
            stems.add(stem)
    return stems

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='./online_datasets/breakfast/breakfast/features')
    ap.add_argument('--dst', default='./online_datasets/breakfast/breakfast/features_t')
    ap.add_argument('--bundle', default='./online_datasets/breakfast/breakfast/splits/test.split1.bundle', help='Optional split bundle to restrict stems')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    allow: set[str] | None = read_split_stems(args.bundle)

    npy_files = sorted(src.glob('*.npy'))
    ok = 0
    for p in npy_files:
        stem = p.stem
        if allow is not None and stem not in allow:
            continue
        out = dst / f'{stem}.npy'
        if out.exists() and not args.overwrite:
            print(f'[SKIP] {out} exists')
            ok += 1
            continue
        try:
            X = np.load(p)
            if X.ndim != 2:
                print(f'[WARN] {p.name} has ndim={X.ndim}, skip')
                continue
            # Ensure output is (T, 2048): if channel dim (2048) is first, transpose
            if X.shape[1] == 2048:
                Y = X
            elif X.shape[0] == 2048:
                Y = X.T
            else:
                print(f'[WARN] {p.name} unexpected shape {X.shape}, trying to place 2048 as last dim')
                # Heuristic: if any dim equals 2048, move it to last
                if 2048 in X.shape:
                    if X.shape[0] == 2048:
                        Y = X.T
                    elif X.shape[1] == 2048:
                        Y = X
                    else:
                        # Rare: multi-dim, fallback keep
                        Y = X
                else:
                    Y = X
            np.save(out, Y.astype(np.float32, copy=False))
            print(f'[OK] {out} shape={Y.shape}')
            ok += 1
        except Exception as e:
            print(f'[ERR] {p}: {e}')
    print(f'Done, wrote {ok} files into {dst}')

if __name__ == '__main__':
    main()

