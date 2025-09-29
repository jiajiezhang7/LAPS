import json
from pathlib import Path
from typing import Dict, Any, Generator, Iterable, List, Tuple, Optional

import numpy as np
from tqdm import tqdm


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def read_json_sample(fp: Path) -> Dict[str, Any]:
    with open(fp, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    meta = obj.get('meta', {})
    d = _safe_int(meta.get('encoder_sequence_len', 0))
    starts = obj.get('window_starts', [])
    Wn = len(starts)

    codes_windows = obj.get('codes_windows', None)
    codes_2d = obj.get('code_sequences', None)
    codes_flat = obj.get('code_sequences_flat', None)
    if codes_windows is not None and codes_flat is None:
        try:
            codes_flat = [int(v) for window in codes_windows for v in window]
        except Exception:
            codes_flat = None
    if codes_2d is not None and codes_flat is None:
        # 展平
        try:
            codes_flat = [int(v) for row in codes_2d for v in row]
        except Exception:
            codes_flat = None

    lm = None
    if 'latent_matrix' in obj:
        try:
            shape = obj.get('latent_matrix_shape', None)
            lm_list = obj['latent_matrix']
            if shape is not None and len(shape) == 2:
                Wd, D = int(shape[0]), int(shape[1])
                lm = np.asarray(lm_list, dtype=np.float32).reshape(Wd, D)
            else:
                lm = np.asarray(lm_list, dtype=np.float32)
        except Exception:
            lm = None

    return {
        'meta': meta,
        'num_windows': Wn,
        'd': d,
        'codes_windows': codes_windows,
        'codes_2d': codes_2d,
        'codes_flat': codes_flat,
        'latent_matrix': lm,  # shape (Wn*d, D) or None
    }


def iter_json_samples(json_root: Path) -> Generator[Dict[str, Any], None, None]:
    json_root = Path(json_root)
    for fp in sorted(json_root.rglob('*.json')):
        try:
            rel = fp.relative_to(json_root)
            label = rel.parts[0]  # 顶层子文件夹名作为标签
        except Exception:
            label = 'unknown'
        data = read_json_sample(fp)
        yield {
            'path': fp,
            'label': label,
            **data,
        }


def infer_codebook_size(meta: Dict[str, Any], codes_flat: Optional[List[int]], default_k: int) -> int:
    # 优先 meta['codebook_size']
    k = meta.get('codebook_size', None)
    if k is not None:
        return _safe_int(k, default_k)
    # 其次根据 levels 乘积
    levels = meta.get('codebook_levels', None)
    if isinstance(levels, (list, tuple)) and len(levels) > 0:
        prod = 1
        try:
            for L in levels:
                prod *= int(L)
            return int(prod)
        except Exception:
            pass
    # 兜底：codes 最大值 + 1
    if codes_flat is not None and len(codes_flat) > 0:
        try:
            return int(max(codes_flat)) + 1
        except Exception:
            pass
    return int(default_k)


def compute_bow_feature(codes_flat: List[int], K: int, normalize: bool = True, smooth: float = 1e-9) -> np.ndarray:
    if codes_flat is None or len(codes_flat) == 0:
        # 全零特征
        x = np.zeros((int(K),), dtype=np.float32)
        return x
    idx = np.asarray(codes_flat, dtype=np.int64)
    x = np.bincount(idx, minlength=int(K)).astype(np.float32)
    if smooth and smooth > 0:
        x += float(smooth)
    if normalize and x.sum() > 0:
        x = x / x.sum()
    return x


def compute_vector_avg(latent_matrix: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if latent_matrix is None:
        return None
    if not isinstance(latent_matrix, np.ndarray) or latent_matrix.size == 0:
        return None
    try:
        return latent_matrix.mean(axis=0).astype(np.float32)
    except Exception:
        return None


def build_label_mapping(samples: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    names = sorted({s['label'] for s in samples})
    return {name: i for i, name in enumerate(names)}


def build_dataset(json_root: Path, feature: str, config: Dict[str, Any], expected_classes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Path], List[str]]:
    """
    返回：X, y_true, paths, label_names
    feature: 'bow' or 'avg'
    """
    # We need to know the total number of files for tqdm, so we glob first.
    json_files = sorted(Path(json_root).rglob('*.json'))
    if not json_files:
        raise RuntimeError(f'No JSON files found under: {json_root}')

    # Create a generator for samples from the file list
    def sample_generator(files):
        for fp in files:
            try:
                rel = fp.relative_to(json_root)
                label = rel.parts[0]
            except Exception:
                label = 'unknown'
            data = read_json_sample(fp)
            yield {'path': fp, 'label': label, **data}

    # Build label mapping (with optional label merging)
    label_proc_cfg = config.get('label_processing', {}) if isinstance(config, dict) else {}
    merge_map: Dict[str, str] = label_proc_cfg.get('merge_map', {}) or {}

    # Raw labels from top-level folders
    all_labels_raw = {p.relative_to(json_root).parts[0] for p in json_files}
    # Apply merge_map to get final (merged) label names
    merged_labels_set = {merge_map.get(name, name) for name in all_labels_raw}
    label_names = sorted(list(merged_labels_set))
    label_map = {name: i for i, name in enumerate(label_names)}

    if expected_classes is not None and len(label_map) != int(expected_classes):
        print(f"[Warn] Detected {len(label_map)} merged classes from folders: {label_names}, but expected {expected_classes}.")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    p_list: List[Path] = []
    n_skipped = 0

    # Use tqdm on the generator, with total count known
    pbar = tqdm(sample_generator(json_files), total=len(json_files), desc=f"Building dataset for feature '{feature}'")
    for s in pbar:
        meta = s['meta']
        # Map raw label to merged label if specified
        raw_label = s['label']
        merged_label = merge_map.get(raw_label, raw_label)
        y = label_map.get(merged_label, 0)

        if feature == 'bow':
            K = infer_codebook_size(meta, s.get('codes_flat'), default_k=config['features']['bow']['default_codebook_size'])
            x = compute_bow_feature(s.get('codes_flat'), K, smooth=config['features']['bow']['smoothing'])
        elif feature == 'avg':
            x = compute_vector_avg(s.get('latent_matrix'))
        else:
            raise ValueError(f'Unknown feature: {feature}')

        if x is None:
            n_skipped += 1
            continue

        X_list.append(x)
        y_list.append(int(y))
        p_list.append(s['path'])

    if len(X_list) == 0:
        raise RuntimeError(f'All samples skipped for feature={feature}. Check JSON export mode or data integrity.')

    if n_skipped > 0:
        print(f"[Info] Skipped {n_skipped} samples due to missing feature '{feature}'.")

    X = np.stack(X_list, axis=0)
    y_true = np.asarray(y_list, dtype=np.int64)
    return X, y_true, p_list, label_names
