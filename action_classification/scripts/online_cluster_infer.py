import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import hdbscan  # type: ignore
    from hdbscan import prediction as hdb_pred  # type: ignore
except Exception:
    hdbscan = None
    hdb_pred = None

from sklearn.preprocessing import normalize

from action_classification.data.features import read_json_sample
from action_classification.embedding.common import SeqEncoder


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_seq_encoder(model_path: Path, device: torch.device) -> Tuple[SeqEncoder, Dict[str, Any]]:
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get('config', {})
    label_names = ckpt.get('label_names', None)

    # Determine num_tokens and pad_id
    # Prefer reading from checkpoint weights to avoid mismatch
    try:
        emb_weight = ckpt['state_dict']['embed.weight']
        num_tokens = int(emb_weight.shape[0])
    except Exception:
        codebook_size = int(cfg.get('codebook_size', 2048))
        num_tokens = int(codebook_size + 1)  # +1 for PAD offset
    model_cfg = cfg.get('model', {})
    pad_id = int(cfg.get('sequence', {}).get('pad_id', 0))

    model = SeqEncoder(
        num_tokens=num_tokens,
        d_model=int(model_cfg.get('d_model', 128)),
        rnn_type=str(model_cfg.get('rnn_type', 'lstm')),
        num_layers=int(model_cfg.get('num_layers', 2)),
        dropout=float(model_cfg.get('dropout', 0.1)),
        bidirectional=bool(model_cfg.get('bidirectional', True)),
        pad_id=pad_id,
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model, cfg


def embed_codes(model: SeqEncoder, codes: List[int], device: torch.device, pad_id: int = 0) -> np.ndarray:
    # Mirror training preprocessing: +1 offset for tokens, 0 reserved for PAD
    seq = [int(x) + 1 for x in codes]
    x = torch.tensor(seq, dtype=torch.long, device=device)[None, :]  # [1, T]
    with torch.no_grad():
        logits, h = model(x)
        # mean-pool over valid positions
        mask = (x != model.pad_id).float().unsqueeze(-1)  # [1, T, 1]
        masked = h * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = masked.sum(dim=1) / denom  # [1, H]
        vec = torch.nn.functional.normalize(pooled, p=2, dim=1).squeeze(0)  # L2-normalize
    return vec.detach().cpu().numpy()


def load_cluster_bundle(cluster_dir: Path) -> Tuple[Dict[str, Any], Any, Optional[Any]]:
    meta_path = cluster_dir / 'cluster_meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing cluster_meta.json under {cluster_dir}")
    meta = json.loads(meta_path.read_text(encoding='utf-8'))

    method = meta.get('method', 'hdbscan')
    model = None
    pca = None
    if method == 'hdbscan':
        model_path = cluster_dir / 'model_hdbscan.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Missing HDBSCAN model file: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        pca_path = cluster_dir / 'preprocessor_pca.pkl'
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
    elif method == 'bayes_gmm':
        model_path = cluster_dir / 'model_bayes_gmm.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Missing Bayesian GMM model file: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # optional PCA preprocessor name kept consistent
        pca_path = cluster_dir / 'preprocessor_pca.pkl'
        if pca_path.exists():
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
    else:
        raise ValueError(f"Unknown clustering method in meta: {method}")

    return meta, model, pca


def predict_one(vec: np.ndarray, meta: Dict[str, Any], model: Any, pca: Optional[Any], prob_thr: float) -> Dict[str, Any]:
    # Preprocess
    if meta.get('preprocess', {}).get('l2_normalize', True):
        vec = normalize(vec.reshape(1, -1), norm='l2').reshape(-1)
    if pca is not None:
        vec = pca.transform(vec.reshape(1, -1)).reshape(-1)

    method = meta.get('method', 'hdbscan')
    if method == 'hdbscan':
        if hdb_pred is None:
            raise ImportError("hdbscan is not available but required by the cluster method.")
        labels, strengths = hdb_pred.approximate_predict(model, vec.reshape(1, -1))
        label = int(labels[0])
        prob = float(strengths[0])
        is_anomaly = (label == -1) or (prob < prob_thr)
        return {'cluster_id': label, 'prob': prob, 'anomaly': bool(is_anomaly), 'method': 'hdbscan'}
    elif method == 'bayes_gmm':
        # Use max posterior as confidence
        comp = int(model.predict(vec.reshape(1, -1))[0])
        post = model.predict_proba(vec.reshape(1, -1))[0]
        prob = float(np.max(post))
        is_anomaly = (prob < prob_thr)
        return {'cluster_id': comp, 'prob': prob, 'anomaly': bool(is_anomaly), 'method': 'bayes_gmm'}
    else:
        raise ValueError(f"Unsupported method: {method}")


def infer_over_json(json_path: Path) -> List[int]:
    obj = read_json_sample(json_path)
    codes_flat = obj.get('codes_flat')
    if not codes_flat:
        # fallback to codes_2d
        codes_2d = obj.get('codes_2d')
        if codes_2d:
            codes_flat = [int(v) for row in codes_2d for v in row]
        else:
            codes_flat = []
    return [int(v) for v in codes_flat]


def main():
    ap = argparse.ArgumentParser(description='Online inference: LSTM embedding -> clustering (HDBSCAN/BayesGMM) -> label/anomaly')
    ap.add_argument('--encoder-model', type=str, required=True, help='Path to LSTM encoder checkpoint (model_best.pt)')
    ap.add_argument('--cluster-dir', type=str, required=True, help='Directory containing cluster_meta.json and model_*.pkl')
    ap.add_argument('--prob-thr', type=float, default=0.2, help='Membership/Posterior probability threshold for anomaly alert')
    ap.add_argument('--device', type=str, default=None, help='cuda | cpu (auto)')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--json', type=str, help='Path to a single JSON file produced by Motion Tokenizer inference')
    group.add_argument('--json-root', type=str, help='Root directory; run inference on all JSON files recursively')
    group.add_argument('--codes', type=str, help='Comma-separated integer codes (e.g., 1,2,3,4)')
    ap.add_argument('--out-jsonl', type=str, default=None, help='If provided, save results to this JSONL path')
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load encoder
    encoder, enc_cfg = load_seq_encoder(Path(args.encoder_model).resolve(), device)
    pad_id = int(enc_cfg.get('sequence', {}).get('pad_id', 0))

    # Load cluster model bundle
    meta, cluster_model, pca = load_cluster_bundle(Path(args.cluster_dir).resolve())

    # Optional semantic mapping
    sem_map_path = Path(args.cluster_dir).resolve() / 'cluster_id_to_semantic.json'
    sem_map = None
    if sem_map_path.exists():
        try:
            sem_map = json.loads(sem_map_path.read_text(encoding='utf-8'))
        except Exception:
            sem_map = None

    # Collect inputs
    inputs: List[Tuple[str, List[int]]] = []
    if args.json:
        jp = Path(args.json).resolve()
        inputs.append((str(jp), infer_over_json(jp)))
    elif args.json_root:
        root = Path(args.json_root).resolve()
        for fp in sorted(root.rglob('*.json')):
            inputs.append((str(fp), infer_over_json(fp)))
    else:  # codes
        seq = [int(x.strip()) for x in args.codes.split(',') if x.strip()]
        inputs.append(('codes_cli', seq))

    # Run inference
    results: List[Dict[str, Any]] = []
    for src, codes in inputs:
        if not codes:
            results.append({'source': src, 'error': 'empty_sequence'})
            continue
        vec = embed_codes(encoder, codes, device=device, pad_id=pad_id)
        pred = predict_one(vec, meta=meta, model=cluster_model, pca=pca, prob_thr=float(args.prob_thr))
        if sem_map is not None and str(pred['cluster_id']) in sem_map:
            pred['semantic'] = sem_map[str(pred['cluster_id'])]
        results.append({'source': src, **pred})

    # Output
    if args.out_jsonl:
        outp = Path(args.out_jsonl).resolve()
        ensure_dir(outp.parent)
        with open(outp, 'w', encoding='utf-8') as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"[Saved] Results: {outp}")
    else:
        for rec in results:
            print(json.dumps(rec, ensure_ascii=False))


if __name__ == '__main__':
    main()
