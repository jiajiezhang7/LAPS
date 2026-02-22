from pathlib import Path
from typing import List, Optional, Union
import json

import numpy as np
import torch


def append_codes_jsonl(jsonl_path: Union[str, Path], window: int, codes: List[int]) -> None:
    p = Path(jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"window": int(window), "codes": [int(v) for v in codes]}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False))
        f.write("\n")


def append_energy_jsonl(
    jsonl_path: Union[str, Path],
    window: int,
    energy: float,
    source: str,
    mode: str,
) -> None:
    p = Path(jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "window": int(window),
        "energy": float(energy),
        "source": str(source),
        "mode": str(mode),
    }
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False))
        f.write("\n")


def export_prequant_npy(out_dir: Union[str, Path], win_idx: int, to_quantize: torch.Tensor) -> Optional[str]:
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        Z_np = to_quantize.squeeze(0).detach().cpu().to(torch.float32).numpy()  # (d, D)
        path = out_dir / f"prequant_win_{int(win_idx):06d}.npy"
        with open(path, "wb") as f:
            np.save(f, Z_np)
        return str(path)
    except Exception:
        return None
