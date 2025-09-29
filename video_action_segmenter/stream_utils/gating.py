from typing import List, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch


def pre_gate_check(
    frames: List[np.ndarray],
    resize_shorter: int = 128,
    pixel_diff_thr: float = 0.01,
    mad_thr: float = 0.003,
    min_active_ratio: float = 0.002,
    method: str = "mad",
    debug: bool = False,
) -> Tuple[bool, float, float]:
    """Lightweight pre-tracking pixel-diff gate.

    Returns (dropped, mad_val, active_ratio).
    """
    try:
        gray_list: List[np.ndarray] = []
        for img in frames:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if resize_shorter and resize_shorter > 0:
                h, w = g.shape[:2]
                s = min(h, w)
                if s != resize_shorter:
                    scale = float(resize_shorter) / float(s)
                    nh, nw = int(round(h * scale)), int(round(w * scale))
                    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
            gray_list.append(g)
        diffs = []
        ratios = []
        for i in range(1, len(gray_list)):
            d = cv2.absdiff(gray_list[i], gray_list[i - 1]).astype(np.float32) / 255.0
            diffs.append(float(d.mean()))
            ratios.append(float((d > pixel_diff_thr).mean()))
        mad_val = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
        active_ratio = float(np.mean(ratios)) if len(ratios) > 0 else 0.0
        dropped = (mad_val < float(mad_thr)) and (active_ratio < float(min_active_ratio))
        return dropped, mad_val, active_ratio
    except Exception as e:
        if debug:
            print(f"[PreGate][WARN] diff calc failed: {e}")
        return False, 0.0, 0.0


def motion_gate_check(
    vel_norm: torch.Tensor,
    vel_norm_thr: float = 0.012,
    min_active_ratio: float = 0.01,
    debug: bool = False,
) -> Tuple[bool, float, float]:
    """Velocity-based motion gate after tracking.

    Args:
        vel_norm: (T-1,N,2) normalized velocities
    Returns:
        (dropped, l2_mean, active_ratio)
    """
    try:
        vnorm = torch.linalg.vector_norm(vel_norm, dim=-1)  # (T-1, N)
        v_l2_mean = float(vnorm.mean().item())
        active_ratio = float((vnorm > float(vel_norm_thr)).float().mean().item())
        dropped = (v_l2_mean < float(vel_norm_thr)) and (active_ratio < float(min_active_ratio))
        return dropped, v_l2_mean, active_ratio
    except Exception as e:
        if debug:
            print(f"[Gate][WARN] motion stats failed: {e}")
        return False, 0.0, 0.0
