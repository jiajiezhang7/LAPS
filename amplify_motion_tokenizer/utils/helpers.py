import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_str: str) -> torch.device:
    ds = (device_str or "auto").lower()
    if ds == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ds == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def velocity_to_labels(velocities: torch.Tensor, W: int) -> torch.Tensor:
    """
    将归一化的速度映射到 WxW 局部窗口中的离散类别标签。
    实现论文中公式 `omega_t = Omega(u_t)`。

    Args:
        velocities: 归一化到 [-1, 1] 的速度, shape (..., 2)
        W: 局部窗口的大小 (如 15)

    Returns:
        labels: 离散类别标签, shape (...)
    """
    if velocities.shape[-1] != 2:
        raise ValueError(f"velocities last dim must be 2, got {velocities.shape}")

    # 1) 将归一化的速度 [-1, 1] 映射到像素位移
    pixel_displacement = velocities * (W - 1) / 2.0

    # 2) 将位移映射到窗口坐标系 [0, W-1]
    coords = pixel_displacement + (W - 1) / 2.0

    # 3) 四舍五入并裁剪到有效范围内
    coords = torch.round(coords).long()
    coords = torch.clamp(coords, 0, W - 1)

    # 4) 将 2D 坐标 (x, y) 转换为 1D 类别索引 (label = y * W + x)
    labels = coords[..., 1] * W + coords[..., 0]
    return labels
