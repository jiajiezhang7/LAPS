"""Configuration and utility functions for video action segmentation."""

from pathlib import Path
from typing import Optional, Dict, Any

import torch
import yaml


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration, or empty dict if path is None/invalid
    """
    if config_path is None:
        return {}
    
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            return {}
        
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        if not isinstance(data, dict):
            print(f"[WARN] 配置文件格式非法（需为映射）：{config_path}")
            return {}
        
        return data
    except Exception as e:
        print(f"[WARN] 加载配置文件失败 {config_path}: {e}")
        return {}


def _normalize_velocities(vel_tensor: torch.Tensor, decoder_window_size: int) -> torch.Tensor:
    """Normalize velocity tensor by decoder window size.
    
    This function scales velocities based on the decoder window size to make
    motion thresholds consistent across different temporal resolutions.
    
    Args:
        vel_tensor: Velocity tensor of shape (T-1, N, 2) in pixel coordinates
        decoder_window_size: Reference window size (typically 15 for Motion Tokenizer)
        
    Returns:
        Normalized velocity tensor with same shape as input
    """
    if decoder_window_size <= 0:
        return vel_tensor
    
    # Scale factor: normalize velocities based on decoder window size
    # Smaller window -> smaller expected velocities -> scale up threshold
    scale_factor = 15.0 / float(decoder_window_size)
    
    return vel_tensor * scale_factor
