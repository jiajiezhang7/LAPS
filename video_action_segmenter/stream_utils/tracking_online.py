from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


_MODEL_CACHE = {}


def to_clip_tensor(frames: List[np.ndarray], device: torch.device) -> Tuple[torch.Tensor, int, int]:
    """Convert list of BGR uint8 frames to (T,3,H,W) float32 tensor in [0,1] on device."""
    arr = np.stack(frames, axis=0)
    H, W = arr.shape[1], arr.shape[2]
    arr = arr[:, :, :, ::-1]  # BGR -> RGB
    arr = arr.astype(np.float32) / 255.0
    # Avoid torch.from_numpy to sidestep NumPy C-API compatibility issues; make an explicit tensor copy
    clip = torch.tensor(arr, dtype=torch.float32, device=device).permute(0, 3, 1, 2)  # T, 3, H, W
    return clip, H, W


def get_cotracker_online(device: torch.device):
    """Load CoTracker online model with local hub cache fallback; cache per device index."""
    # Cache per device index to avoid cross-device tensor residency and memory growth
    dev_index = device.index if getattr(device, "index", None) is not None else 0
    key = f"online_{device.type}_{dev_index}"
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    try:
        hub_dir = Path(torch.hub.get_dir()).expanduser()
    except Exception:
        hub_dir = Path.home() / ".cache/torch/hub"
    local_repo = hub_dir / "facebookresearch_co-tracker_main"

    try:
        if local_repo.exists():
            print(f"[CoTracker] Using local Hub cache: {local_repo}")
            model = torch.hub.load(str(local_repo), "cotracker3_online", source="local").to(device)
        else:
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)
    except Exception as e:
        if local_repo.exists():
            print(f"[CoTracker][WARN] Remote load failed ({e}), retrying with local cache: {local_repo}")
            model = torch.hub.load(str(local_repo), "cotracker3_online", source="local").to(device)
        else:
            raise

    model.eval()
    _MODEL_CACHE[key] = model
    return model


def track_window_with_online(frames: List[np.ndarray], grid_size: int, device: torch.device) -> torch.Tensor:
    """Run CoTracker online over a window of frames.

    The grid is re-initialized on the first frame of the window. Returns velocities (T-1,N,2) on CPU.
    """
    if len(frames) == 0:
        raise RuntimeError("Empty frames for tracking")
    clip, H, W = to_clip_tensor(frames, device)
    model = get_cotracker_online(device)
    with torch.inference_mode():
        # Initialize queries on first frame
        first_frame_clip, _, _ = to_clip_tensor([frames[0]], device)
        model(
            video_chunk=first_frame_clip.unsqueeze(0),  # (1, 1, 3, H, W)
            is_first_step=True,
            grid_size=grid_size,
            grid_query_frame=0,
        )
        # Process the whole window
        video = clip.unsqueeze(0)  # (1, T, 3, H, W)
        pred_tracks, pred_visibility = model(
            video_chunk=video,
            is_first_step=False,
        )
        tracks = pred_tracks[0].detach().cpu()  # (T, N, 2)
        velocities = tracks[1:] - tracks[:-1]  # (T-1, N, 2)
    # Explicitly delete intermediate tensors to drop references quickly
    try:
        del first_frame_clip  # GPU tensor
    except Exception:
        pass
    try:
        del video, pred_tracks, pred_visibility  # GPU tensors
    except Exception:
        pass
    try:
        del clip  # GPU tensor created from frames
    except Exception:
        pass
    try:
        del tracks  # CPU tensor
    except Exception:
        pass
    return velocities
