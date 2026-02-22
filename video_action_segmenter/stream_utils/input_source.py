from pathlib import Path
from typing import Optional, Tuple
import os
import cv2

from .batch import run_batch_over_folder


def open_input_capture(input_cfg: dict, params_path: str) -> Tuple[Optional[cv2.VideoCapture], str, str, bool]:
    """Open input source according to config. Mirrors behavior in stream_inference.py.

    Returns:
      (cap, src_type, video_name, handled_folder)
      - handled_folder=True means the function already executed folder batch logic; caller should return.
    """
    # EOF timeout is handled by the caller; only source opening is done here.
    src_type = str(input_cfg.get("type", "camera")).lower()  # camera | file | rtsp | folder

    # Subprocess override: MT_OVERRIDE_INPUT_PATH forces file mode on a single path
    _override_path_env = os.environ.get("MT_OVERRIDE_INPUT_PATH", "").strip()
    if _override_path_env:
        src_type = "file"

    cap: Optional[cv2.VideoCapture] = None
    video_name = "unknown"

    if src_type == "camera":
        cam_index = int(input_cfg.get("camera_index", 0))
        cap = cv2.VideoCapture(cam_index)
        video_name = f"camera_{cam_index}"

    elif src_type == "file":
        path_env = os.environ.get("MT_OVERRIDE_INPUT_PATH", "").strip()
        path_cfg = str(input_cfg.get("path", "")).strip()
        path = path_env or path_cfg
        cap = cv2.VideoCapture(path)
        if path:
            video_name = Path(path).stem
        else:
            video_name = "unknown_file"
        if path and os.path.isdir(path):
            # Switch to folder mode
            src_type = "folder"
            cap.release()
            cap = None

    elif src_type == "rtsp":
        url = str(input_cfg.get("url", ""))
        cap = cv2.VideoCapture(url)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.path:
                video_name = Path(parsed.path).stem or "rtsp_stream"
            else:
                video_name = "rtsp_stream"
        except Exception:
            video_name = "rtsp_stream"

    elif src_type == "folder":
        cap = None
        video_name = "batch_folder"
    else:
        raise ValueError(f"不支持的输入类型: {src_type}")

    # Folder batch mode
    if src_type == "folder":
        in_dir = str(input_cfg.get("dir", "")).strip()
        # Backward compatible: if user passed file.path but it's a dir
        if not in_dir:
            path_cfg = str(input_cfg.get("path", "")).strip()
            if path_cfg and os.path.isdir(path_cfg):
                in_dir = path_cfg
        if not in_dir:
            raise RuntimeError("folder 模式需要提供 input.dir 或 input.path 为目录")

        exts = input_cfg.get("video_exts", [".mp4", ".mov", ".avi", ".mkv"])
        try:
            exts = [str(e).lower() for e in exts]
        except Exception:
            exts = [".mp4", ".mov", ".avi", ".mkv"]
        recursive = bool(input_cfg.get("recursive", True))

        batch_cfg = input_cfg.get("batch", {}) if isinstance(input_cfg.get("batch", {}), dict) else {}
        enable_parallel = bool(batch_cfg.get("enable_parallel", False))
        gpu_ids_cfg = batch_cfg.get("gpu_ids", None)
        if isinstance(gpu_ids_cfg, (list, tuple)):
            gpu_ids = list(gpu_ids_cfg)
        else:
            gpu_ids = None
        try:
            max_procs_per_gpu = int(batch_cfg.get("max_procs_per_gpu", 1))
        except Exception:
            max_procs_per_gpu = 1
        try:
            poll_interval = float(batch_cfg.get("poll_interval_seconds", 0.2))
        except Exception:
            poll_interval = 0.2

        run_batch_over_folder(
            Path(in_dir),
            params_path,
            exts=exts,
            recursive=recursive,
            enable_parallel=enable_parallel,
            gpu_ids=gpu_ids,
            max_procs_per_gpu=max_procs_per_gpu,
            poll_interval=poll_interval,
        )
        return None, "folder", video_name, True

    return cap, src_type, video_name, False

