import argparse
import math
import multiprocessing as mp
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from video_action_segmenter.stream_utils.gating import pre_gate_check

# Cache for CoTracker models per process to avoid re-loading per window
_MODEL_CACHE: dict = {}


@dataclass
class Cfg:
    T: int
    grid_size: int
    num_points: int
    out_dir: Path
    target_fps: int
    resize_shorter: int
    stride: int
    dec_window_size: int
    batch_size: int
    pre_gate_enable: bool
    pre_gate_resize_shorter: int
    pre_gate_pixel_diff_thr: float
    pre_gate_mad_thr: float
    pre_gate_min_active_ratio: float
    pre_gate_method: str
    pre_gate_debug: bool


def _read_frames_resampled(video_path: Path, target_fps: int, resize_shorter: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps_in = float(fps_in)
    step = 1
    if target_fps > 0 and fps_in > 0:
        step = max(1, int(round(fps_in / target_fps)))

    frames: List[np.ndarray] = []
    idx = 0
    ok, frame = cap.read()
    while ok:
        if idx % step == 0:
            # Resize while keeping aspect ratio
            if resize_shorter and resize_shorter > 0:
                h, w = frame.shape[:2]
                shorter = min(h, w)
                if shorter != resize_shorter:
                    scale = resize_shorter / float(shorter)
                    nh, nw = int(round(h * scale)), int(round(w * scale))
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1
        ok, frame = cap.read()
    cap.release()
    return frames




def _frames_to_tensor(frames: List[np.ndarray]) -> Tuple[torch.Tensor, int, int]:
    """
    将帧列表一次性转换为 (T, 3, H, W) 的 float32 Tensor（值域 [0,1]，RGB）。

    为了后续窗口滑动时避免重复的数据拷贝和 CPU→GPU 传输开销，
    这里会确保结果 `contiguous`，方便批处理堆叠。
    """

    arr = np.stack(frames, axis=0)  # T, H, W, 3 (BGR)
    H, W = arr.shape[1], arr.shape[2]
    arr = arr[:, :, :, ::-1]  # BGR -> RGB
    clip = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
    return clip, H, W


def _normalize_velocities(vel: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将像素位移归一化到 [-1, 1]，按模型解码窗口半径进行缩放。

    目标：使 1 像素位移约等价于窗口坐标系中的 1 个“半步”（即 0.5 个 bin），
    从而 velocity_to_labels() 中的映射 pixel_displacement = v * (W-1)/2 能恢复到原位移尺度。

    公式：v_norm = clamp( vel_pixels / ((W_dec - 1) / 2), -1, 1 )

    Args:
        vel: (T-1, N, 2) in pixel displacement
        window_size: 模型解码局部窗口大小 (W_dec)
    Returns:
        vel_norm: 归一化到 [-1, 1] 的速度张量
    """
    scale = max(1.0, (float(window_size) - 1.0) / 2.0)
    vel_norm = torch.clamp(vel / scale, -1.0, 1.0)
    return vel_norm


def _get_cotracker(device: torch.device):
    """Return a cached CoTracker model on the given device."""
    key = str(device)
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    # Use torch.hub to load the recommended offline model
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _track_one_window(clip: torch.Tensor, grid_size: int, device: torch.device) -> Optional[torch.Tensor]:
    """Run CoTracker on a single window. Returns velocities (T-1, N, 2) on CPU or None on failure."""
    try:
        model = _get_cotracker(device)
        with torch.no_grad():
            video = clip.contiguous().unsqueeze(0).contiguous().to(device, non_blocking=True)  # 1, T, C, H, W
            # New API uses grid_size and returns tracks and visibility
            tracks, _ = model(video, grid_size=grid_size)
            tracks = tracks[0]  # T, N, 2
            velocities = tracks[1:] - tracks[:-1]
            return velocities.cpu()
    except Exception as e:
        print(f"[WARN] CoTracker window tracking failed: {e}")
        return None

def _track_windows_batch(clips: torch.Tensor, grid_size: int, device: torch.device) -> Optional[torch.Tensor]:
    """
    对一个批次的窗口进行并行跟踪。输入为 (B, T, 3, H, W)，返回 (B, T-1, N, 2)。

    若出现异常则返回 None。
    """

    try:
        model = _get_cotracker(device)
        with torch.no_grad():
            video = clips.contiguous().to(device, non_blocking=True)
            tracks, _ = model(video, grid_size=grid_size)  # B, T, N, 2
            velocities = tracks[:, 1:] - tracks[:, :-1]
            return velocities.cpu()
    except Exception as e:
        print(f"[WARN] CoTracker batch tracking failed: {e}")
        return None


def _process_segment(segment_path: Path, cfg: Cfg, device_str: str,
                     max_windows: Optional[int], fallback_random: bool, seed: int) -> int:
    device = torch.device(device_str)
    rng = np.random.default_rng(seed)

    cap = cv2.VideoCapture(str(segment_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {segment_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps_in = float(fps_in)
    step = 1
    if cfg.target_fps > 0 and fps_in > 0:
        step = max(1, int(round(fps_in / cfg.target_fps)))

    def _resize_frame(frame: np.ndarray) -> np.ndarray:
        if not cfg.resize_shorter or cfg.resize_shorter <= 0:
            return frame
        h, w = frame.shape[:2]
        shorter = min(h, w)
        if shorter == cfg.resize_shorter:
            return frame
        scale = cfg.resize_shorter / float(shorter)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    buffer: deque[np.ndarray] = deque(maxlen=cfg.T)
    batch: List[Tuple[int, torch.Tensor]] = []
    processed = 0
    raw_idx = 0
    sampled_idx = 0
    limit_reached = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if raw_idx % step == 0:
            frame = _resize_frame(frame)
            buffer.append(frame)
            sampled_idx += 1

            if len(buffer) == cfg.T:
                start = sampled_idx - cfg.T
                if start % cfg.stride == 0:
                    window_frames = list(buffer)
                    if cfg.pre_gate_enable:
                        dropped, mad_val, active_ratio = pre_gate_check(
                            window_frames,
                            resize_shorter=cfg.pre_gate_resize_shorter,
                            pixel_diff_thr=cfg.pre_gate_pixel_diff_thr,
                            mad_thr=cfg.pre_gate_mad_thr,
                            min_active_ratio=cfg.pre_gate_min_active_ratio,
                            method=cfg.pre_gate_method,
                            debug=cfg.pre_gate_debug,
                        )
                        if dropped:
                            if cfg.pre_gate_debug:
                                print(
                                    f"[PreGate][DROP] {segment_path.name} start={start} "
                                    f"mad={mad_val:.4f} active_ratio={active_ratio:.4f}"
                                )
                            continue

                    clip, _, _ = _frames_to_tensor(window_frames)
                    batch.append((start, clip.contiguous()))

                    if len(batch) >= cfg.batch_size:
                        processed += _process_batch(batch, segment_path, cfg, device, rng, fallback_random)
                        batch.clear()
                        if max_windows is not None and processed >= max_windows:
                            limit_reached = True
                            break

        if limit_reached:
            break

        raw_idx += 1

    cap.release()

    if batch and not limit_reached:
        processed += _process_batch(batch, segment_path, cfg, device, rng, fallback_random)
    elif batch and max_windows is not None and processed < max_windows:
        processed += _process_batch(batch, segment_path, cfg, device, rng, fallback_random)

    return processed


def _process_batch(batch: List[Tuple[int, torch.Tensor]], segment_path: Path, cfg: Cfg,
                   device: torch.device, rng: np.random.Generator, fallback_random: bool) -> int:
    if not batch:
        return 0

    starts = [item[0] for item in batch]
    clips = torch.stack([item[1] for item in batch], dim=0).contiguous()

    velocities_batch = _track_windows_batch(clips, cfg.grid_size, device)
    processed = 0

    for idx, start in enumerate(starts):
        velocities = None
        if velocities_batch is not None:
            velocities = velocities_batch[idx]
        else:
            # 如果批次失败，尝试逐个窗口回退
            velocities = _track_one_window(batch[idx][1], cfg.grid_size, device)

        if velocities is None:
            if not fallback_random:
                continue
            velocities = torch.from_numpy(
                rng.uniform(-1.0, 1.0, size=(cfg.T - 1, cfg.num_points, 2)).astype(np.float32)
            )
        else:
            velocities = _normalize_velocities(velocities, cfg.dec_window_size)

        out_name = f"{segment_path.stem}_start{start:06d}.pt"
        out_path = cfg.out_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(velocities, out_path)
        processed += 1

    torch.cuda.synchronize(device) if device.type == "cuda" else None

    return processed


def _gather_segments(segments_root: Path, exts: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in segments_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in [e.lower() for e in exts]:
            files.append(p)
    return sorted(files)


def main():
    # Set start method for multiprocessing
    # NOTE: this must be done before any CUDA context is created.
    # We do it here at the top of main().
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # 'spawn' may have been set already, which is fine.
        pass

    parser = argparse.ArgumentParser(description="Preprocess video segments in parallel to generate velocities")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=None, help="Override parallel workers (default from config)")
    parser.add_argument("--max-windows-per-segment", type=int, default=None)
    parser.add_argument("--fallback-random", action="store_true", help="Use random velocities if tracking fails")
    parser.add_argument("--use-segments", action="store_true", help="Read from segments_dir instead of source videos")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards for distributed runs")
    parser.add_argument("--shard-idx", type=int, default=0, help="Index of this shard [0..num-shards-1]")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory for .pt files")
    parser.add_argument("--window-batch-size", type=int, default=None, help="Number of windows per CoTracker forward pass")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    data_cfg = config['data']
    pre_cfg = config.get('preprocess', {})
    model_cfg = config.get('model', {})

    T = int(data_cfg['sequence_length'])
    grid_size = int(data_cfg['grid_size'])
    num_points = int(data_cfg['num_points'])
    # output directory, allow override by CLI
    out_dir = Path(args.output_dir) if args.output_dir else Path(data_cfg['preprocess_output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    target_fps = int(pre_cfg.get('target_fps', 15))
    resize_shorter = int(pre_cfg.get('resize_shorter', 480))
    stride = int(pre_cfg.get('window_stride', 4))
    segments_dir = Path(pre_cfg.get('segments_dir', './amplify_motion_tokenizer/data/video_segments'))
    parallel_workers = int(args.workers or pre_cfg.get('parallel_workers', 4))
    batch_size = int(args.window_batch_size or pre_cfg.get('window_batch_size', 1))

    pre_gate_cfg = pre_cfg.get('pre_gate', {}) if isinstance(pre_cfg.get('pre_gate', {}), dict) else {}
    pre_gate_enable = bool(pre_gate_cfg.get('enable', False))
    pre_gate_resize_shorter = int(pre_gate_cfg.get('resize_shorter', 128))
    pre_gate_pixel_diff_thr = float(pre_gate_cfg.get('pixel_diff_thr', 0.02))
    pre_gate_mad_thr = float(pre_gate_cfg.get('mad_thr', 0.005))
    pre_gate_min_active_ratio = float(pre_gate_cfg.get('min_active_ratio', 0.05))
    pre_gate_method = str(pre_gate_cfg.get('diff_method', 'mad')).lower()
    pre_gate_debug = bool(pre_gate_cfg.get('debug', False))

    exts = pre_cfg.get('video_exts', [".mp4", ".mov", ".avi", ".mkv"])

    dec_window_size = int(model_cfg.get('decoder_window_size', 15))

    cfg = Cfg(
        T=T,
        grid_size=grid_size,
        num_points=num_points,
        out_dir=out_dir,
        target_fps=target_fps,
        resize_shorter=resize_shorter,
        stride=stride,
        dec_window_size=dec_window_size,
        batch_size=max(1, batch_size),
        pre_gate_enable=pre_gate_enable,
        pre_gate_resize_shorter=pre_gate_resize_shorter,
        pre_gate_pixel_diff_thr=pre_gate_pixel_diff_thr,
        pre_gate_mad_thr=pre_gate_mad_thr,
        pre_gate_min_active_ratio=pre_gate_min_active_ratio,
        pre_gate_method=pre_gate_method,
        pre_gate_debug=pre_gate_debug,
    )

    if args.use_segments:
        inputs = _gather_segments(segments_dir, exts)
        in_desc = str(segments_dir)
    else:
        # fall back to source videos (process full video sequentially)
        inputs = _gather_segments(Path(data_cfg['video_source_dir']), exts)
        in_desc = str(data_cfg['video_source_dir'])

    if not inputs:
        print(f"No input videos found at {in_desc}")
        return

    # Shard the input list so that multiple concurrent invocations can process disjoint subsets
    num_shards = max(1, int(args.num_shards))
    shard_idx = int(args.shard_idx)
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"shard-idx must be in [0, {num_shards-1}], got {shard_idx}")
    if num_shards > 1:
        inputs = [p for i, p in enumerate(inputs) if i % num_shards == shard_idx]
        in_desc += f" | shard {shard_idx}/{num_shards} -> {len(inputs)} items"
        print(f"Sharded inputs: {in_desc}")

    # If using GPU, it's safer to run workers=1 unless you know what you're doing
    if device_str == "cuda" and parallel_workers > 1:
        print("[WARN] Using GPU with multiple workers may cause OOM or contention. Consider --workers 1.")

    total = 0
    if parallel_workers <= 1:
        for p in tqdm(inputs, desc="Preprocessing"):
            total += _process_segment(
                segment_path=p,
                cfg=cfg,
                device_str=device_str,
                max_windows=args.max_windows_per_segment,
                fallback_random=args.fallback_random,
                seed=args.seed,
            )
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
            futures = []
            for i, p in enumerate(inputs):
                futures.append(ex.submit(
                    _process_segment,
                    p,
                    cfg,
                    device_str,
                    args.max_windows_per_segment,
                    args.fallback_random,
                    args.seed + i,
                ))
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing (parallel)"):
                try:
                    total += fut.result()
                except Exception as e:
                    print(f"[ERR] segment failed: {e}")

    print(f"Done. Generated windows: {total}")


if __name__ == "__main__":
    main()
