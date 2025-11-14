#!/usr/bin/env python3
"""
按窗口(T=16)滑动处理视频，支持单文件或目录批量模式：
- 重采样到目标 FPS（默认 20Hz），短边缩放到 480（保持纵横比）
- 以窗口长度 T=16 和步长 stride（默认 4）滑动
- 每个窗口重新初始化 CoTracker 的网格关键点（离线模型），对该窗口进行一次独立跟踪
- 将该窗口（原始帧）叠加关键点与轨迹后，保存为独立的视频文件 win_{start}.mp4

输入模式：
- 单文件：通过 YAML 的 `video` 或 CLI 的 `--video` 指定单个视频
- 目录：通过 YAML 的 `input_dir` 或 CLI 的 `--input-dir` 指定根目录；也支持 `video` 指向目录
  - `recursive` 控制是否递归搜索
  - `include_exts` 控制匹配的视频扩展名列表
  - `max_files` 控制最多处理的文件数（0 表示无限制）

依赖：torch, opencv-python, numpy（与 co-tracker hub 可用）
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import argparse
import os

import cv2
import numpy as np
import torch
from video_action_segmenter.stream_utils import (
    load_config,
    _normalize_velocities,
    pre_gate_check,
    motion_gate_check,
)


# -----------------------------
# I/O 与时间重采样/缩放
# -----------------------------

class VideoFrameReader:
    """流式视频帧读取器，支持时间重采样和缩放。"""
    
    def __init__(self, video_path: Path, target_fps: int, resize_shorter: int):
        self.video_path = video_path
        self.target_fps = target_fps
        self.resize_shorter = resize_shorter
        
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        self.fps_in = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        # 时间采样参数
        if target_fps > 0 and self.fps_in > 0:
            self.step_sec = 1.0 / float(target_fps)
            self.use_time_sampling = True
        else:
            self.step_sec = 0.0
            self.use_time_sampling = False
        
        self.current_idx = 0
        self.next_sample_time = 0.0
        self._cached_frames = []  # 缓存少量帧用于窗口处理
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """缩放帧到指定短边尺寸。"""
        if not self.resize_shorter or self.resize_shorter <= 0:
            return frame
        
        h, w = frame.shape[:2]
        shorter = min(h, w)
        if shorter != self.resize_shorter:
            scale = self.resize_shorter / float(shorter)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        return frame
    
    def read_next_sampled_frame(self) -> Optional[np.ndarray]:
        """读取下一个采样帧。"""
        if self.cap is None:
            return None
            
        while True:
            ret, frame = self.cap.read()
            if not ret:
                return None
                
            if not self.use_time_sampling:
                # 不使用时间采样，直接返回每一帧
                return self._resize_frame(frame)
            
            # 使用时间采样
            current_time = self.current_idx / self.fps_in
            if current_time + 1e-6 >= self.next_sample_time:
                self.next_sample_time += self.step_sec
                self.current_idx += 1
                return self._resize_frame(frame)
            
            self.current_idx += 1
    
    def get_total_sampled_frames(self) -> int:
        """估算采样后的总帧数。"""
        if not self.use_time_sampling:
            return self.total_frames
        
        if self.fps_in <= 0:
            return self.total_frames
            
        video_duration = self.total_frames / self.fps_in
        return int(video_duration * self.target_fps)
    
    def read_window_frames(self, start_frame: int, window_size: int) -> List[np.ndarray]:
        """读取指定窗口的帧。
        
        Args:
            start_frame: 窗口开始帧索引（采样后的索引）
            window_size: 窗口大小
            
        Returns:
            窗口帧列表
        """
        # 如果缓存中有足够的帧，直接使用
        if len(self._cached_frames) > start_frame + window_size:
            return self._cached_frames[start_frame:start_frame + window_size]
        
        # 需要读取更多帧
        target_frames = start_frame + window_size
        
        # 如果缓存不足，继续读取
        while len(self._cached_frames) < target_frames:
            frame = self.read_next_sampled_frame()
            if frame is None:
                break
            self._cached_frames.append(frame)
        
        # 返回窗口帧
        if len(self._cached_frames) <= start_frame:
            return []
        
        end_idx = min(start_frame + window_size, len(self._cached_frames))
        return self._cached_frames[start_frame:end_idx]
    
    def clear_cache_before(self, frame_idx: int):
        """清理指定帧索引之前的缓存，释放内存。"""
        if frame_idx > 0 and len(self._cached_frames) > frame_idx:
            # 保留一些重叠帧以支持窗口滑动
            keep_from = max(0, frame_idx - 5)  # 保留5帧重叠
            self._cached_frames = self._cached_frames[keep_from:]


def _read_frames_resampled(video_path: Path, target_fps: int, resize_shorter: int) -> Tuple[List[np.ndarray], float]:
    """读取视频帧，按目标帧率进行基于时间轴的抽样，
    并将短边缩放到 resize_shorter（保持纵横比）。
    
    注意：此函数保留用于向后兼容，但建议使用 VideoFrameReader 进行流式处理。

    返回 (frames[BGR uint8], input_fps)。
    """
    with VideoFrameReader(video_path, target_fps, resize_shorter) as reader:
        frames = []
        while True:
            frame = reader.read_next_sampled_frame()
            if frame is None:
                break
            frames.append(frame)
        return frames, reader.fps_in


# -----------------------------
# CoTracker 加载与窗口跟踪（离线）
# -----------------------------

_MODEL_CACHE: dict = {}


def _get_cotracker_offline(device: torch.device):
    """优先尝试本地 hub 缓存加载 CoTracker 离线版，否则回退到 GitHub。"""
    key = f"offline_{device.type}"
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    # 优先本地 hub 缓存
    try:
        hub_dir = Path(torch.hub.get_dir()).expanduser()
    except Exception:
        hub_dir = Path.home() / ".cache/torch/hub"
    local_repo = hub_dir / "facebookresearch_co-tracker_main"

    try:
        if local_repo.exists():
            print(f"[CoTracker] Using local Hub cache: {local_repo}")
            model = torch.hub.load(str(local_repo), "cotracker3_offline", source="local").to(device)
        else:
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    except Exception as e:
        if local_repo.exists():
            print(f"[CoTracker][WARN] Remote load failed ({e}), retrying with local cache: {local_repo}")
            model = torch.hub.load(str(local_repo), "cotracker3_offline", source="local").to(device)
        else:
            raise

    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _to_clip_tensor(frames: List[np.ndarray], device: torch.device) -> Tuple[torch.Tensor, int, int]:
    # frames: list of HxWxC in BGR uint8 -> tensor (T, 3, H, W) float32 in [0, 1]
    arr = np.stack(frames, axis=0)
    H, W = arr.shape[1], arr.shape[2]
    arr = arr[:, :, :, ::-1]  # BGR -> RGB
    arr = arr.astype(np.float32) / 255.0
    clip = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)  # T, 3, H, W
    return clip, H, W


def track_window_offline(frames: List[np.ndarray], grid_size: int, device: torch.device) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """对给定窗口帧执行一次离线 CoTracker 跟踪。
    返回：
        tracks_np: (T, N, 2) float32 像素坐标
        visibility_np: (T, N) float32 可见度（若模型返回）或 None
    """
    if len(frames) == 0:
        raise RuntimeError("Empty frames for tracking")
    clip, H, W = _to_clip_tensor(frames, device)
    with torch.no_grad():
        video = clip.unsqueeze(0)  # (1, T, 3, H, W)
        model = _get_cotracker_offline(device)
        pred = model(video, grid_size=grid_size)
        if isinstance(pred, (tuple, list)) and len(pred) >= 1:
            tracks = pred[0]
            visibility = pred[1] if len(pred) > 1 else None
        else:
            tracks = pred
            visibility = None
        tracks = tracks[0].detach().cpu()  # (T, N, 2)
        tracks_np = tracks.numpy().astype(np.float32)
        if isinstance(visibility, torch.Tensor):
            visibility_np = visibility[0].detach().cpu().numpy().astype(np.float32)
        else:
            visibility_np = None
    return tracks_np, visibility_np


# -----------------------------
# 可视化叠加并保存（不弹出窗口）
# -----------------------------

def _make_colors(num: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    hues = np.linspace(0, 179, num=num, endpoint=False, dtype=np.float32)
    colors = []
    for h in hues:
        s = 200 + rng.integers(0, 56)  # 200..255
        v = 200 + rng.integers(0, 56)
        color = np.uint8([[[h, s, v]]])  # HSV
        bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        colors.append(bgr)
    return np.array(colors, dtype=np.uint8)


def _filter_static_points(tracks: np.ndarray, threshold: float) -> np.ndarray:
    """根据位移过滤静态点，返回动态点的索引。

    Args:
        tracks: (T, N, 2) 轨迹数据
        threshold: 最大位移阈值（像素），小于等于该值的点被视为静态

    Returns:
        (K,) 动态点的索引数组
    """
    if threshold <= 0:
        return np.arange(tracks.shape[1])

    # 计算每个点相对于其初始位置的最大位移
    # initial_positions: (1, N, 2)
    initial_positions = tracks[0:1, :, :]
    # displacements: (T, N, 2)
    displacements = tracks - initial_positions
    # max_displacements_per_point: (N,)
    max_displacements_per_point = np.max(np.linalg.norm(displacements, axis=2), axis=0)

    dynamic_indices = np.where(max_displacements_per_point > threshold)[0]
    return dynamic_indices.astype(np.int32)


def draw_tracks_and_save(
    frames: List[np.ndarray],
    tracks: np.ndarray,           # (T, N, 2)
    visibility: Optional[np.ndarray],  # (T, N) or None
    draw_indices: np.ndarray,     # (K,)
    trail: int,
    save_path: Path,
    display_fps: int,
    point_radius: int,
    line_thickness: int,
    overlay_text: Optional[str] = None,
) -> None:
    """将轨迹叠加到窗口帧并保存为视频；不显示窗口。"""
    T, N, _ = tracks.shape
    assert T == len(frames), "tracks length must match number of frames"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if save_path.suffix.lower() in [".mp4", ".m4v"] else cv2.VideoWriter_fourcc(*"XVID")
    h, w = frames[0].shape[:2]
    out_fps = max(1, int(display_fps))
    writer = cv2.VideoWriter(str(save_path), fourcc, out_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {save_path}")

    colors = _make_colors(len(draw_indices))

    for t in range(T):
        frame = frames[t].copy()
        img_h, img_w = frame.shape[:2]
        t0 = max(0, t - trail)
        seg = tracks[t0:t+1, draw_indices, :]  # (L, K, 2)
        if visibility is not None:
            vis_seg = visibility[t0:t+1, draw_indices]
            vis_now = (visibility[t, draw_indices] > 0.5)
        else:
            vis_seg = None
            vis_now = np.ones(len(draw_indices), dtype=bool)

        for k in range(len(draw_indices)):
            if not bool(vis_now[k]):
                continue
            pts = seg[:, k, :].astype(np.int32)
            # 画线段
            for i in range(1, pts.shape[0]):
                if vis_seg is not None and (vis_seg[i-1, k] <= 0.5 or vis_seg[i, k] <= 0.5):
                    continue
                x1, y1 = int(pts[i-1, 0]), int(pts[i-1, 1])
                x2, y2 = int(pts[i, 0]), int(pts[i, 1])
                cv2.line(frame, (x1, y1), (x2, y2), color=colors[k].tolist(), thickness=line_thickness, lineType=cv2.LINE_AA)
            # 画当前点
            x, y = int(pts[-1, 0]), int(pts[-1, 1])
            if 0 <= x < img_w and 0 <= y < img_h:
                cv2.circle(frame, (x, y), radius=point_radius, color=colors[k].tolist(), thickness=-1, lineType=cv2.LINE_AA)

        # 叠加文本
        cv2.putText(frame, f"t={t+1}/{T}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        if overlay_text:
            cv2.putText(frame, overlay_text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()


# -----------------------------
# 主流程
# -----------------------------

def _collect_video_files(input_dir: Path, include_exts: List[str], recursive: bool) -> List[Path]:
    """从目录中收集视频文件列表。

    Args:
        input_dir: 输入根目录
        include_exts: 允许的扩展名（如 [".mp4", ".avi"]），大小写不敏感，可不带点（"mp4" 也可）
        recursive: 是否递归搜索

    Returns:
        视频文件路径列表（按字典序排序）
    """
    # 标准化扩展名到以点开头的小写
    norm_exts = []
    for e in include_exts or []:
        if not e:
            continue
        e = e.lower()
        if not e.startswith('.'):
            e = '.' + e
        norm_exts.append(e)
    if not norm_exts:
        norm_exts = ['.mp4', '.m4v', '.mov', '.avi', '.mkv']

    candidates = input_dir.rglob('*') if recursive else input_dir.glob('*')
    files = [p for p in candidates if p.is_file() and p.suffix.lower() in norm_exts]
    files = sorted(files)
    return files

def build_argparser():
    p = argparse.ArgumentParser(description="Track an action video with sliding windows (T=16) and save each window with overlaid tracks")
    # 支持从 YAML 加载，CLI 可覆盖
    p.add_argument("--params", type=str, default=str(Path(__file__).resolve().with_name("params_window_track.yaml")), help="YAML 配置文件路径")
    # 将可覆盖项的默认设为 None，以便区分“未在 CLI 指定”的情况
    p.add_argument("--video", type=str, default=None, help="输入视频路径（可在 YAML 中设置 video；若为目录则按目录模式处理）")
    p.add_argument("--input-dir", type=str, default=None, help="输入目录（可在 YAML 中设置 input_dir）")
    p.add_argument("--recursive", type=lambda x: str(x).lower() in ["1", "true", "yes", "y"], default=None, help="是否递归搜索（true/false）")
    p.add_argument("--include-exts", type=str, default=None, help="逗号分隔的扩展名列表，如: mp4,mov,avi（YAML 亦可为列表）")
    p.add_argument("--max-files", type=int, default=None, help="最多处理的文件数；0 表示不限制")
    p.add_argument("--output-dir", type=str, default=None, help="输出根目录")
    p.add_argument("--target-fps", type=int, default=None, help="重采样目标 FPS")
    p.add_argument("--resize-shorter", type=int, default=None, help="短边缩放至该尺寸，保持纵横比")
    p.add_argument("--grid-size", type=int, default=None, help="CoTracker 网格大小 (N=grid_size^2)")
    p.add_argument("--T", type=int, default=None, help="窗口长度")
    p.add_argument("--stride", type=int, default=None, help="窗口步长")
    p.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"], help="计算设备")
    p.add_argument("--gpu-id", type=int, default=None, help="当 device=cuda 时，使用的 GPU ID")
    p.add_argument("--max-windows", type=int, default=None, help="最多处理的窗口数量（0 表示全部）")
    p.add_argument("--trail", type=int, default=None, help="轨迹历史长度（帧）")
    p.add_argument("--display-fps", type=int, default=None, help="输出视频 FPS（用于保存节奏）")
    p.add_argument("--point-radius", type=int, default=None, help="关键点绘制半径")
    p.add_argument("--line-thickness", type=int, default=None, help="轨迹线宽")
    p.add_argument("--static-filter-thresh", type=float, default=None, help="静态点过滤阈值(像素)")
    return p

def process_single_video(
    video_path: Path,
    output_root: Path,
    target_fps: int,
    resize_shorter: int,
    grid_size: int,
    T: int,
    stride: int,
    device: torch.device,
    max_windows: int,
    trail: int,
    display_fps: int,
    point_radius: int,
    line_thickness: int,
    static_filter_thresh: float,
    pre_gate_params: Dict[str, Any],
    motion_gate_params: Dict[str, Any],
    decoder_window_size: int,
    relative_parent: Optional[Path] = None,
) -> int:
    out_root = output_root
    if relative_parent and relative_parent != Path('.'):
        out_root = out_root / relative_parent
    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[I/O] input={video_path}")
    print(f"[I/O] out_dir={out_dir}")

    with VideoFrameReader(video_path, target_fps=target_fps, resize_shorter=resize_shorter) as reader:
        total_sampled_frames = reader.get_total_sampled_frames()
        _ = reader.fps_in

        if total_sampled_frames < T:
            print(f"[Skip] 估算帧数不足 T={T}，预计 {total_sampled_frames} 帧")
            return 0

        starts = list(range(0, total_sampled_frames - T + 1, stride))
        if max_windows and max_windows > 0:
            starts = starts[: max_windows]
        if not starts:
            print(f"[Skip] 无有效窗口（长度={total_sampled_frames}，T={T}，stride={stride}）")
            return 0

        print(f"[Windows] total={len(starts)}, estimated_frames={total_sampled_frames}")

        # 预热加载模型
        _ = _get_cotracker_offline(device)

        processed = 0
        pre_gate_enable = bool(pre_gate_params.get("enable", False))
        pre_gate_resize_shorter = int(pre_gate_params.get("resize_shorter", 128))
        pre_gate_method = str(pre_gate_params.get("diff_method", "mad"))
        pre_gate_pixel_diff_thr = float(pre_gate_params.get("pixel_diff_thr", 0.01))
        pre_gate_mad_thr = float(pre_gate_params.get("mad_thr", 0.003))
        pre_gate_min_active_ratio = float(pre_gate_params.get("min_active_ratio", 0.002))
        pre_gate_debug = bool(pre_gate_params.get("debug", False))

        motion_gate_enable = bool(motion_gate_params.get("enable", False))
        motion_gate_vel_thr = float(motion_gate_params.get("vel_norm_thr", 0.012))
        motion_gate_min_active_ratio = float(motion_gate_params.get("min_active_ratio", 0.01))
        motion_gate_debug = bool(motion_gate_params.get("debug", False))

        for i, s in enumerate(starts):
            clear_cache_needed = (i % 10 == 0 and i > 0)
            # 流式读取窗口帧
            sub = reader.read_window_frames(s, T)
            if len(sub) < T:
                print(f"[Window][SKIP] start={s} 实际帧数不足: {len(sub)} < {T}")
                if clear_cache_needed:
                    reader.clear_cache_before(s)
                continue

            # 1) 前置像素差分门控，避免进入 CoTracker 的静止窗口
            if pre_gate_enable:
                dropped, mad_val, act_ratio = pre_gate_check(
                    sub,
                    resize_shorter=pre_gate_resize_shorter,
                    pixel_diff_thr=pre_gate_pixel_diff_thr,
                    mad_thr=pre_gate_mad_thr,
                    min_active_ratio=pre_gate_min_active_ratio,
                    method=pre_gate_method,
                    debug=pre_gate_debug,
                )
                if dropped:
                    if pre_gate_debug:
                        print(
                            f"[PreGate] DROP start={s}: mad={mad_val:.6f}(<{pre_gate_mad_thr}) ratio={act_ratio:.4f}(<{pre_gate_min_active_ratio})"
                        )
                    if clear_cache_needed:
                        reader.clear_cache_before(s)
                    continue
            try:
                tracks, vis = track_window_offline(sub, grid_size=grid_size, device=device)
            except Exception as e:
                print(f"[Window][ERR] start={s} 跟踪失败: {e}")
                if clear_cache_needed:
                    reader.clear_cache_before(s)
                continue

            N = tracks.shape[1]

            # 2) 运动门控：基于归一化速度判断是否静止窗口
            if motion_gate_enable:
                try:
                    vel_pix = tracks[1:] - tracks[:-1]
                    vel_tensor = torch.from_numpy(vel_pix)
                    vel_norm = _normalize_velocities(vel_tensor, decoder_window_size)
                    dropped, v_l2_mean, active_ratio = motion_gate_check(
                        vel_norm,
                        vel_norm_thr=motion_gate_vel_thr,
                        min_active_ratio=motion_gate_min_active_ratio,
                        debug=motion_gate_debug,
                    )
                except Exception as e:
                    if motion_gate_debug:
                        print(f"[Gate][WARN] start={s} 运动门控失败: {e}")
                    dropped = False
                if dropped:
                    if motion_gate_debug:
                        print(
                            f"[Gate] DROP start={s}: mean={v_l2_mean:.6f}(<{motion_gate_vel_thr}) ratio={active_ratio:.4f}(<{motion_gate_min_active_ratio})"
                        )
                    if clear_cache_needed:
                        reader.clear_cache_before(s)
                    continue

            # 1. 过滤静态点
            dynamic_indices = _filter_static_points(tracks, static_filter_thresh)

            # 2. 使用全部动态点进行可视化（若无动态点则回退到全部点）
            if dynamic_indices.size > 0:
                draw_idx = dynamic_indices
            else:
                draw_idx = np.arange(N, dtype=np.int32)
            save_path = out_dir / f"win_{s:06d}.mp4"
            overlay = f"start={s} T={T} stride={stride} grid={grid_size} N={N}"
            try:
                draw_tracks_and_save(
                    frames=sub,
                    tracks=tracks,
                    visibility=vis,
                    draw_indices=draw_idx,
                    trail=trail,
                    save_path=save_path,
                    display_fps=display_fps,
                    point_radius=point_radius,
                    line_thickness=line_thickness,
                    overlay_text=overlay,
                )
                print(f"[Window][OK] start={s} -> {save_path.name} (N={N}, draw={len(draw_idx)})")
                processed += 1
            except Exception as e:
                print(f"[Window][SAVE-ERR] start={s}: {e}")
                if clear_cache_needed:
                    reader.clear_cache_before(s)
                continue

            # 定期清理缓存以释放内存
            if clear_cache_needed:
                reader.clear_cache_before(s)

    print(f"[Done] processed={processed}/{len(starts)} windows, output_dir={out_dir}")
    return processed


def main():
    args = build_argparser().parse_args()

    # 1) 加载 YAML 配置（若存在）
    cfg: Dict[str, Any] = {}
    params_path = Path(args.params) if isinstance(args.params, str) and len(args.params) > 0 else None
    if params_path is not None and params_path.exists():
        try:
            cfg = load_config(str(params_path))  # dict
            if not isinstance(cfg, dict):
                cfg = {}
        except Exception as e:
            print(f"[WARN] 无法加载配置文件 {params_path}: {e}")
            cfg = {}
    else:
        if params_path is not None:
            print(f"[INFO] 配置文件不存在，使用 CLI/内置默认: {params_path}")

    # 2) 合并：YAML 默认 + CLI 覆盖
    def pick(key: str, default: Any = None):
        v_cli = getattr(args, key.replace('-', '_')) if hasattr(args, key.replace('-', '_')) else None
        return v_cli if v_cli is not None else cfg.get(key, default)

    # 输入解析：支持单文件或目录
    video_str = pick('video', None)
    input_dir_str = pick('input_dir', None)
    if not video_str and not input_dir_str:
        print("[ERR] 未指定输入。请在 YAML/CLI 指定 'video'（文件或目录）或 'input_dir'。")
        return 1

    output_dir = pick('output_dir', str(Path(__file__).resolve().with_name("inference_outputs") / "windows"))
    target_fps = int(pick('target_fps', 20))
    resize_shorter = int(pick('resize_shorter', 480))
    grid_size = int(pick('grid_size', 20))
    T = int(pick('T', 16))
    stride = int(pick('stride', 4))
    device_str = pick('device', 'auto')
    gpu_id = pick('gpu_id', None)
    max_windows = int(pick('max_windows', 0) or 0)
    trail = int(pick('trail', 15))
    display_fps = int(pick('display_fps', target_fps))
    point_radius = int(pick('point_radius', 2))
    line_thickness = int(pick('line_thickness', 1))
    static_filter_thresh = float(pick('static_filter_thresh', 0.0))
    decoder_window_size = int(pick('decoder_window_size', 15))
    pre_gate_cfg = cfg.get('pre_gate', {}) if isinstance(cfg.get('pre_gate', {}), dict) else {}
    motion_gate_cfg = cfg.get('motion_gate', {}) if isinstance(cfg.get('motion_gate', {}), dict) else {}
    # 预 Gate 参数
    pre_gate_params: Dict[str, Any] = {
        'enable': bool(pre_gate_cfg.get('enable', True)),
        'resize_shorter': int(pre_gate_cfg.get('resize_shorter', 128)) if pre_gate_cfg.get('resize_shorter', None) is not None else 128,
        'diff_method': str(pre_gate_cfg.get('diff_method', 'mad')).lower(),
        'pixel_diff_thr': float(pre_gate_cfg.get('pixel_diff_thr', 0.01)) if pre_gate_cfg.get('pixel_diff_thr', None) is not None else 0.01,
        'mad_thr': float(pre_gate_cfg.get('mad_thr', 0.003)) if pre_gate_cfg.get('mad_thr', None) is not None else 0.003,
        'min_active_ratio': float(pre_gate_cfg.get('min_active_ratio', 0.002)) if pre_gate_cfg.get('min_active_ratio', None) is not None else 0.002,
        'debug': bool(pre_gate_cfg.get('debug', False)),
    }
    # 运动 Gate 参数
    motion_gate_params: Dict[str, Any] = {
        'enable': bool(motion_gate_cfg.get('enable', True)),
        'vel_norm_thr': float(motion_gate_cfg.get('vel_norm_thr', 0.012)) if motion_gate_cfg.get('vel_norm_thr', None) is not None else 0.012,
        'min_active_ratio': float(motion_gate_cfg.get('min_active_ratio', 0.01)) if motion_gate_cfg.get('min_active_ratio', None) is not None else 0.01,
        'debug': bool(motion_gate_cfg.get('debug', False)),
    }
    # 目录模式相关配置
    recursive = bool(pick('recursive', False))
    include_exts_cfg = pick('include_exts', None)
    if isinstance(include_exts_cfg, str) and include_exts_cfg:
        include_exts = [s.strip() for s in include_exts_cfg.split(',') if s.strip()]
    elif isinstance(include_exts_cfg, (list, tuple)):
        include_exts = list(include_exts_cfg)
    else:
        include_exts = ['.mp4', '.m4v', '.mov', '.avi', '.mkv']
    max_files = int(pick('max_files', 0) or 0)

    # 3) 设备解析
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if device_str == "cuda" else "cpu")
    if device.type == "cuda" and gpu_id is not None:
        try:
            torch.cuda.set_device(int(gpu_id))
            device = torch.device(f"cuda:{int(gpu_id)}")
        except Exception:
            pass

    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[Config] device={device}, T={T}, stride={stride}, grid_size={grid_size}, fps={target_fps}, resize_shorter={resize_shorter}")

    total_processed_windows = 0
    processed_videos = 0

    # 决定执行模式
    dir_mode = False
    dir_path: Optional[Path] = None
    if video_str:
        vpath = Path(video_str)
        if not vpath.exists():
            raise FileNotFoundError(f"Video path not found: {vpath}")
        if vpath.is_dir():
            dir_mode = True
            dir_path = vpath
        else:
            # 单文件模式
            total_processed_windows += process_single_video(
                video_path=vpath,
                output_root=out_root,
                target_fps=target_fps,
                resize_shorter=resize_shorter,
                grid_size=grid_size,
                T=T,
                stride=stride,
                device=device,
                max_windows=max_windows,
                trail=trail,
                display_fps=display_fps,
                point_radius=point_radius,
                line_thickness=line_thickness,
                static_filter_thresh=static_filter_thresh,
                pre_gate_params=pre_gate_params,
                motion_gate_params=motion_gate_params,
                decoder_window_size=decoder_window_size,
                relative_parent=None,
            )
            processed_videos = 1
    if input_dir_str is not None:
        # 如果显式给出 input_dir，则强制目录模式
        dpath = Path(input_dir_str)
        if not dpath.exists() or not dpath.is_dir():
            raise FileNotFoundError(f"Input directory not found: {dpath}")
        dir_mode = True
        dir_path = dpath

    if dir_mode and dir_path is not None:
        files = _collect_video_files(dir_path, include_exts=include_exts, recursive=recursive)
        if max_files > 0:
            files = files[:max_files]
        if not files:
            print(f"[Batch][Skip] 目录中未找到匹配的视频: dir={dir_path}, exts={include_exts}, recursive={recursive}")
            return 1
        print(f"[Batch] found {len(files)} video files in {dir_path}")
        for fp in files:
            relative_parent: Optional[Path] = None
            try:
                relative_parent = fp.parent.relative_to(dir_path)
            except ValueError:
                # 如果文件不在 dir_path 内（符号链接等情况），保持默认结构
                relative_parent = None
            try:
                processed = process_single_video(
                    video_path=fp,
                    output_root=out_root,
                    target_fps=target_fps,
                    resize_shorter=resize_shorter,
                    grid_size=grid_size,
                    T=T,
                    stride=stride,
                    device=device,
                    max_windows=max_windows,
                    trail=trail,
                    display_fps=display_fps,
                    point_radius=point_radius,
                    line_thickness=line_thickness,
                    static_filter_thresh=static_filter_thresh,
                    pre_gate_params=pre_gate_params,
                    motion_gate_params=motion_gate_params,
                    decoder_window_size=decoder_window_size,
                    relative_parent=relative_parent,
                )
                total_processed_windows += processed
                processed_videos += 1
            except Exception as e:
                print(f"[Batch][ERR] file={fp}: {e}")
                continue

        print(f"[Batch][Done] videos={processed_videos}, total_processed_windows={total_processed_windows}, output_root={out_root}")
        return 0 if total_processed_windows > 0 else 1

    # 单文件模式已在前面执行过
    return 0 if total_processed_windows > 0 else 1


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    raise SystemExit(main())
