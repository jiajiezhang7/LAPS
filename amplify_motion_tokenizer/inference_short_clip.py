import argparse
import os
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import logging

import cv2
import numpy as np
import torch
import sys
from tqdm import tqdm
import torch.multiprocessing as mp

import re
from datetime import datetime
from amplify_motion_tokenizer.models.motion_tokenizer import MotionTokenizer
from amplify_motion_tokenizer.utils.helpers import load_config, get_device


# -----------------------------
# I/O 与跟踪/窗口处理工具
# -----------------------------

@dataclass
class InferCfg:
    checkpoint_dir: Path
    checkpoint_name: str
    model_config: Path
    video_root: Path
    output_dir: Path
    target_fps: int
    resize_shorter: int
    stride: int
    device: torch.device
    video_exts: List[str]
    save_codes: bool
    max_files: int
    save_json: bool
    json_round_decimals: int
    json_export_mode: str  # 'codes' | 'matrix' | 'both'
    save_jsonl: bool
    jsonl_filename: str
    debug_fsq: bool
    gpu_ids: Optional[List[int]]
    batch_size: int
    amp: bool
    progress_mininterval: float


def _init_logger(base_dir: Path, device: torch.device, rank: int, world_size: int) -> logging.Logger:
    """为当前进程初始化专属 logger。日志写入 run_output_dir/logs/ 下。
    文件名规则：timing_cpu.log 或 timing_gpu<ID>.log；若多进程，则附带 rank。
    """
    logs_dir = base_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    if device.type == 'cuda':
        dev_id = device.index if device.index is not None else rank
        fname = f"timing_gpu{dev_id}.log" if world_size > 1 else f"timing_gpu{dev_id}.log"
    else:
        fname = "timing_cpu.log" if world_size <= 1 else f"timing_cpu_rank{rank}.log"
    log_path = logs_dir / fname

    logger = logging.getLogger(f"infer_timing_{fname}")
    logger.setLevel(logging.INFO)
    # 避免重复添加 handler
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class _Timer:
    """简单计时器，支持 with 语法或手动 start/stop。"""
    def __init__(self):
        self.t0 = None
        self.elapsed = 0.0
    def start(self):
        self.t0 = time.perf_counter()
        return self
    def stop(self):
        if self.t0 is not None:
            self.elapsed += time.perf_counter() - self.t0
            self.t0 = None
        return self.elapsed
    def reset(self):
        self.t0 = None
        self.elapsed = 0.0
        return self

def _gather_videos(root: Path, exts: List[str]) -> List[Path]:
    exts = [e.lower() for e in exts]
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


# Debug 全局控制，限制打印次数，避免刷屏
_FSQ_DEBUG_MAX_PRINTS = 4
_FSQ_DEBUG_COUNT = 0

# 强制 stdout 行缓冲，确保 tqdm/print 实时输出
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _read_frames_resampled(video_path: Path, target_fps: int, resize_shorter: int) -> Tuple[List[np.ndarray], float]:
    """读取视频帧，按目标帧率进行基于时间轴的抽样（确保 25->20 等非整数倍率也能正确转换），
    并将短边缩放到 resize_shorter（保持纵横比）。

    返回 (frames[BGR uint8], input_fps)。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []

    if target_fps <= 0 or fps_in <= 0:
        # 直接读取全部帧
        ok, frame = cap.read()
        while ok:
            if resize_shorter and resize_shorter > 0:
                h, w = frame.shape[:2]
                shorter = min(h, w)
                if shorter != resize_shorter:
                    scale = resize_shorter / float(shorter)
                    nh, nw = int(round(h * scale)), int(round(w * scale))
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            ok, frame = cap.read()
        cap.release()
        return frames, fps_in

    # 基于时间的抽样：按 1/target_fps 的时间步选择最接近的输入帧
    step_sec = 1.0 / float(target_fps)
    next_t = 0.0
    idx = 0
    ok, frame = cap.read()
    while ok:
        t = idx / fps_in
        if t + 1e-6 >= next_t:
            # 保留该帧
            if resize_shorter and resize_shorter > 0:
                h, w = frame.shape[:2]
                shorter = min(h, w)
                if shorter != resize_shorter:
                    scale = resize_shorter / float(shorter)
                    nh, nw = int(round(h * scale)), int(round(w * scale))
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            next_t += step_sec
        idx += 1
        ok, frame = cap.read()

    cap.release()
    return frames, fps_in


def _to_clip_tensor(frames: List[np.ndarray], device: torch.device) -> Tuple[torch.Tensor, int, int]:
    # frames: list of HxWxC in BGR uint8 -> tensor (T, 3, H, W) float32 in [0, 1]
    arr = np.stack(frames, axis=0)  # T, H, W, 3
    H, W = arr.shape[1], arr.shape[2]
    arr = arr[:, :, :, ::-1]  # BGR -> RGB
    arr = arr.astype(np.float32) / 255.0
    clip = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)  # T, 3, H, W
    return clip, H, W


_MODEL_CACHE: Dict[str, torch.nn.Module] = {}

def _get_cotracker(device: torch.device):
    key = f"offline_{device.type}"
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model
    # 需要联网从 torch.hub 下载，或本地已缓存
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _track_full_clip(frames: List[np.ndarray], grid_size: int, device: torch.device) -> torch.Tensor:
    """对整段视频进行一次离线跟踪，返回轨迹张量 tracks: (T, N, 2) on CPU。"""
    if len(frames) == 0:
        raise RuntimeError("Empty frames for tracking")
    clip, H, W = _to_clip_tensor(frames, device)
    with torch.no_grad():
        video = clip.unsqueeze(0)  # 1, T, 3, H, W
        tracks, _ = _get_cotracker(device)(video, grid_size=grid_size)
        tracks = tracks[0].detach().cpu()  # T, N, 2
    return tracks


def _track_window(frames: List[np.ndarray], start: int, T: int, grid_size: int, device: torch.device) -> torch.Tensor:
    """对以 start 开始、长度为 T 的窗口执行一次离线跟踪，返回窗口速度 (T-1, N, 2) (CPU)。
    注意：离线模型在 clip 的首帧初始化网格，因此将 clip 设为 frames[start:start+T] 即可满足“每窗重置关键点”的要求。
    """
    end = start + T
    if end > len(frames):
        raise ValueError(f"Window out of range: start={start}, T={T}, total_frames={len(frames)}")
    sub = frames[start:end]
    clip, H, W = _to_clip_tensor(sub, device)
    with torch.no_grad():
        video = clip.unsqueeze(0)  # 1, T, 3, H, W
        tracks, _ = _get_cotracker(device)(video, grid_size=grid_size)
        tracks = tracks[0].detach().cpu()  # T, N, 2
        velocities = tracks[1:] - tracks[:-1]  # (T-1, N, 2)
    return velocities


def _normalize_velocities(vel: torch.Tensor, window_size: int) -> torch.Tensor:
    """将像素位移归一化到 [-1, 1]，按模型解码窗口半径进行缩放。
    v_norm = clamp( vel_pixels / ((W_dec - 1) / 2), -1, 1 )
    Args:
        vel: (T-1, N, 2) in pixel displacement
        window_size: 模型解码局部窗口大小 (W_dec)
    Returns:
        vel_norm: 归一化到 [-1, 1] 的速度张量 (同形状)
    """
    scale = max(1.0, (float(window_size) - 1.0) / 2.0)
    vel_norm = torch.clamp(vel / scale, -1.0, 1.0)
    return vel_norm


# 组装 FSQ 多进制 digits -> 单整数 ID（范围 [0, ∏levels - 1]）
def _fsq_digits_to_ids(fsq_idx: torch.Tensor, levels: List[int]) -> torch.Tensor:
    """将 FSQ 返回的离散索引转换为单一的码本 ID。
    兼容多种形状：
      - (d,)             已经是单整数 ID，直接返回
      - (d, k)           每 token k 个 digits（k=len(levels)）
      - (1, d) / (1, d, k)  去掉 batch 维
    映射采用小端混合进制：id = sum_{j=0..k-1} digit_j * prod(levels[:j])。
    """
    if fsq_idx is None:
        raise ValueError("fsq_idx is None; 量化器未返回离散索引")
    x = fsq_idx.detach().cpu().to(torch.long)
    # 去 batch 维
    if x.dim() == 3 and x.size(0) == 1:
        x = x.squeeze(0)
    if x.dim() == 2 and x.size(0) == 1:
        x = x.squeeze(0)
    # (d,) 直接返回
    if x.dim() == 1:
        return x
    # (d, k) 或 (k, d)
    if x.dim() == 2:
        d0, d1 = x.shape
        if d1 == len(levels):
            digits = x  # (d, k)
        elif d0 == len(levels):
            digits = x.transpose(0, 1)  # (d, k)
        else:
            raise ValueError(f"无法判定 FSQ 索引的 digits 维度, shape={tuple(x.shape)}, levels={levels}")

        # 预计算混合进制权重（小端）
        weights = torch.ones((len(levels),), dtype=torch.long)
        cur = 1
        for j, L in enumerate(levels):
            weights[j] = cur
            cur *= int(L)
        ids = (digits * weights.view(1, -1)).sum(dim=-1)  # (d,)
        return ids.to(torch.long)

    raise ValueError(f"不支持的 FSQ 索引形状: {tuple(x.shape)}")


# -----------------------------
# 模型推理：提取 FSQ 代码与标签
# -----------------------------

def extract_latent_and_labels(model: MotionTokenizer, window_vel: torch.Tensor, debug_fsq: bool = False, dbg_tag: str = "") -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """对单个窗口 (T-1,N,2) 进行前向编码，返回 (latent_tokens, fsq_indices or None, labels_per_point)。
    - latent_tokens: 量化后的记忆序列 (d, D)，即每个窗口的潜在表示（可作为 Latent Action Sequence）。
    - fsq_indices: 若后端提供离散码索引，则返回相应索引（形状依具体实现而定），否则为 None。
    - labels_per_point: (N,) 每个关键点的预测类别（decoder 输出经 argmax）。

    注意：模型的 decoder 输出在时间维度是复制的，因此 labels 在各时间步相同，这里返回 (N,) 的每点标签。
    """
    device = next(model.parameters()).device
    # 组装与 model.forward 一致的计算图，但捕获 quantizer indices
    with torch.no_grad():
        Tm1, N, _ = window_vel.shape
        B = 1
        x = window_vel.unsqueeze(0).reshape(B, Tm1 * N, 2).to(device)  # (1, S, 2)
        x = model.input_projection(x)
        x = x + model.pos_embed  # (1, S, D)
        encoded = model.encoder(x, mask=model.causal_mask.to(device))  # (1, S, D)
        proj_in = encoded.transpose(1, 2)  # (1, D, S)
        proj_out = model.encoder_output_projection(proj_in)  # (1, D, d)
        to_quantize = proj_out.transpose(1, 2)  # (1, d, D)

        # 尝试稳定获取量化索引
        fsq_indices = None
        try:
            quant_out = model.quantizer(to_quantize, return_indices=True)
        except TypeError:
            quant_out = model.quantizer(to_quantize)

        if isinstance(quant_out, (tuple, list)):
            # 选择浮点张量作为 quantized，整型张量作为 indices
            quantized = None
            for item in quant_out:
                if isinstance(item, torch.Tensor):
                    if item.dtype in (torch.float16, torch.float32, torch.float64):
                        quantized = item
                    elif item.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                        fsq_indices = item
            if quantized is None:
                # 回退到第一个元素
                quantized = quant_out[0] if isinstance(quant_out[0], torch.Tensor) else torch.as_tensor(quant_out[0])
        else:
            quantized = quant_out

        # Debug：打印量化返回的形状/类型/范围
        global _FSQ_DEBUG_COUNT
        if debug_fsq and _FSQ_DEBUG_COUNT < _FSQ_DEBUG_MAX_PRINTS:
            _FSQ_DEBUG_COUNT += 1
            try:
                print(f"[FSQ-DEBUG] {dbg_tag} to_quantize: shape={tuple(to_quantize.shape)} dtype={to_quantize.dtype}")
                if isinstance(quant_out, (tuple, list)):
                    print(f"[FSQ-DEBUG] {dbg_tag} quant_out list len={len(quant_out)}")
                    for idx, it in enumerate(quant_out):
                        if isinstance(it, torch.Tensor):
                            msg = f"tensor shape={tuple(it.shape)} dtype={it.dtype}"
                            try:
                                if it.numel() > 0:
                                    msg += f" min={float(it.min())} max={float(it.max())}"
                            except Exception:
                                pass
                            print(f"[FSQ-DEBUG] {dbg_tag} quant_out[{idx}]: {msg}")
                        else:
                            print(f"[FSQ-DEBUG] {dbg_tag} quant_out[{idx}]: type={type(it)}")
                else:
                    print(f"[FSQ-DEBUG] {dbg_tag} quant_out: shape={tuple(quant_out.shape) if isinstance(quant_out, torch.Tensor) else 'NA'} dtype={quant_out.dtype if isinstance(quant_out, torch.Tensor) else type(quant_out)}")
                if fsq_indices is not None:
                    fi = fsq_indices
                    if fi.dim() == 3 and fi.size(0) == 1:
                        fi = fi.squeeze(0)
                    if fi.dim() == 2 and fi.size(0) == 1:
                        fi = fi.squeeze(0)
                    msg = f"shape={tuple(fi.shape)} dtype={fi.dtype}"
                    try:
                        msg += f" min={int(fi.min())} max={int(fi.max())} uniq={fi.unique().numel()}"
                    except Exception:
                        pass
                    print(f"[FSQ-DEBUG] {dbg_tag} fsq_indices: {msg}")
            except Exception as e:
                print(f"[FSQ-DEBUG] error: {e}")

        queries = model.decoder_queries.expand(B, -1, -1)
        decoded = model.decoder(tgt=queries, memory=quantized)  # (1, N, D)
        logits = model.output_projection(decoded)  # (1, N, C)
        labels_per_point = logits.argmax(dim=-1).squeeze(0).detach().cpu()  # (N,)

    # 提取量化后的潜在序列 (d, D)，移到 CPU 以便保存
    latent_tokens = quantized.squeeze(0).detach().cpu()  # (d, D)
    if fsq_indices is not None:
        return latent_tokens, fsq_indices.detach().cpu(), labels_per_point
    return latent_tokens, None, labels_per_point


def extract_latent_and_labels_batched(
    model: MotionTokenizer,
    window_vel_list: List[torch.Tensor],  # list of (T-1,N,2) on CPU
    debug_fsq: bool = False,
    dbg_tag: str = "",
    amp: bool = False,
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], List[torch.Tensor]]:
    """批量对多个窗口进行编码/解码。
    返回：latent_list[(d,D)]、fsq_list(可 None)、labels_list[(N,)].
    """
    if len(window_vel_list) == 0:
        return [], [], []
    device = next(model.parameters()).device
    # pad-free reshape 合并 batch
    Tm1, N, _ = window_vel_list[0].shape
    x = torch.stack(window_vel_list, dim=0).reshape(len(window_vel_list), Tm1 * N, 2).to(device)

    latent_list: List[torch.Tensor] = []
    fsq_list: List[Optional[torch.Tensor]] = []
    labels_list: List[torch.Tensor] = []

    with torch.no_grad():
        try:
            autocast_ctx = torch.cuda.amp.autocast(enabled=amp and device.type == 'cuda')
        except Exception:
            class _Dummy:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc, tb):
                    return False
            autocast_ctx = _Dummy()

        with autocast_ctx:
            x = model.input_projection(x)  # (B,S,D)
            x = x + model.pos_embed  # (B,S,D)
            encoded = model.encoder(x, mask=model.causal_mask.to(device))  # (B,S,D)
            proj_in = encoded.transpose(1, 2)  # (B,D,S)
            proj_out = model.encoder_output_projection(proj_in)  # (B,D,d)
            to_quantize = proj_out.transpose(1, 2)  # (B,d,D)

            fsq_indices = None
            try:
                quant_out = model.quantizer(to_quantize, return_indices=True)
            except TypeError:
                quant_out = model.quantizer(to_quantize)

            if isinstance(quant_out, (tuple, list)):
                quantized = None
                for item in quant_out:
                    if isinstance(item, torch.Tensor):
                        if item.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                            quantized = item
                        elif item.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                            fsq_indices = item
                if quantized is None:
                    quantized = quant_out[0] if isinstance(quant_out[0], torch.Tensor) else torch.as_tensor(quant_out[0], device=device)
            else:
                quantized = quant_out

            global _FSQ_DEBUG_COUNT
            if debug_fsq and _FSQ_DEBUG_COUNT < _FSQ_DEBUG_MAX_PRINTS:
                _FSQ_DEBUG_COUNT += 1
                try:
                    print(f"[FSQ-DEBUG] {dbg_tag} B={len(window_vel_list)} to_quantize={tuple(to_quantize.shape)}", flush=True)
                except Exception:
                    pass

            queries = model.decoder_queries.expand(len(window_vel_list), -1, -1)
            decoded = model.decoder(tgt=queries, memory=quantized)  # (B,N,D)
            logits = model.output_projection(decoded)  # (B,N,C)
            labels_batch = logits.argmax(dim=-1).detach().cpu()  # (B,N)

    q_cpu = quantized.detach().cpu()  # (B,d,D)
    fi_cpu = fsq_indices.detach().cpu() if isinstance(fsq_indices, torch.Tensor) else None
    for i in range(q_cpu.size(0)):
        latent_list.append(q_cpu[i])
        labels_list.append(labels_batch[i])
        fsq_list.append(None if fi_cpu is None else fi_cpu[i])

    return latent_list, fsq_list, labels_list


# -----------------------------
# 主流程
# -----------------------------

def load_infer_cfg(cfg_path: Path) -> InferCfg:
    cfg = load_config(str(cfg_path))
    device = get_device(cfg.get("device", "auto"))
    return InferCfg(
        checkpoint_dir=Path(cfg["checkpoint_dir"]).resolve(),
        checkpoint_name=str(cfg.get("checkpoint_name", "best.pth")),
        model_config=Path(cfg["model_config"]).resolve(),
        video_root=Path(cfg["video_root"]).resolve(),
        output_dir=Path(cfg["output_dir"]).resolve(),
        target_fps=int(cfg.get("target_fps", 20)),
        resize_shorter=int(cfg.get("resize_shorter", 480)),
        stride=int(cfg.get("stride", 8)),
        device=device,
        video_exts=[str(x).lower() for x in cfg.get("video_exts", [".mp4", ".mov", ".avi", ".mkv"])],
        save_codes=bool(cfg.get("save_codes", True)),
        max_files=int(cfg.get("max_files", 0) or 0),
        save_json=bool(cfg.get("save_json", True)),
        json_round_decimals=int(cfg.get("json_round_decimals", 4)),
        json_export_mode=str(cfg.get("json_export_mode", "codes")),
        save_jsonl=bool(cfg.get("save_jsonl", True)),
        jsonl_filename=str(cfg.get("jsonl_filename", "codes.jsonl")),
        debug_fsq=bool(cfg.get("debug_fsq", False)),
        gpu_ids=[int(x) for x in str(cfg.get("gpu_ids", "")).split(',') if str(x).strip().isdigit()] if cfg.get("gpu_ids", None) else None,
        batch_size=int(cfg.get("batch_size", 1) or 1),
        amp=bool(cfg.get("amp", False)),
        progress_mininterval=float(cfg.get("progress_mininterval", 0.1)),
    )


def _process_videos(
    videos: List[Path],
    infer_cfg: InferCfg,
    model_cfg: Dict,
    run_output_dir: Path,
    rank: int = 0,
    world_size: int = 1,
) -> int:
    """在当前进程/设备上处理一组视频。返回处理数量。"""
    # 设备与模型
    if infer_cfg.device.type == 'cuda':
        try:
            gpu_index = infer_cfg.device.index if infer_cfg.device.index is not None else 0
            torch.cuda.set_device(gpu_index)
        except Exception:
            pass
    # 初始化日志器
    logger = _init_logger(run_output_dir, infer_cfg.device, rank, world_size)

    total_timer = _Timer().start()
    model = MotionTokenizer(model_cfg).to(infer_cfg.device)
    ckpt_path = infer_cfg.checkpoint_dir / infer_cfg.checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到权重文件: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=infer_cfg.device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 解析模型/数据配置
    T = int(model_cfg['data']['sequence_length'])
    grid_size = int(model_cfg['data']['grid_size'])
    N = int(model_cfg['data']['num_points'])
    assert N == grid_size * grid_size, f"num_points ({N}) 必须等于 grid_size^2 ({grid_size*grid_size})"
    W_dec = int(model_cfg['model']['decoder_window_size'])
    num_classes = int(model_cfg['model']['num_classes'])
    d = int(model_cfg['model']['encoder_sequence_len'])
    fsq_levels = list(model_cfg['model'].get('fsq_levels', []))
    codebook_size = 1
    for L in fsq_levels:
        codebook_size *= int(L)

    # JSONL 文件名（多进程安全）：为每个 rank 追加后缀
    jsonl_filename = infer_cfg.jsonl_filename
    if world_size > 1:
        base, ext = os.path.splitext(jsonl_filename)
        jsonl_filename = f"{base}_gpu{infer_cfg.device.index if infer_cfg.device.index is not None else rank}{ext or '.jsonl'}"

    # 视频级进度条（仅主进程显示）
    show_video_bar = (world_size == 1) or (rank == 0)
    video_iter = videos
    if show_video_bar:
        video_iter = tqdm(videos, desc="Videos", unit="video", dynamic_ncols=True, mininterval=infer_cfg.progress_mininterval)

    total_files = 0
    for vid in video_iter:
        per_video_timer = _Timer().start()
        t_read = 0.0
        t_track = 0.0
        t_norm = 0.0
        t_forward = 0.0
        t_fsq_merge = 0.0
        t_save_pt = 0.0
        t_save_json = 0.0
        t_save_jsonl = 0.0
        num_windows = 0

        try:
            _t = time.perf_counter()
            frames, fps_in = _read_frames_resampled(vid, target_fps=infer_cfg.target_fps, resize_shorter=infer_cfg.resize_shorter)
            t_read += time.perf_counter() - _t
        except Exception as e:
            tqdm.write(f"[Infer][GPU {infer_cfg.device}] 读取失败，跳过 {vid.name}: {e}")
            logger.info(f"VIDEO {vid} | status=read_failed | error={e}")
            continue

        if len(frames) < T:
            tqdm.write(f"[Infer][GPU {infer_cfg.device}] {vid.name} 帧数不足 T={T} ({len(frames)} 帧)，跳过")
            continue

        # 滑动窗口（按帧）：每个窗口单独初始化 CoTracker 网格
        starts = list(range(0, len(frames) - T + 1, infer_cfg.stride))
        if not starts:
            tqdm.write(f"[Infer][GPU {infer_cfg.device}] {vid.name} 无有效窗口（长度={len(frames)}，T={T}，stride={infer_cfg.stride}），跳过")
            continue

        latent_tokens_list: List[torch.Tensor] = []  # 每窗口 (d, D)
        fsq_indices_list: List[Optional[torch.Tensor]] = []
        labels_list: List[torch.Tensor] = []  # 每窗口 (N,)
        code_ids_list: List[torch.Tensor] = []  # 每窗口 (d,)
        used_digits_merge = False

        # 窗口级进度条
        window_bar = tqdm(starts, desc=f"[GPU {infer_cfg.device}] {vid.name}", unit="win", dynamic_ncols=True, mininterval=infer_cfg.progress_mininterval, leave=False)

        batch_buf: List[torch.Tensor] = []
        batch_starts: List[int] = []
        for s in window_bar:
            try:
                _t = time.perf_counter()
                vel_pix_win = _track_window(frames, start=s, T=T, grid_size=grid_size, device=infer_cfg.device)  # (T-1,N,2)
                t_track += time.perf_counter() - _t
                tqdm.write(f"[GPU {infer_cfg.device}] Window {s}: raw max_abs={torch.abs(vel_pix_win).max().item():.4f}, mean={vel_pix_win.mean().item():.4f}, std={vel_pix_win.std().item():.4f}")
            except Exception as e:
                tqdm.write(f"[Infer][GPU {infer_cfg.device}] 窗口跟踪失败 [start={s}]: {e}")
                continue
            _t = time.perf_counter()
            window_vel = _normalize_velocities(vel_pix_win, window_size=W_dec)  # [-1,1]
            t_norm += time.perf_counter() - _t
            tqdm.write(f"[GPU {infer_cfg.device}] Window {s}: norm max_abs={torch.abs(window_vel).max().item():.4f}, mean={window_vel.mean().item():.4f}, std={window_vel.std().item():.4f}")
            batch_buf.append(window_vel)
            batch_starts.append(s)

            if len(batch_buf) >= max(1, infer_cfg.batch_size):
                _t = time.perf_counter()
                latents, fsqs, labels_batch = extract_latent_and_labels_batched(
                    model, batch_buf, debug_fsq=infer_cfg.debug_fsq, dbg_tag=f"{vid.name}:batch-start={batch_starts[0]}", amp=infer_cfg.amp
                )
                t_forward += time.perf_counter() - _t
                for latent_tok, fsq_idx, labels in zip(latents, fsqs, labels_batch):
                    latent_tokens_list.append(latent_tok)
                    fsq_indices_list.append(fsq_idx)
                    labels_list.append(labels)
                    if fsq_idx is not None and len(fsq_levels) > 0:
                        try:
                            _t_merge = time.perf_counter()
                            code_ids = _fsq_digits_to_ids(fsq_idx, fsq_levels)  # (d,)
                            xi = fsq_idx.detach().cpu() if isinstance(fsq_idx, torch.Tensor) else torch.as_tensor(fsq_idx)
                            if xi.dim() == 2 and (xi.shape[-1] == len(fsq_levels) or xi.shape[0] == len(fsq_levels)):
                                used_digits_merge = True
                            code_ids_list.append(code_ids.to(torch.int16))
                            t_fsq_merge += time.perf_counter() - _t_merge
                        except Exception as e:
                            tqdm.write(f"[Infer] FSQ 索引转换失败 [batch start={batch_starts[0]}]: {e}")
                batch_buf.clear()
                batch_starts.clear()
                num_windows += len(latents)

        # 处理剩余 batch
        if len(batch_buf) > 0:
            _t = time.perf_counter()
            latents, fsqs, labels_batch = extract_latent_and_labels_batched(
                model, batch_buf, debug_fsq=infer_cfg.debug_fsq, dbg_tag=f"{vid.name}:batch-start={batch_starts[0]}", amp=infer_cfg.amp
            )
            t_forward += time.perf_counter() - _t
            for latent_tok, fsq_idx, labels in zip(latents, fsqs, labels_batch):
                latent_tokens_list.append(latent_tok)
                fsq_indices_list.append(fsq_idx)
                labels_list.append(labels)
                if fsq_idx is not None and len(fsq_levels) > 0:
                    try:
                        _t_merge = time.perf_counter()
                        code_ids = _fsq_digits_to_ids(fsq_idx, fsq_levels)  # (d,)
                        xi = fsq_idx.detach().cpu() if isinstance(fsq_idx, torch.Tensor) else torch.as_tensor(fsq_idx)
                        if xi.dim() == 2 and (xi.shape[-1] == len(fsq_levels) or xi.shape[0] == len(fsq_levels)):
                            used_digits_merge = True
                        code_ids_list.append(code_ids.to(torch.int16))
                        t_fsq_merge += time.perf_counter() - _t_merge
                    except Exception as e:
                        tqdm.write(f"[Infer] FSQ 索引转换失败 [batch start={batch_starts[0]}]: {e}")
            num_windows += len(latents)

        # 组织与保存输出
        rel = None
        try:
            rel = vid.relative_to(infer_cfg.video_root)
        except Exception:
            # 若不在根目录下，兜底使用文件名
            rel = Path(vid.name)

        # 分离输出子目录
        pt_out_path = run_output_dir / 'pt' / rel.with_suffix('.pt')
        json_out_path = run_output_dir / 'json' / rel.with_suffix('.json')
        pt_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.parent.mkdir(parents=True, exist_ok=True)

        save_obj: Dict = {
            'meta': {
                'video_path': str(vid),
                'input_fps': float(fps_in),
                'target_fps': infer_cfg.target_fps,
                'frames_used': len(frames),
                'T': T,
                'stride': infer_cfg.stride,
                'grid_size': grid_size,
                'num_points': N,
                'decoder_window_size': W_dec,
                'encoder_sequence_len': d,
                'num_classes': num_classes,
                'checkpoint': str(ckpt_path),
                'model_config': str(infer_cfg.model_config),
                'codebook_levels': fsq_levels,
                'codebook_size': int(codebook_size),
                'output_run_dir': str(run_output_dir),
            },
            'window_starts': [int(x) for x in starts],
            'labels_per_point': [x.to(torch.int16) for x in labels_list],  # list of (N,)
            'latent_tokens': [lt.to(torch.float16) for lt in latent_tokens_list],  # list of (d, D) 半精度以节省空间
            'code_sequences': [cid for cid in code_ids_list],  # list of (d,) int16
        }

        if infer_cfg.save_codes:
            # 存储 FSQ 索引（若可用），否则忽略该字段
            if any(x is not None for x in fsq_indices_list):
                # 允许不同实现返回不同形状的 codes，这里按 list 存储，保证通用
                save_obj['fsq_indices'] = [x for x in fsq_indices_list if x is not None]
            else:
                save_obj['fsq_indices'] = []

        _t = time.perf_counter()
        torch.save(save_obj, pt_out_path)
        t_save_pt += time.perf_counter() - _t
        print(f"[Infer] PT 保存: {pt_out_path}")

        # JSON 导出 Latent Action Sequence（按导出模式）
        # - 连续向量矩阵 latent_matrix: [num_windows * d, D]（当 mode 包含 'matrix' 时）
        # - 离散码序列 code_sequences: [num_windows, d] 及其展平版 [num_windows * d]（当 mode 包含 'codes' 时）
        if infer_cfg.save_json and len(latent_tokens_list) > 0:
            try:
                _t = time.perf_counter()
                json_obj = {
                    'meta': {**save_obj['meta']},
                    'window_starts': [int(x) for x in starts],
                }

                # 导出离散 codes（按需）
                if infer_cfg.json_export_mode in ("codes", "both") and len(code_ids_list) == len(starts):
                    codes_2d = torch.stack(code_ids_list, dim=0)  # (W, d)
                    codes_list = [[int(v) for v in row.tolist()] for row in codes_2d]
                    codes_flat = [int(v) for v in codes_2d.reshape(-1).tolist()]
                    json_obj['code_sequences'] = codes_list
                    json_obj['code_sequences_flat'] = codes_flat
                    # 诊断信息
                    eff_size = int(codes_2d.max().item()) + 1 if codes_2d.numel() > 0 else 0
                    json_obj['codes_source'] = 'digits_merge' if used_digits_merge else 'fsq_single_index'
                    json_obj['effective_codebook_size'] = eff_size
                    # 控制台打印，便于快速判断是否塌陷
                    try:
                        uniq_all = int(codes_2d.unique().numel()) if codes_2d.numel() > 0 else 0
                        uniq_first = int(codes_2d[0].unique().numel()) if codes_2d.size(0) > 0 else 0
                    except Exception:
                        uniq_all, uniq_first = None, None
                    print(
                        f"[Infer][Codes] windows={int(codes_2d.size(0)) if codes_2d.dim()==2 else 0}, "
                        f"d={int(codes_2d.size(1)) if codes_2d.dim()==2 else 0}, "
                        f"effective_codebook_size={eff_size}/{int(codebook_size)}, "
                        f"uniq_all={uniq_all}, uniq_first_window={uniq_first}, "
                        f"source={json_obj['codes_source']}"
                    )
                    if eff_size < int(codebook_size):
                        json_obj['indices_warning'] = (
                            f"effective_codebook_size={eff_size} < configured codebook_size={int(codebook_size)}; "
                            "FSQ 返回的索引可能是粗粒度/子码本索引，或库版本差异导致未返回完整 digits"
                        )
                    # 附带 fsq_indices 的 dtype/shape（基于第一窗）
                    if len(fsq_indices_list) > 0 and fsq_indices_list[0] is not None:
                        fi = fsq_indices_list[0]
                        if isinstance(fi, torch.Tensor):
                            tfi = fi.detach().cpu()
                        else:
                            tfi = torch.as_tensor(fi)
                        info_shape = tuple(tfi.shape)
                        try:
                            info_min = int(tfi.min().item())
                            info_max = int(tfi.max().item())
                        except Exception:
                            info_min = None
                            info_max = None
                        json_obj['fsq_indices_info'] = {
                            'dtype': str(tfi.dtype),
                            'shape': list(info_shape),
                            'min': info_min,
                            'max': info_max,
                        }

                # 导出连续向量矩阵（按需）
                if infer_cfg.json_export_mode in ("matrix", "both"):
                    lt_stack = torch.stack(latent_tokens_list, dim=0).to(torch.float32)  # (W, d, D)
                    lt_np = lt_stack.cpu().numpy()
                    if infer_cfg.json_round_decimals is not None and infer_cfg.json_round_decimals >= 0:
                        lt_np = lt_np.round(decimals=int(infer_cfg.json_round_decimals))
                    W, d_tokens, D_dim = lt_np.shape
                    lt2d = lt_np.reshape(W * d_tokens, D_dim)
                    lt2d_list = lt2d.tolist()
                    json_obj['latent_matrix_shape'] = [int(W * d_tokens), int(D_dim)]
                    json_obj['latent_matrix'] = lt2d_list

                with open(json_out_path, 'w', encoding='utf-8') as f:
                    json.dump(json_obj, f, ensure_ascii=False, indent=2)
                t_save_json += time.perf_counter() - _t
                print(f"[Infer] JSON 保存: {json_out_path}")
            except Exception as e:
                print(f"[Infer] JSON 导出失败: {e}")

        # 追加写入聚合 JSONL：每行一个视频片段对象 {label, codes}
        if infer_cfg.save_jsonl and len(code_ids_list) == len(starts):
            try:
                _t = time.perf_counter()
                jsonl_dir = run_output_dir / 'json'
                jsonl_path = jsonl_dir / jsonl_filename
                jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                codes_2d = torch.stack(code_ids_list, dim=0)  # (W, d)
                codes_list = [[int(v) for v in row.tolist()] for row in codes_2d]
                # 解析标签：取视频相对路径的第一层目录名；若以 'videos_' 开头则剥去前缀
                try:
                    label_name = rel.parts[0] if len(rel.parts) > 0 else 'unknown'
                except Exception:
                    label_name = 'unknown'
                if isinstance(label_name, str) and label_name.startswith('videos_'):
                    label_name = label_name[len('videos_'):]
                rec = {"label": str(label_name), "codes": codes_list}
                with open(jsonl_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False))
                    f.write("\n")
                t_save_jsonl += time.perf_counter() - _t
            except Exception as e:
                print(f"[Infer] JSONL 追加失败: {e}")
        total_files += 1

        # 每视频汇总日志
        per_video_elapsed = per_video_timer.stop()
        logger.info(
            (
                f"VIDEO {vid} | status=done | windows={num_windows} | fps_in={fps_in:.3f} | "
                f"read={t_read:.4f}s | track={t_track:.4f}s | norm={t_norm:.4f}s | "
                f"forward={t_forward:.4f}s | fsq_merge={t_fsq_merge:.4f}s | "
                f"save_pt={t_save_pt:.4f}s | save_json={t_save_json:.4f}s | save_jsonl={t_save_jsonl:.4f}s | "
                f"total={per_video_elapsed:.4f}s"
            )
        )

    total_elapsed = total_timer.stop()
    logger.info(f"PROCESS SUMMARY | device={infer_cfg.device} | videos={total_files} | total_time={total_elapsed:.4f}s")
    print(f"[Infer][GPU {infer_cfg.device}] 完成。共处理 {total_files} 个视频片段。")
    return total_files


def _mp_worker(rank: int, gpu_id: int, videos_subset: List[str], infer_cfg: InferCfg, model_cfg: Dict, run_output_dir: str, ret_queue):
    try:
        # 准备每进程的设备配置
        local_cfg = replace(infer_cfg, device=torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu"))
        vids = [Path(v) for v in videos_subset]
        count = _process_videos(vids, local_cfg, model_cfg, Path(run_output_dir), rank=rank, world_size=max(1, torch.cuda.device_count()))
        ret_queue.put(count)
    except Exception as e:
        tqdm.write(f"[Worker-{rank}] error: {e}")
        ret_queue.put(0)


def run_inference(args):
    # 1) 加载推理与模型配置
    infer_cfg = load_infer_cfg(Path(args.config))
    # CLI 覆盖 device/批量/AMP/GPU（若提供）
    if hasattr(args, "device") and args.device:
        infer_cfg.device = get_device(args.device)
    if hasattr(args, "batch_size") and args.batch_size:
        infer_cfg.batch_size = int(args.batch_size)
    if hasattr(args, "amp") and args.amp:
        infer_cfg.amp = True
    if hasattr(args, "gpu_ids") and args.gpu_ids:
        infer_cfg.gpu_ids = [int(x) for x in str(args.gpu_ids).split(',') if str(x).strip().isdigit()]
    if hasattr(args, "progress_mininterval") and args.progress_mininterval is not None:
        infer_cfg.progress_mininterval = float(args.progress_mininterval)

    model_cfg = load_config(str(infer_cfg.model_config))

    # 为本次推理创建基于当前时间戳的结果子目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = infer_cfg.output_dir / f"result_{ts}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Infer] 输出将写入目录: {run_output_dir}")

    # 3) 收集视频文件
    videos = _gather_videos(infer_cfg.video_root, infer_cfg.video_exts)
    if infer_cfg.max_files > 0:
        videos = videos[: infer_cfg.max_files]
    if not videos:
        print(f"[Infer] 未在 {infer_cfg.video_root} 下找到视频：{infer_cfg.video_exts}")
        return 0

    # 多 GPU 分配
    gpu_ids = None
    if infer_cfg.gpu_ids is not None and len(infer_cfg.gpu_ids) > 0:
        gpu_ids = infer_cfg.gpu_ids
    elif infer_cfg.device.type == 'cuda':
        try:
            n = torch.cuda.device_count()
            if n > 1:
                gpu_ids = list(range(n))
        except Exception:
            gpu_ids = None

    if not gpu_ids or len(gpu_ids) <= 1:
        # 单 GPU 或 CPU 路径
        return _process_videos(videos, infer_cfg, model_cfg, run_output_dir, rank=0, world_size=1)

    # 多 GPU：按轮询切分视频
    shards: List[List[str]] = [[] for _ in range(len(gpu_ids))]
    for i, v in enumerate(videos):
        shards[i % len(gpu_ids)].append(str(v))
    print(f"[Infer] 启动多 GPU 进程: {gpu_ids}; 每个进程分配视频数={[len(s) for s in shards]}")

    ret_queue = mp.Queue()
    procs: List[mp.Process] = []
    for rank, gid in enumerate(gpu_ids):
        p = mp.Process(target=_mp_worker, args=(rank, gid, shards[rank], infer_cfg, model_cfg, str(run_output_dir), ret_queue), daemon=False)
        p.start()
        procs.append(p)

    total = 0
    for _ in procs:
        try:
            total += int(ret_queue.get())
        except Exception:
            pass
    for p in procs:
        p.join()

    # 合并 JSONL 文件
    try:
        json_dir = run_output_dir / 'json'
        base, ext = os.path.splitext(infer_cfg.jsonl_filename)
        merged_path = json_dir / infer_cfg.jsonl_filename
        with open(merged_path, 'w', encoding='utf-8') as fout:
            for gid in gpu_ids:
                part_path = json_dir / f"{base}_gpu{gid}{ext or '.jsonl'}"
                if part_path.exists():
                    with open(part_path, 'r', encoding='utf-8') as fin:
                        for line in fin:
                            fout.write(line)
        print(f"[Infer] 合并 JSONL 完成: {merged_path}")
    except Exception as e:
        print(f"[Infer] 合并 JSONL 失败: {e}")

    print(f"[Infer] 完成。共处理 {total} 个视频片段。")
    return total


def build_argparser():
    p = argparse.ArgumentParser(description="Infer latent action sequences on short clips using Motion Tokenizer")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[0] / "configs" / "inference_config.yaml"),
        help="Inference yaml 配置路径",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="覆盖配置中的设备选择：auto/cuda/cpu",
    )
    p.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="逗号分隔的 GPU ID 列表，例如 '0,1,2'。留空则自动使用全部或单卡。",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="窗口级批大小（仅作用于 MotionTokenizer 前向；跟踪仍逐窗）。",
    )
    p.add_argument(
        "--amp",
        action='store_true',
        help="启用自动混合精度 (AMP) 以加速推理（CUDA 有效）。",
    )
    p.add_argument(
        "--progress-mininterval",
        type=float,
        default=None,
        help="tqdm 刷新最小时间间隔（秒）。",
    )
    return p


if __name__ == "__main__":
    # 多进程启动方式（Linux/CUDA 建议 spawn）
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    args = build_argparser().parse_args()
    run_inference(args)
