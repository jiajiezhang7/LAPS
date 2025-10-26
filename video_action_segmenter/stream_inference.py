import argparse
import time
from pathlib import Path
from typing import Deque, List, Optional
from collections import deque

import cv2
import numpy as np
import torch
import os
import json
import threading
import sys
import subprocess
from queue import Queue, Empty

from omegaconf import OmegaConf
import yaml
from amplify.models.motion_tokenizer import MotionTokenizer
from amplify.utils.cfg_utils import get_device
from amplify.utils.train import unwrap_compiled_state_dict
from video_action_segmenter.stream_utils import (
    TimeResampler,
    resize_shorter_keep_aspect,
    track_window_with_online,
    draw_energy_plot,
    draw_energy_plot_two,
    draw_energy_plot_enhanced,
    draw_energy_plot_enhanced_dual,
    compute_energy,
    append_codes_jsonl,
    append_energy_jsonl,
    export_prequant_npy,
    CausalSmoother1D,
    # 新增模块化工具
    pre_gate_check,
    motion_gate_check,
    compute_per_video_energy_jsonl_path,
    should_skip_video_outputs,
    ensure_output_dirs,
    cleanup_segment_and_codes,
    should_save_codes,
    export_codes_for_segment,
    run_batch_over_folder,
)


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def _normalize_velocities(vel_pix, window_size):
    """Normalize velocities to [-1, 1] range based on window size"""
    # vel_pix: (T-1, N, 2) on CPU
    return vel_pix / (window_size / 2.0)

def build_argparser():
    p = argparse.ArgumentParser(description="Real-time streaming inference for Motion Tokenizer (20Hz resample, T=16, stride=4)")
    p.add_argument("--params", type=str, default=str(Path(__file__).resolve().with_name("params.yaml")), help="YAML 配置文件路径")
    p.add_argument("--device", type=str, default=None, help="覆盖配置中的设备选择：auto/cuda/cpu")
    p.add_argument("--gpu-id", type=int, default=None, help="当 device=cuda 时，强制使用的 GPU ID")
    return p


def run_streaming(args):
    cfg = load_config(args.params)

    # 设备
    device_str = args.device or cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    if device.type == "cuda" and args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
        try:
            torch.cuda.set_device(device.index)
        except Exception:
            pass

    # 模型加载 - 新的 amplify 框架
    checkpoint_path = Path(cfg["checkpoint_path"]).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到权重文件: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = OmegaConf.create(checkpoint['config'])
    
    # Handle compiled models
    if model_cfg.get('compile', False) or "_orig_mod." in str(list(checkpoint['model'].keys())[0]):
        checkpoint['model'] = unwrap_compiled_state_dict(checkpoint['model'])
    
    # Create model
    model = MotionTokenizer(model_cfg, load_encoder=True, load_decoder=True).to(device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    
    print(f"[Model] Loaded checkpoint from: {checkpoint_path}")
    print(f"[Model] Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # 关键参数 - 从新配置结构中提取
    T = int(model_cfg.track_pred_horizon) - 1  # -1 because velocity
    grid_size = int(np.sqrt(model_cfg.num_tracks))  # Infer from num_tracks
    N = int(model_cfg.num_tracks)
    W_dec = 480  # Default window size for normalization (can be overridden in cfg)
    if 'decoder_window_size' in cfg:
        W_dec = int(cfg['decoder_window_size'])
    
    # Get FSQ levels from codebook size
    codebook_size = int(model_cfg.codebook_size)
    power = int(np.log2(codebook_size))
    if power == 11:  # 2048 = 2^11
        fsq_levels = [8, 8, 6, 5]  # From get_fsq_level function
    elif power == 10:  # 1024
        fsq_levels = [8, 5, 5, 5]
    elif power == 12:  # 4096
        fsq_levels = [7, 5, 5, 5, 5]
    else:
        fsq_levels = []  # Will be inferred from data if needed
    
    print(f"[Model] T={T}, N={N}, grid_size={grid_size}, codebook_size={codebook_size}, fsq_levels={fsq_levels}")

    target_fps = int(cfg.get("target_fps", 20))
    resize_shorter = int(cfg.get("resize_shorter", 480))
    stride = int(cfg.get("stride", 4))
    # 限制最大窗口数（可选）。<=0 或未配置表示无限制。
    _mw = cfg.get("max_windows", None)
    try:
        max_windows = int(_mw) if _mw is not None else None
    except Exception:
        max_windows = None
    if isinstance(max_windows, int) and max_windows <= 0:
        max_windows = None
    # 计算实时预算（每窗可用时间）：stride / target_fps（秒）
    budget_per_window = (float(stride) / float(target_fps)) if target_fps > 0 else 0.0

    amp = bool(cfg.get("amp", True))
    jsonl_output = bool(cfg.get("jsonl_output", False))
    jsonl_path = Path(cfg.get("jsonl_path", "./video_action_segmenter/inference_outputs/stream_codes.jsonl")).resolve()

    # 可视化播放窗口配置
    visualize = bool(cfg.get("visualize", True))
    show_overlay = bool(cfg.get("show_overlay", True))
    window_title = str(cfg.get("window_title", "MotionTokenizer Stream"))
    display_fps = int(cfg.get("display_fps", target_fps))
    if jsonl_output:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # 能量与导出配置（均为可选，默认安全关闭或提供合理默认值）
    energy_cfg = cfg.get("energy", {}) if isinstance(cfg.get("energy", {}), dict) else {}
    energy_enable = bool(energy_cfg.get("enable", True))
    energy_source = str(energy_cfg.get("source", "prequant")).lower()  # prequant | quantized | velocity
    energy_mode = str(energy_cfg.get("mode", "l2_mean")).lower()       # l2_mean | token_diff_l2_mean
    energy_jsonl_output = bool(energy_cfg.get("jsonl_output", False))
    energy_jsonl_path = Path(energy_cfg.get("jsonl_path", "./video_action_segmenter/inference_outputs/stream_energy.jsonl")).resolve()
    if energy_jsonl_output:
        energy_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # 能量可视化配置
    energy_visualize = bool(energy_cfg.get("visualize", True))
    energy_window_title = str(energy_cfg.get("window_title", "Energy Curve"))
    energy_viz_window = int(energy_cfg.get("viz_window", 300))
    energy_y_min = energy_cfg.get("y_min", None)
    energy_y_max = energy_cfg.get("y_max", None)
    try:
        energy_y_min = float(energy_y_min) if energy_y_min is not None else None
    except Exception:
        energy_y_min = None
    try:
        energy_y_max = float(energy_y_max) if energy_y_max is not None else None
    except Exception:
        energy_y_max = None
    energy_plot_w = int(energy_cfg.get("plot_width", 600))
    energy_plot_h = int(energy_cfg.get("plot_height", 200))
    _col = energy_cfg.get("plot_color", [0, 255, 0])
    try:
        energy_color = (int(_col[0]), int(_col[1]), int(_col[2]))
    except Exception:
        energy_color = (0, 255, 0)
    # 新增：能量可视化样式与主题（兼容多种配置位置）
    _viz_style_val = energy_cfg.get("viz_style", None)
    if _viz_style_val is None:
        _viz_style_val = energy_cfg.get("energy_viz_style", None)
    if _viz_style_val is None:
        _viz_style_val = cfg.get("energy_viz_style", "basic")
    energy_viz_style = str(_viz_style_val).lower()  # basic | enhanced

    _theme_val = energy_cfg.get("theme", None)
    if _theme_val is None:
        _theme_val = energy_cfg.get("energy_theme", None)
    if _theme_val is None:
        _theme_val = cfg.get("energy_theme", "academic_blue")
    energy_theme = str(_theme_val)

    # 能量平滑配置（可选，默认关闭）
    smoothing_cfg = energy_cfg.get("smoothing", {}) if isinstance(energy_cfg.get("smoothing", {}), dict) else {}
    smoothing_enable = bool(smoothing_cfg.get("enable", False))
    smoothing_method = str(smoothing_cfg.get("method", "ema")).lower()
    try:
        smoothing_alpha = float(smoothing_cfg.get("alpha", 0.4))
    except Exception:
        smoothing_alpha = 0.4
    try:
        smoothing_window = int(smoothing_cfg.get("window", 3))
    except Exception:
        smoothing_window = 3
    smoothing_use_for_seg = bool(smoothing_cfg.get("use_for_seg", True))
    smoothing_visualize_both = bool(smoothing_cfg.get("visualize_both", True))

    # 前置像素差分快速门控（在 CoTracker 之前，极低开销）
    pre_gate_cfg = cfg.get("pre_gate", {}) if isinstance(cfg.get("pre_gate", {}), dict) else {}
    pre_gate_enable = bool(pre_gate_cfg.get("enable", True))
    try:
        pre_gate_resize_shorter = int(pre_gate_cfg.get("resize_shorter", 128))
    except Exception:
        pre_gate_resize_shorter = 128
    pre_gate_method = str(pre_gate_cfg.get("diff_method", "mad")).lower()  # 目前仅支持 mad
    try:
        pre_gate_pixel_diff_thr = float(pre_gate_cfg.get("pixel_diff_thr", 0.01))  # 灰度归一化[0,1]
    except Exception:
        pre_gate_pixel_diff_thr = 0.01
    try:
        pre_gate_mad_thr = float(pre_gate_cfg.get("mad_thr", 0.003))
    except Exception:
        pre_gate_mad_thr = 0.003
    try:
        pre_gate_min_active_ratio = float(pre_gate_cfg.get("min_active_ratio", 0.002))
    except Exception:
        pre_gate_min_active_ratio = 0.002
    pre_gate_debug = bool(pre_gate_cfg.get("debug", False))

    # 静止窗口过滤（基于 CoTracker 速度阈值）
    motion_gate_cfg = cfg.get("motion_gate", {}) if isinstance(cfg.get("motion_gate", {}), dict) else {}
    motion_gate_enable = bool(motion_gate_cfg.get("enable", True))
    try:
        motion_gate_vel_thr = float(motion_gate_cfg.get("vel_norm_thr", 0.012))
    except Exception:
        motion_gate_vel_thr = 0.012
    try:
        motion_gate_min_active_ratio = float(motion_gate_cfg.get("min_active_ratio", 0.01))
    except Exception:
        motion_gate_min_active_ratio = 0.01
    motion_gate_debug = bool(motion_gate_cfg.get("debug", False))

    # 分割与片段保存配置（默认关闭，启用时才生效）
    seg_cfg = cfg.get("segmentation", {}) if isinstance(cfg.get("segmentation", {}), dict) else {}
    seg_enable = bool(seg_cfg.get("enable", False))
    seg_mode = str(seg_cfg.get("mode", "fixed")).lower()  # fixed | report
    seg_threshold = float(seg_cfg.get("threshold", 10.625))
    seg_report_path = Path(seg_cfg.get("report_path", "./video_action_segmenter/energy_sweep_report/best_threshold_quantized_token_diff.json")).resolve()
    seg_report_key = str(seg_cfg.get("report_key", "quantized_token_diff_best.best_f1.thr"))
    seg_min_len_windows = int(seg_cfg.get("min_len_windows", 2))  # 片段最少包含的正窗口数
    seg_output_dir = Path(seg_cfg.get("output_dir", "./video_action_segmenter/inference_outputs/segments")).resolve()
    seg_codec = str(seg_cfg.get("codec", "mp4v"))  # OpenCV fourcc，如 mp4v, avc1
    seg_ext = str(seg_cfg.get("ext", ".mp4"))
    seg_align = str(seg_cfg.get("align", "center")).lower()  # start | center | end
    # 新增：若输出目录下已存在以视频名命名的专属文件夹且包含片段/索引，则跳过该视频
    seg_skip_if_exists = bool(seg_cfg.get("skip_if_exists", True))

    # 分割护栏参数
    seg_hysteresis_ratio = float(seg_cfg.get("hysteresis_ratio", 0.95))  # thr_off = thr_on * ratio
    seg_up_count = int(seg_cfg.get("up_count", 2))        # 连续 >=thr_on 才启动
    seg_down_count = int(seg_cfg.get("down_count", 2))    # 连续 <thr_off 才结束
    seg_cooldown_windows = int(seg_cfg.get("cooldown_windows", 1))  # 结束后冷却窗口数
    # 最大时长限制：优先秒，次选窗口
    seg_max_duration_seconds = float(seg_cfg.get("max_duration_seconds", 2.0))
    seg_max_duration_windows_cfg = int(seg_cfg.get("max_duration_windows", 0))
    # 片段 codes 导出配置：使用在线流推的每窗 codes，按与片段重叠比例筛选整窗并拼接导出
    seg_export_codes = bool(seg_cfg.get("export_codes", True))
    seg_codes_min_overlap_ratio = float(seg_cfg.get("codes_min_overlap_ratio", 0.25))
    # 新增：是否允许保存与片段重叠的所有候选窗口（不去重叠）
    seg_allow_overlap = bool(seg_cfg.get("allow_overlap", False))
    # 未开启分割时无需缓存大量窗口；开启时保留小历史窗口用于触发稳定判定
    seg_history_keep = max(4, int(seg_up_count) + int(seg_cooldown_windows) + 1)

    # 可选：显存调试与碎片缓解（默认关闭）
    debug_mem_cfg = cfg.get("debug_memory", {}) if isinstance(cfg.get("debug_memory", {}), dict) else {}
    debug_mem_enable = bool(debug_mem_cfg.get("enable", False))
    try:
        debug_mem_interval = int(debug_mem_cfg.get("interval_windows", 50))
    except Exception:
        debug_mem_interval = 50
    debug_mem_empty_cache = bool(debug_mem_cfg.get("empty_cache", False))
    try:
        debug_mem_empty_interval = int(debug_mem_cfg.get("empty_cache_interval", 0))  # 0 代表关闭
    except Exception:
        debug_mem_empty_interval = 0

    # 如指定从报告文件读取阈值，则覆盖 seg_threshold
    if seg_enable and seg_mode in ("report", "from_report", "best_from_report"):
        try:
            with open(seg_report_path, "r", encoding="utf-8") as f:
                rep = json.load(f)
            cur = rep
            for k in seg_report_key.split('.'):
                if isinstance(cur, dict) and (k in cur):
                    cur = cur[k]
                else:
                    cur = None
                    break
            if isinstance(cur, (int, float)):
                seg_threshold = float(cur)
                print(f"[Seg] Loaded threshold {seg_threshold} from report: {seg_report_path} key={seg_report_key}")
            else:
                print(f"[Seg][WARN] Could not read threshold from report key={seg_report_key}, use fallback threshold={seg_threshold}")
        except Exception as e:
            print(f"[Seg][WARN] Failed to load report {seg_report_path}: {e}. Use fallback threshold={seg_threshold}")

    export_cfg = cfg.get("export", {}) if isinstance(cfg.get("export", {}), dict) else {}
    export_prequant = bool(export_cfg.get("prequant", False))
    prequant_dir = Path(export_cfg.get("prequant_dir", "./video_action_segmenter/inference_outputs/stream_prequant")).resolve()
    if export_prequant:
        prequant_dir.mkdir(parents=True, exist_ok=True)

    # 输入源
    input_cfg = cfg.get("input", {})
    # 当输入为单视频文件时，加入 EOF 超时检测，避免无穷等待
    try:
        eof_timeout_seconds = float(input_cfg.get("eof_timeout_seconds", 1.0))
    except Exception:
        eof_timeout_seconds = 1.0
    src_type = str(input_cfg.get("type", "camera")).lower()  # camera | file | rtsp | folder
    # 子进程批处理覆盖：若设置了 MT_OVERRIDE_INPUT_PATH，则无论原配置为何，强制按 file 模式处理单个视频
    _override_path_env = os.environ.get("MT_OVERRIDE_INPUT_PATH", "").strip()
    if _override_path_env:
        src_type = "file"
    cap: Optional[cv2.VideoCapture] = None
    video_name = "unknown"  # 默认视频名称
    
    if src_type == "camera":
        cam_index = int(input_cfg.get("camera_index", 0))
        cap = cv2.VideoCapture(cam_index)
        video_name = f"camera_{cam_index}"
    elif src_type == "file":
        # 支持通过环境变量覆盖单视频路径（用于 folder 批处理子进程传参）
        path_env = os.environ.get("MT_OVERRIDE_INPUT_PATH", "").strip()
        path_cfg = str(input_cfg.get("path", "")).strip()
        path = path_env or path_cfg
        cap = cv2.VideoCapture(path)
        # 从文件路径提取视频名称（不含扩展名）
        if path:
            video_name = Path(path).stem
        else:
            video_name = "unknown_file"
        # 当给定路径为目录时，切换到 folder 批处理模式
        if path and os.path.isdir(path):
            src_type = "folder"
            cap.release()
            cap = None
    elif src_type == "rtsp":
        url = str(input_cfg.get("url", ""))
        cap = cv2.VideoCapture(url)
        # 从RTSP URL提取名称或使用默认名称
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
        # 在下方批处理分支中处理，这里仅占位避免被判为不支持
        cap = None
        video_name = "batch_folder"
    else:
        raise ValueError(f"不支持的输入类型: {src_type}")

    # 目录批处理模式：依次对文件夹中的所有视频运行当前脚本（子进程方式），通过环境变量传递路径
    if src_type == "folder":
        in_dir = str(input_cfg.get("dir", "")).strip() or (path if 'path' in locals() else "")
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
            args.params,
            exts=exts,
            recursive=recursive,
            enable_parallel=enable_parallel,
            gpu_ids=gpu_ids,
            max_procs_per_gpu=max_procs_per_gpu,
            poll_interval=poll_interval,
        )
        return

    # 若为单文件模式且启用跳过逻辑：检查目标输出是否已有该视频名的结果，若有则直接跳过
    if seg_enable and seg_skip_if_exists and src_type == "file":
        try:
            if should_skip_video_outputs(seg_output_dir, video_name, seg_ext):
                print(f"[Skip] Detected existing outputs for video '{video_name}' under: {Path(seg_output_dir) / video_name}. Skip processing.")
                return
        except Exception:
            # 出现异常则不做跳过，继续正常处理
            pass

    # 动态设置每视频一份的能量 JSONL 输出路径（基于 video_name 的子目录）
    if energy_jsonl_output:
        try:
            energy_jsonl_path = compute_per_video_energy_jsonl_path(
                base_path=energy_jsonl_path,
                video_name=video_name,
                energy_source=str(energy_source),
                energy_mode=str(energy_mode),
                seg_enable=seg_enable,
                seg_output_dir=seg_output_dir,
            )
            print(f"[Energy] JSONL per-video path: {energy_jsonl_path}")
        except Exception as e:
            print(f"[Energy][WARN] Failed to set per-video JSONL path, use original: {energy_jsonl_path} | {e}")

    if not cap or not cap.isOpened():
        raise RuntimeError("无法打开输入视频流")

    print(f"[Stream] device={device}, target_fps={target_fps}, T={T}, stride={stride}, resize_shorter={resize_shorter}, grid_size={grid_size}, max_windows={max_windows if max_windows is not None else 'unlimited'}")
    print(f"[Budget] per-window budget = {budget_per_window:.3f}s (stride={stride}, target_fps={target_fps})")
    print(f"[Video] Processing video: {video_name} (type: {src_type})")

    # 初始化可视化窗口
    window_ok = False
    energy_window_ok = False
    if visualize:
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            window_ok = True
        except Exception as e:
            print(f"[Stream][WARN] 无法创建可视化窗口: {e}. 将在无窗口模式下运行。")
            window_ok = False
        # 能量窗口（可选）
        if energy_visualize:
            try:
                cv2.namedWindow(energy_window_title, cv2.WINDOW_NORMAL)
                energy_window_ok = True
            except Exception as e:
                print(f"[Stream][WARN] 无法创建能量窗口: {e}.")
                energy_window_ok = False

    # 后台计算线程：解耦显示/采样 与 重计算
    job_q: Queue = Queue(maxsize=1)        # 仅保留最新待处理窗口，防止积压
    result_q: Queue = Queue(maxsize=16)    # 结果缓冲
    stop_event = threading.Event()

    def compute_worker():
        # 可选：设置 CUDA 设备，防止多线程环境下设备不一致
        try:
            if device.type == 'cuda' and device.index is not None:
                torch.cuda.set_device(device.index)
        except Exception:
            pass

        while not stop_event.is_set():
            try:
                job = job_q.get(timeout=0.1)
            except Empty:
                continue
            if job is None:
                break
            win_idx: int = job["win_idx"]
            frames: List[np.ndarray] = job["frames"]

            # 跟踪 + 前向
            try:
                # 1) 前置像素差分 门控（在 CoTracker 前）
                if pre_gate_enable:
                    dropped, mad_val, act_ratio = pre_gate_check(
                        frames,
                        resize_shorter=pre_gate_resize_shorter,
                        pixel_diff_thr=pre_gate_pixel_diff_thr,
                        mad_thr=pre_gate_mad_thr,
                        min_active_ratio=pre_gate_min_active_ratio,
                        method=pre_gate_method,
                        debug=pre_gate_debug,
                    )
                    if dropped:
                        if pre_gate_debug:
                            print(f"[PreGate] DROP win#{win_idx}: mad={mad_val:.6f}(<{pre_gate_mad_thr}) ratio={act_ratio:.4f}(<{pre_gate_min_active_ratio})")
                        energy_val_pg = 0.0 if energy_enable else None
                        result = {
                            "win_idx": win_idx,
                            "t_track": 0.0,
                            "t_forward": 0.0,
                            "d": 0,
                            "D": 0,
                            "codes": None,
                            "used_digits_merge": False,
                            "energy": energy_val_pg if energy_val_pg is not None else None,
                            "energy_source": str(energy_source),
                            "energy_mode": energy_mode,
                            "prequant_path": None,
                            "pre_gate": {
                                "dropped": True,
                                "mad": mad_val,
                                "active_ratio": act_ratio,
                                "mad_thr": pre_gate_mad_thr,
                                "pixel_diff_thr": pre_gate_pixel_diff_thr,
                                "min_active_ratio": pre_gate_min_active_ratio,
                            },
                        }
                        try:
                            result_q.put_nowait(result)
                        except Exception:
                            try:
                                _ = result_q.get_nowait()
                            except Empty:
                                pass
                            try:
                                result_q.put_nowait(result)
                            except Exception:
                                pass
                        continue

                # 2) CoTracker 跟踪
                t0 = time.perf_counter()
                vel_pix = track_window_with_online(frames, grid_size=grid_size, device=device)  # (T-1,N,2) on CPU
                t_track = time.perf_counter() - t0

                t1 = time.perf_counter()
                vel_norm = _normalize_velocities(vel_pix, window_size=W_dec)
                # 速度阈值门控：检测“全静止”窗口，跳过后续前向以节省计算
                if motion_gate_enable:
                    dropped, v_l2_mean, active_ratio = motion_gate_check(
                        vel_norm,
                        vel_norm_thr=float(motion_gate_vel_thr),
                        min_active_ratio=float(motion_gate_min_active_ratio),
                        debug=motion_gate_debug,
                    )
                    if dropped:
                        if motion_gate_debug:
                            print(f"[Gate] DROP win#{win_idx}: mean={v_l2_mean:.6f}(<{motion_gate_vel_thr}) ratio={active_ratio:.4f}(<{motion_gate_min_active_ratio})")
                        energy_val_gate: Optional[float] = None
                        used_energy_source = str(energy_source)
                        if energy_enable and str(energy_source).lower() == "velocity":
                            try:
                                energy_val_gate = compute_energy(
                                    source="velocity",
                                    mode=energy_mode,
                                    to_quantize=None,
                                    quantized=None,
                                    vel_norm=vel_norm,
                                )
                            except Exception:
                                energy_val_gate = None
                            else:
                                used_energy_source = "velocity"
                        if energy_val_gate is None:
                            energy_val_gate = 0.0

                        result = {
                            "win_idx": win_idx,
                            "t_track": t_track,
                            "t_forward": 0.0,
                            "d": 0,
                            "D": 0,
                            "codes": None,
                            "used_digits_merge": False,
                            "energy": energy_val_gate,
                            "energy_source": used_energy_source,
                            "energy_mode": energy_mode,
                            "prequant_path": None,
                            "motion_gate": {
                                "dropped": True,
                                "l2_mean": v_l2_mean,
                                "active_ratio": active_ratio,
                                "vel_norm_thr": motion_gate_vel_thr,
                                "min_active_ratio": motion_gate_min_active_ratio,
                            },
                        }
                        try:
                            result_q.put_nowait(result)
                        except Exception:
                            try:
                                _ = result_q.get_nowait()
                            except Empty:
                                pass
                            try:
                                result_q.put_nowait(result)
                            except Exception:
                                pass
                        continue  # 跳过后续模型前向

                # 使用新模型的 encode/quantize API
                # 新模型期望输入: (B, V, T, N, D) 其中 V=views, T=时间步, N=点数, D=2(xy坐标)
                # vel_norm: (T-1, N, 2) -> reshape to (1, 1, T-1, N, 2)
                Tm1, N_local, _ = vel_norm.shape
                x_input = vel_norm.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T-1, N, 2)

                # AMP autocast（仅 CUDA 且 amp=True 有效）
                try:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=amp and device.type == 'cuda')
                except Exception:
                    class _Dummy:
                        def __enter__(self):
                            return None
                        def __exit__(self, exc_type, exc, tb):
                            return False
                    autocast_ctx = _Dummy()

                # 推理模式可进一步减少版本计数与内存占用
                with torch.inference_mode():
                    with autocast_ctx:
                        # Encode to latent space (before quantization)
                        to_quantize = model.encode(x_input, cond=None)  # (B, seq_len, hidden_dim)
                        
                        # Quantize using FSQ
                        quantized, fsq_indices = model.quantize(to_quantize)  # FSQ returns (quantized, indices)

                t_forward = time.perf_counter() - t1

                # 计算 codes（如可用）
                # 新的 FSQ 直接返回合并后的 code IDs: (B, seq_len)
                code_ids_seq_list = None
                used_digits_merge = False
                if isinstance(fsq_indices, torch.Tensor):
                    try:
                        # fsq_indices: (1, seq_len) -> flatten to (seq_len,)
                        code_ids_seq = fsq_indices.squeeze(0).to(torch.int16)  # (seq_len,)
                        code_ids_seq_list = [int(v) for v in code_ids_seq.tolist()]
                        used_digits_merge = True  # FSQ already merged digits internally
                    except Exception as e:
                        print(f"[Stream][WARN] FSQ code conversion failed: {e}")
                        code_ids_seq_list = None
                
                # 在 GPU 上的中间张量：显式转移到 CPU，尽快释放显存
                to_quantize_cpu = to_quantize.detach().cpu() if isinstance(to_quantize, torch.Tensor) else None
                quantized_cpu = quantized.detach().cpu() if isinstance(quantized, torch.Tensor) else None
                
                # 显式删除大中间变量，减少 CUDA caching allocator 的峰值
                try:
                    del x_input, to_quantize, quantized, fsq_indices
                except Exception:
                    pass

                # 计算能量（可选）
                energy_val: Optional[float] = None
                if energy_enable:
                    energy_val = compute_energy(
                        source=energy_source,
                        mode=energy_mode,
                        to_quantize=to_quantize_cpu,
                        quantized=quantized_cpu,
                        vel_norm=vel_norm,
                    )

                # 可选导出未量化 latent（to_quantize）
                prequant_path = export_prequant_npy(prequant_dir, win_idx, to_quantize_cpu) if export_prequant and to_quantize_cpu is not None else None

                # 维度从 CPU 张量读取（若存在）
                if to_quantize_cpu is not None:
                    d = int(to_quantize_cpu.shape[1])
                    D_dim = int(to_quantize_cpu.shape[2])
                else:
                    d = 0
                    D_dim = 0

                result = {
                    "win_idx": win_idx,
                    "t_track": t_track,
                    "t_forward": t_forward,
                    "d": d,
                    "D": D_dim,
                    "codes": code_ids_seq_list,
                    "used_digits_merge": used_digits_merge,
                    "energy": energy_val,
                    "energy_source": energy_source,
                    "energy_mode": energy_mode,
                    "prequant_path": prequant_path,
                }
                try:
                    result_q.put_nowait(result)
                except Exception:
                    # 若结果队列满，丢弃最旧结果以保证最新状态
                    try:
                        _ = result_q.get_nowait()
                    except Empty:
                        pass
                    try:
                        result_q.put_nowait(result)
                    except Exception:
                        pass
            except Exception as e:
                # 将异常作为一条结果返回，便于主线程打印
                err = {"win_idx": win_idx, "error": str(e)}
                try:
                    result_q.put_nowait(err)
                except Exception:
                    pass
                # 异常情况下的清理，尽量丢弃潜在的 GPU/CPU 大对象引用
                for _nm in ("first_frame_clip", "video", "pred_tracks", "pred_visibility", "clip", "x", "encoded", "proj_in", "proj_out", "to_quantize", "quantized", "fsq_indices"):
                    try:
                        if _nm in locals():
                            del locals()[_nm]
                    except Exception:
                        pass

    worker = threading.Thread(target=compute_worker, daemon=True)
    worker.start()

    resampler = TimeResampler(target_fps)
    ring: Deque[np.ndarray] = deque(maxlen=T)
    energy_ring: Deque[float] = deque(maxlen=energy_viz_window)
    energy_smooth_ring: Deque[float] = deque(maxlen=energy_viz_window)
    smoother = CausalSmoother1D(method=smoothing_method, alpha=smoothing_alpha, window=smoothing_window) if smoothing_enable else None

    # 窗口帧缓存：按 win_idx 保存 T 帧，用于阈值分割写出
    window_frames_store = {}
    # 每窗 codes 与帧范围缓存（用于片段 codes 导出）
    window_codes_store = {}
    window_frame_range_store = {}

    # 分割状态
    seg_active = False
    seg_writer = None
    seg_start_win = -1
    seg_last_pos_win = -1
    seg_written_windows = 0
    seg_segment_count = 0
    seg_current_path = None
    seg_fourcc = cv2.VideoWriter_fourcc(*seg_codec) if seg_enable else None
    # 护栏状态
    seg_thr_on = float(seg_threshold)
    seg_thr_off = float(seg_threshold) * float(seg_hysteresis_ratio)
    pos_run = 0
    neg_run = 0
    cooldown_left = 0
    seg_len_windows = 0
    # 片段帧级区间（用于基于重叠比例筛选整窗 codes）
    seg_start_frame = -1
    seg_end_frame = -1
    # 计算窗口级最大时长
    windows_per_second = max(1.0, float(target_fps) / max(1, int(stride)))
    if seg_max_duration_windows_cfg and seg_max_duration_windows_cfg > 0:
        seg_max_duration_windows = int(seg_max_duration_windows_cfg)
    else:
        seg_max_duration_windows = int(max(1, round(float(seg_max_duration_seconds) * windows_per_second))) if seg_max_duration_seconds and seg_max_duration_seconds > 0 else 0
    if seg_enable:
        seg_videos_dir, seg_codes_dir = ensure_output_dirs(seg_output_dir, video_name)

    frames_emitted = 0  # 重采样后累计的帧数
    windows_done = 0     # 已完成的窗口（结果已收到）
    windows_enqueued = 0 # 已投递到后台的窗口数

    # 计时统计（滑动平均）
    avg_track = 0.0
    avg_forward = 0.0
    avg_total = 0.0
    alpha = 0.1  # EMA 系数
    
    reached_max_windows = False
    # EOF 侦测：记录最后一次成功读取帧的时间与连续失败次数
    last_ok_read_ts = time.perf_counter()
    consecutive_failed_reads = 0

    try:
        while True:
            # 时间重采样到 target_fps：未到发帧时间不读取帧，避免提前消费导致片段“加速”
            now_ts = time.perf_counter()
            if not resampler.should_emit(now_ts):
                # 在等待阶段也泵一下窗口事件，避免 GUI 被窗口管理器判定为未响应
                try:
                    if window_ok or energy_window_ok:
                        cv2.waitKey(1)
                except Exception:
                    window_ok = False
                    energy_window_ok = False
                if not visualize:
                    # 避免空转占满 CPU，略作让步
                    time.sleep(0.001)
                continue

            ok, frame = cap.read()
            if not ok:
                # 读取失败：针对文件输入，若长时间无新帧则判定为 EOF，优雅退出
                consecutive_failed_reads += 1
                now_read = time.perf_counter()
                if src_type == "file" and (now_read - last_ok_read_ts) >= float(eof_timeout_seconds):
                    print(f"[Stream] Detected end of video '{video_name}' (no frames for {now_read - last_ok_read_ts:.2f}s). Finishing...")
                    break
                time.sleep(0.005)
                continue
            else:
                last_ok_read_ts = time.perf_counter()
                consecutive_failed_reads = 0

            frame = resize_shorter_keep_aspect(frame, resize_shorter)
            ring.append(frame)
            frames_emitted += 1

            # 可视化：每次发出采样帧时进行播放
            if window_ok:
                try:
                    disp = frame.copy()
                    if show_overlay:
                        # 左上角叠加基本状态
                        y0, dy = 24, 20
                        cv2.putText(disp, f"T={T} stride={stride} grid={grid_size}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
                        cv2.putText(disp, f"win={windows_done} track(EMA)={avg_track:.3f}s fwd(EMA)={avg_forward:.3f}s", (10, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1, cv2.LINE_AA)
                        if seg_enable:
                            seg_color = (0, 0, 255) if seg_active else (180, 180, 180)
                            cv2.putText(disp, f"SEG={'ON' if seg_active else 'OFF'} thr={seg_threshold:.3f}", (10, y0+2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, seg_color, 1, cv2.LINE_AA)
                    cv2.imshow(window_title, disp)
                    # 更新能量曲线窗口
                    if energy_window_ok:
                        if energy_viz_style == "enhanced":
                            if smoothing_enable and smoothing_visualize_both and (len(energy_ring) > 0 or len(energy_smooth_ring) > 0):
                                energy_img = draw_energy_plot_enhanced_dual(
                                    raw_values=list(energy_ring),
                                    smooth_values=list(energy_smooth_ring),
                                    width=energy_plot_w, height=energy_plot_h,
                                    y_min=energy_y_min, y_max=energy_y_max,
                                    theme=energy_theme,
                                    show_grid=True, show_labels=True, show_legend=True, show_statistics=True,
                                    title=energy_window_title,
                                )
                                cv2.imshow(energy_window_title, energy_img)
                            elif len(energy_ring) > 0:
                                values_to_plot = list(energy_smooth_ring) if smoothing_enable else list(energy_ring)
                                energy_img = draw_energy_plot_enhanced(
                                    values=values_to_plot,
                                    width=energy_plot_w, height=energy_plot_h,
                                    y_min=energy_y_min, y_max=energy_y_max,
                                    theme=energy_theme,
                                    show_grid=True, show_labels=True, show_statistics=True,
                                    title=energy_window_title,
                                )
                                cv2.imshow(energy_window_title, energy_img)
                        else:
                            if smoothing_enable and smoothing_visualize_both and (len(energy_ring) > 0 or len(energy_smooth_ring) > 0):
                                energy_img = draw_energy_plot_two(
                                    list(energy_ring), list(energy_smooth_ring),
                                    width=energy_plot_w, height=energy_plot_h,
                                    y_min=energy_y_min, y_max=energy_y_max,
                                    color_raw=energy_color, color_smooth=(0, 165, 255)
                                )
                                cv2.imshow(energy_window_title, energy_img)
                            elif len(energy_ring) > 0:
                                values_to_plot = list(energy_smooth_ring) if smoothing_enable else list(energy_ring)
                                energy_img = draw_energy_plot(values_to_plot, width=energy_plot_w, height=energy_plot_h,
                                                              y_min=energy_y_min, y_max=energy_y_max, color=energy_color)
                                cv2.imshow(energy_window_title, energy_img)
                    # 轻量 waitKey 以刷新窗口并响应按键
                    key = cv2.waitKey(max(1, int(1000/max(1, display_fps)))) & 0xFF
                    if key == ord('q'):
                        break
                except Exception:
                    # 若图形后端异常，自动降级为无窗口模式
                    window_ok = False
                    energy_window_ok = False

            # 尝试从后台取回结果并打印
            while True:
                try:
                    res = result_q.get_nowait()
                except Empty:
                    break
                if res is None:
                    continue
                if "error" in res:
                    print(f"[Stream][ERR] win#{res.get('win_idx', -1)}: {res['error']}")
                    continue

                t_track = float(res["t_track"])
                t_forward = float(res["t_forward"])
                t_total = t_track + t_forward
                avg_track = (1 - alpha) * avg_track + alpha * t_track if windows_done > 0 else t_track
                avg_forward = (1 - alpha) * avg_forward + alpha * t_forward if windows_done > 0 else t_forward
                avg_total = (1 - alpha) * avg_total + alpha * t_total if windows_done > 0 else t_total

                d, D = int(res["d"]), int(res["D"])
                codes = res.get("codes", None)
                used_digits_merge = bool(res.get("used_digits_merge", False))
                codes_info = f"codes(d={len(codes)}) source={'digits_merge' if used_digits_merge else 'fsq_single_index'}" if isinstance(codes, list) else "codes=None"
                energy_val = res.get("energy", None)
                energy_smooth_val = None
                energy_info = f"energy={energy_val:.6f} ({res.get('energy_source','-')}/{res.get('energy_mode','-')})" if isinstance(energy_val, (float, int)) else "energy=None"

                # 预算判定
                slack = budget_per_window - t_total
                status = "OK" if slack >= 0 else f"LAG(+{abs(slack):.3f}s)"

                pre_meta = res.get("pre_gate", None)
                pre_info = ""
                if isinstance(pre_meta, dict) and pre_meta.get("dropped", False):
                    mv = pre_meta.get("mad", None)
                    ar0 = pre_meta.get("active_ratio", None)
                    pre_info = f" | GATE(pre):DROP mad={mv:.6f} ratio={ar0:.4f}" if isinstance(mv, (float,int)) and isinstance(ar0, (float,int)) else " | GATE(pre):DROP"

                gate_meta = res.get("motion_gate", None)
                gate_info = ""
                if isinstance(gate_meta, dict) and gate_meta.get("dropped", False):
                    lm = gate_meta.get("l2_mean", None)
                    ar = gate_meta.get("active_ratio", None)
                    gate_info = f" | GATE(post):DROP mean={lm:.6f} ratio={ar:.4f}" if isinstance(lm, (float,int)) and isinstance(ar, (float,int)) else " | GATE(post):DROP"

                print(
                    f"[Stream] win#{windows_done} | track={t_track:.3f}s (EMA={avg_track:.3f}) | "
                    f"fwd={t_forward:.3f}s (EMA={avg_forward:.3f}) | total={t_total:.3f}s (EMA={avg_total:.3f}) | "
                    f"budget={budget_per_window:.3f}s {status}{pre_info}{gate_info} | "
                    f"latent(d,D)=({d},{D}) | {codes_info} | {energy_info}"
                )

                # 可选：写入 JSONL（流式）
                if jsonl_output and isinstance(codes, list):
                    try:
                        append_codes_jsonl(jsonl_path, window=int(windows_done), codes=[int(v) for v in codes])
                    except Exception as e:
                        print(f"[Stream][WARN] JSONL 写入失败: {e}")
                # 缓存每窗 codes（用于片段导出）
                try:
                    if seg_enable and seg_export_codes and isinstance(codes, list):
                        window_codes_store[int(windows_done)] = [int(v) for v in codes]
                except Exception:
                    pass

                # 可选：写入能量 JSONL（与 codes JSONL 相互独立）——保持写入原始值以与离线分析兼容
                if energy_jsonl_output and isinstance(energy_val, (float, int)):
                    try:
                        append_energy_jsonl(
                            energy_jsonl_path,
                            window=int(windows_done),
                            energy=float(energy_val),
                            source=str(res.get('energy_source', 'prequant')),
                            mode=str(res.get('energy_mode', 'l2_mean')),
                        )
                    except Exception as e:
                        print(f"[Stream][WARN] Energy JSONL 写入失败: {e}")

                # 更新能量环形缓冲，用于实时绘制
                if isinstance(energy_val, (float, int)):
                    try:
                        energy_ring.append(float(energy_val))
                        if smoothing_enable and smoother is not None:
                            energy_smooth_val = float(smoother.update(float(energy_val)))
                            energy_smooth_ring.append(energy_smooth_val)
                    except Exception:
                        pass

                # 基于能量阈值的实时分割与片段写出
                if seg_enable and isinstance(energy_val, (float, int)):
                    energy_for_seg = float(energy_smooth_val) if (smoothing_enable and smoothing_use_for_seg and energy_smooth_val is not None) else float(energy_val)
                    frames_cur = window_frames_store.get(int(windows_done), None)

                    # 冷却倒计时
                    if cooldown_left > 0:
                        cooldown_left = max(0, int(cooldown_left) - 1)

                    if not seg_active:
                        # 去抖：仅当 >= thr_on 时累计
                        if energy_for_seg >= seg_thr_on:
                            pos_run += 1
                        else:
                            pos_run = 0

                        if cooldown_left == 0 and pos_run >= max(1, int(seg_up_count)):
                            # 开始新片段
                            seg_writer = None
                            seg_current_path = None
                            if isinstance(frames_cur, list) and len(frames_cur) > 0:
                                h, w = frames_cur[0].shape[:2]
                                seg_current_path = seg_videos_dir / f"segment_{seg_segment_count:04d}_startwin_{int(windows_done):06d}{seg_ext}"
                                try:
                                    seg_writer = cv2.VideoWriter(str(seg_current_path), seg_fourcc, float(target_fps), (w, h))
                                except Exception:
                                    seg_writer = None
                                    seg_current_path = None
                            
                            # 只有当成功创建了视频写入器时，才启动片段
                            if seg_writer is not None and seg_current_path is not None:
                                seg_active = True
                                seg_start_win = int(windows_done)
                                seg_written_windows = 0
                                seg_len_windows = 0
                                seg_segment_count += 1
                                pos_run = 0
                                neg_run = 0
                                print(f"[Seg] START win#{int(windows_done)} thr_on={seg_thr_on:.3f} thr_off={seg_thr_off:.3f} E={float(energy_for_seg):.3f}")
                                # 记录片段起始帧（与首窗写帧策略一致）
                                try:
                                    if int(seg_start_win) in window_frame_range_store:
                                        ws, we = window_frame_range_store[int(seg_start_win)]
                                        if seg_align == "center":
                                            seg_start_frame = int(ws) + (int(T) // 2)
                                        elif seg_align == "end":
                                            k = max(1, int(stride))
                                            seg_start_frame = int(we) - k + 1
                                        else:  # start
                                            seg_start_frame = int(ws)
                                    else:
                                        seg_start_frame = -1
                                except Exception:
                                    seg_start_frame = -1
                            else:
                                print(f"[Seg][WARN] Failed to create video writer for segment, skipping...")

                    if seg_active:
                        # 写入当前窗口帧：首窗写 T，后续写 stride 帧
                        if isinstance(frames_cur, list) and len(frames_cur) > 0 and seg_writer is not None:
                            T_cur = len(frames_cur)
                            if seg_written_windows == 0:
                                if seg_align == "center":
                                    start_idx = T_cur // 2
                                    frames_to_write = frames_cur[start_idx:]
                                elif seg_align == "end":
                                    k = max(1, int(stride))
                                    frames_to_write = frames_cur[T_cur - k:]
                                else:  # start
                                    frames_to_write = frames_cur
                            else:
                                k = max(0, int(stride))
                                frames_to_write = frames_cur[T_cur - k:] if k > 0 else []
                            for fimg in frames_to_write:
                                try:
                                    seg_writer.write(fimg)
                                except Exception:
                                    pass

                        seg_written_windows += 1
                        seg_len_windows += 1
                        # 更新片段结束帧 = 当前窗末帧
                        try:
                            if int(windows_done) in window_frame_range_store:
                                seg_end_frame = int(window_frame_range_store[int(windows_done)][1])
                        except Exception:
                            pass

                        # 结束条件 1：最大时长
                        if seg_max_duration_windows and seg_len_windows >= int(seg_max_duration_windows):
                            try:
                                if seg_writer is not None:
                                    seg_writer.release()
                            except Exception:
                                pass
                            print(f"[Seg] END (max_duration) win#{int(windows_done)} len_windows={seg_written_windows} path={seg_current_path}")
                            # 导出片段 codes（整窗拼接，选择与片段重叠比例>=阈值且互不重叠的窗口）
                            try:
                                if seg_export_codes and should_save_codes(seg_current_path):
                                    export_codes_for_segment(
                                        seg_codes_dir=seg_codes_dir,
                                        seg_current_path=seg_current_path,
                                        window_frame_range_store=window_frame_range_store,
                                        window_codes_store=window_codes_store,
                                        seg_start_frame=seg_start_frame,
                                        seg_end_frame=seg_end_frame,
                                        seg_start_win=seg_start_win,
                                        T=T,
                                        stride=stride,
                                        target_fps=target_fps,
                                        seg_align=seg_align,
                                        seg_codes_min_overlap_ratio=seg_codes_min_overlap_ratio,
                                        allow_overlap=seg_allow_overlap,
                                    )
                            except Exception as e:
                                print(f"[Seg][WARN] export codes failed: {e}")
                            seg_active = False
                            seg_writer = None
                            seg_start_win = -1
                            seg_last_pos_win = -1
                            seg_written_windows = 0
                            seg_current_path = None
                            seg_len_windows = 0
                            cooldown_left = max(cooldown_left, int(seg_cooldown_windows))
                        else:
                            # 结束条件 2：滞回 + 去抖
                            if energy_for_seg < seg_thr_off:
                                neg_run += 1
                            else:
                                neg_run = 0
                            if neg_run >= max(1, int(seg_down_count)):
                                try:
                                    if seg_writer is not None:
                                        seg_writer.release()
                                except Exception:
                                    pass
                                # 若长度不足最小窗口数，删除文件；否则导出 codes
                                try:
                                    if seg_written_windows < max(1, int(seg_min_len_windows)) and seg_current_path is not None:
                                        cleanup_segment_and_codes(seg_current_path)
                                    else:
                                        print(f"[Seg] END win#{int(windows_done)} len_windows={seg_written_windows} path={seg_current_path}")
                                        try:
                                            if seg_export_codes and should_save_codes(seg_current_path):
                                                export_codes_for_segment(
                                                    seg_codes_dir=seg_codes_dir,
                                                    seg_current_path=seg_current_path,
                                                    window_frame_range_store=window_frame_range_store,
                                                    window_codes_store=window_codes_store,
                                                    seg_start_frame=seg_start_frame,
                                                    seg_end_frame=seg_end_frame,
                                                    seg_start_win=seg_start_win,
                                                    T=T,
                                                    stride=stride,
                                                    target_fps=target_fps,
                                                    seg_align=seg_align,
                                                    seg_codes_min_overlap_ratio=seg_codes_min_overlap_ratio,
                                                    allow_overlap=seg_allow_overlap,
                                                )
                                        except Exception as e:
                                            print(f"[Seg][WARN] export codes failed: {e}")
                                except Exception:
                                    pass
                                seg_active = False
                                seg_writer = None
                                seg_start_win = -1
                                seg_last_pos_win = -1
                                seg_written_windows = 0
                                seg_current_path = None
                                seg_len_windows = 0
                                cooldown_left = max(cooldown_left, int(seg_cooldown_windows))

                # 释放当前窗口帧缓存
                try:
                    if seg_enable and int(windows_done) in window_frames_store:
                        del window_frames_store[int(windows_done)]
                except Exception:
                    pass

                if seg_enable:
                    try:
                        cur_win_idx = int(windows_done)
                        if seg_active and int(seg_start_win) >= 0:
                            keep_from = int(seg_start_win)
                        else:
                            keep_from = max(0, cur_win_idx - int(seg_history_keep))
                        codes_drop = [k for k in window_codes_store.keys() if k < keep_from]
                        for k in codes_drop:
                            window_codes_store.pop(k, None)
                        range_drop = [k for k in window_frame_range_store.keys() if k < keep_from]
                        for k in range_drop:
                            window_frame_range_store.pop(k, None)
                    except Exception:
                        pass

                windows_done += 1
                # 若达到最大窗口数限制，则提前停止
                if max_windows is not None and windows_done >= max_windows:
                    print(f"[Stream] Reached max_windows={max_windows}. Stopping...")
                    try:
                        stop_event.set()
                    except Exception:
                        pass
                    reached_max_windows = True
                    break

                # 可选：周期性打印显存信息，并按需释放缓存
                if debug_mem_enable:
                    try:
                        if windows_done % max(1, int(debug_mem_interval)) == 0:
                            if device.type == 'cuda' and torch.cuda.is_available():
                                dev_idx = device.index if device.index is not None else 0
                                alloc = torch.cuda.memory_allocated(dev_idx) / (1024.0 ** 2)
                                reserv = torch.cuda.memory_reserved(dev_idx) / (1024.0 ** 2)
                                max_reserv = torch.cuda.max_memory_reserved(dev_idx) / (1024.0 ** 2)
                                print(f"[Mem][GPU{dev_idx}] win#{windows_done}: allocated={alloc:.1f}MB reserved={reserv:.1f}MB max_reserved={max_reserv:.1f}MB")
                            else:
                                print(f"[Mem][CPU] win#{windows_done}")
                    except Exception:
                        pass
                    try:
                        if debug_mem_empty_cache and device.type == 'cuda' and torch.cuda.is_available() and debug_mem_empty_interval and debug_mem_empty_interval > 0:
                            if windows_done % int(debug_mem_empty_interval) == 0:
                                dev_idx = device.index if device.index is not None else 0
                                torch.cuda.synchronize(dev_idx)
                                torch.cuda.empty_cache()
                                print(f"[Mem][GPU{dev_idx}] empty_cache() called at win#{windows_done}")
                    except Exception:
                        pass
                
            if reached_max_windows:
                break
            # 尚未积满 T 帧
            if len(ring) < T:
                continue

            # 按 stride 触发窗口
            if frames_emitted % stride != 0:
                continue

            # 拷贝当前窗口帧，投递给后台线程处理
            win_frames = list(ring)  # 长度 T
            job = {"win_idx": int(windows_enqueued), "frames": win_frames}
            try:
                job_q.put_nowait(job)
                if seg_enable:
                    window_frames_store[int(windows_enqueued)] = win_frames
                    # 记录该窗覆盖的帧区间 [start, end]（重采样 20Hz 的帧计数）。
                    try:
                        ws = int(frames_emitted) - int(T)
                        we = int(frames_emitted) - 1
                        window_frame_range_store[int(windows_enqueued)] = (max(0, ws), max(0, we))
                    except Exception:
                        pass
            except Exception:
                # 队列满时，丢弃最旧任务以保持最新
                try:
                    old_job = job_q.get_nowait()
                    if seg_enable and isinstance(old_job, dict) and ("win_idx" in old_job):
                        try:
                            del window_frames_store[int(old_job["win_idx"])]
                        except Exception:
                            pass
                        try:
                            if int(old_job["win_idx"]) in window_frame_range_store:
                                del window_frame_range_store[int(old_job["win_idx"])]
                        except Exception:
                            pass
                except Empty:
                    pass
                try:
                    job_q.put_nowait(job)
                    if seg_enable:
                        window_frames_store[int(windows_enqueued)] = win_frames
                        try:
                            ws = int(frames_emitted) - int(T)
                            we = int(frames_emitted) - 1
                            window_frame_range_store[int(windows_enqueued)] = (max(0, ws), max(0, we))
                        except Exception:
                            pass
                except Exception:
                    pass
            windows_enqueued += 1
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            stop_event.set()
        except Exception:
            pass
        try:
            # 发送哨兵以尽快唤醒线程
            job_q.put_nowait(None)
        except Exception:
            pass
        try:
            worker.join(timeout=2.0)
        except Exception:
            pass
        # 分割清理：若退出时仍在录制，关闭并按最小长度判定
        try:
            if 'seg_enable' in locals() and seg_enable:
                try:
                    if 'seg_writer' in locals() and seg_writer is not None:
                        seg_writer.release()
                except Exception:
                    pass
                try:
                    if 'seg_active' in locals() and seg_active:
                        if seg_written_windows < max(1, int(seg_min_len_windows)) and seg_current_path is not None:
                            cleanup_segment_and_codes(seg_current_path)
                        else:
                            print(f"[Seg] END on exit len_windows={seg_written_windows} path={seg_current_path}")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if window_ok:
                cv2.destroyWindow(window_title)
            if energy_window_ok:
                try:
                    cv2.destroyWindow(energy_window_title)
                except Exception:
                    pass
        except Exception:
            pass


if __name__ == "__main__":
    # Linux/CUDA 推荐 spawn
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    parser = build_argparser()
    run_streaming(parser.parse_args())
