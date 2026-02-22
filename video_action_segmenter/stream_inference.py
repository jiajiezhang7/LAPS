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
import math
import statistics
try:
    import resource
except Exception:
    resource = None

import threading
import sys
import subprocess
from queue import Queue, Empty
import logging
from datetime import datetime, timezone

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
from video_action_segmenter.stream_utils import create_compute_worker, open_input_capture, render_and_handle_windows



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

    # 性能监控配置（JSONL 每窗 + 汇总）
    perf_cfg = cfg.get("perf", {}) if isinstance(cfg.get("perf", {}), dict) else {}
    perf_jsonl_output = bool(perf_cfg.get("jsonl_output", True))
    perf_summary_output = bool(perf_cfg.get("summary_output", True))
    perf_jsonl_path_cfg = perf_cfg.get("jsonl_path", None)
    perf_summary_dir_cfg = perf_cfg.get("summary_dir", None)

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
    seg_min_save_windows = max(int(seg_min_len_windows), 3)  # 强制过滤 <=2 窗口的片段
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
    # codes 窗口数过滤：少于此数量的片段视为噪声，不保存
    seg_min_codes_windows = int(seg_cfg.get("min_codes_windows", 2))
    # 未开启分割时无需缓存大量窗口；开启时保留小历史窗口用于触发稳定判定
    seg_history_keep = max(4, int(seg_up_count) + int(seg_cooldown_windows) + 1)
    # 片段元数据导出配置
    seg_export_segments_json = bool(seg_cfg.get("export_segments_json", True))
    seg_segments_json_suffix = str(seg_cfg.get("segments_json_suffix", "_segments"))

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
    cap, src_type, video_name, handled_folder = open_input_capture(input_cfg, args.params)
    if handled_folder:
        return

    # 可选：将视频名映射到自定义模板（用于匹配特定数据集的 GT 命名，例如 Breakfast）
    try:
        name_tmpl = str(input_cfg.get("video_name_template", "")).strip()
    except Exception:
        name_tmpl = ""
    if name_tmpl:
        try:
            _raw_name = str(video_name)
            _id_part = _raw_name.split('_', 1)[0] if '_' in _raw_name else _raw_name
            video_name_mapped = name_tmpl.format(name=_raw_name, id=_id_part)
            print(f"[Input] Remap video_name: '{_raw_name}' -> '{video_name_mapped}' via template='{name_tmpl}'")
            video_name = video_name_mapped
        except Exception as e:
            print(f"[Input][WARN] Failed to apply video_name_template on '{video_name}': {e}")

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

    # 原视频信息与元数据时间基
    try:
        orig_fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap is not None else 0.0
    except Exception:
        orig_fps = 0.0
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap is not None else 0
    except Exception:
        total_frames = 0
    # 使用视频时间基：文件输入优先用原视频 fps，否则回退到 target_fps（实时源）
    if src_type == "file" and orig_fps and orig_fps > 0:
        meta_timebase_fps = float(orig_fps)
    else:
        meta_timebase_fps = float(target_fps)
    # 输入视频完整文件名（含扩展名，仅 file 模式可靠）
    try:
        _path_env = os.environ.get("MT_OVERRIDE_INPUT_PATH", "").strip()
        _path_cfg = str(cfg.get("input", {}).get("path", "")).strip()
        input_video_filename = Path(_path_env or _path_cfg).name if src_type == "file" else str(video_name)
    except Exception:
        input_video_filename = str(video_name)

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
    worker, job_q, result_q, stop_event = create_compute_worker(
        model=model,
        device=device,
        grid_size=grid_size,
        W_dec=W_dec,
        amp=amp,
        energy_enable=energy_enable,
        energy_source=energy_source,
        energy_mode=energy_mode,
        export_prequant=export_prequant,
        prequant_dir=prequant_dir,
        seg_enable=seg_enable,
        seg_export_codes=seg_export_codes,
        pre_gate_enable=pre_gate_enable,
        pre_gate_resize_shorter=pre_gate_resize_shorter,
        pre_gate_pixel_diff_thr=pre_gate_pixel_diff_thr,
        pre_gate_mad_thr=pre_gate_mad_thr,
        pre_gate_min_active_ratio=pre_gate_min_active_ratio,
        pre_gate_method=pre_gate_method,
        pre_gate_debug=pre_gate_debug,
        motion_gate_enable=motion_gate_enable,
        motion_gate_vel_thr=motion_gate_vel_thr,
        motion_gate_min_active_ratio=motion_gate_min_active_ratio,
        motion_gate_debug=motion_gate_debug,
    )

    resampler = TimeResampler(target_fps)
    ring: Deque[np.ndarray] = deque(maxlen=T)
    energy_ring: Deque[float] = deque(maxlen=energy_viz_window)
    energy_smooth_ring: Deque[float] = deque(maxlen=energy_viz_window)
    smoother = CausalSmoother1D(method=smoothing_method, alpha=smoothing_alpha, window=smoothing_window) if smoothing_enable else None

    # 窗口帧缓存：按 win_idx 保存 T 帧，用于阈值分割写出
    window_frames_store = {}
    # 每窗 codes 与帧范围缓存（用于片段 codes 导出）
    window_codes_store = {}
    window_quantized_store = {}
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
    # 片段元数据收集
    segments_records: List[dict] = []
    # 性能指标收集与输出初始化
    job_enqueue_ts = {}
    perf_t_track: List[float] = []
    perf_t_forward: List[float] = []
    perf_t_total: List[float] = []
    perf_latency: List[float] = []
    ok_count = 0
    lag_count = 0
    wall_start = time.perf_counter()
    # 预清理 GPU 峰值显存计数
    try:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass
    # 解析并确定性能输出路径
    if 'perf_jsonl_output' in locals() and (perf_jsonl_output or perf_summary_output):
        try:
            perf_dir = (seg_videos_dir.parent if seg_enable else Path("./video_action_segmenter/inference_outputs").resolve())
        except Exception:
            perf_dir = Path("./video_action_segmenter/inference_outputs").resolve()
        try:
            perf_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        perf_jsonl_path = (Path(perf_jsonl_path_cfg).resolve() if (perf_jsonl_output and perf_jsonl_path_cfg) else (perf_dir / "stream_perf.jsonl"))
        perf_summary_path = (Path(perf_summary_dir_cfg).resolve() / f"{video_name}_perf_summary.json") if perf_summary_dir_cfg else (perf_dir / f"{video_name}_perf_summary.json")
        # 追加写入工具
        def append_perf_jsonl(path: Path, record: dict):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass
    # 增量写入函数
    def _write_segments_json_incremental():
        if not seg_export_segments_json:
            return
        try:
            segments_dir_local = seg_videos_dir
        except Exception:
            segments_dir_local = None
        if segments_dir_local is None:
            return
        try:
            segments_dir_local.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        meta_obj = {
            "video": str(input_video_filename),
            "segments": segments_records,
            "video_duration_sec": (float(total_frames) / float(orig_fps) if (orig_fps and orig_fps > 0 and total_frames > 0) else None),
            "fps": float(orig_fps) if orig_fps else None,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "segmentation_params": {
                "threshold": float(seg_threshold),
                "mode": str(seg_mode),
                "hysteresis_ratio": float(seg_hysteresis_ratio),
                "up_count": int(seg_up_count),
                "down_count": int(seg_down_count),
                "cooldown_windows": int(seg_cooldown_windows),
                "min_len_windows": int(seg_min_len_windows),
                "stride": int(stride),
                "target_fps": int(target_fps),
                "orig_fps": float(orig_fps) if orig_fps else None,
            },
        }
        out_path = segments_dir_local / f"{video_name}{seg_segments_json_suffix}.json"
        try:
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        except Exception:
            tmp_path = None
        try:
            if tmp_path:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(meta_obj, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, out_path)
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(meta_obj, f, ensure_ascii=False, indent=2)
            print(f"[SegMeta] Written: {out_path}")
        except Exception as e:
            logging.warning(f"[SegMeta][WARN] 写入失败: {e}")

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
                window_ok, energy_window_ok, _quit = render_and_handle_windows(
                    frame=frame,
                    window_ok=window_ok,
                    energy_window_ok=energy_window_ok,
                    visualize=visualize,
                    show_overlay=show_overlay,
                    window_title=window_title,
                    seg_enable=seg_enable,
                    seg_active=seg_active,
                    seg_threshold=seg_threshold,
                    avg_track=avg_track,
                    avg_forward=avg_forward,
                    windows_done=windows_done,
                    T=T,
                    stride=stride,
                    grid_size=grid_size,
                    energy_window_title=energy_window_title,
                    energy_viz_style=energy_viz_style,
                    smoothing_enable=smoothing_enable,
                    smoothing_visualize_both=smoothing_visualize_both,
                    energy_ring=energy_ring,
                    energy_smooth_ring=energy_smooth_ring,
                    energy_plot_w=energy_plot_w,
                    energy_plot_h=energy_plot_h,
                    energy_y_min=energy_y_min,
                    energy_y_max=energy_y_max,
                    energy_theme=energy_theme,
                    display_fps=display_fps,
                    energy_color=energy_color,
                )
                if _quit:
                    break

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
                # 性能记录（JSONL/汇总）
                try:
                    if 'perf_jsonl_output' in locals() and (perf_jsonl_output or perf_summary_output):
                        if slack >= 0:
                            ok_count += 1
                        else:
                            lag_count += 1
                        perf_t_track.append(float(t_track))
                        perf_t_forward.append(float(t_forward))
                        perf_t_total.append(float(t_total))
                        st = job_enqueue_ts.pop(int(res.get("win_idx", -1)), None)
                        latency = (time.perf_counter() - st) if st is not None else None
                        if latency is not None:
                            perf_latency.append(float(latency))
                        if perf_jsonl_output:
                            append_perf_jsonl(perf_jsonl_path, {
                                "video": str(video_name),
                                "win": int(res.get("win_idx", windows_done)),
                                "t_track_s": float(t_track),
                                "t_forward_s": float(t_forward),
                                "t_total_s": float(t_total),
                                "budget_s": float(budget_per_window),
                                "slack_s": float(slack),
                                "status": str(status),
                                "latency_s": (float(latency) if latency is not None else None),
                            })
                except Exception:
                    pass

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

                try:
                    qseq = res.get("quantized_vectors", None)
                    if seg_enable and seg_export_codes and isinstance(qseq, list):
                        window_quantized_store[int(windows_done)] = qseq
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
                            if seg_written_windows < int(seg_min_save_windows):
                                if seg_current_path is not None:
                                    cleanup_segment_and_codes(seg_current_path)
                                print(
                                    f"[Seg] DROP (max_duration) win#{int(windows_done)} len_windows={seg_written_windows} path={seg_current_path}"
                                )
                            else:
                                print(
                                    f"[Seg] END (max_duration) win#{int(windows_done)} len_windows={seg_written_windows} path={seg_current_path}"
                                )
                                # 导出片段 codes（整窗拼接，选择与片段重叠比例>=阈值且互不重叠的窗口）
                                try:
                                    if seg_export_codes and should_save_codes(seg_current_path):
                                        export_codes_for_segment(
                                            seg_codes_dir=seg_codes_dir,
                                            seg_current_path=seg_current_path,
                                            window_frame_range_store=window_frame_range_store,
                                            window_codes_store=window_codes_store,
                                            window_quantized_store=window_quantized_store,
                                            seg_start_frame=seg_start_frame,
                                            seg_end_frame=seg_end_frame,
                                            seg_start_win=seg_start_win,
                                            T=T,
                                            stride=stride,
                                            target_fps=target_fps,
                                            seg_align=seg_align,
                                            seg_codes_min_overlap_ratio=seg_codes_min_overlap_ratio,
                                            allow_overlap=seg_allow_overlap,
                                            min_codes_windows=seg_min_codes_windows,
                                        )
                                    # 记录片段元数据（仅保存成功的片段）
                                    try:
                                        if seg_export_segments_json and seg_current_path is not None and os.path.isfile(seg_current_path):
                                            n_frames = max(0, int(seg_end_frame) - int(seg_start_frame) + 1)
                                            start_sec = float(int(seg_start_frame)) / float(meta_timebase_fps) if meta_timebase_fps > 0 else 0.0
                                            end_sec = start_sec + (float(n_frames) / float(meta_timebase_fps) if meta_timebase_fps > 0 else 0.0)
                                            segments_records.append({
                                                "start_sec": float(start_sec),
                                                "end_sec": float(end_sec),
                                                "label": f"segment_{max(0, seg_segment_count-1)}",
                                            })
                                            try:
                                                _write_segments_json_incremental()
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
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
                                    if seg_written_windows < int(seg_min_save_windows) and seg_current_path is not None:
                                        cleanup_segment_and_codes(seg_current_path)
                                        print(
                                            f"[Seg] DROP win#{int(windows_done)} len_windows={seg_written_windows} path={seg_current_path}"
                                        )
                                    else:
                                        print(
                                            f"[Seg] END win#{int(windows_done)} len_windows={seg_written_windows} path={seg_current_path}"
                                        )
                                        try:
                                            if seg_export_codes and should_save_codes(seg_current_path):
                                                export_codes_for_segment(
                                                    seg_codes_dir=seg_codes_dir,
                                                    seg_current_path=seg_current_path,
                                                    window_frame_range_store=window_frame_range_store,
                                                    window_codes_store=window_codes_store,
                                                    window_quantized_store=window_quantized_store,
                                                    seg_start_frame=seg_start_frame,
                                                    seg_end_frame=seg_end_frame,
                                                    seg_start_win=seg_start_win,
                                                    T=T,
                                                    stride=stride,
                                                    target_fps=target_fps,
                                                    seg_align=seg_align,
                                                    seg_codes_min_overlap_ratio=seg_codes_min_overlap_ratio,
                                                    allow_overlap=seg_allow_overlap,
                                                    min_codes_windows=seg_min_codes_windows,
                                                )
                                            # 记录片段元数据（仅保存成功的片段）
                                            try:
                                                if seg_export_segments_json and seg_current_path is not None and os.path.isfile(seg_current_path):
                                                    n_frames = max(0, int(seg_end_frame) - int(seg_start_frame) + 1)
                                                    start_sec = float(int(seg_start_frame)) / float(meta_timebase_fps) if meta_timebase_fps > 0 else 0.0
                                                    end_sec = start_sec + (float(n_frames) / float(meta_timebase_fps) if meta_timebase_fps > 0 else 0.0)
                                                    segments_records.append({
                                                        "start_sec": float(start_sec),
                                                        "end_sec": float(end_sec),
                                                        "label": f"segment_{max(0, seg_segment_count-1)}",
                                                    })
                                                    try:
                                                        _write_segments_json_incremental()
                                                    except Exception:
                                                        pass
                                            except Exception:
                                                pass
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
                        q_drop = [k for k in window_quantized_store.keys() if k < keep_from]
                        for k in q_drop:
                            window_quantized_store.pop(k, None)
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
                job_enqueue_ts[int(windows_enqueued)] = time.perf_counter()
                job_q.put_nowait(job)
                if seg_enable:
                    window_frames_store[int(windows_enqueued)] = win_frames
                    # 记录该窗覆盖的帧区间 [start, end]（当前实现为解码出的原视频帧计数）。
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
                        if seg_written_windows < int(seg_min_save_windows) and seg_current_path is not None:
                            cleanup_segment_and_codes(seg_current_path)
                            print(
                                f"[Seg] DROP on exit len_windows={seg_written_windows} path={seg_current_path}"
                            )
                        else:
                            print(f"[Seg] END on exit len_windows={seg_written_windows} path={seg_current_path}")
                            # on-exit 情况下补充记录片段元数据（仅保存成功的片段）
                            try:
                                if seg_export_segments_json and seg_current_path is not None and os.path.isfile(seg_current_path):
                                    n_frames = max(0, int(seg_end_frame) - int(seg_start_frame) + 1)
                                    start_sec = float(int(seg_start_frame)) / float(meta_timebase_fps) if meta_timebase_fps > 0 else 0.0
                                    end_sec = start_sec + (float(n_frames) / float(meta_timebase_fps) if meta_timebase_fps > 0 else 0.0)
                                    segments_records.append({
                                        "start_sec": float(start_sec),
                                        "end_sec": float(end_sec),
                                        "label": f"segment_{max(0, seg_segment_count-1)}",
                                    })
                                    try:
                                        _write_segments_json_incremental()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass
        # 写出性能汇总（如启用）
        try:
            if 'perf_summary_output' in locals() and perf_summary_output and ('perf_t_total' in locals()) and len(perf_t_total) > 0:
                def _quantile(xs, q):
                    xs = sorted(xs)
                    if not xs:
                        return None
                    k = (len(xs)-1)*q
                    f = int(k // 1)
                    c = int(k // 1 + 1) if f < len(xs)-1 else f
                    if f == c:
                        return float(xs[f])
                    return float(xs[f] + (xs[c]-xs[f])*(k-f))
                wall_time = time.perf_counter() - wall_start if 'wall_start' in locals() else None
                realized_wps = float(len(perf_t_total)) / float(wall_time) if wall_time and wall_time > 0 else None
                realized_fps = float(realized_wps * float(stride)) if realized_wps is not None else None
                ok_ratio = float(ok_count) / float(len(perf_t_total)) if len(perf_t_total) > 0 else None
                gpu_mem_mb = None; gpu_mem_reserved_mb = None
                try:
                    if device.type == 'cuda':
                        try:
                            gpu_mem_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0*1024.0)
                        except Exception:
                            pass
                        try:
                            gpu_mem_reserved_mb = float(torch.cuda.max_memory_reserved(device)) / (1024.0*1024.0)
                        except Exception:
                            pass
                except Exception:
                    pass
                cpu_max_rss_mb = None
                try:
                    if 'resource' in globals() and resource is not None:
                        cpu_max_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
                except Exception:
                    pass
                try:
                    param_count = int(sum(p.numel() for p in model.parameters()))
                except Exception:
                    param_count = None
                summary = {
                    "video": str(input_video_filename),
                    "video_name": str(video_name),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "target_fps": int(target_fps),
                    "stride": int(stride),
                    "budget_per_window_s": float(budget_per_window),
                    "windows_done": int(len(perf_t_total)),
                    "wall_time_s": (float(wall_time) if wall_time is not None else None),
                    "ok_ratio": (float(ok_ratio) if ok_ratio is not None else None),
                    "realized_windows_per_sec": (float(realized_wps) if realized_wps is not None else None),
                    "throughput_fps_estimate": (float(realized_fps) if realized_fps is not None else None),
                    "t_total_s": {"mean": float(sum(perf_t_total)/len(perf_t_total)), "p50": (_quantile(perf_t_total,0.5) if len(perf_t_total)>0 else None), "p95": (_quantile(perf_t_total,0.95) if len(perf_t_total)>0 else None), "max": float(max(perf_t_total))},
                    "t_track_s": {"mean": float(sum(perf_t_track)/len(perf_t_track)), "p50": (_quantile(perf_t_track,0.5) if len(perf_t_track)>0 else None), "p95": (_quantile(perf_t_track,0.95) if len(perf_t_track)>0 else None), "max": float(max(perf_t_track))},
                    "t_forward_s": {"mean": float(sum(perf_t_forward)/len(perf_t_forward)), "p50": (_quantile(perf_t_forward,0.5) if len(perf_t_forward)>0 else None), "p95": (_quantile(perf_t_forward,0.95) if len(perf_t_forward)>0 else None), "max": float(max(perf_t_forward))},
                    "latency_s": {"mean": (float(sum(perf_latency)/len(perf_latency)) if perf_latency else None), "p50": (_quantile(perf_latency,0.5) if perf_latency else None), "p95": (_quantile(perf_latency,0.95) if perf_latency else None), "max": (float(max(perf_latency)) if perf_latency else None)},
                    "gpu_max_mem_allocated_mb": gpu_mem_mb,
                    "gpu_max_mem_reserved_mb": gpu_mem_reserved_mb,
                    "cpu_max_rss_mb": cpu_max_rss_mb,
                    "param_count": param_count,
                }
                try:
                    perf_summary_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                with open(perf_summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                print(f"[Perf] Summary written: {perf_summary_path}")
        except Exception:
            pass

        # 写出每视频 JSON 元数据文件
        try:
            if 'seg_enable' in locals() and seg_enable and seg_export_segments_json:
                try:
                    segments_dir = seg_videos_dir if 'seg_videos_dir' in locals() else None
                except Exception:
                    segments_dir = None
                if segments_dir is not None:
                    try:
                        segments_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    meta_obj = {
                        "video": str(input_video_filename),
                        "segments": segments_records,
                        "video_duration_sec": (float(total_frames) / float(orig_fps) if (orig_fps and orig_fps > 0 and total_frames > 0) else None),
                        "fps": float(orig_fps) if orig_fps else None,
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "segmentation_params": {
                            "threshold": float(seg_threshold),
                            "mode": str(seg_mode),
                            "hysteresis_ratio": float(seg_hysteresis_ratio),
                            "up_count": int(seg_up_count),
                            "down_count": int(seg_down_count),
                            "cooldown_windows": int(seg_cooldown_windows),
                            "min_len_windows": int(seg_min_len_windows),
                            "stride": int(stride),
                            "target_fps": int(target_fps),
                            "orig_fps": float(orig_fps) if orig_fps else None,
                        },
                    }
                    out_path = segments_dir / f"{video_name}{seg_segments_json_suffix}.json"
                    try:
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(meta_obj, f, ensure_ascii=False, indent=2)
                        print(f"[SegMeta] Written: {out_path}")
                    except Exception as e:
                        logging.warning(f"[SegMeta][WARN] 写入失败: {e}")
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
