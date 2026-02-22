import time
from typing import List, Optional, Tuple
from queue import Queue, Empty

import numpy as np
import torch

from .gating import pre_gate_check, motion_gate_check
from .tracking_online import track_window_with_online
from .energy import compute_energy
from .stream_io import export_prequant_npy


def _normalize_velocities_local(vel_pix: np.ndarray, window_size: int):
    """Keep exactly the same normalization behavior as in stream_inference.py.
    vel_pix: (T-1, N, 2) on CPU
    returns torch.Tensor (T-1, N, 2)
    """
    # vel_pix may be np.ndarray; convert to torch tensor on CPU for consistency
    if not isinstance(vel_pix, torch.Tensor):
        vel_pix = torch.as_tensor(vel_pix)
    return vel_pix / (float(window_size) / 2.0)


def create_compute_worker(
    *,
    model: torch.nn.Module,
    device: torch.device,
    grid_size: int,
    W_dec: int,
    amp: bool,
    energy_enable: bool,
    energy_source: str,
    energy_mode: str,
    export_prequant: bool,
    prequant_dir,
    seg_enable: bool,
    seg_export_codes: bool,
    # pre-gate
    pre_gate_enable: bool,
    pre_gate_resize_shorter: int,
    pre_gate_pixel_diff_thr: float,
    pre_gate_mad_thr: float,
    pre_gate_min_active_ratio: float,
    pre_gate_method: str,
    pre_gate_debug: bool,
    # motion-gate
    motion_gate_enable: bool,
    motion_gate_vel_thr: float,
    motion_gate_min_active_ratio: float,
    motion_gate_debug: bool,
):
    """Create and start the background compute worker thread.

    Returns: (thread, job_q, result_q, stop_event)
    - job_q: put {"win_idx": int, "frames": List[np.ndarray]}
    - result_q: get result dicts (same schema as original stream_inference)
    """
    job_q: Queue = Queue(maxsize=1)
    result_q: Queue = Queue(maxsize=16)
    import threading as _th
    stop_event = _th.Event()

    def compute_worker():
        # Optional: set CUDA device in this thread
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
            win_idx: int = int(job["win_idx"])  # window index
            frames: List[np.ndarray] = job["frames"]

            try:
                # 1) Pre-gate on raw frames (very cheap)
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

                # 2) CoTracker tracking
                t0 = time.perf_counter()
                vel_pix = track_window_with_online(frames, grid_size=grid_size, device=device)
                try:
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                except Exception:
                    pass
                t_track = time.perf_counter() - t0

                # 3) Normalize velocities
                t1 = time.perf_counter()
                vel_norm = _normalize_velocities_local(vel_pix, window_size=W_dec)

                # 4) Motion gate on velocities
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
                        used_energy_source = str(energy_source)
                        energy_val_gate: Optional[float] = None
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
                        continue

                # 5) Encode + Quantize (FSQ)
                Tm1, N_local, _ = vel_norm.shape
                x_input = vel_norm.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T-1, N, 2)

                # AMP autocast (CUDA only)
                try:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=amp and device.type == 'cuda')
                except Exception:
                    class _Dummy:
                        def __enter__(self):
                            return None
                        def __exit__(self, exc_type, exc, tb):
                            return False
                    autocast_ctx = _Dummy()

                with torch.inference_mode():
                    with autocast_ctx:
                        to_quantize = model.encode(x_input, cond=None)
                        quantized, fsq_indices = model.quantize(to_quantize)

                try:
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                except Exception:
                    pass
                t_forward = time.perf_counter() - t1

                # codes extraction from fsq_indices
                code_ids_seq_list = None
                used_digits_merge = False
                if isinstance(fsq_indices, torch.Tensor):
                    try:
                        code_ids_seq = fsq_indices.squeeze(0).to(torch.int16)
                        code_ids_seq_list = [int(v) for v in code_ids_seq.tolist()]
                        used_digits_merge = True
                    except Exception:
                        code_ids_seq_list = None

                # move to CPU ASAP
                to_quantize_cpu = to_quantize.detach().cpu() if isinstance(to_quantize, torch.Tensor) else None
                quantized_cpu = quantized.detach().cpu() if isinstance(quantized, torch.Tensor) else None

                quantized_seq_list = None
                if seg_enable and seg_export_codes and isinstance(quantized_cpu, torch.Tensor):
                    try:
                        q = quantized_cpu.squeeze(0)  # (seq_len, hidden_dim)
                        quantized_seq_list = q.to(torch.float32).numpy().tolist()
                    except Exception:
                        quantized_seq_list = None

                # explicit delete to reduce peak memory
                try:
                    del x_input, to_quantize, quantized, fsq_indices
                except Exception:
                    pass

                # 6) energy (optional)
                energy_val: Optional[float] = None
                if energy_enable:
                    energy_val = compute_energy(
                        source=energy_source,
                        mode=energy_mode,
                        to_quantize=to_quantize_cpu,
                        quantized=quantized_cpu,
                        vel_norm=vel_norm,
                    )

                # 7) export prequant (optional)
                prequant_path = export_prequant_npy(prequant_dir, win_idx, to_quantize_cpu) if export_prequant and to_quantize_cpu is not None else None

                # shapes
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
                    "quantized_vectors": quantized_seq_list,
                    "energy": energy_val,
                    "energy_source": energy_source,
                    "energy_mode": energy_mode,
                    "prequant_path": prequant_path,
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
            except Exception as e:
                # return error object
                err = {"win_idx": win_idx, "error": str(e)}
                try:
                    result_q.put_nowait(err)
                except Exception:
                    pass
                # best-effort cleanup of large objects
                for _nm in (
                    "first_frame_clip", "video", "pred_tracks", "pred_visibility", "clip", "x",
                    "encoded", "proj_in", "proj_out", "to_quantize", "quantized", "fsq_indices"
                ):
                    try:
                        if _nm in locals():
                            del locals()[_nm]
                    except Exception:
                        pass

    import threading
    worker = threading.Thread(target=compute_worker, daemon=True)
    worker.start()
    return worker, job_q, result_q, stop_event

