"""Utilities for streaming inference modularization.

This package groups helpers into thematic modules:
- video: time resampling and aspect-ratio preserving resize
- tracking_online: CoTracker online loader and per-window tracking
- energy: energy computation and visualization helpers
- stream_io: JSONL appenders and array export helpers
- config_utils: configuration loading and utility functions
"""
from .video import TimeResampler, resize_shorter_keep_aspect
from .tracking_online import (
    to_clip_tensor,
    get_cotracker_online,
    track_window_with_online,
)
from .energy import (
    draw_energy_plot,
    draw_energy_plot_two,
    draw_energy_plot_enhanced,
    draw_energy_plot_enhanced_dual,
    compute_energy,
    CausalSmoother1D,
    apply_smoothing_1d,
)
from .stream_io import (
    append_codes_jsonl,
    append_energy_jsonl,
    export_prequant_npy,
)
from .gating import (
    pre_gate_check,
    motion_gate_check,
)
from .paths import (
    compute_per_video_energy_jsonl_path,
    should_skip_video_outputs,
)
from .segmentation import (
    cleanup_segment_and_codes,
    should_save_codes,
    ensure_output_dirs,
    export_codes_for_segment,
)
from .batch import (
    run_batch_over_folder,
)
from .config_utils import (
    load_config,
    _normalize_velocities,
)
from .inference_worker import create_compute_worker
from .input_source import open_input_capture
from .visualization import render_and_handle_windows
