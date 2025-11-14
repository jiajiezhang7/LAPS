#!/usr/bin/env python3
"""
Plot Optical Flow Energy vs Action Energy (Quantized) with Segmentor Output.

This script generates a publication-quality figure comparing:
- E_action (quantized token diff L2 mean) in blue
- Optical Flow magnitude mean in red
- GT action boundaries as green dashed lines
- Segmentor output boundaries as purple solid lines

Output: High-resolution PNG suitable for paper Figure 3.

Usage:
  conda run -n laps python -m video_action_segmenter.scripts.plot_energy_comparison \
    --optical-flow-jsonl /path/to/stream_energy_optical_flow_mag_mean.jsonl \
    --action-energy-jsonl /path/to/stream_energy_quantized_token_diff_l2_mean.jsonl \
    --gt-json /path/to/gt_segments.json \
    --segmentor-json /path/to/segmented_videos/segments.json \
    --output-dir /path/to/output \
    --start-sec 0 --duration-sec 60 \
    --dpi 300
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


def load_energy_jsonl(jsonl_path: Path) -> Dict[int, float]:
    """Load energy JSONL file and return dict: window_idx -> energy value."""
    energies = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    window = int(rec["window"])
                    energy = float(rec["energy"])
                    energies[window] = energy
    except Exception as e:
        print(f"[WARN] Failed to load {jsonl_path}: {e}")
    return energies


def load_gt_segments(gt_json_path: Path) -> List[Tuple[float, float]]:
    """Load GT segments JSON and return list of (start_sec, end_sec) tuples."""
    segments = []
    try:
        with open(gt_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for seg in data.get("segments", []):
                start = float(seg["start_sec"])
                end = float(seg["end_sec"])
                segments.append((start, end))
    except Exception as e:
        print(f"[WARN] Failed to load GT segments from {gt_json_path}: {e}")
    return segments


def load_segmentor_segments(segmentor_json_path: Optional[Path]) -> List[Tuple[float, float]]:
    """Load segmentor output JSON and return list of (start_sec, end_sec) tuples.

    Returns empty list if path is None or file doesn't exist.
    """
    if segmentor_json_path is None:
        return []

    segments = []
    try:
        with open(segmentor_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for seg in data.get("segments", []):
                start = float(seg["start_sec"])
                end = float(seg["end_sec"])
                segments.append((start, end))
    except Exception as e:
        print(f"[WARN] Failed to load segmentor segments from {segmentor_json_path}: {e}")
    return segments


def window_to_sec(window_idx: int, stride: int = 4, target_fps: float = 10.0) -> float:
    """Convert window index to seconds.

    Formula: sec = window_idx * stride / target_fps
    Default: stride=4, target_fps=10 (from LAPS pipeline)
    """
    return float(window_idx) * float(stride) / float(target_fps)


def sec_to_window(sec: float, stride: int = 4, target_fps: float = 10.0) -> int:
    """Convert seconds to window index (inverse of window_to_sec)."""
    return int(round(sec * float(target_fps) / float(stride)))


# --- Time mapping helpers and parameter resolver ---

def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def resolve_params_from_segmentor_json(segmentor_json_path: Optional[Path]):
    """Resolve stride/target_fps/orig_fps/video_duration_sec from a segmentor JSON.
    Returns (stride, target_fps, orig_fps, video_duration_sec) with None when unavailable.
    """
    if segmentor_json_path is None:
        return None, None, None, None
    try:
        with open(segmentor_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None, None, None, None

    segp = (meta.get("segmentation_params") or {}) if isinstance(meta, dict) else {}
    stride = _safe_int(segp.get("stride"))
    target_fps = _safe_float(segp.get("target_fps"))

    orig_fps = _safe_float(meta.get("orig_fps"))
    if orig_fps is None:
        orig_fps = _safe_float(meta.get("fps"))
    if orig_fps is None:
        orig_fps = _safe_float(segp.get("orig_fps"))

    video_duration_sec = _safe_float(meta.get("video_duration_sec"))
    return stride, target_fps, orig_fps, video_duration_sec


# Mapping for action energy (window advances by stride at target_fps)
# Keep the existing window_to_sec/sec_to_window as action-energy mapping.
# Add dedicated mapping for optical flow (sample index at target_fps, no stride).

def window_to_sec_of(window_idx: int, target_fps: float = 10.0) -> float:
    """Optical flow energy: each window is one emitted sample at target_fps."""
    return float(window_idx) / float(target_fps)


def sec_to_window_of(sec: float, target_fps: float = 10.0) -> int:
    return int(round(float(sec) * float(target_fps)))


def extract_time_range_action(
    energies: Dict[int, float],
    start_sec: float,
    duration_sec: float,
    stride: int = 4,
    target_fps: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    start_window = sec_to_window(start_sec, stride, target_fps)
    end_window = sec_to_window(start_sec + duration_sec, stride, target_fps)
    times, values = [], []
    for w in range(start_window, end_window + 1):
        if w in energies:
            times.append(window_to_sec(w, stride, target_fps))
            values.append(energies[w])
    return np.array(times), np.array(values)


def extract_time_range_of(
    energies: Dict[int, float],
    start_sec: float,
    duration_sec: float,
    target_fps: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    start_window = sec_to_window_of(start_sec, target_fps)
    end_window = sec_to_window_of(start_sec + duration_sec, target_fps)
    times, values = [], []
    for w in range(start_window, end_window + 1):
        if w in energies:
            times.append(window_to_sec_of(w, target_fps))
            values.append(energies[w])
    return np.array(times), np.array(values)


def extract_time_range(
    energies: Dict[int, float],
    start_sec: float,
    duration_sec: float,
    stride: int = 4,
    target_fps: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract energy values for a specific time range.

    Returns:
        (time_array, energy_array) where time_array is in seconds
    """
    start_window = sec_to_window(start_sec, stride, target_fps)
    end_window = sec_to_window(start_sec + duration_sec, stride, target_fps)

    times = []
    values = []
    for w in range(start_window, end_window + 1):
        if w in energies:
            t = window_to_sec(w, stride, target_fps)
            times.append(t)
            values.append(energies[w])

    return np.array(times), np.array(values)


def plot_energy_comparison(
    optical_flow_jsonl: Optional[Path],
    action_energy_jsonl: Path,
    gt_json: Path,
    output_dir: Path,
    start_sec: float = 0.0,
    duration_sec: float = 60.0,
    stride: int = 4,
    target_fps: float = 10.0,
    dpi: int = 300,
    figsize: Tuple[float, float] = (14, 6),
    segmentor_json: Optional[Path] = None,
    plot_optical_flow: bool = True,
) -> None:
    """Generate publication-quality comparison plot.

    Args:
        optical_flow_jsonl: Optional path to optical flow energy JSONL
        action_energy_jsonl: Path to action energy (quantized) JSONL
        gt_json: Path to GT segments JSON
        output_dir: Output directory for PNG
        start_sec: Start time in seconds
        duration_sec: Duration in seconds (default 60s for paper)
        stride: Window stride (default 4)
        target_fps: Target FPS (default 10)
        dpi: Output DPI (default 300 for publication)
        figsize: Figure size in inches
        segmentor_json: Optional path to segmentor output JSON
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve params from segmentor JSON if provided
    orig_fps: Optional[float] = None
    video_duration_sec: Optional[float] = None
    if segmentor_json is not None:
        s_res, t_res, o_res, vdur_res = resolve_params_from_segmentor_json(segmentor_json)
        if s_res is not None:
            stride = int(s_res)
        if t_res is not None:
            target_fps = float(t_res)
        if o_res is not None:
            orig_fps = float(o_res)
        if vdur_res is not None:
            video_duration_sec = float(vdur_res)
        print(f"[INFO] Params from segmentor_json: stride={stride}, target_fps={target_fps}, orig_fps={orig_fps}, video_duration_sec={video_duration_sec}")

    # Load data
    print("[INFO] Loading energy signals...")
    optical_flow = None
    if plot_optical_flow:
        if optical_flow_jsonl is None:
            print("[ERR] Optical flow plotting enabled but no file provided")
            return
        optical_flow = load_energy_jsonl(optical_flow_jsonl)
    action_energy = load_energy_jsonl(action_energy_jsonl)
    gt_segments = load_gt_segments(gt_json)
    segmentor_segments = load_segmentor_segments(segmentor_json)

    # Sanity-check energy axis lengths vs video duration; auto-calibrate target_fps if mismatched
    ae_target_fps_eff = float(target_fps)
    of_target_fps_eff = float(target_fps)
    try:
        if video_duration_sec is not None and video_duration_sec > 0:
            if action_energy:
                max_w_ae = max(action_energy.keys())
                est_ae_sec = window_to_sec(max_w_ae, stride, target_fps)
                if abs(est_ae_sec - video_duration_sec) > 5.0 and stride and stride > 0:
                    implied_tfps = (float(max_w_ae) / float(video_duration_sec)) * float(stride)
                    print(f"[WARN] Action energy axis mismatch: ~{est_ae_sec:.1f}s vs {video_duration_sec:.1f}s; using implied target_fps={implied_tfps:.3f} for plotting")
                    ae_target_fps_eff = float(implied_tfps)
            if plot_optical_flow and optical_flow:
                max_w_of = max(optical_flow.keys())
                est_of_sec = window_to_sec_of(max_w_of, target_fps)
                if abs(est_of_sec - video_duration_sec) > 5.0:
                    implied_tfps_of = (float(max_w_of) / float(video_duration_sec)) if video_duration_sec > 0 else float(target_fps)
                    print(f"[WARN] Optical flow axis mismatch: ~{est_of_sec:.1f}s vs {video_duration_sec:.1f}s; using implied target_fps={implied_tfps_of:.3f} for plotting")
                    of_target_fps_eff = float(implied_tfps_of)
    except Exception as e:
        print(f"[WARN] Sanity-check failed: {e}")


    if plot_optical_flow and not optical_flow:
        print("[ERR] No optical flow data loaded")
        return
    if not action_energy:
        print("[ERR] No action energy data loaded")
        return

    if segmentor_json and segmentor_segments:
        print(f"[INFO] Loaded {len(segmentor_segments)} segmentor output segments")
    elif segmentor_json:
        print("[WARN] Segmentor JSON specified but no segments loaded")

    # Extract time range
    print(f"[INFO] Extracting time range: {start_sec}s - {start_sec + duration_sec}s")
    if plot_optical_flow:
        of_times, of_values = extract_time_range_of(
            optical_flow, start_sec, duration_sec, of_target_fps_eff
        )
    else:
        of_times, of_values = np.array([]), np.array([])
    ae_times, ae_values = extract_time_range_action(
        action_energy, start_sec, duration_sec, stride, ae_target_fps_eff
    )

    if len(ae_times) == 0:
        print("[ERR] No action energy data in the specified time range")
        return
    if plot_optical_flow and len(of_times) == 0:
        print("[ERR] No optical flow data in the specified time range")
        return

    # Filter GT segments to the time range
    end_sec = start_sec + duration_sec
    visible_gt_segments = [
        (s, e) for s, e in gt_segments
        if not (e < start_sec or s > end_sec)
    ]
    visible_segmentor_segments = [
        (s, e) for s, e in segmentor_segments
        if not (e < start_sec or s > end_sec)
    ]

    # Create figure
    print("[INFO] Creating figure...")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Normalize both signals to [0, 1] for better visual comparison
    ae_min, ae_max = np.min(ae_values), np.max(ae_values)

    if plot_optical_flow:
        of_min, of_max = np.min(of_values), np.max(of_values)
        of_norm = (of_values - of_min) / (of_max - of_min + 1e-8)
    ae_norm = (ae_values - ae_min) / (ae_max - ae_min + 1e-8)

    # Plot energy signals
    if plot_optical_flow:
        ax.plot(
            of_times, of_norm,
            color="#F5A9A9",  # Light Red
            linewidth=2.5,
            label="Optical Flow Magnitude",
            alpha=0.85,
            zorder=2,
        )
    ax.plot(
        ae_times, ae_norm,
        color="#3498DB",  # Blue
        linewidth=2.5,
        label="$E_{\\text{action}}$ (Quantized)",
        alpha=0.85,
        zorder=2,
    )

    # Plot GT boundaries as shaded regions only (green)
    for start, end in visible_gt_segments:
        # Clip to visible range
        seg_start = max(start, start_sec)
        seg_end = min(end, end_sec)

        # Shade the segment region lightly
        ax.axvspan(
            float(seg_start), float(seg_end),
            color="#1E8449",
            alpha=0.15,
            zorder=0,
        )

    # Plot segmentor output boundaries as vertical solid lines (purple)
    for start, end in visible_segmentor_segments:
        # Clip to visible range
        seg_start = max(start, start_sec)
        seg_end = min(end, end_sec)

        # Draw boundary lines (start and end)
        ax.axvline(
            float(seg_start),
            color="#9B59B6",  # Purple
            linestyle="-",
            linewidth=1.8,
            alpha=0.7,
            zorder=2,
        )
        ax.axvline(
            float(seg_end),
            color="#9B59B6",  # Purple
            linestyle="-",
            linewidth=1.8,
            alpha=0.7,
            zorder=2,
        )

    # Formatting
    ax.set_xlabel("Time (seconds)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Normalized Energy", fontsize=16, fontweight="bold")

    # Set x-axis limits
    ax.set_xlim(start_sec, end_sec)
    ax.set_ylim(-0.05, 1.1)

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend with custom entries
    legend_elements = [
        plt.Line2D([0], [0], color="#3498DB", linewidth=2.5, label="$E_{\\text{action}}$ (Quantized)"),
        mpatches.Patch(
            facecolor="#1E8449",
            alpha=0.15,
            edgecolor="none",
            label="GT Action Regions",
        ),
    ]

    if plot_optical_flow:
        legend_elements.insert(
            0,
            plt.Line2D(
                [0], [0], color="#F5A9A9", linewidth=2.5, label="Optical Flow Magnitude"
            ),
        )

    # Add segmentor output to legend if available
    if visible_segmentor_segments:
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                color="#9B59B6",
                linestyle="-",
                linewidth=1.8,
                label="Segmentor Output",
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=13,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
    )

    # Tight layout
    plt.tight_layout()

    # Save
    output_path = output_dir / f"energy_comparison_{start_sec:.0f}s_{duration_sec:.0f}s.png"
    print(f"[INFO] Saving figure to: {output_path}")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    # Print statistics
    if plot_optical_flow:
        print(f"\n[STATS] Optical Flow:")
        print(f"  - Min: {of_min:.4f}, Max: {of_max:.4f}, Mean: {np.mean(of_values):.4f}")
        print(f"  - Std: {np.std(of_values):.4f}")
    print(f"\n[STATS] Action Energy:")
    print(f"  - Min: {ae_min:.4f}, Max: {ae_max:.4f}, Mean: {np.mean(ae_values):.4f}")
    print(f"  - Std: {np.std(ae_values):.4f}")
    print(f"\n[STATS] GT Segments in range: {len(visible_gt_segments)}")
    if visible_segmentor_segments:
        print(f"[STATS] Segmentor Output Segments in range: {len(visible_segmentor_segments)}")
    print(f"[OK] Figure saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot Optical Flow vs Action Energy comparison for paper Figure 3"
    )
    ap.add_argument(
        "--optical-flow-jsonl",
        type=str,
        default=None,
        help="Path to optical flow energy JSONL file (omit with --no-optical-flow)",
    )
    ap.add_argument(
        "--action-energy-jsonl",
        type=str,
        required=True,
        help="Path to action energy (quantized) JSONL file",
    )
    ap.add_argument(
        "--gt-json",
        type=str,
        required=True,
        help="Path to GT segments JSON file",
    )
    ap.add_argument(
        "--segmentor-json",
        type=str,
        default=None,
        help="Optional path to segmentor output JSON file",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for PNG figure",
    )
    ap.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0)",
    )
    ap.add_argument(
        "--duration-sec",
        type=float,
        default=60.0,
        help="Duration in seconds (default: 60 for paper)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Window stride (default: 4)",
    )
    ap.add_argument(
        "--target-fps",
        type=float,
        default=10.0,
        help="Target FPS (default: 10)",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300 for publication)",
    )
    ap.add_argument(
        "--figsize-width",
        type=float,
        default=14,
        help="Figure width in inches (default: 14)",
    )
    ap.add_argument(
        "--figsize-height",
        type=float,
        default=6,
        help="Figure height in inches (default: 6)",
    )
    ap.add_argument(
        "--no-optical-flow",
        action="store_true",
        help="Disable optical flow plotting even if a file is provided",
    )

    args = ap.parse_args()

    plot_energy_comparison(
        optical_flow_jsonl=Path(args.optical_flow_jsonl) if args.optical_flow_jsonl else None,
        action_energy_jsonl=Path(args.action_energy_jsonl),
        gt_json=Path(args.gt_json),
        output_dir=Path(args.output_dir),
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        stride=args.stride,
        target_fps=args.target_fps,
        dpi=args.dpi,
        figsize=(args.figsize_width, args.figsize_height),
        segmentor_json=Path(args.segmentor_json) if args.segmentor_json else None,
        plot_optical_flow=not args.no_optical_flow,
    )


if __name__ == "__main__":
    main()
