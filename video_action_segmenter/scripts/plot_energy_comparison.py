#!/usr/bin/env python3
"""
Plot Optical Flow Energy vs Action Energy (Quantized) for Q1 comparison.

This script generates a publication-quality figure comparing:
- E_action (quantized token diff L2 mean) in blue
- Optical Flow magnitude mean in red
- GT action boundaries as vertical dashed lines

Output: High-resolution PNG suitable for paper Figure 3.

Usage:
  conda run -n laps python -m video_action_segmenter.scripts.plot_energy_comparison \
    --optical-flow-jsonl /path/to/stream_energy_optical_flow_mag_mean.jsonl \
    --action-energy-jsonl /path/to/stream_energy_quantized_token_diff_l2_mean.jsonl \
    --gt-json /path/to/gt_segments.json \
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


def window_to_sec(window_idx: int, stride: int = 4, target_fps: float = 10.0) -> float:
    """Convert window index to seconds.
    
    Formula: sec = window_idx * stride / target_fps
    Default: stride=4, target_fps=10 (from LAPS pipeline)
    """
    return float(window_idx) * float(stride) / float(target_fps)


def sec_to_window(sec: float, stride: int = 4, target_fps: float = 10.0) -> int:
    """Convert seconds to window index (inverse of window_to_sec)."""
    return int(round(sec * float(target_fps) / float(stride)))


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
    optical_flow_jsonl: Path,
    action_energy_jsonl: Path,
    gt_json: Path,
    output_dir: Path,
    start_sec: float = 0.0,
    duration_sec: float = 60.0,
    stride: int = 4,
    target_fps: float = 10.0,
    dpi: int = 300,
    figsize: Tuple[float, float] = (14, 6),
) -> None:
    """Generate publication-quality comparison plot.
    
    Args:
        optical_flow_jsonl: Path to optical flow energy JSONL
        action_energy_jsonl: Path to action energy (quantized) JSONL
        gt_json: Path to GT segments JSON
        output_dir: Output directory for PNG
        start_sec: Start time in seconds
        duration_sec: Duration in seconds (default 60s for paper)
        stride: Window stride (default 4)
        target_fps: Target FPS (default 10)
        dpi: Output DPI (default 300 for publication)
        figsize: Figure size in inches
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("[INFO] Loading energy signals...")
    optical_flow = load_energy_jsonl(optical_flow_jsonl)
    action_energy = load_energy_jsonl(action_energy_jsonl)
    gt_segments = load_gt_segments(gt_json)
    
    if not optical_flow:
        print("[ERR] No optical flow data loaded")
        return
    if not action_energy:
        print("[ERR] No action energy data loaded")
        return
    
    # Extract time range
    print(f"[INFO] Extracting time range: {start_sec}s - {start_sec + duration_sec}s")
    of_times, of_values = extract_time_range(
        optical_flow, start_sec, duration_sec, stride, target_fps
    )
    ae_times, ae_values = extract_time_range(
        action_energy, start_sec, duration_sec, stride, target_fps
    )
    
    if len(of_times) == 0 or len(ae_times) == 0:
        print("[ERR] No data in the specified time range")
        return
    
    # Filter GT segments to the time range
    end_sec = start_sec + duration_sec
    visible_segments = [
        (s, e) for s, e in gt_segments
        if not (e < start_sec or s > end_sec)
    ]
    
    # Create figure
    print("[INFO] Creating figure...")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Normalize both signals to [0, 1] for better visual comparison
    of_min, of_max = np.min(of_values), np.max(of_values)
    ae_min, ae_max = np.min(ae_values), np.max(ae_values)
    
    of_norm = (of_values - of_min) / (of_max - of_min + 1e-8)
    ae_norm = (ae_values - ae_min) / (ae_max - ae_min + 1e-8)
    
    # Plot energy signals
    ax.plot(
        of_times, of_norm,
        color="#E74C3C",  # Red
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
    
    # Plot GT boundaries as vertical dashed lines
    for start, end in visible_segments:
        # Clip to visible range
        seg_start = max(start, start_sec)
        seg_end = min(end, end_sec)
        
        # Draw boundary lines (start and end)
        ax.axvline(
            seg_start,
            color="#2ECC71",  # Green
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            zorder=1,
        )
        ax.axvline(
            seg_end,
            color="#2ECC71",  # Green
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            zorder=1,
        )
        
        # Shade the segment region lightly
        ax.axvspan(
            seg_start, seg_end,
            color="#2ECC71",
            alpha=0.08,
            zorder=0,
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
        plt.Line2D([0], [0], color="#E74C3C", linewidth=2.5, label="Optical Flow Magnitude"),
        plt.Line2D([0], [0], color="#3498DB", linewidth=2.5, label="$E_{\\text{action}}$ (Quantized)"),
        plt.Line2D(
            [0], [0],
            color="#2ECC71",
            linestyle="--",
            linewidth=1.5,
            label="GT Action Boundaries",
        ),
    ]
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
    print(f"\n[STATS] Optical Flow:")
    print(f"  - Min: {of_min:.4f}, Max: {of_max:.4f}, Mean: {np.mean(of_values):.4f}")
    print(f"  - Std: {np.std(of_values):.4f}")
    print(f"\n[STATS] Action Energy:")
    print(f"  - Min: {ae_min:.4f}, Max: {ae_max:.4f}, Mean: {np.mean(ae_values):.4f}")
    print(f"  - Std: {np.std(ae_values):.4f}")
    print(f"\n[STATS] GT Segments in range: {len(visible_segments)}")
    print(f"[OK] Figure saved: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot Optical Flow vs Action Energy comparison for paper Figure 3"
    )
    ap.add_argument(
        "--optical-flow-jsonl",
        type=str,
        required=True,
        help="Path to optical flow energy JSONL file",
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
    
    args = ap.parse_args()
    
    plot_energy_comparison(
        optical_flow_jsonl=Path(args.optical_flow_jsonl),
        action_energy_jsonl=Path(args.action_energy_jsonl),
        gt_json=Path(args.gt_json),
        output_dir=Path(args.output_dir),
        start_sec=float(args.start_sec),
        duration_sec=float(args.duration_sec),
        stride=int(args.stride),
        target_fps=float(args.target_fps),
        dpi=int(args.dpi),
        figsize=(float(args.figsize_width), float(args.figsize_height)),
    )


if __name__ == "__main__":
    main()
