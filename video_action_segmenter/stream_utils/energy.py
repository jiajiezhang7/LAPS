from typing import List, Optional, Tuple, Sequence, Union

import cv2
import numpy as np
import torch
from collections import deque


def _ensure_color(color: Optional[Union[Tuple[int, int, int], Sequence[int], str]], default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if color is None:
        return default
    if isinstance(color, str):
        name = color.strip().lower()
        color_map = {
            "yellow": (0, 255, 255),
            "light_yellow": (60, 240, 255),
            "orange": (0, 165, 255),
            "red": (0, 0, 255),
            "green": (0, 200, 0),
            "blue": (255, 0, 0),
            "purple": (255, 0, 255),
            "gray": (128, 128, 128),
        }
        return color_map.get(name, default)
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        try:
            return (int(color[0]), int(color[1]), int(color[2]))
        except Exception:
            return default
    return default


def _normalize_threshold_lines(
    threshold_lines: Optional[Sequence],
    default_color: Tuple[int, int, int],
    default_thickness: int = 2,
    default_style: str = "solid",
) -> List[dict]:
    if not threshold_lines:
        return []
    normalized = []
    for entry in threshold_lines:
        value = float("nan")
        label = ""
        color = default_color
        thickness = default_thickness
        style = default_style
        if isinstance(entry, dict):
            try:
                value = float(entry.get("value", float("nan")))
            except Exception:
                value = float("nan")
            label = str(entry.get("label", ""))
            color = _ensure_color(entry.get("color"), default_color)
            try:
                thickness = int(entry.get("thickness", default_thickness))
            except Exception:
                thickness = default_thickness
            style = str(entry.get("style", default_style)).lower()
        elif isinstance(entry, (tuple, list)):
            if len(entry) >= 1:
                try:
                    value = float(entry[0])
                except Exception:
                    value = float("nan")
            if len(entry) >= 2 and entry[1] is not None:
                label = str(entry[1])
            if len(entry) >= 3:
                color = _ensure_color(entry[2], default_color)
            if len(entry) >= 4 and entry[3] is not None:
                try:
                    thickness = int(entry[3])
                except Exception:
                    thickness = default_thickness
            if len(entry) >= 5 and entry[4] is not None:
                style = str(entry[4]).lower()
        else:
            try:
                value = float(entry)
            except Exception:
                continue

        if not np.isfinite(value):
            continue

        style_norm = style.lower()
        if style_norm in ("dashed", "dash", "--"):
            style_norm = "dashed"
        elif style_norm in ("dotted", "dot", ":", "."):
            style_norm = "dotted"
        else:
            style_norm = "solid"

        normalized.append(
            {
                "value": float(value),
                "label": label,
                "color": _ensure_color(color, default_color),
                "thickness": int(max(1, thickness)),
                "style": style_norm,
            }
        )
    return normalized


def _draw_styled_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    style: str = "solid",
    dash_length: int = 12,
    gap_length: int = 6,
):
    style = (style or "solid").lower()
    if style == "solid":
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        return

    p1 = np.array(pt1, dtype=np.float32)
    p2 = np.array(pt2, dtype=np.float32)
    diff = p2 - p1
    length = float(np.linalg.norm(diff))
    if length <= 0.5:
        return
    direction = diff / length

    if style == "dotted":
        radius = max(1, thickness // 2 + 1)
        step = max(1.0, float(dash_length) + float(gap_length))
        t = 0.0
        while t <= length:
            center = p1 + direction * t
            cv2.circle(img, tuple(np.round(center).astype(int)), radius, color, -1, cv2.LINE_AA)
            t += step
        return

    dash_len = max(1.0, float(dash_length))
    gap_len = max(1.0, float(gap_length))
    step = dash_len + gap_len
    t = 0.0
    while t < length:
        start = p1 + direction * t
        end = p1 + direction * min(length, t + dash_len)
        cv2.line(
            img,
            tuple(np.round(start).astype(int)),
            tuple(np.round(end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        t += step


def draw_energy_plot(
    values: List[float],
    width: int = 600,
    height: int = 200,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Render a simple line plot of 1D energy values using OpenCV."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if values is None or len(values) == 0:
        return img
    vals = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vals).any():
        return img

    vmin = float(np.nanmin(vals)) if y_min is None else float(y_min)
    vmax = float(np.nanmax(vals)) if y_max is None else float(y_max)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.05
    vmin_p, vmax_p = vmin - pad, vmax + pad

    n = len(vals)
    xs = np.linspace(0, width - 1, n, dtype=np.int32)
    ys_float = (1.0 - (vals - vmin_p) / max(1e-6, (vmax_p - vmin_p))) * (height - 1)
    ys = np.clip(ys_float, 0, height - 1).astype(np.int32)

    grid_color = (60, 60, 60)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (90, 90, 90), 1)
    for gx in range(0, width, max(1, width // 10)):
        cv2.line(img, (gx, 0), (gx, height - 1), grid_color, 1)
    for gy in range(0, height, max(1, height // 5)):
        cv2.line(img, (0, gy), (width - 1, gy), grid_color, 1)

    for i in range(1, n):
        cv2.line(img, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), color, thickness)

    try:
        cur = float(vals[-1])
        cv2.putText(img, f"E={cur:.4f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(img, f"[{vmin:.4f},{vmax:.4f}]", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    except Exception:
        pass

    return img


def _l2_mean_over_tokens(Z: torch.Tensor) -> float:
    Z2 = Z
    if Z2.dim() == 3 and Z2.size(0) == 1:
        Z2 = Z2.squeeze(0)
    return float(torch.linalg.vector_norm(Z2, dim=-1).mean().item())


def _token_diff_l2_mean(Z: torch.Tensor) -> float:
    Z2 = Z
    if Z2.dim() == 3 and Z2.size(0) == 1:
        Z2 = Z2.squeeze(0)
    if Z2.size(0) >= 2:
        return float(torch.linalg.vector_norm(Z2[1:] - Z2[:-1], dim=-1).mean().item())
    else:
        return float(torch.linalg.vector_norm(Z2, dim=-1).mean().item())


def _velocity_l2_mean(V: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(V, dim=-1).mean().item())


def compute_energy(
    source: str,
    mode: str,
    to_quantize: Optional[torch.Tensor] = None,
    quantized: Optional[torch.Tensor] = None,
    vel_norm: Optional[torch.Tensor] = None,
) -> Optional[float]:
    """Compute energy value given the configured source and mode.

    - source: 'prequant' | 'quantized' | 'velocity'
    - mode: 'l2_mean' | 'token_diff_l2_mean'
    - to_quantize: (1, d, D) or (d, D)
    - quantized: (1, d, D) or (d, D)
    - vel_norm: (T-1, N, 2)
    """
    try:
        source = str(source).lower()
        mode = str(mode).lower()
        if source == "prequant" and to_quantize is not None:
            Z = to_quantize
            if mode == "l2_mean":
                return _l2_mean_over_tokens(Z)
            elif mode == "token_diff_l2_mean":
                return _token_diff_l2_mean(Z)
            else:
                return _l2_mean_over_tokens(Z)
        elif source == "quantized" and quantized is not None:
            Q = quantized
            if mode == "l2_mean":
                return _l2_mean_over_tokens(Q)
            elif mode == "token_diff_l2_mean":
                return _token_diff_l2_mean(Q)
            else:
                return _l2_mean_over_tokens(Q)
        elif source == "velocity" and vel_norm is not None:
            return _velocity_l2_mean(vel_norm)
    except Exception:
        return None
    return None


class CausalSmoother1D:
    """Stateful causal smoother for streaming 1D sequences.

    Supports:
    - EMA: y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    - MA (causal window): y[t] = mean(x[max(0,t-w+1):t+1])
    """

    def __init__(self, method: str = "ema", alpha: float = 0.4, window: int = 3):
        method = str(method).lower()
        if method not in ("ema", "ma"):
            method = "ema"
        self.method = method
        self.alpha = float(alpha)
        self.window = max(1, int(window))

        # Internal state
        self._ema: Optional[float] = None
        self._buf = deque()
        self._sum = 0.0

    def reset(self):
        self._ema = None
        self._buf.clear()
        self._sum = 0.0

    def update(self, x: float) -> float:
        xv = float(x)
        if self.method == "ema":
            if self._ema is None:
                self._ema = xv
            else:
                a = float(self.alpha)
                if not np.isfinite(a):
                    a = 0.4
                a = float(np.clip(a, 0.0, 1.0))
                self._ema = a * xv + (1.0 - a) * self._ema
            return float(self._ema)
        else:
            self._buf.append(xv)
            self._sum += xv
            if len(self._buf) > self.window:
                self._sum -= self._buf.popleft()
            return float(self._sum / max(1, len(self._buf)))


def apply_smoothing_1d(values: Sequence[float], method: str = "ema", alpha: float = 0.4, window: int = 3) -> np.ndarray:
    """Apply causal smoothing to a 1D numeric sequence and return float32 array.

    - EMA: causal exponential moving average
    - MA: causal moving average (window)
    """
    if values is None:
        return np.asarray([], dtype=np.float32)
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v.astype(np.float32)
    m = str(method).lower()
    if m == "ema":
        a = float(alpha)
        if not np.isfinite(a):
            a = 0.4
        a = float(np.clip(a, 0.0, 1.0))
        out = np.empty_like(v, dtype=np.float64)
        ema = float(v[0])
        out[0] = ema
        for i in range(1, v.size):
            ema = a * float(v[i]) + (1.0 - a) * ema
            out[i] = ema
        return out.astype(np.float32)
    elif m == "ma":
        w = max(1, int(window))
        out = np.empty_like(v, dtype=np.float64)
        cs = np.cumsum(v)
        for i in range(v.size):
            j = max(0, i - w + 1)
            s = cs[i] - (cs[j - 1] if j > 0 else 0.0)
            out[i] = s / float(i - j + 1)
        return out.astype(np.float32)
    else:
        return v.astype(np.float32)


def draw_energy_plot_two(
    raw_values: List[float],
    smooth_values: List[float],
    width: int = 600,
    height: int = 200,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    color_raw: Tuple[int, int, int] = (0, 255, 0),
    color_smooth: Tuple[int, int, int] = (0, 165, 255),
    thickness_raw: int = 1,
    thickness_smooth: int = 2,
) -> np.ndarray:
    """Render two overlaid line plots (raw vs smoothed) using OpenCV."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if (raw_values is None or len(raw_values) == 0) and (smooth_values is None or len(smooth_values) == 0):
        return img
    r = np.asarray(raw_values if raw_values is not None else [], dtype=np.float32)
    s = np.asarray(smooth_values if smooth_values is not None else [], dtype=np.float32)
    n = int(max(len(r), len(s)))
    if n <= 0:
        return img
    # right-align to same length
    if len(r) < n:
        r = np.pad(r, (n - len(r), 0), mode='edge') if len(r) > 0 else np.zeros((n,), dtype=np.float32)
    if len(s) < n:
        s = np.pad(s, (n - len(s), 0), mode='edge') if len(s) > 0 else np.zeros((n,), dtype=np.float32)

    vals = np.concatenate([r, s]) if len(r) > 0 and len(s) > 0 else (r if len(r) > 0 else s)
    vmin = float(np.nanmin(vals)) if y_min is None and vals.size > 0 else float(y_min if y_min is not None else 0.0)
    vmax = float(np.nanmax(vals)) if y_max is None and vals.size > 0 else float(y_max if y_max is not None else 1.0)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.05
    vmin_p, vmax_p = vmin - pad, vmax + pad

    xs = np.linspace(0, width - 1, n, dtype=np.int32)
    def to_y(a: np.ndarray) -> np.ndarray:
        ys_float = (1.0 - (a - vmin_p) / max(1e-6, (vmax_p - vmin_p))) * (height - 1)
        return np.clip(ys_float, 0, height - 1).astype(np.int32)

    yr = to_y(r) if len(r) > 0 else None
    ys2 = to_y(s) if len(s) > 0 else None

    grid_color = (60, 60, 60)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (90, 90, 90), 1)
    for gx in range(0, width, max(1, width // 10)):
        cv2.line(img, (gx, 0), (gx, height - 1), grid_color, 1)
    for gy in range(0, height, max(1, height // 5)):
        cv2.line(img, (0, gy), (width - 1, gy), grid_color, 1)

    if yr is not None and len(yr) > 1:
        for i in range(1, n):
            cv2.line(img, (xs[i - 1], yr[i - 1]), (xs[i], yr[i]), color_raw, thickness_raw)
    if ys2 is not None and len(ys2) > 1:
        for i in range(1, n):
            cv2.line(img, (xs[i - 1], ys2[i - 1]), (xs[i], ys2[i]), color_smooth, thickness_smooth)

    try:
        cur_r = float(r[-1]) if len(r) > 0 else None
        cur_s = float(s[-1]) if len(s) > 0 else None
        if cur_r is not None:
            cv2.putText(img, f"E_raw={cur_r:.4f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
        if cur_s is not None:
            cv2.putText(img, f"E_smooth={cur_s:.4f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv2.LINE_AA)
    except Exception:
        pass

    return img


# ==================== Enhanced Visualization for Academic Papers ====================

def draw_energy_plot_enhanced(
    values: List[float],
    width: int = 720,
    height: int = 520,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    theme: str = "academic_blue",
    show_grid: bool = True,
    show_labels: bool = True,
    show_statistics: bool = True,
    line_width: int = 2,
    title: str = "Energy Curve",
    threshold_lines: Optional[Sequence] = None,
) -> np.ndarray:
    """Enhanced energy plot visualization suitable for academic papers.
    
    Args:
        values: List of energy values to plot
        width: Plot width in pixels
        height: Plot height in pixels  
        y_min: Minimum y-axis value (auto if None)
        y_max: Maximum y-axis value (auto if None)
        theme: Color theme ('academic_blue', 'nature_green', 'ieee_red', 'minimal_gray')
        show_grid: Whether to show grid lines
        show_labels: Whether to show axis labels and ticks
        show_statistics: Whether to show current value and statistics
        line_width: Width of the energy curve line
        title: Plot title
        
    Returns:
        np.ndarray: Rendered plot image (H, W, 3) in BGR format
    """
    # Define color themes suitable for academic papers
    themes = {
        "academic_blue": {
            "bg": (250, 250, 250),           # Light gray background
            "line": (180, 100, 50),          # Academic blue
            "grid": (220, 220, 220),         # Light grid
            "axis": (100, 100, 100),         # Dark gray axes
            "text": (50, 50, 50),            # Dark text
            "accent": (220, 150, 80),        # Accent color
        },
        "nature_green": {
            "bg": (252, 252, 252),           # Almost white
            "line": (100, 150, 50),          # Nature green
            "grid": (230, 230, 230),         # Very light grid
            "axis": (80, 80, 80),            # Medium gray axes
            "text": (40, 40, 40),            # Very dark text
            "accent": (150, 180, 70),        # Light green accent
        },
        "ieee_red": {
            "bg": (248, 248, 248),           # Light background
            "line": (50, 50, 180),           # IEEE red
            "grid": (225, 225, 225),         # Light grid
            "axis": (90, 90, 90),            # Gray axes
            "text": (60, 60, 60),            # Dark gray text
            "accent": (80, 80, 200),         # Light red accent
        },
        "minimal_gray": {
            "bg": (255, 255, 255),           # Pure white
            "line": (80, 80, 80),            # Dark gray line
            "grid": (240, 240, 240),         # Very light grid
            "axis": (120, 120, 120),         # Medium gray axes
            "text": (60, 60, 60),            # Dark text
            "accent": (140, 140, 140),       # Medium gray accent
        }
    }
    
    # Get theme colors
    if theme not in themes:
        theme = "academic_blue"
    colors = themes[theme]
    
    # Create image with background color
    img = np.full((height, width, 3), colors["bg"], dtype=np.uint8)
    
    if not values or len(values) == 0:
        return img
        
    vals = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vals).any():
        return img
    
    # Calculate plot area (leave margins for labels)
    margin_left = 90 if show_labels else 24
    margin_right = 30
    margin_top = 50 if title else 24
    margin_bottom = 70 if show_labels else 24
    
    plot_x = margin_left
    plot_y = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    
    # Calculate value range
    vmin = float(np.nanmin(vals)) if y_min is None else float(y_min)
    vmax = float(np.nanmax(vals)) if y_max is None else float(y_max)
    
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0
    
    # Add small padding to range
    vrange = vmax - vmin
    pad = 0.05 * vrange if vrange > 0 else 0.1
    vmin_plot = vmin - pad
    vmax_plot = vmax + pad
    
    # Draw plot border
    cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), colors["axis"], 2)
    
    # Draw grid
    if show_grid:
        # Vertical grid lines
        for i in range(1, 10):
            x = plot_x + int(i * plot_w / 10)
            cv2.line(img, (x, plot_y), (x, plot_y + plot_h), colors["grid"], 1)
        
        # Horizontal grid lines  
        for i in range(1, 6):
            y = plot_y + int(i * plot_h / 6)
            cv2.line(img, (plot_x, y), (plot_x + plot_w, y), colors["grid"], 1)
    
    # Draw Y-axis ticks and labels
    if show_labels:
        for i in range(7):  # 7 ticks including endpoints
            y_pos = plot_y + int(i * plot_h / 6)
            tick_val = vmax_plot - (i / 6.0) * (vmax_plot - vmin_plot)

            # Tick mark
            cv2.line(img, (plot_x - 10, y_pos), (plot_x, y_pos), colors["axis"], 2)
            
            # Tick label
            label = f"{tick_val:.3f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(img, label, 
                       (plot_x - text_size[0] - 14, y_pos + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors["text"], 1, cv2.LINE_AA)
        
        # X-axis ticks and labels
        n_ticks = min(8, len(vals))
        for i in range(n_ticks + 1):
            x_pos = plot_x + int(i * plot_w / n_ticks)
            tick_val = int(i * len(vals) / n_ticks)
            
            # Tick mark
            cv2.line(img, (x_pos, plot_y + plot_h), (x_pos, plot_y + plot_h + 8), colors["axis"], 2)
            
            # Tick label
            label = str(tick_val)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(img, label,
                       (x_pos - text_size[0] // 2, plot_y + plot_h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors["text"], 1, cv2.LINE_AA)
        
        # Axis labels
        cv2.putText(img, "Energy", (18, plot_y + plot_h // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["text"], 1, cv2.LINE_AA)
        
        text_size = cv2.getTextSize("Time Window", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        label_x_pos = plot_x + (plot_w - text_size[0]) // 2
        cv2.putText(img, "Time Window", (label_x_pos, plot_y + plot_h + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["text"], 1, cv2.LINE_AA)
    
    # Draw title
    if title:
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        title_x = (width - text_size[0]) // 2
        cv2.putText(img, title, (title_x, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors["text"], 2, cv2.LINE_AA)
    
    # Convert values to plot coordinates and draw curve
    n = len(vals)
    if n > 1:
        xs = np.linspace(plot_x, plot_x + plot_w, n)
        ys = plot_y + plot_h - ((vals - vmin_plot) / (vmax_plot - vmin_plot)) * plot_h
        
        # Clip y coordinates to plot area
        ys = np.clip(ys, plot_y, plot_y + plot_h)
        
        # Draw the energy curve with anti-aliasing
        points = np.column_stack([xs, ys]).astype(np.int32)
        
        # Draw line segments with thickness
        for i in range(len(points) - 1):
            cv2.line(img, tuple(points[i]), tuple(points[i + 1]), 
                    colors["line"], line_width, cv2.LINE_AA)
        
        # Draw accent points for better visibility (sample to avoid clutter)
        sample_step = max(1, len(points) // 50)
        for i in range(0, len(points), sample_step):
            cv2.circle(img, tuple(points[i]), max(1, line_width // 2), 
                      colors["accent"], -1, cv2.LINE_AA)
    
    # Draw threshold lines (if provided)
    threshold_info = _normalize_threshold_lines(threshold_lines, colors.get("accent", (180, 100, 50)))
    if threshold_info:
        for idx, info in enumerate(threshold_info):
            thr_val = info["value"]
            if vmax_plot == vmin_plot:
                continue
            y_thr = plot_y + plot_h - ((thr_val - vmin_plot) / (vmax_plot - vmin_plot)) * plot_h
            y_thr = float(np.clip(y_thr, plot_y, plot_y + plot_h))
            y_int = int(round(y_thr))
            _draw_styled_line(
                img,
                (plot_x, y_int),
                (plot_x + plot_w, y_int),
                color=info["color"],
                thickness=info["thickness"],
                style=info["style"],
            )
            label = info.get("label", "")
            if label:
                text = f"{label}: {thr_val:.3f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                text_x = plot_x + plot_w - text_size[0] - 10
                text_y = y_int - 6 - idx * (text_size[1] + 6)
                if text_y < plot_y + 15:
                    text_y = int(min(plot_y + plot_h - 10, y_thr + (idx + 1) * (text_size[1] + 6)))
                cv2.putText(
                    img,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    colors["text"],
                    1,
                    cv2.LINE_AA,
                )

    # Draw statistics box
    if show_statistics and len(vals) > 0:
        stats_x = plot_x + plot_w - 200
        stats_y = plot_y + 20
        
        current_val = float(vals[-1])
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        
        stats_text = [
            f"Current: {current_val:.4f}",
            f"Mean: {mean_val:.4f}",
            f"Std: {std_val:.4f}",
            f"Range: [{vmin:.3f}, {vmax:.3f}]"
        ]
        
        # Draw semi-transparent background for stats
        stats_bg_x1 = stats_x - 10
        stats_bg_y1 = stats_y - 15
        stats_bg_x2 = stats_x + 180
        stats_bg_y2 = stats_y + len(stats_text) * 18 + 5
        
        overlay = img.copy()
        cv2.rectangle(overlay, (stats_bg_x1, stats_bg_y1), (stats_bg_x2, stats_bg_y2), 
                     colors["bg"], -1)
        cv2.rectangle(overlay, (stats_bg_x1, stats_bg_y1), (stats_bg_x2, stats_bg_y2), 
                     colors["grid"], 1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        
        # Draw statistics text
        for i, text in enumerate(stats_text):
            cv2.putText(img, text, (stats_x, stats_y + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors["text"], 1, cv2.LINE_AA)
    
    return img


def draw_energy_plot_enhanced_dual(
    raw_values: List[float],
    smooth_values: List[float],
    width: int = 720,
    height: int = 520,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    theme: str = "academic_blue",
    show_grid: bool = True,
    show_labels: bool = True,
    show_legend: bool = True,
    show_statistics: bool = True,
    line_width_raw: int = 2,
    line_width_smooth: int = 2,
    title: str = "Energy Curve (Raw vs Smoothed)",
    threshold_lines: Optional[Sequence] = None,
) -> np.ndarray:
    """Enhanced dual energy plot visualization for raw and smoothed curves."""
    
    # Define color themes with dual line support
    themes = {
        "academic_blue": {
            "bg": (250, 250, 250),
            "line_raw": (200, 200, 100),      # Light blue for raw
            "line_smooth": (180, 100, 50),    # Darker blue for smooth
            "grid": (220, 220, 220),
            "axis": (100, 100, 100),
            "text": (50, 50, 50),
        },
        "nature_green": {
            "bg": (252, 252, 252),
            "line_raw": (150, 200, 100),      # Light green for raw
            "line_smooth": (100, 150, 50),    # Darker green for smooth
            "grid": (230, 230, 230),
            "axis": (80, 80, 80),
            "text": (40, 40, 40),
        },
        "ieee_red": {
            "bg": (248, 248, 248),
            "line_raw": (100, 100, 200),      # Light red for raw
            "line_smooth": (50, 50, 180),     # Darker red for smooth
            "grid": (225, 225, 225),
            "axis": (90, 90, 90),
            "text": (60, 60, 60),
        },
    }
    
    if theme not in themes:
        theme = "academic_blue"
    colors = themes[theme]
    
    # Create image with background color
    img = np.full((height, width, 3), colors["bg"], dtype=np.uint8)
    
    if (not raw_values or len(raw_values) == 0) and (not smooth_values or len(smooth_values) == 0):
        return img
    
    # Process input arrays
    r = np.asarray(raw_values if raw_values else [], dtype=np.float32)
    s = np.asarray(smooth_values if smooth_values else [], dtype=np.float32)
    
    # Determine the length and align arrays
    n = max(len(r), len(s))
    if n == 0:
        return img
    
    # Right-align arrays to same length
    if len(r) < n and len(r) > 0:
        r = np.pad(r, (n - len(r), 0), mode='edge')
    elif len(r) == 0:
        r = np.zeros(n, dtype=np.float32)
        
    if len(s) < n and len(s) > 0:
        s = np.pad(s, (n - len(s), 0), mode='edge')
    elif len(s) == 0:
        s = np.zeros(n, dtype=np.float32)
    
    # Calculate plot area
    margin_left = 90 if show_labels else 24
    margin_right = 30
    margin_top = 55 if title else 28
    margin_bottom = 90 if (show_labels and show_legend) else (65 if show_labels else 28)
    
    plot_x = margin_left
    plot_y = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    
    # Calculate value range from both arrays
    all_vals = np.concatenate([r[np.isfinite(r)], s[np.isfinite(s)]])
    if len(all_vals) == 0:
        return img
        
    vmin = float(np.min(all_vals)) if y_min is None else float(y_min)
    vmax = float(np.max(all_vals)) if y_max is None else float(y_max)
    
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    
    # Add padding
    vrange = vmax - vmin
    pad = 0.05 * vrange
    vmin_plot = vmin - pad
    vmax_plot = vmax + pad
    
    # Draw plot border
    cv2.rectangle(img, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), colors["axis"], 2)
    
    # Draw grid
    if show_grid:
        for i in range(1, 10):
            x = plot_x + int(i * plot_w / 10)
            cv2.line(img, (x, plot_y), (x, plot_y + plot_h), colors["grid"], 1)
        for i in range(1, 6):
            y = plot_y + int(i * plot_h / 6)
            cv2.line(img, (plot_x, y), (plot_x + plot_w, y), colors["grid"], 1)

    # Draw axis ticks and labels
    if show_labels:
        # Y-axis ticks and labels
        for i in range(7):
            y_pos = plot_y + int(i * plot_h / 6)
            tick_val = vmax_plot - (i / 6.0) * (vmax_plot - vmin_plot)

            cv2.line(img, (plot_x - 10, y_pos), (plot_x, y_pos), colors["axis"], 2)

            label = f"{tick_val:.3f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(
                img,
                label,
                (plot_x - text_size[0] - 14, y_pos + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colors["text"],
                1,
                cv2.LINE_AA,
            )

        # X-axis ticks and labels
        n_ticks = min(8, n)
        if n_ticks > 0:
            for i in range(n_ticks + 1):
                x_pos = plot_x + int(i * plot_w / n_ticks)
                tick_val = int(i * n / n_ticks)

                cv2.line(img, (x_pos, plot_y + plot_h), (x_pos, plot_y + plot_h + 8), colors["axis"], 2)

                label = str(tick_val)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.putText(
                    img,
                    label,
                    (x_pos - text_size[0] // 2, plot_y + plot_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    colors["text"],
                    1,
                    cv2.LINE_AA,
                )

        # Axis labels
        text_size = cv2.getTextSize("Time Window", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        label_x_pos = plot_x + (plot_w - text_size[0]) // 2
        cv2.putText(
            img,
            "Time Window",
            (label_x_pos, plot_y + plot_h + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colors["text"],
            1,
            cv2.LINE_AA,
        )
    
    # Draw title
    if title:
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        title_x = (width - text_size[0]) // 2
        cv2.putText(img, title, (title_x, 34),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors["text"], 2, cv2.LINE_AA)
    
    # Draw curves
    if n > 1:
        xs = np.linspace(plot_x, plot_x + plot_w, n)
        
        # Raw curve
        if len(raw_values) > 0:
            ys_r = plot_y + plot_h - ((r - vmin_plot) / (vmax_plot - vmin_plot)) * plot_h
            ys_r = np.clip(ys_r, plot_y, plot_y + plot_h)
            points_r = np.column_stack([xs, ys_r]).astype(np.int32)
            
            for i in range(len(points_r) - 1):
                cv2.line(img, tuple(points_r[i]), tuple(points_r[i + 1]), 
                        colors["line_raw"], line_width_raw, cv2.LINE_AA)
        
        # Smooth curve
        if len(smooth_values) > 0:
            ys_s = plot_y + plot_h - ((s - vmin_plot) / (vmax_plot - vmin_plot)) * plot_h
            ys_s = np.clip(ys_s, plot_y, plot_y + plot_h)
            points_s = np.column_stack([xs, ys_s]).astype(np.int32)
            
            for i in range(len(points_s) - 1):
                cv2.line(img, tuple(points_s[i]), tuple(points_s[i + 1]), 
                        colors["line_smooth"], line_width_smooth, cv2.LINE_AA)
    
    # Draw threshold lines (if provided)
    threshold_info = _normalize_threshold_lines(threshold_lines, colors.get("axis", (100, 100, 100)))
    if threshold_info:
        for idx, info in enumerate(threshold_info):
            thr_val = info["value"]
            if vmax_plot == vmin_plot:
                continue
            y_thr = plot_y + plot_h - ((thr_val - vmin_plot) / (vmax_plot - vmin_plot)) * plot_h
            y_thr = float(np.clip(y_thr, plot_y, plot_y + plot_h))
            y_int = int(round(y_thr))
            _draw_styled_line(
                img,
                (plot_x, y_int),
                (plot_x + plot_w, y_int),
                color=info["color"],
                thickness=info["thickness"],
                style=info["style"],
            )
            label = info.get("label", "")
            if label:
                text = f"{label}: {thr_val:.3f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                text_x = plot_x + plot_w - text_size[0] - 10
                text_y = y_int - 6 - idx * (text_size[1] + 6)
                if text_y < plot_y + 15:
                    text_y = int(min(plot_y + plot_h - 10, y_thr + (idx + 1) * (text_size[1] + 6)))
                cv2.putText(
                    img,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    colors.get("text", (60, 60, 60)),
                    1,
                    cv2.LINE_AA,
                )

    # Draw legend
    if show_legend:
        legend_x = plot_x + 20
        legend_y = plot_y + plot_h + 55
        
        # Raw line legend
        if len(raw_values) > 0:
            cv2.line(img, (legend_x, legend_y), (legend_x + 30, legend_y), 
                    colors["line_raw"], line_width_raw, cv2.LINE_AA)
            cv2.putText(img, "Raw", (legend_x + 40, legend_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors["text"], 1, cv2.LINE_AA)
        
        # Smooth line legend
        if len(smooth_values) > 0:
            legend_x_smooth = legend_x + 150
            cv2.line(img, (legend_x_smooth, legend_y), (legend_x_smooth + 30, legend_y), 
                    colors["line_smooth"], line_width_smooth, cv2.LINE_AA)
            cv2.putText(img, "Smoothed", (legend_x_smooth + 40, legend_y + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors["text"], 1, cv2.LINE_AA)
    
    return img
