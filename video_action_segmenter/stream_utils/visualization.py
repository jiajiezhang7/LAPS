from typing import Deque, List, Tuple
import cv2

from .energy import (
    draw_energy_plot,
    draw_energy_plot_two,
    draw_energy_plot_enhanced,
    draw_energy_plot_enhanced_dual,
)


def render_and_handle_windows(
    *,
    frame,
    window_ok: bool,
    energy_window_ok: bool,
    visualize: bool,
    show_overlay: bool,
    window_title: str,
    seg_enable: bool,
    seg_active: bool,
    seg_threshold: float,
    avg_track: float,
    avg_forward: float,
    windows_done: int,
    T: int,
    stride: int,
    grid_size: int,
    energy_window_title: str,
    energy_viz_style: str,
    smoothing_enable: bool,
    smoothing_visualize_both: bool,
    energy_ring: Deque[float],
    energy_smooth_ring: Deque[float],
    energy_plot_w: int,
    energy_plot_h: int,
    energy_y_min,
    energy_y_max,
    energy_theme: str,
    display_fps: int,
    energy_color,
) -> Tuple[bool, bool, bool]:
    """Render main and energy windows, return (window_ok, energy_window_ok, quit_flag).
    Mirrors the logic in stream_inference.py without altering behavior or visuals.
    """
    if not visualize or not window_ok:
        return window_ok, energy_window_ok, False

    try:
        disp = frame.copy()
        if show_overlay:
            y0, dy = 24, 20
            cv2.putText(disp, f"T={T} stride={stride} grid={grid_size}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
            cv2.putText(disp, f"win={windows_done} track(EMA)={avg_track:.3f}s fwd(EMA)={avg_forward:.3f}s", (10, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1, cv2.LINE_AA)
            if seg_enable:
                seg_color = (0, 0, 255) if seg_active else (180, 180, 180)
                cv2.putText(disp, f"SEG={'ON' if seg_active else 'OFF'} thr={seg_threshold:.3f}", (10, y0+2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, seg_color, 1, cv2.LINE_AA)
        cv2.imshow(window_title, disp)

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

        key = cv2.waitKey(max(1, int(1000/max(1, display_fps)))) & 0xFF
        if key == ord('q'):
            return window_ok, energy_window_ok, True
        return window_ok, energy_window_ok, False
    except Exception:
        # degrade to headless mode on any GUI error
        return False, False, False

