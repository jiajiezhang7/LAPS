from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import os


def cleanup_segment_and_codes(seg_current_path: Optional[Path]) -> None:
    """删除视频片段文件及其对应的 codes.json 文件（兼容旧/新路径）。"""
    if seg_current_path is None:
        return
    try:
        if os.path.isfile(seg_current_path):
            try:
                os.remove(seg_current_path)
                print(f"[Seg] DROP video segment: {seg_current_path}")
            except Exception as e:
                print(f"[Seg][WARN] Failed to remove video: {e}")
        # 删除 sidecar codes
        try:
            candidates = []
            candidates.append(seg_current_path.parent / f"{seg_current_path.stem}.codes.json")
            candidates.append(seg_current_path.parent.parent / "code_indices" / f"{seg_current_path.stem}.codes.json")
            for sidecar_path in candidates:
                if os.path.isfile(sidecar_path):
                    try:
                        os.remove(sidecar_path)
                        print(f"[Seg] DROP codes JSON: {sidecar_path}")
                    except Exception as e:
                        print(f"[Seg][WARN] Failed to remove codes JSON: {e}")
        except Exception:
            pass
    except Exception:
        pass


def should_save_codes(seg_current_path: Optional[Path]) -> bool:
    """检查是否应该保存 codes.json（视频文件必须存在）。"""
    try:
        return (seg_current_path is not None) and os.path.isfile(seg_current_path)
    except Exception:
        return False


def ensure_output_dirs(seg_output_dir: Path, video_name: str) -> Tuple[Path, Path]:
    """创建分割输出目录结构，返回 (seg_videos_dir, seg_codes_dir)。"""
    try:
        seg_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    video_specific_dir = seg_output_dir / video_name
    try:
        video_specific_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    seg_videos_dir = video_specific_dir / "segmented_videos"
    seg_codes_dir = video_specific_dir / "code_indices"
    try:
        seg_videos_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        seg_codes_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    print(f"[Seg] Output directories created for video '{video_name}':")
    print(f"[Seg]   Videos: {seg_videos_dir}")
    print(f"[Seg]   Codes:  {seg_codes_dir}")
    return seg_videos_dir, seg_codes_dir


def _select_non_overlapping_windows(
    window_frame_range_store: Dict[int, Tuple[int, int]],
    window_codes_store: Dict[int, List[int]],
    T: int,
    seg_start_frame: int,
    seg_end_frame: int,
    seg_codes_min_overlap_ratio: float,
) -> Tuple[List[int], List[List[int]]]:
    """基于片段帧区间选择互不重叠且重叠比例>=阈值的整窗 codes。"""
    candidates: List[Tuple[int, int, int, float]] = []
    s0 = int(seg_start_frame) if seg_start_frame >= 0 else None
    e0 = int(seg_end_frame) if seg_end_frame >= 0 else None
    for w_idx, (ws, we) in sorted(window_frame_range_store.items()):
        if w_idx not in window_codes_store:
            continue
        ss = int(ws) if s0 is None else max(int(ws), s0)
        ee = int(we) if e0 is None else min(int(we), e0)
        overlap = max(0, ee - ss + 1)
        ratio = overlap / float(int(T)) if int(T) > 0 else 0.0
        if overlap > 0 and ratio >= float(seg_codes_min_overlap_ratio):
            candidates.append((int(w_idx), int(ws), int(we), ratio))
    # 贪心选择互不重叠窗口（按起点排序）
    candidates.sort(key=lambda x: x[1])
    selected: List[int] = []
    last_end = -1
    for w_idx, ws, we, ratio in candidates:
        if ws > last_end:
            selected.append(w_idx)
            last_end = we
    codes_windows: List[List[int]] = [window_codes_store[w] for w in selected if w in window_codes_store]
    return selected, codes_windows


def export_codes_for_segment(
    seg_codes_dir: Path,
    seg_current_path: Path,
    window_frame_range_store: Dict[int, Tuple[int, int]],
    window_codes_store: Dict[int, List[int]],
    seg_start_frame: int,
    seg_end_frame: int,
    seg_start_win: int,
    T: int,
    stride: int,
    target_fps: int,
    seg_align: str,
    seg_codes_min_overlap_ratio: float,
    allow_overlap: bool = False,
) -> bool:
    """从缓存中筛选整窗 codes 并写入 sidecar JSON 文件。返回是否成功写入。"""
    try:
        if allow_overlap:
            # 收集所有满足重叠比例阈值的候选窗口（按起点排序），不做去重叠
            candidates = []
            s0 = int(seg_start_frame) if seg_start_frame >= 0 else None
            e0 = int(seg_end_frame) if seg_end_frame >= 0 else None
            for w_idx, (ws, we) in sorted(window_frame_range_store.items()):
                if w_idx not in window_codes_store:
                    continue
                ss = int(ws) if s0 is None else max(int(ws), s0)
                ee = int(we) if e0 is None else min(int(we), e0)
                overlap = max(0, ee - ss + 1)
                ratio = overlap / float(int(T)) if int(T) > 0 else 0.0
                if overlap > 0 and ratio >= float(seg_codes_min_overlap_ratio):
                    candidates.append((int(w_idx), int(ws), int(we), ratio))
            candidates.sort(key=lambda x: x[1])
            selected = [w for (w, _ws, _we, _r) in candidates]
            codes_windows = [window_codes_store[w] for w in selected if w in window_codes_store]
        else:
            # 原有逻辑：选择互不重叠的窗口
            selected, codes_windows = _select_non_overlapping_windows(
                window_frame_range_store=window_frame_range_store,
                window_codes_store=window_codes_store,
                T=T,
                seg_start_frame=seg_start_frame,
                seg_end_frame=seg_end_frame,
                seg_codes_min_overlap_ratio=seg_codes_min_overlap_ratio,
            )
        sidecar_path = seg_codes_dir / f"{seg_current_path.stem}.codes.json"
        meta = {
            "codes_windows": codes_windows,
            "selected_win_idxs": selected,
            "overlap_ratio_threshold": float(seg_codes_min_overlap_ratio),
            "segment": {
                "start_frame": int(seg_start_frame),
                "end_frame": int(seg_end_frame),
                "start_win": int(seg_start_win),
            },
            "window": {
                "T": int(T),
                "stride": int(stride),
                "target_fps": int(target_fps),
            },
            "align": str(seg_align),
            "video_segment_path": str(seg_current_path),
            "source": "stream_online",
            "allow_overlap": bool(allow_overlap),
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[Seg] CODES saved: {sidecar_path} (windows={len(codes_windows)})")
        return True
    except Exception as e:
        print(f"[Seg][WARN] failed to save codes JSON: {e}")
        return False
