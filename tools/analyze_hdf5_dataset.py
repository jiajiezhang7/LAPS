import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
from h5py import h5s, h5t
import numpy as np

try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover
    yaml = None


VIDEO_EXTS = [
    ".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV",
]


def load_cfg(cfg_path: Optional[str]) -> Dict:
    if not cfg_path:
        return {}
    if yaml is None:
        return {}
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_h5_files(root_dir: str, max_files: Optional[int] = None) -> List[str]:
    all_paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.hdf5"), recursive=True))
    if max_files is not None and max_files > 0:
        all_paths = all_paths[: max_files]
    return all_paths


def _rel_path_after_root(path: str, root_dir: str) -> str:
    try:
        return os.path.relpath(path, start=root_dir)
    except Exception:
        return os.path.basename(path)


def find_video_path(h5_path: str, root_dir: str, video_root: Optional[str]) -> Optional[str]:
    stem = Path(h5_path).stem
    h5_dir = Path(h5_path).parent

    for ext in VIDEO_EXTS:
        cand = h5_dir / f"{stem}{ext}"
        if cand.exists():
            return str(cand)

    if video_root:
        try:
            rel = os.path.relpath(h5_path, start=root_dir)
        except ValueError:
            rel = Path(h5_path).name

        for ext in VIDEO_EXTS:
            cand = Path(video_root) / Path(rel)
            cand = cand.with_suffix(ext)
            if cand.exists():
                return str(cand)

        pattern = f"**/{stem}*"
        try:
            hits = sorted(Path(video_root).glob(pattern))
        except Exception:
            hits = []
        for p in hits:
            if p.suffix in VIDEO_EXTS and p.is_file():
                return str(p)

    return None


def sample_window_indices(T: int, k: int) -> List[int]:
    if T <= 0:
        return []
    if k >= T:
        return list(range(T))
    # spread roughly uniformly
    step = max(1, T // k)
    idxs = list(range(0, T, step))[:k]
    if len(idxs) == 0:
        idxs = [0]
    return idxs


def summarize_velocities(tracks_win: np.ndarray, movement_threshold_px: float = 0.5) -> Dict[str, float]:
    # tracks_win: (1, horizon, N, 2) or (horizon, N, 2)
    x = tracks_win
    if x.ndim == 4:
        x = x[0]
    # (horizon, N, 2)
    if x.shape[0] < 2:
        return {
            "median_vel_px": 0.0,
            "nonzero_step_frac": 0.0,
            "oob_frac": 0.0,
            "nan_frac": 0.0,
        }
    diffs = np.diff(x, axis=0)  # (h-1, N, 2)
    mags = np.linalg.norm(diffs, axis=-1)  # (h-1, N)
    median_per_step = np.median(mags, axis=1)  # (h-1,)
    median_vel_px = float(np.median(median_per_step))
    nonzero_step_frac = float(np.mean(median_per_step > movement_threshold_px))
    nan_frac = float(np.mean(np.isnan(x)))
    # oob_frac to be set outside when img_size is known
    return {
        "median_vel_px": median_vel_px,
        "nonzero_step_frac": nonzero_step_frac,
        "oob_frac": 0.0,
        "nan_frac": nan_frac,
    }


def is_oob(tracks_win: np.ndarray, img_h: int, img_w: int) -> float:
    x = tracks_win
    if x.ndim == 4:
        x = x[0]
    if img_h <= 0 or img_w <= 0:
        return 0.0
    y = x[..., 0]
    c = x[..., 1]
    bad = (y < 0) | (y >= img_h) | (c < 0) | (c >= img_w)
    return float(np.mean(bad))


def analyze_h5_file(
    h5_path: str,
    sample_windows: int,
    movement_threshold_px: float,
    root_dir: str,
    video_root: Optional[str],
) -> Dict:
    info: Dict = {
        "file": h5_path,
        "views": [],
        "T": None,
        "horizon": None,
        "N": None,
        "has_vis": False,
        "img_size": None,
        "video_found": False,
        "sample": {
            "median_vel_px": [],
            "nonzero_step_frac": [],
            "oob_frac": [],
            "nan_frac": [],
        },
    }
    try:
        with h5py.File(h5_path, "r") as f:
            if "root" not in f:
                return info
            views = list(f["root"].keys())
            if len(views) == 0:
                return info
            info["views"] = views
            use_view = views[0]
            grp = f[f"root/{use_view}"]
            if "tracks" not in grp:
                return info
            dset = grp["tracks"]
            shape = dset.shape  # (T, horizon, N, 2)
            if len(shape) != 4 or shape[-1] != 2:
                return info
            T, horizon, N, _ = shape
            info["T"], info["horizon"], info["N"] = int(T), int(horizon), int(N)
            info["has_vis"] = "vis" in grp
            try:
                h_attr = int(grp.attrs.get("height"))
                w_attr = int(grp.attrs.get("width"))
                if h_attr > 0 and w_attr > 0:
                    info["img_size"] = [h_attr, w_attr]
            except Exception:
                pass

            idxs = sample_window_indices(T, sample_windows)
            for t0 in idxs:
                # try high-level slice; fallback to low-level HDF5 read with explicit IEEE_F32
                try:
                    win = dset[[t0]]  # (1, horizon, N, 2)
                except Exception:
                    # Low-level fallback: read into preallocated float32 via HDF5 API
                    file_start = (int(t0), 0, 0, 0)
                    file_count = (1, int(horizon), int(N), 2)
                    file_space = dset.id.get_space()
                    try:
                        file_space.select_hyperslab(start=file_start, count=file_count)
                    except TypeError:
                        file_space.select_hyperslab(file_start, file_count)
                    mem_space = h5s.create_simple(file_count)
                    win = np.empty(file_count, dtype=np.float32)
                    try:
                        dset.id.read(mem_space, file_space, win, mtype=h5t.IEEE_F32LE)
                    except TypeError:
                        dset.id.read(mem_space, file_space, win)
                s = summarize_velocities(win, movement_threshold_px)
                if info["img_size"] is not None:
                    s["oob_frac"] = is_oob(win, info["img_size"][0], info["img_size"][1])
                for k, v in s.items():
                    info["sample"][k].append(float(v))
    except Exception:
        return info

    # video mapping check (best-effort)
    vpath = find_video_path(h5_path, root_dir, video_root)
    info["video_found"] = bool(vpath and os.path.exists(vpath))
    return info


def aggregate_reports(reports: List[Dict]) -> Dict:
    agg: Dict = {}
    valid = [r for r in reports if r.get("T") is not None]
    agg["num_files_scanned"] = len(reports)
    agg["num_files_valid"] = len(valid)

    # basic distributions
    horizons = Counter([r["horizon"] for r in valid if r.get("horizon") is not None])
    tracksN = Counter([r["N"] for r in valid if r.get("N") is not None])
    views = Counter([v for r in valid for v in r.get("views", [])])
    img_sizes = Counter([tuple(r["img_size"]) for r in valid if r.get("img_size") is not None])

    agg["horizon_dist"] = dict(horizons)
    agg["tracksN_dist"] = dict(tracksN)
    agg["view_names_dist"] = dict(views)
    agg["img_sizes_dist"] = {f"{k[0]}x{k[1]}": v for k, v in img_sizes.items()}

    # totals
    total_T = sum([r["T"] for r in valid])
    agg["total_rollout_T"] = int(total_T)
    agg["avg_T_per_file"] = float(total_T / max(len(valid), 1))

    # vis / video / img_size presence
    agg["has_vis_ratio"] = float(np.mean([r.get("has_vis", False) for r in valid]) if valid else 0.0)
    agg["video_found_ratio"] = float(np.mean([r.get("video_found", False) for r in valid]) if valid else 0.0)
    agg["has_img_size_ratio"] = float(np.mean([r.get("img_size") is not None for r in valid]) if valid else 0.0)

    # sample stats
    def _collect(k: str) -> List[float]:
        out: List[float] = []
        for r in valid:
            out.extend(r.get("sample", {}).get(k, []))
        return out

    for k in ["median_vel_px", "nonzero_step_frac", "oob_frac", "nan_frac"]:
        vals = _collect(k)
        if len(vals) == 0:
            agg[f"{k}_count"] = 0
            agg[f"{k}_mean"] = 0.0
            agg[f"{k}_p50"] = 0.0
            agg[f"{k}_p90"] = 0.0
        else:
            agg[f"{k}_count"] = len(vals)
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_p50"] = float(np.percentile(vals, 50))
            agg[f"{k}_p90"] = float(np.percentile(vals, 90))

    return agg


def compare_with_cfg(agg: Dict, reports: List[Dict], cfg: Dict) -> Dict:
    recs: List[str] = []

    # cfg fields (best-effort)
    img_shape = tuple(cfg.get("img_shape", [])) if isinstance(cfg.get("img_shape"), list) else None
    true_horizon = cfg.get("true_horizon")
    track_pred_horizon = cfg.get("track_pred_horizon")
    num_tracks = cfg.get("num_tracks")
    cond_cameraviews = cfg.get("cond_cameraviews") or []

    # horizon
    if agg.get("horizon_dist"):
        common_h = max(agg["horizon_dist"].items(), key=lambda kv: kv[1])[0]
        if true_horizon is not None and int(true_horizon) != int(common_h):
            recs.append(
                f"HDF5 horizon={common_h} 与 cfg.true_horizon={true_horizon} 不一致；训练会在 loader 截断或插值，建议对齐以减少误差。"
            )
        if track_pred_horizon is not None and int(track_pred_horizon) != int(true_horizon or common_h):
            recs.append(
                f"cfg.track_pred_horizon={track_pred_horizon} 与 true_horizon={true_horizon or common_h} 不一致；将触发插值，注意插值方法的影响。"
            )

    # tracks
    if agg.get("tracksN_dist"):
        common_n = max(agg["tracksN_dist"].items(), key=lambda kv: kv[1])[0]
        if num_tracks is not None and int(num_tracks) != int(common_n):
            recs.append(
                f"HDF5 轨迹数 N={common_n} 与 cfg.num_tracks={num_tracks} 不一致；建议在预处理阶段统一为相同 N（如 uniform_400）。"
            )

    # image size
    if img_shape and agg.get("img_sizes_dist"):
        img_keys = list(agg["img_sizes_dist"].keys())
        mism = [k for k in img_keys if k != f"{img_shape[0]}x{img_shape[1]}"]
        if len(mism) > 0:
            recs.append(
                f"数据内存在多种图像尺寸 {img_keys}，与 cfg.img_shape={img_shape} 不完全一致；loader 会缩放轨迹，建议统一尺寸以减少缩放误差。"
            )
    if agg.get("has_img_size_ratio", 0.0) < 0.8:
        recs.append("多数样本缺少 (height,width) 属性，建议在预处理阶段写入，便于精确缩放。")

    # views
    if cond_cameraviews:
        data_views = set(agg.get("view_names_dist", {}).keys())
        cfg_views = set(cond_cameraviews)
        if not cfg_views.issubset(data_views):
            recs.append(f"cfg.cond_cameraviews={sorted(cfg_views)} 与数据中可用视角 {sorted(data_views)} 不一致；建议在预处理或训练参数中对齐视角集合。")

    # motion statistics
    nz_mean = agg.get("nonzero_step_frac_mean", 0.0)
    if nz_mean < 0.1:
        recs.append(
            "相邻帧位移极小（nonzero_step_frac 平均<0.1），建议：降低 target_fps（如 20→10），或在采样时增大步长/窗口，提升运动信号。"
        )

    oob_mean = agg.get("oob_frac_mean", 0.0)
    if oob_mean > 0.05:
        recs.append("存在较多越界坐标（>5%），建议检查归一化/尺寸映射与轨迹有效性筛选。")

    nan_mean = agg.get("nan_frac_mean", 0.0)
    if nan_mean > 0.0:
        recs.append("存在 NaN/Inf 轨迹值，建议在预处理阶段清理或剔除对应窗口。")

    # video-path mapping
    if agg.get("video_found_ratio", 1.0) < 0.7:
        recs.append("多数 HDF5 未能映射到视频文件；检查 video_root 或保持同名同目录视频以便可视化对齐。")

    # class imbalance
    med_vel = agg.get("median_vel_px_mean", 0.0)
    if med_vel < 0.5:
        recs.append(
            "速度中位数很低，分类相对位移极度偏中心；建议结合训练侧的径向类权重/焦点损失，或在数据侧进行运动阈值过滤。"
        )

    return {"recommendations": recs}


def main():
    parser = argparse.ArgumentParser(description="Analyze HDF5 dataset quality and config matching.")
    parser.add_argument("--root-dir", required=True, help="Root directory containing preprocessed .hdf5 files")
    parser.add_argument("--cfg", default="amplify/cfg/train_motion_tokenizer.yaml", help="Training cfg yaml to compare")
    parser.add_argument("--video-root", default=None, help="Optional video root for path mapping checks")
    parser.add_argument("--max-files", type=int, default=200, help="Max number of .hdf5 files to scan")
    parser.add_argument("--sample-windows", type=int, default=5, help="Windows per file to sample for motion stats")
    parser.add_argument("--movement-threshold-px", type=float, default=0.5, help="Velocity threshold (pixels) per-step median to deem nonzero")
    parser.add_argument("--out", default=None, help="Path to write JSON report (default: <root-dir>/analysis_report.json)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root_dir = args.root_dir
    video_root = args.video_root or os.getenv("AMPLIFY_VIDEO_ROOT") or os.getenv("VIDEO_ROOT")

    cfg = load_cfg(args.cfg)

    h5_list = find_h5_files(root_dir, max_files=args.max_files)
    if args.verbose:
        print(f"Found {len(h5_list)} hdf5 files under {root_dir}")

    reports: List[Dict] = []
    for i, h5_path in enumerate(h5_list):
        rep = analyze_h5_file(
            h5_path=h5_path,
            sample_windows=args.sample_windows,
            movement_threshold_px=args.movement_threshold_px,
            root_dir=root_dir,
            video_root=video_root,
        )
        reports.append(rep)
        if args.verbose and (i + 1) % 20 == 0:
            print(f"Scanned {i+1}/{len(h5_list)} files")

    agg = aggregate_reports(reports)
    comp = compare_with_cfg(agg, reports, cfg)

    out = {
        "root_dir": root_dir,
        "cfg_path": args.cfg,
        "cfg_digest": {
            k: cfg.get(k)
            for k in [
                "img_shape",
                "true_horizon",
                "track_pred_horizon",
                "num_tracks",
                "cond_cameraviews",
                "interp_method",
            ]
            if k in cfg
        },
        "aggregate": agg,
        "recommendations": comp.get("recommendations", []),
        "files": reports[:50],  # include first 50 file-level entries for inspection
    }

    out_path = args.out or os.path.join(root_dir, "analysis_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "out": out_path,
        "num_files_scanned": agg.get("num_files_scanned", 0),
        "num_files_valid": agg.get("num_files_valid", 0),
        "video_found_ratio": agg.get("video_found_ratio", 0.0),
        "median_vel_px_mean": agg.get("median_vel_px_mean", 0.0),
        "nonzero_step_frac_mean": agg.get("nonzero_step_frac_mean", 0.0),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
