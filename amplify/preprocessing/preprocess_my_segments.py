"""
Preprocess custom video segments into Motion Tokenizer training HDF5 files.

This module mirrors the design of `preprocess_libero.py` but operates on raw
video files stored in a nested directory. Each input video becomes a single
sample whose point tracks are computed with CoTracker and written under
`root/<view>/{tracks, vis}` inside an output HDF5 file, matching the training
expectations of `train_motion_tokenizer.py`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import h5py
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

from preprocessing.preprocess_base import (
    PreprocessDataset,
    Sample,
    SampleSkipError,
    TrackProcessor,
    run_dataset,
    run_dataset_prefetch,
    run_dataset_mgpu,
)


class PreprocessMySegments(PreprocessDataset):
    """Preprocess a directory of video segments for Motion Tokenizer training."""

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.mode != "tracks":
            raise ValueError("preprocess_my_segments currently supports only mode='tracks'")

        # Resolve paths relative to the original working directory (Hydra safe).
        self.source_dir = Path(to_absolute_path(cfg.source)).expanduser()
        self.dataset_name = getattr(cfg, "dataset_name", "custom_segments")
        self.output_dir_override = getattr(cfg, "output_dir", None)
        self.dest_root = Path(to_absolute_path(cfg.dest)).expanduser()
        self.recursive = bool(getattr(cfg, "recursive", True))
        self.video_exts = self._normalize_exts(getattr(cfg, "video_exts", [".mp4", ".mov", ".avi", ".mkv"]))
        self.view_name = getattr(cfg, "view_name", "default")
        self.target_fps = float(getattr(cfg, "target_fps", 0))
        self.resize_shorter = int(getattr(cfg, "resize_shorter", 0))
        self.max_files = getattr(cfg, "max_files", None)
        self.verbose = bool(getattr(cfg, "verbose", True))
        self.video_progress_desc = getattr(cfg, "video_progress_desc", "Videos")

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {self.source_dir}")

        # Output sub-folder mirrors the CoTracker configuration, same as LIBERO pipeline.
        reinit_str = f"_reinit_{cfg.horizon}" if cfg.reinit else ""
        self.extension = f"{cfg.init_queries}_{cfg.n_tracks}{reinit_str}"
        if self.output_dir_override:
            self.output_base = Path(to_absolute_path(self.output_dir_override)).expanduser()
        else:
            self.output_base = self.dest_root / self.dataset_name / self.extension

        self.video_paths = self._collect_videos()
        if not self.video_paths:
            raise ValueError(f"No video files found under {self.source_dir} with extensions {self.video_exts}")

        # Optional: control OpenCV internal threads to avoid oversubscription
        try:
            opencv_threads = int(getattr(cfg, "opencv_num_threads", 0) or 0)
            if opencv_threads > 0:
                cv2.setNumThreads(opencv_threads)
        except Exception:
            pass

    @staticmethod
    def _normalize_exts(exts: Iterable[str]) -> List[str]:
        norm = []
        for ext in exts:
            if not ext:
                continue
            if not ext.startswith('.'):
                norm.append(f".{ext.lower()}")
            else:
                norm.append(ext.lower())
        return norm or ['.mp4']

    def _collect_videos(self) -> List[Path]:
        globber = self.source_dir.rglob if self.recursive else self.source_dir.glob
        candidates = [p for p in globber("**/*" if self.recursive else "*") if p.is_file() and p.suffix.lower() in self.video_exts]
        candidates.sort()
        max_files = self.max_files if isinstance(self.max_files, int) and self.max_files > 0 else None
        if max_files is not None:
            candidates = candidates[:max_files]
        return candidates

    # --------- Required PreprocessDataset interface ---------
    def build_models(self, cfg) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "mps":
            device = torch.device("cpu")
        # Speed up cuDNN convolution algorithms selection for varying input sizes
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        if cfg.reinit:
            cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
        else:
            cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
        cotracker = cotracker.eval()

        parallel_cfg = getattr(cfg, "parallel", None)
        enable_multi_gpu = bool(getattr(parallel_cfg, "enable_multi_gpu", False))
        device_ids = None
        if enable_multi_gpu and torch.cuda.is_available():
            total_gpus = torch.cuda.device_count()
            requested_ids = getattr(parallel_cfg, "multi_gpu_device_ids", None)
            if requested_ids is not None:
                # Filter out-of-range ids gracefully
                device_ids = [int(i) for i in requested_ids if 0 <= int(i) < total_gpus]
            if not device_ids:
                device_ids = list(range(total_gpus))
            if device_ids:
                primary = torch.device(f"cuda:{device_ids[0]}")
                cotracker = cotracker.to(primary)
                if len(device_ids) > 1:
                    cotracker = torch.nn.DataParallel(cotracker, device_ids=device_ids)
                    print(f"Using DataParallel for CoTracker on GPUs: {device_ids}")
            else:
                cotracker = cotracker.to(device)
        else:
            cotracker = cotracker.to(device)

        models = {"cotracker": cotracker}
        return models

    def build_processors(self, cfg, models: Dict[str, Any]) -> Dict[str, TrackProcessor]:
        processors: Dict[str, TrackProcessor] = {
            "tracks": TrackProcessor(
                model=models["cotracker"],
                init_queries=cfg.init_queries,
                reinit=cfg.reinit,
                horizon=cfg.horizon,
                n_tracks=cfg.n_tracks,
                batch_size=cfg.batch_size,
                precision=getattr(cfg, "precision", "fp32"),
                auto_batch=bool(getattr(cfg, "auto_batch", True)),
                min_batch_size=int(getattr(cfg, "min_batch_size", 1)),
            )
        }
        return processors

    def iter_items(self, cfg) -> Iterable[Any]:
        iterator = self.video_paths
        if self.verbose:
            iterator = tqdm(self.video_paths, desc=self.video_progress_desc, unit="video")

        for video_path in iterator:
            rel_path = video_path.relative_to(self.source_dir)
            # Early skip if output already exists and appears complete
            if bool(getattr(cfg, "skip_exist", True)) and self._is_outfile_complete(rel_path):
                continue
            yield {
                "video_path": video_path,
                "relative_path": rel_path,
                "view_name": self.view_name,
            }

    def to_sample(self, item: Dict[str, Any], cfg) -> Sample:
        video_path: Path = item["video_path"]
        try:
            frames_thwc, meta = self._load_video(video_path, item["relative_path"])
        except RuntimeError as exc:
            raise SampleSkipError(
                f"Skipping video due to load failure: {video_path} ({exc})",
                item=item,
            ) from exc

        sample_id = self._sample_id(item["relative_path"])
        videos = {item["view_name"]: frames_thwc}
        meta.update(
            {
                "video_path": str(video_path),
                "relative_path": str(item["relative_path"]),
                "view_name": item["view_name"],
            }
        )
        return Sample(id=sample_id, videos=videos, meta=meta)

    def output_path(self, sample: Sample, cfg) -> str:
        rel_path = Path(sample.meta["relative_path"])
        save_dir = self.output_base / rel_path.parent
        filename = f"{rel_path.stem}.hdf5"
        return str(save_dir / filename)

    # --------- Helpers ---------
    def _sample_id(self, rel_path: Path) -> str:
        parts = list(rel_path.parts)
        if parts:
            parts[-1] = Path(parts[-1]).stem
        safe = "__".join(parts) if parts else rel_path.stem
        return safe.replace(" ", "_")

    def _is_outfile_complete(self, rel_path: Path) -> bool:
        """Return True if the target HDF5 already contains the expected view group.

        We mirror the skip logic used by `inital_save_h5` in preprocessing_utils:
        if `root/<view_name>` exists, treat as processed and skip.
        """
        save_dir = self.output_base / rel_path.parent
        filename = f"{rel_path.stem}.hdf5"
        out_path = save_dir / filename
        if not out_path.exists():
            return False
        try:
            with h5py.File(out_path, "r") as f:
                return f"root/{self.view_name}" in f
        except Exception:
            return False

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_shorter and self.resize_shorter > 0:
            h, w = frame.shape[:2]
            shorter = min(h, w)
            if shorter != self.resize_shorter:
                scale = self.resize_shorter / float(shorter)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def _load_video(self, video_path: Path, rel_path: Path) -> tuple[np.ndarray, Dict[str, Any]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps_in = float(fps_in)
        step = 1
        if self.target_fps > 0 and fps_in > 0:
            step = max(1, int(round(fps_in / self.target_fps)))
        effective_fps = fps_in / step if fps_in > 0 else (self.target_fps if self.target_fps > 0 else 0.0)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pbar: Optional[tqdm] = None
        if self.verbose:
            desc = f"{rel_path.as_posix()}"
            pbar = tqdm(total=total_frames if total_frames > 0 else None,
                        desc=desc,
                        unit="frame",
                        leave=False,
                        dynamic_ncols=True)

        frames: List[np.ndarray] = []
        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                frame_bgr = self._resize_frame(frame_bgr)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            if pbar is not None:
                pbar.update(1)
            frame_idx += 1
        cap.release()
        if pbar is not None:
            pbar.close()

        if not frames:
            raise RuntimeError(f"No frames extracted from video: {video_path}")

        video_thwc = np.stack(frames, axis=0)  # (T, H, W, C) uint8
        meta = {
            "fps_input": fps_in,
            "fps_effective": effective_fps,
            "frame_step": step,
            "num_frames": video_thwc.shape[0],
            "height": video_thwc.shape[1],
            "width": video_thwc.shape[2],
        }
        return video_thwc.astype(np.uint8), meta


@hydra.main(config_path="../cfg/preprocessing", config_name="preprocess_my_segments", version_base="1.2")
def main(cfg) -> None:
    os.makedirs(to_absolute_path(cfg.dest), exist_ok=True)

    dataset = PreprocessMySegments(cfg)
    output_dir = dataset.output_base
    os.makedirs(output_dir, exist_ok=True)

    # Save resolved config alongside outputs for reproducibility.
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)

    print(f"Processing {len(dataset.video_paths)} videos from {dataset.source_dir}")
    print(f"Outputs will be written under {output_dir}")
    if dataset.verbose:
        print("Verbose progress enabled (tqdm). Use `verbose=false` to disable.")

    # Choose runner: multiprocess multi-GPU, prefetch, or simple
    use_mgpu = False
    use_prefetch = False
    try:
        par = getattr(cfg, "parallel", None)
        use_mgpu = bool(par is not None and getattr(par, "enable_multiprocess_gpu", False))
        use_prefetch = bool(par is not None and getattr(par, "enable_io_prefetch", False))
    except Exception:
        pass

    if use_mgpu:
        print("Multiprocess multi-GPU runner: each process binds to one GPU.")
        run_dataset_mgpu(dataset, cfg)
    elif use_prefetch:
        print("Prefetch-enabled runner: IO and GPU compute will be overlapped.")
        run_dataset_prefetch(dataset, cfg)
    else:
        run_dataset(dataset, cfg)
    print("Done!")


if __name__ == "__main__":
    main()
