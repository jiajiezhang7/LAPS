"""
Preprocessing Core v2 — minimal, dataset-agnostic API.

How to use
- Subclass PreprocessDataset and implement these methods:
  - build_models(cfg) -> Dict[str, Any]
  - build_processors(cfg, models) -> Dict[str, Processor]
  - iter_items(cfg) -> Iterable[Any]
  - to_sample(item, cfg) -> Sample
  - output_path(sample, cfg) -> str (absolute or workspace‑relative output .hdf5 path)

This module makes no dataset assumptions; all specifics live in the subclass.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, List
import multiprocessing as mp

import h5py
import numpy as np
from einops import rearrange
import torch

from amplify.utils.preprocessing_utils import inital_save_h5, tracks_from_video

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

@dataclass
class Sample:
    """A single training example to write.

    Required
    - id: Unique identifier used in output file naming
    - videos: dict mapping view -> THWC uint8/float32 arrays

    Optional
    - text: str instruction for this sample
    - actions: np.ndarray action sequence for this sample
    - meta: arbitrary metadata
    """

    id: str
    videos: Dict[str, np.ndarray]
    text: Optional[str] = None
    actions: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None


class Processor(ABC):
    """Writes a modality for one Sample into an opened HDF5 file."""
    @abstractmethod
    def process(self, out_h5: h5py.File, sample: Sample) -> None:
        ...


class TrackProcessor(Processor):
    """Computes dense tracks per view and writes root/<view>/{tracks,vis}."""

    def __init__(
        self,
        model,
        init_queries: str = "uniform",
        reinit: bool = True,
        horizon: int = 16,
        n_tracks: int = 400,
        batch_size: int = 8,
        precision: str = "fp32",
        auto_batch: bool = True,
        min_batch_size: int = 1,
    ):
        self.model = model
        self.init_queries = init_queries
        self.reinit = reinit
        self.horizon = horizon
        self.n_tracks = n_tracks
        self.batch_size = batch_size
        self.precision = precision
        self.auto_batch = auto_batch
        self.min_batch_size = min_batch_size

    def process(self, out_h5: h5py.File, sample: Sample) -> None:
        root = out_h5.create_group("root") if "root" not in out_h5 else out_h5["root"]
        for view, video_thwc in sample.videos.items():
            video_tchw = rearrange(video_thwc, "t h w c -> t c h w")
            tracks, vis = tracks_from_video(
                video=video_tchw,
                track_model=self.model,
                init_queries=self.init_queries,
                reinit=self.reinit,
                horizon=self.horizon,
                n_tracks=self.n_tracks,
                batch_size=self.batch_size,
                dim_order="tchw",
                precision=self.precision,
                auto_batch=self.auto_batch,
                min_batch_size=self.min_batch_size,
            )
            vg = root.create_group(view) if view not in root else root[view]
            # Persist the actual resized image size used for tracking (H, W) for downstream normalization
            try:
                h_resized = int(video_thwc.shape[1])
                w_resized = int(video_thwc.shape[2])
                vg.attrs["height"] = h_resized
                vg.attrs["width"] = w_resized
            except Exception:
                # Attributes are optional; training will fallback to cfg.img_shape
                pass
            if "tracks" in vg:
                vg.__delitem__("tracks")
            if "vis" in vg:
                vg.__delitem__("vis")
            vg.create_dataset("tracks", data=tracks, dtype="float32")
            vg.create_dataset("vis", data=vis, dtype="float32")


class DepthProcessor(Processor):
    """Runs a provided depth function per frame and writes root/<view>/depth.

    Note: Depth is not used by AMPLIFY training by default; this is provided
    for completeness and for users who want to export this modality.
    """

    def __init__(self, depth_fn):
        """depth_fn: Callable[[np.ndarray (T,H,W,C)], np.ndarray (T,H,W)]"""
        self.depth_fn = depth_fn

    def process(self, out_h5: h5py.File, sample: Sample) -> None:
        root = out_h5.create_group("root") if "root" not in out_h5 else out_h5["root"]
        for view, video_thwc in sample.videos.items():
            depth = self.depth_fn(video_thwc)
            vg = root.create_group(view) if view not in root else root[view]
            if "depth" in vg:
                vg.__delitem__("depth")
            vg.create_dataset("depth", data=depth, dtype="float32")


class TextEmbeddingProcessor(Processor):
    """Writes top-level text_emb for a sample (if sample.text is set)."""

    def __init__(self, text_encoder):
        self.text_encoder = text_encoder

    def process(self, out_h5: h5py.File, sample: Sample) -> None:
        if not sample.text:
            return
        emb = self.text_encoder([sample.text]).cpu().numpy()
        if "text_emb" in out_h5:
            out_h5.__delitem__("text_emb")
        out_h5.create_dataset("text_emb", data=emb, dtype="float32")

class PreprocessDataset(ABC):
    """Implement these five methods to define your dataset preprocessing."""

    @abstractmethod
    def build_models(self, cfg) -> Dict[str, Any]:
        """Create and return any models needed by processors (e.g., trackers)."""
        ...

    @abstractmethod
    def build_processors(self, cfg, models: Dict[str, Any]) -> Dict[str, Processor]:
        """Return a dict of processors, e.g., {"tracks": TrackProcessor(...)}."""
        ...

    @abstractmethod
    def iter_items(self, cfg) -> Iterable[Any]:
        """Yield raw items to be converted into Samples (e.g., file paths/keys)."""
        ...

    @abstractmethod
    def to_sample(self, item: Any, cfg) -> Sample:
        """Map a raw item to a Sample (id, videos, optional text/actions/meta)."""
        ...

    @abstractmethod
    def output_path(self, sample: Sample, cfg) -> str:
        """Return the full output .hdf5 file path for this sample."""
        ...

    # Optional hooks
    def on_begin(self, cfg, models: Dict[str, Any]) -> None:
        pass

    def on_end(self, cfg, models: Dict[str, Any]) -> None:
        pass


def _open_outfile(save_path: str, view_names: Iterable[str], skip_exist: bool) -> Optional[h5py.File]:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return inital_save_h5(save_path, skip_exist, view_names=list(view_names))


class SampleSkipError(Exception):
    """Signal that the current item should be skipped without aborting the run."""

    def __init__(self, message: str, *, item: Any | None = None):
        super().__init__(message)
        self.item = item


def run_dataset(defn: PreprocessDataset, cfg) -> None:
    """Run preprocessing: build models, iterate items, write outputs."""
    models = defn.build_models(cfg)
    processors = defn.build_processors(cfg, models)
    defn.on_begin(cfg, models)

    try:
        for item in defn.iter_items(cfg):
            try:
                sample = defn.to_sample(item, cfg)
            except SampleSkipError as exc:
                print(f"[WARN] {exc}")
                continue

            if sample is None:
                continue
            save_path = defn.output_path(sample, cfg)
            out_h5 = _open_outfile(save_path, sample.videos.keys(), getattr(cfg, "skip_exist", True))
            if out_h5 is None:
                continue
            try:
                for proc in processors.values():
                    proc.process(out_h5, sample)
            finally:
                out_h5.close()
    finally:
        defn.on_end(cfg, models)


def _split_list_round_robin(items: List[Any], num_buckets: int) -> List[List[Any]]:
    buckets: List[List[Any]] = [[] for _ in range(max(1, num_buckets))]
    for i, it in enumerate(items):
        buckets[i % max(1, num_buckets)].append(it)
    return buckets


def _worker_mgpu(proc_rank: int, cuda_id: int, items: List[Any], cfg, defn_cls) -> None:
    """Child process: bind to a single GPU and process its shard sequentially."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    except Exception:
        pass

    # Ensure single-GPU context inside the process
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass

    # Disable in-process DataParallel when using multiprocess GPU
    try:
        if hasattr(cfg, "parallel") and cfg.parallel is not None:
            setattr(cfg.parallel, "enable_multi_gpu", False)
    except Exception:
        pass

    # Recreate dataset in the child to keep internal state consistent
    defn: PreprocessDataset = defn_cls(cfg)
    models = defn.build_models(cfg)
    processors = defn.build_processors(cfg, models)
    defn.on_begin(cfg, models)

    try:
        for item in items:
            try:
                sample = defn.to_sample(item, cfg)
            except SampleSkipError as exc:
                print(f"[WARN][GPU{cuda_id}] {exc}")
                continue
            if sample is None:
                continue
            save_path = defn.output_path(sample, cfg)
            out_h5 = _open_outfile(save_path, sample.videos.keys(), getattr(cfg, "skip_exist", True))
            if out_h5 is None:
                continue
            try:
                for proc in processors.values():
                    proc.process(out_h5, sample)
            finally:
                out_h5.close()
    finally:
        defn.on_end(cfg, models)


def run_dataset_mgpu(defn: PreprocessDataset, cfg) -> None:
    """Multiprocess multi-GPU runner: each process binds to one GPU and processes
    a shard of items. Outputs and behavior match run_dataset().

    Controlled by cfg.parallel.mp_device_ids (optional) or cfg.parallel.multi_gpu_device_ids.
    """
    if not torch.cuda.is_available():
        # Fallback to regular runner if no CUDA
        return run_dataset(defn, cfg)

    # Collect all items in main process (cheap: paths + metadata only)
    items: List[Any] = list(defn.iter_items(cfg))
    if len(items) == 0:
        return

    # Resolve device ids
    total = torch.cuda.device_count()
    par = getattr(cfg, "parallel", None)
    device_ids = None
    if par is not None:
        device_ids = getattr(par, "mp_device_ids", None) or getattr(par, "multi_gpu_device_ids", None)
        if device_ids is not None:
            device_ids = [int(i) for i in device_ids if 0 <= int(i) < total]
    if not device_ids:
        device_ids = list(range(total))

    if len(device_ids) <= 1:
        # Single GPU visible -> run single-process
        return run_dataset(defn, cfg)

    shards = _split_list_round_robin(items, len(device_ids))
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # already set by parent
        pass

    procs: List[mp.Process] = []
    for rank, (cuda_id, shard) in enumerate(zip(device_ids, shards)):
        p = mp.Process(target=_worker_mgpu, args=(rank, cuda_id, shard, cfg, defn.__class__))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


def run_dataset_prefetch(defn: PreprocessDataset, cfg) -> None:
    """Run preprocessing with threaded IO prefetch to overlap video decode and GPU compute.

    Behavior and outputs are identical to `run_dataset`. Only scheduling differs.
    Controlled by cfg.parallel:{enable_io_prefetch, io_workers, prefetch_buffer}.
    """
    models = defn.build_models(cfg)
    processors = defn.build_processors(cfg, models)
    defn.on_begin(cfg, models)

    # Parallel settings with safe defaults
    parallel = getattr(cfg, "parallel", None)
    io_workers = int(getattr(parallel, "io_workers", 4) or 4)
    prefetch_buffer = int(getattr(parallel, "prefetch_buffer", 4) or 4)
    max_pending = max(1, max(io_workers, prefetch_buffer))

    def _to_sample(item):
        return defn.to_sample(item, cfg)

    try:
        items_iter = iter(defn.iter_items(cfg))
        pending = []  # list of (future,)
        with ThreadPoolExecutor(max_workers=io_workers) as executor:
            # Prime the queue
            try:
                for _ in range(max_pending):
                    item = next(items_iter)
                    pending.append(executor.submit(_to_sample, item))
            except StopIteration:
                pass

            while pending:
                # Wait for any completed sample
                for fut in as_completed(pending):
                    pending.remove(fut)
                    try:
                        sample = fut.result()
                    except SampleSkipError as exc:
                        print(f"[WARN] {exc}")
                        try:
                            item = next(items_iter)
                            pending.append(executor.submit(_to_sample, item))
                        except StopIteration:
                            pass
                        break

                    if sample is None:
                        try:
                            item = next(items_iter)
                            pending.append(executor.submit(_to_sample, item))
                        except StopIteration:
                            pass
                        break
                    save_path = defn.output_path(sample, cfg)
                    out_h5 = _open_outfile(save_path, sample.videos.keys(), getattr(cfg, "skip_exist", True))
                    if out_h5 is None:
                        # Immediately try to enqueue next item when skipping
                        try:
                            item = next(items_iter)
                            pending.append(executor.submit(_to_sample, item))
                        except StopIteration:
                            pass
                        continue
                    try:
                        for proc in processors.values():
                            proc.process(out_h5, sample)
                    finally:
                        out_h5.close()

                    # Enqueue next item to keep pipeline full
                    try:
                        item = next(items_iter)
                        pending.append(executor.submit(_to_sample, item))
                    except StopIteration:
                        # No more items; loop will drain remaining futures
                        pass
                    # Break to outer while to re-enter as_completed with updated 'pending'
                    break
    finally:
        defn.on_end(cfg, models)
