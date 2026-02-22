"""
LIBERO Dataset Definition v2 — implements the minimal PreprocessDataset API.

Notes on modalities
- Tracks: used in AMPLIFY training.
- Text: stored once per task in `dest/suite/text/<task>.hdf5`.
- Depth / GT segmentation / GT depth: not used by AMPLIFY training; provided for
  completeness and for users who want to export these modalities.
"""

import os
import sys
import json
from typing import Any, Dict, Iterable, List

import h5py
import hydra
import numpy as np
import torch
from tqdm import tqdm
from natsort import natsorted
from omegaconf import OmegaConf

from preprocessing.preprocess_base import (
    PreprocessDataset,
    Sample,
    TrackProcessor,
    DepthProcessor,
    TextEmbeddingProcessor,
    run_dataset,
)
from amplify.utils.libero_utils.env_utils import build_libero_env
from amplify.utils.preprocessing_utils import load_depth_anything_v2

# Add local Depth-Anything-V2 to path if present
sys.path.append(os.path.join(os.getcwd(), "Depth-Anything-V2"))


def _libero_task_files(cfg) -> (List[str], List[str]):
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    b = benchmark_dict[cfg.suite]()
    num = b.get_num_tasks()
    return [b.get_task_demonstration(i) for i in range(num)], b.get_task_names()


def _parse_range(range_str: str, n: int) -> List[int]:
    try:
        a, b = range_str.split("-")
        return list(range(int(a), int(b) + 1))
    except Exception:
        return list(range(n))


class PreprocessLibero(PreprocessDataset):
    def __init__(self, cfg):
        self.cfg = cfg
        # compute save path extension based on mode
        if cfg.mode == "tracks":
            reinit_str = f"_reinit_{cfg.horizon}" if cfg.reinit else ""
            self.extension = f"{cfg.init_queries}_{cfg.n_tracks}{reinit_str}"
        elif cfg.mode == "depth":
            self.extension = "depth_anything_v2" + ("_metric" if cfg.metric_depth else "")
        elif cfg.mode == "gt_depth":
            self.extension = "gt_depth"
        elif cfg.mode == "gt_segmentation":
            self.extension = "gt_segmentation"
        else:
            self.extension = ""

        # Enumerate once to reuse in multiple methods
        src_root = os.path.join(os.getcwd(), cfg.source)
        task_rel_paths, task_names = _libero_task_files(cfg)
        self.task_rel_paths = task_rel_paths
        self.task_names = task_names
        self.src_root = src_root
        self.task_indices = _parse_range(cfg.range, len(task_rel_paths))

    def build_models(self, cfg) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        if cfg.mode == "tracks":
            # MPS is not supported by CoTracker due to border padding mode
            device = torch.device("cuda" if torch.cuda.is_available() else ("cpu"))
            if device.type == "mps":
                device = torch.device("cpu")
            if not cfg.reinit:
                cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
            else:
                cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
            models["cotracker"] = cotracker.eval().to(device)
        elif cfg.mode == "depth":
            # Load Depth-Anything V2
            models['depth_model'] = load_depth_anything_v2(
                metric_depth=cfg.metric_depth,
                encoder=cfg.depth_encoder,
                checkpoints_dir=cfg.depth_checkpoint_dir,
            )
        elif cfg.mode == "text":
            from amplify.models.encoders.t5 import T5
            device = torch.device("cuda" if torch.cuda.is_available() else ("cpu"))
            models["text_encoder"] = T5(**cfg.text_encoder).eval().to(device)
        return models

    def build_processors(self, cfg, models: Dict[str, Any]) -> Dict[str, Any]:
        if cfg.mode == "tracks":
            return {
                "tracks": TrackProcessor(
                    model=models["cotracker"],
                    init_queries=cfg.init_queries,
                    reinit=cfg.reinit,
                    horizon=cfg.horizon,
                    n_tracks=cfg.n_tracks,
                    batch_size=cfg.batch_size,
                )
            }
        elif cfg.mode == "depth":
            return {"depth": LiberoDAV2DepthProcessor(models["depth_model"])}
        elif cfg.mode == "text":
            # Wrote one embedding per task in on_begin; avoid per-demo files.
            return {}
        elif cfg.mode == "gt_depth":
            return {"gt_depth": LiberoGtDepthProcessor(cfg)}
        elif cfg.mode == "gt_segmentation":
            return {"gt_segmentation": LiberoGtSegProcessor(cfg)}
        else:
            return {}

    def iter_items(self, cfg) -> Iterable[Any]:
        # Yield (task_name, abs_task_h5, demo_key)
        if cfg.mode == "text":
            # No per-demo iteration needed for text mode
            return []
        for idx in self.task_indices:
            task_name = self.task_names[idx]
            task_h5 = os.path.join(self.src_root, self.task_rel_paths[idx])
            with h5py.File(task_h5, "r") as f:
                demos = f["data"]
                demo_keys = natsorted(list(demos.keys()))
                # Save views for this task file
                try:
                    attrs = json.loads(demos.attrs["env_args"])
                    views = attrs["env_kwargs"]["camera_names"]
                    views.sort()
                    views = [name.replace("robot0_", "") if name.endswith("eye_in_hand") else name for name in views]
                except Exception:
                    views = ["agentview", "eye_in_hand"]
                for dk in demo_keys:
                    yield {"task_name": task_name, "task_h5": task_h5, "demo_key": dk, "views": views, "task_idx": idx}

    def to_sample(self, item: Any, cfg) -> Sample:
        # For tracks/depth: load THWC videos per view (flip vertically). For gt_*: no videos needed.
        task_h5 = item["task_h5"]
        demo_key = item["demo_key"]
        views = item["views"]
        meta = {"task_name": item["task_name"], "task_h5": task_h5, "demo_key": demo_key, "views": views, "task_idx": item["task_idx"]}
        if self.cfg.mode in ("gt_depth", "gt_segmentation"):
            videos = {v: None for v in views}
        else:
            videos = {}
            with h5py.File(task_h5, "r") as f:
                for view in views:
                    arr = np.array(f["data"][demo_key][f"obs/{view}_rgb"])  # (T,H,W,C)
                    videos[view] = arr[:, ::-1, :, :].copy()
        return Sample(id=demo_key, videos=videos, meta=meta)

    def output_path(self, sample: Sample, cfg) -> str:
        task_name = sample.meta["task_name"]
        base = os.path.join(os.getcwd(), cfg.dest, cfg.suite, self.extension)
        return os.path.join(base, f"{task_name}_demo", f"{sample.id}.hdf5")

    # ---------- Optional hooks ----------
    def on_begin(self, cfg, models: Dict[str, Any]) -> None:
        # Text mode: write a single embedding per task into dest/suite/text/<task_name>.hdf5
        if cfg.mode != "text":
            return
        from libero.libero.benchmark import grab_language_from_filename

        text_dir = os.path.join(os.getcwd(), cfg.dest, cfg.suite, "text")
        os.makedirs(text_dir, exist_ok=True)

        for idx in self.task_indices:
            task_rel = self.task_rel_paths[idx]
            task_abs = os.path.join(self.src_root, task_rel)
            task_stem = os.path.splitext(os.path.basename(task_abs))[0]
            task_name = task_stem.replace("_demo", "")
            text = grab_language_from_filename(task_stem.replace("_demo", ".bddl"))
            text_emb = models["text_encoder"]([text]).cpu().numpy()

            out_path = os.path.join(text_dir, f"{task_name}.hdf5")
            with h5py.File(out_path, "w") as f:
                f.create_dataset("text_emb", data=text_emb, dtype="float32")
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset("text", data=np.array(text, dtype=dt), dtype=dt)


class LiberoGtDepthProcessor(DepthProcessor):
    """Simulate ground-truth depths by replaying states in LIBERO env.

    Note: Not used by AMPLIFY training; provided as a utility exporter.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.env_cache: Dict[int, Any] = {}

    def _get_env(self, task_idx: int):
        if task_idx in self.env_cache:
            return self.env_cache[task_idx]
        env_cfg = {
            "task_suite": self.cfg.suite,
            "task_no": task_idx,
            "img_size": self.cfg.img_size,
            "action_dim": 7,
            "n_envs": 1,
            "use_depth": True,
            "dataset_path": os.path.join(os.getcwd(), self.cfg.source),
        }
        env, _, _ = build_libero_env(**env_cfg)
        self.env_cache[task_idx] = env
        return env

    def process(self, out_h5: h5py.File, sample: Sample) -> None:
        views = sample.meta["views"]
        task_idx = sample.meta["task_idx"]
        env = self._get_env(task_idx)
        with h5py.File(sample.meta["task_h5"], "r") as f:
            states = f["data"][sample.meta["demo_key"]]["states"]
            T = len(states)
            depth_video = np.empty((len(views), T, self.cfg.img_size, self.cfg.img_size), dtype=np.float32)
            env.reset()
            for t in tqdm(range(T), desc=f"GT depth {sample.meta['task_name']}/{sample.meta['demo_key']}", leave=True):
                obs = env.regenerate_obs_from_state(states[t])
                depth_img = obs["depth"][0]  # (V, H, W)
                depth_video[:, t] = depth_img

        root = out_h5.create_group("root") if "root" not in out_h5 else out_h5["root"]
        for i, view in enumerate(views):
            vg = root.create_group(view) if view not in root else root[view]
            if "depth" in vg:
                vg.__delitem__("depth")
            vg.create_dataset("depth", data=depth_video[i], dtype="float32")


class LiberoGtSegProcessor(DepthProcessor):
    """Simulate ground-truth segmentation by replaying states in LIBERO env.

    Note: Not used by AMPLIFY training; provided as a utility exporter.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.env_cache: Dict[int, Any] = {}

    def _get_env(self, task_idx: int):
        if task_idx in self.env_cache:
            return self.env_cache[task_idx]
        env_cfg = {
            "task_suite": self.cfg.suite,
            "task_no": task_idx,
            "img_size": self.cfg.img_size,
            "action_dim": 7,
            "n_envs": 1,
            "use_depth": False,
            "segmentation_level": 'instance',
            "dataset_path": os.path.join(os.getcwd(), self.cfg.source),
        }
        env, _, _ = build_libero_env(**env_cfg)
        self.env_cache[task_idx] = env
        return env

    def process(self, out_h5: h5py.File, sample: Sample) -> None:
        views = sample.meta["views"]
        task_idx = sample.meta["task_idx"]
        env = self._get_env(task_idx)
        with h5py.File(sample.meta["task_h5"], "r") as f:
            states = f["data"][sample.meta["demo_key"]]["states"]
            T = len(states)
            seg_video = np.empty((len(views), T, self.cfg.img_size, self.cfg.img_size), dtype=np.uint8)
            env.reset()
            for t in tqdm(range(T), desc=f"GT seg {sample.meta['task_name']}/{sample.meta['demo_key']}", leave=True):
                obs = env.regenerate_obs_from_state(states[t])
                seg_img = obs["segmentation"][0]  # (V, H, W)
                seg_video[:, t] = seg_img

        root = out_h5.create_group("root") if "root" not in out_h5 else out_h5["root"]
        for i, view in enumerate(views):
            vg = root.create_group(view) if view not in root else root[view]
            if "segmentation" in vg:
                vg.__delitem__("segmentation")
            vg.create_dataset("segmentation", data=seg_video[i], dtype="uint8")


@hydra.main(config_path="../cfg/preprocessing", config_name="preprocess_libero", version_base="1.2")
def main(cfg):
    # Save resolved config into the destination root
    # Note: identical pathing to v1 for easy drop‑in replacement
    reinit_str = f"_reinit_{cfg.horizon}" if (cfg.mode == "tracks" and cfg.reinit) else ""
    if cfg.mode == "tracks":
        extension = f"{cfg.init_queries}_{cfg.n_tracks}{reinit_str}"
    elif cfg.mode == "depth":
        extension = "depth_anything_v2" + ("_metric" if cfg.metric_depth else "")
    elif cfg.mode == "gt_depth":
        extension = "gt_depth"
    elif cfg.mode == "gt_segmentation":
        extension = "gt_segmentation"
    else:
        extension = ""

    save_dir = os.path.join(os.getcwd(), cfg.dest, cfg.suite, extension)
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

    # Run with our dataset definition
    dataset = PreprocessLibero(cfg)
    run_dataset(dataset, cfg)
    print("Done!")


class LiberoDAV2DepthProcessor(DepthProcessor):
    """Depth-Anything V2 depth with informative per-demo progress.

    We keep semantics identical to DepthProcessor but move the tqdm to this
    processor so the description can include task/demo identifiers.
    """

    def __init__(self, model):
        self.model = model

    def process(self, out_h5: h5py.File, sample: Sample) -> None:
        root = out_h5.create_group("root") if "root" not in out_h5 else out_h5["root"]
        for view, video_thwc in sample.videos.items():
            T, H, W, C = video_thwc.shape
            depth = np.empty((T, H, W), dtype=np.float32)
            desc = f"Depth {sample.meta.get('task_name','')}/{sample.meta.get('demo_key', sample.id)}/{view}"
            for t in tqdm(range(T), desc=desc, leave=True):
                depth[t] = self.model.infer_image(video_thwc[t])
            vg = root.create_group(view) if view not in root else root[view]
            if "depth" in vg:
                vg.__delitem__("depth")
            vg.create_dataset("depth", data=depth, dtype="float32")


if __name__ == "__main__":
    main()
