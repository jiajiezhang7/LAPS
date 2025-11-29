import os, time, json, re
import numpy as np
from pathlib import Path

from amplify.loaders.custom_segments_dataset import CustomSegmentsDataset

ROOT_DIR = "./data/preprocessed_gtea_m10/split1"
IMG_SHAPE = (480, 771)
DATASET_NAMES = ["custom_segments"]
COND_VIEWS = ["default"]
TRACK_METHOD = "uniform_400_reinit_16"
TRUE_H = 16
PRED_H = 16
NUM_TRACKS = 400
KEYS_TI = ["tracks", "images"]


def time_steps(ds, steps=10, num_workers=0):
    from torch.utils.data import DataLoader, RandomSampler
    from amplify.utils.train import safe_collate  # ensure same collate

    sampler = RandomSampler(ds, num_samples=steps)
    dl = DataLoader(ds, batch_size=8, sampler=sampler, num_workers=num_workers,
                    persistent_workers=num_workers > 0, collate_fn=safe_collate)

    t0 = time.time()
    times = []
    n = 0
    for batch in dl:
        n += 1
        t1 = time.time()
        times.append(t1 - t0)
        t0 = t1
    return {
        "steps": n,
        "avg_s": float(np.mean(times)) if times else None,
        "p50_s": float(np.median(times)) if times else None,
        "p90_s": float(np.percentile(times, 90)) if times else None,
    }


def wrap_find_video_path(ds):
    # Patch _find_video_path to measure call count and time
    orig = ds._find_video_path
    stats = {"calls": 0, "total_s": 0.0}

    def _wrapped(h5_path):
        t0 = time.time()
        try:
            return orig(h5_path)
        finally:
            stats["calls"] += 1
            stats["total_s"] += (time.time() - t0)
    ds._find_video_path = _wrapped  # type: ignore
    return stats


def sample_img_mean(ds):
    # Sample 2 items to check if images are real frames (mean>0.05) or black placeholders (~0)
    import random
    idxs = random.sample(range(len(ds)), k=min(2, len(ds)))
    means = []
    for i in idxs:
        d = ds[i]
        img = d.get("images")
        if img is None:
            means.append(None)
        else:
            arr = img
            if isinstance(arr, np.ndarray):
                means.append(float(arr.mean()))
            else:
                try:
                    means.append(float(np.array(arr).mean()))
                except Exception:
                    means.append(None)
    return means


def build_dataset(video_root):
    DS = CustomSegmentsDataset if CustomSegmentsDataset else CS2
    ds = DS(
        root_dir=ROOT_DIR,
        dataset_names=DATASET_NAMES,
        track_method=TRACK_METHOD,
        cond_cameraviews=COND_VIEWS,
        keys_to_load=KEYS_TI,
        img_shape=IMG_SHAPE,
        true_horizon=TRUE_H,
        track_pred_horizon=PRED_H,
        interp_method="linear",
        num_tracks=NUM_TRACKS,
        use_cached_index_map=False,
        video_root=video_root,
    )
    return ds


def main():
    out = {
        "env_VIDEO_ROOT": os.environ.get("VIDEO_ROOT"),
        "env_AMPLIFY_VIDEO_ROOT": os.environ.get("AMPLIFY_VIDEO_ROOT"),
    }

    tests = []

    # Test A: mimic current run config (video_root from wandb config default)
    video_root_default = "/home/jay/action_ws/data/raw_video_d01"
    dsA = build_dataset(video_root_default)
    stA = wrap_find_video_path(dsA)
    meansA = sample_img_mean(dsA)
    tA0 = time.time(); profA0 = time_steps(dsA, steps=10, num_workers=0); profA4 = time_steps(dsA, steps=10, num_workers=4); tA1 = time.time()
    tests.append({
        "name": "A_video_root_default",
        "video_root": video_root_default,
        "len": len(dsA),
        "img_means": meansA,
        "find_calls": stA["calls"],
        "find_total_s": stA["total_s"],
        "dl_profile_w0": profA0,
        "dl_profile_w4": profA4,
        "wall_s": tA1 - tA0,
    })

    # Test B: explicit null video_root
    dsB = build_dataset(None)
    stB = wrap_find_video_path(dsB)
    meansB = sample_img_mean(dsB)
    tB0 = time.time(); profB0 = time_steps(dsB, steps=10, num_workers=0); profB4 = time_steps(dsB, steps=10, num_workers=4); tB1 = time.time()
    tests.append({
        "name": "B_video_root_null",
        "video_root": None,
        "len": len(dsB),
        "img_means": meansB,
        "find_calls": stB["calls"],
        "find_total_s": stB["total_s"],
        "dl_profile_w0": profB0,
        "dl_profile_w4": profB4,
        "wall_s": tB1 - tB0,
    })

    # Test C: keys_to_load=['tracks'] with null video_root
    DS = CustomSegmentsDataset
    dsC = DS(
        root_dir=ROOT_DIR,
        dataset_names=DATASET_NAMES,
        track_method=TRACK_METHOD,
        cond_cameraviews=COND_VIEWS,
        keys_to_load=["tracks"],
        img_shape=IMG_SHAPE,
        true_horizon=TRUE_H,
        track_pred_horizon=PRED_H,
        interp_method="linear",
        num_tracks=NUM_TRACKS,
        use_cached_index_map=False,
        video_root=None,
    )
    stC = wrap_find_video_path(dsC)
    tC0 = time.time(); profC0 = time_steps(dsC, steps=10, num_workers=0); profC4 = time_steps(dsC, steps=10, num_workers=4); tC1 = time.time()
    tests.append({
        "name": "C_tracks_only",
        "video_root": None,
        "len": len(dsC),
        "img_means": None,
        "find_calls": stC["calls"],
        "find_total_s": stC["total_s"],
        "dl_profile_w0": profC0,
        "dl_profile_w4": profC4,
        "wall_s": tC1 - tC0,
    })

    out["tests"] = tests
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

