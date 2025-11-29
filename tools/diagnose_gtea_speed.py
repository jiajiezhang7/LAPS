import os
import time
import json
import statistics as stats

import torch
from torch.utils.data import DataLoader
from amplify.utils.train import safe_collate

from amplify.loaders.custom_segments_dataset import CustomSegmentsDataset

ROOT = "./data/preprocessed_gtea_m10/split1"
VIDEO_ROOT = "./online_datasets/gtea/gtea/Videos_train.split1"
IMG_SHAPE = (480, 771)
BATCH_SIZE = 8
NSTEPS = 10  # keep very small to avoid contention with current training


def describe_dataset(keys):
    ds = CustomSegmentsDataset(
        root_dir=ROOT,
        dataset_names=["custom_segments"],
        img_shape=IMG_SHAPE,
        true_horizon=16,
        track_pred_horizon=16,
        keys_to_load=keys,
        use_cached_index_map=True,
        video_root=VIDEO_ROOT,
    )
    length = len(ds)
    # video_path 存在率估计
    idx = ds.index_map
    v_paths = [it.get("video_path") for it in idx]
    v_exist = sum(1 for p in v_paths if p and os.path.exists(p))
    v_total = len(v_paths)
    return ds, {
        "length": length,
        "video_path_exist": v_exist,
        "video_path_total": v_total,
    }


def bench_loader(ds, num_workers):
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=False, collate_fn=safe_collate)
    it = iter(dl)
    times = []
    img_means = []
    steps = 0
    t0 = time.time()
    while steps < NSTEPS:
        t_step0 = time.time()
        try:
            batch = next(it)
        except StopIteration:
            break
        dt = time.time() - t_step0
        times.append(dt)
        steps += 1
        # images 统计（若存在）
        if isinstance(batch, dict) and ("images" in batch):
            im = batch["images"]  # torch.Tensor [B, V, H, W, C] after safe_collate
            try:
                img_means.append(float(im.mean().item()))
            except Exception:
                pass
    total = time.time() - t0
    return {
        "steps": steps,
        "avg_step_s": (sum(times) / len(times)) if times else None,
        "p50_step_s": stats.median(times) if times else None,
        "p90_step_s": (sorted(times)[int(0.9 * len(times))] if times else None),
        "total_s": total,
        "num_workers": num_workers,
        "img_mean_sample": (sum(img_means) / len(img_means)) if img_means else None,
    }


def main():
    out = {}
    # 配置 A：tracks+images
    ds_ti, info_ti = describe_dataset(["tracks", "images"])
    out["tracks+images_info"] = info_ti
    out["tracks+images_w0"] = bench_loader(ds_ti, num_workers=0)

    # 配置 B：tracks only
    ds_t, info_t = describe_dataset(["tracks"])
    out["tracks_only_info"] = info_t
    out["tracks_only_w0"] = bench_loader(ds_t, num_workers=0)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

