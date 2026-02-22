#!/usr/bin/env python3
import argparse
import glob
import os
from collections import Counter, defaultdict
from typing import List, Optional

import h5py
import numpy as np


def scan_hdf5(
    root_dir: str,
    limit: int = 0,
    expect_views: Optional[List[str]] = None,
    require_attrs_size: bool = True,
    expect_horizon: int = -1,
    expect_n_tracks: int = -1,
    require_vis: bool = False,
    check_bounds: bool = True,
    in_bounds_ratio: float = 0.95,
    sample_t: int = 4,
    sample_n: int = 256,
):
    files = sorted(glob.glob(os.path.join(root_dir, "**", "*.hdf5"), recursive=True))
    if limit and limit > 0:
        files = files[:limit]

    summary = {
        "root_dir": root_dir,
        "total_files": len(files),
        "ok_files": 0,
        "no_root": 0,
        "no_views": 0,
        "no_tracks": 0,
        "zero_rollout": 0,
        "exceptions": 0,
        "estimated_index_entries": 0,  # sum of rollout_len across ok files (per-view consistent)
        # Extended checks
        "missing_expected_views": 0,
        "missing_size_attrs": 0,
        "bad_size_attrs": 0,
        "wrong_dtype": 0,
        "wrong_shape": 0,
        "inconsistent_shapes_across_views": 0,
        "missing_vis": 0,
        "coords_not_finite": 0,
        "coords_out_of_bounds": 0,
    }

    # Extra diagnostics
    shapes_ctr = Counter()
    dtypes_ctr = Counter()
    sizes_ctr = Counter()  # (H,W)
    reasons = defaultdict(list)  # reason -> [file,...] (kept small)
    example_ok = []

    for fp in files:
        try:
            with h5py.File(fp, "r") as f:
                if "root" not in f:
                    summary["no_root"] += 1
                    if len(reasons["no_root"]) < 10:
                        reasons["no_root"].append(fp)
                    continue

                views = list(f["root"].keys())
                if len(views) == 0:
                    summary["no_views"] += 1
                    if len(reasons["no_views"]) < 10:
                        reasons["no_views"].append(fp)
                    continue

                # Check expected views (if provided)
                if expect_views:
                    missing = [v for v in expect_views if v not in views]
                    if missing:
                        summary["missing_expected_views"] += 1
                        if len(reasons["missing_expected_views"]) < 10:
                            reasons["missing_expected_views"].append(f"{fp} :: missing {missing}")

                # Collect per-view checks
                view_infos = []  # (view, shape, dtype, T)
                base_shape = None
                for v in views:
                    grp = f[f"root/{v}"]
                    if "tracks" not in grp:
                        continue

                    # size attrs
                    h_attr = grp.attrs.get("height", None)
                    w_attr = grp.attrs.get("width", None)
                    if require_attrs_size and (h_attr is None or w_attr is None):
                        summary["missing_size_attrs"] += 1
                        if len(reasons["missing_size_attrs"]) < 10:
                            reasons["missing_size_attrs"].append(f"{fp}::{v}")
                    else:
                        try:
                            h_attr = int(h_attr) if h_attr is not None else None
                            w_attr = int(w_attr) if w_attr is not None else None
                            if h_attr is not None and w_attr is not None:
                                if h_attr <= 0 or w_attr <= 0:
                                    summary["bad_size_attrs"] += 1
                                    if len(reasons["bad_size_attrs"]) < 10:
                                        reasons["bad_size_attrs"].append(f"{fp}::{v} -> (h={h_attr}, w={w_attr})")
                                else:
                                    sizes_ctr[(h_attr, w_attr)] += 1
                        except Exception:
                            summary["bad_size_attrs"] += 1
                            if len(reasons["bad_size_attrs"]) < 10:
                                reasons["bad_size_attrs"].append(f"{fp}::{v} -> (h={h_attr}, w={w_attr})")

                    # tracks dset
                    dset = grp["tracks"]
                    shape = tuple(int(x) for x in dset.shape)
                    dtype = str(dset.dtype)
                    shapes_ctr[shape] += 1
                    dtypes_ctr[dtype] += 1

                    # Basic shape/dtype expectations
                    wrong = False
                    if dtype != "float32":
                        wrong = True
                        summary["wrong_dtype"] += 1
                        if len(reasons["wrong_dtype"]) < 10:
                            reasons["wrong_dtype"].append(f"{fp}::{v} -> {dtype}")
                    if not (len(shape) == 4 and shape[-1] == 2 and shape[0] > 0 and shape[2] > 0):
                        wrong = True
                        summary["wrong_shape"] += 1
                        if len(reasons["wrong_shape"]) < 10:
                            reasons["wrong_shape"].append(f"{fp}::{v} -> {shape}")
                    if expect_horizon > 0 and len(shape) == 4 and shape[1] != expect_horizon:
                        wrong = True
                        summary["wrong_shape"] += 1
                        if len(reasons["wrong_shape"]) < 10:
                            reasons["wrong_shape"].append(f"{fp}::{v} -> expected horizon {expect_horizon}, got {shape}")
                    if expect_n_tracks > 0 and len(shape) == 4 and shape[2] != expect_n_tracks:
                        wrong = True
                        summary["wrong_shape"] += 1
                        if len(reasons["wrong_shape"]) < 10:
                            reasons["wrong_shape"].append(f"{fp}::{v} -> expected n_tracks {expect_n_tracks}, got {shape}")

                    # vis present (optional)
                    if require_vis and "vis" not in grp:
                        summary["missing_vis"] += 1
                        if len(reasons["missing_vis"]) < 10:
                            reasons["missing_vis"].append(f"{fp}::{v}")
                    if "vis" in grp:
                        vis_shape = tuple(int(x) for x in grp["vis"].shape)
                        # expected (T, horizon, N)
                        if len(shape) == 4 and not (len(vis_shape) == 3 and vis_shape[0] == shape[0] and vis_shape[1] == shape[1] and vis_shape[2] == shape[2]):
                            summary["wrong_shape"] += 1
                            if len(reasons["wrong_shape"]) < 10:
                                reasons["wrong_shape"].append(f"{fp}::{v} -> vis {vis_shape} incompatible with tracks {shape}")

                    # Coordinate bounds sampling check
                    if check_bounds and len(shape) == 4 and shape[0] > 0 and shape[2] > 0:
                        t_s = min(sample_t, shape[0])
                        n_s = min(sample_n, shape[2])
                        try:
                            sample = dset[0:t_s, :, 0:n_s, :]
                            if not np.all(np.isfinite(sample)):
                                summary["coords_not_finite"] += 1
                                if len(reasons["coords_not_finite"]) < 10:
                                    reasons["coords_not_finite"].append(f"{fp}::{v}")
                            # Evaluate in-bounds ratio under both (r,c) and (c,r) interpretations
                            if h_attr is not None and w_attr is not None and h_attr > 0 and w_attr > 0:
                                r = sample[..., 0]
                                c = sample[..., 1]
                                rc_ok = np.mean((r >= 0) & (r < h_attr) & (c >= 0) & (c < w_attr))
                                cr_ok = np.mean((c >= 0) & (c < h_attr) & (r >= 0) & (r < w_attr))
                                if max(rc_ok, cr_ok) < in_bounds_ratio:
                                    summary["coords_out_of_bounds"] += 1
                                    if len(reasons["coords_out_of_bounds"]) < 10:
                                        reasons["coords_out_of_bounds"].append(f"{fp}::{v} -> in_bounds rc={rc_ok:.3f}, cr={cr_ok:.3f} vs thr={in_bounds_ratio}")
                        except Exception as e:
                            # Ignore sampling errors but record
                            summary["exceptions"] += 1
                            if len(reasons["exceptions"]) < 10:
                                reasons["exceptions"].append(f"{fp}::{v} sample :: {type(e).__name__}: {e}")

                    # collect per-view basic info
                    rollout_len = int(shape[0]) if len(shape) > 0 else 0
                    view_infos.append((v, shape, dtype, rollout_len))
                    if base_shape is None:
                        base_shape = shape

                # If no view has tracks
                if not view_infos:
                    summary["no_tracks"] += 1
                    if len(reasons["no_tracks"]) < 10:
                        reasons["no_tracks"].append(fp)
                    continue

                # Consistency across views (if multiple views present with tracks)
                if len(view_infos) > 1:
                    shapes_set = {vi[1][:3] for vi in view_infos}  # compare T,horizon,N (ignore last dim=2)
                    if len(shapes_set) != 1:
                        summary["inconsistent_shapes_across_views"] += 1
                        if len(reasons["inconsistent_shapes_across_views"]) < 10:
                            reasons["inconsistent_shapes_across_views"].append(f"{fp} -> {[vi[1] for vi in view_infos]}")

                # Mark file ok and accumulate rollout from first view
                chosen_view, shape, dtype, rollout_len = view_infos[0]
                if rollout_len <= 0:
                    summary["zero_rollout"] += 1
                    if len(reasons["zero_rollout"]) < 10:
                        reasons["zero_rollout"].append(fp)
                    continue

                summary["ok_files"] += 1
                summary["estimated_index_entries"] += rollout_len
                if len(example_ok) < 5:
                    example_ok.append((fp, chosen_view, shape, dtype, rollout_len))

        except Exception as e:
            summary["exceptions"] += 1
            if len(reasons["exceptions"]) < 10:
                reasons["exceptions"].append(f"{fp} :: {type(e).__name__}: {e}")

    return summary, shapes_ctr, dtypes_ctr, sizes_ctr, reasons, example_ok


def main():
    parser = argparse.ArgumentParser(description="Deep check custom HDF5 structure for Motion Tokenizer")
    parser.add_argument("--root", required=True, help="Root directory to scan recursively for .hdf5")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = all)")
    parser.add_argument("--expect-views", nargs="*", default=None, help="Expected view names (e.g., agentview eye_in_hand)")
    parser.add_argument("--require-size-attrs", action="store_true", help="Require per-view attrs height/width to exist and be >0")
    parser.add_argument("--expect-horizon", type=int, default=-1, help="Expected horizon dimension (tracks shape[1]); -1 to skip")
    parser.add_argument("--expect-n-tracks", type=int, default=-1, help="Expected number of tracks (tracks shape[2]); -1 to skip")
    parser.add_argument("--require-vis", action="store_true", help="Require visibility dataset present with compatible shape")
    parser.add_argument("--no-bounds-check", action="store_true", help="Disable coordinate bounds sampling check")
    parser.add_argument("--bounds-thr", type=float, default=0.95, help="Min fraction of sampled coords inside [0,H/W) (either rc or cr)")
    parser.add_argument("--sample-t", type=int, default=4, help="Sample first T windows for bounds check")
    parser.add_argument("--sample-n", type=int, default=256, help="Sample first N tracks per window for bounds check")
    args = parser.parse_args()

    summary, shapes_ctr, dtypes_ctr, sizes_ctr, reasons, example_ok = scan_hdf5(
        root_dir=args.root,
        limit=args.limit,
        expect_views=args.expect_views,
        require_attrs_size=bool(args.require_size_attrs),
        expect_horizon=args.expect_horizon,
        expect_n_tracks=args.expect_n_tracks,
        require_vis=bool(args.require_vis),
        check_bounds=not args.no_bounds_check,
        in_bounds_ratio=args.bounds_thr,
        sample_t=args.sample_t,
        sample_n=args.sample_n,
    )

    print("=== HDF5 Deep Check Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nTop shapes (tracks):")
    for shape, cnt in shapes_ctr.most_common(10):
        print(f"  {shape}: {cnt}")

    print("\nDtypes (tracks):")
    for dt, cnt in dtypes_ctr.most_common():
        print(f"  {dt}: {cnt}")

    print("\nImage sizes (H,W) from attrs:")
    for sz, cnt in sizes_ctr.most_common(10):
        print(f"  {sz}: {cnt}")

    print("\nExamples of OK files (up to 5):")
    for fp, view, shape, dtype, rollout in example_ok:
        print(f"  {fp} | view={view} | shape={shape} | dtype={dtype} | rollout_len={rollout}")

    print("\nReasons and sample files (up to 10 each):")
    for reason, fps in reasons.items():
        print(f"  {reason}: {len(fps)}")
        for s in fps:
            print(f"    - {s}")


if __name__ == "__main__":
    main()
