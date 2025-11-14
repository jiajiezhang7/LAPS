#!/usr/bin/env python3
"""
Batch runner for energy comparison plotting.

- Scans all video sample folders under D01_LAPS
- Enumerates start_sec in steps of 30s (0, 30, 60, ...) with duration_sec=60s
- Ensures start_sec + duration_sec <= total_sec (default 600s)
- Runs: python -m video_action_segmenter.scripts.plot_energy_comparison ...

Tips:
- Please run this script under conda env "laps".
  e.g., conda run -n laps python -m video_action_segmenter.scripts.batch_plot_energy_comparison
"""

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_VIDEOS_ROOT = Path(
    "/home/johnny/action_ws/datasets/output/segmentation_outputs/D02_LAPS"
)
DEFAULT_OUTPUT_BASE = Path(
    "/home/johnny/action_ws/supplement_output/segmentor/energy_curves"
)
DEFAULT_TOTAL_SEC = 600
DEFAULT_DURATION_SEC = 90
DEFAULT_STEP_SEC = 30
DEFAULT_DPI = 300


def find_video_names(videos_root: Path) -> List[str]:
    names: List[str] = []
    if not videos_root.exists():
        print(f"[ERR] videos_root does not exist: {videos_root}")
        return names
    for p in sorted(videos_root.iterdir()):
        if p.is_dir():
            names.append(p.name)
    return names


def iter_start_secs(total_sec: int, duration_sec: int, step_sec: int) -> Iterable[int]:
    if duration_sec <= 0 or step_sec <= 0 or total_sec <= 0:
        return []
    last_start = total_sec - duration_sec
    s = 0
    while s <= last_start:
        yield s
        s += step_sec


def build_paths(video_name: str) -> dict:
    base = Path("/home/johnny/action_ws")
    paths = {
        "optical_flow_jsonl": base
        / f"datasets/output/segmentation_outputs/D02_optical_flow/{video_name}/stream_energy_optical_flow_mag_mean.jsonl",
        "action_energy_jsonl": base
        / f"datasets/output/segmentation_outputs/D02_LAPS/{video_name}/stream_energy_quantized_token_diff_l2_mean.jsonl",
        "gt_json": base
        / f"datasets/gt_annotations/_temporal_jitter/D02/{video_name}_segments.json",
        "segmentor_json": base
        / f"datasets/output/segmentation_outputs/D02_LAPS/{video_name}/segmented_videos/{video_name}_segments.json",
    }
    return paths


def validate_inputs(paths: dict) -> Optional[str]:
    missing = [k for k, p in paths.items() if k != "segmentor_json" and not Path(p).exists()]
    if missing:
        return (
            "Missing required input files: "
            + ", ".join([f"{m} -> {paths[m]}" for m in missing])
        )
    return None


def run_one(
    paths: dict,
    output_dir: Path,
    start_sec: int,
    duration_sec: int,
    dpi: int,
) -> int:
    cmd = [
        "python",
        "-m",
        "video_action_segmenter.scripts.plot_energy_comparison",
        "--optical-flow-jsonl",
        str(paths["optical_flow_jsonl"]),
        "--action-energy-jsonl",
        str(paths["action_energy_jsonl"]),
        "--gt-json",
        str(paths["gt_json"]),
        "--output-dir",
        str(output_dir),
        "--start-sec",
        str(start_sec),
        "--duration-sec",
        str(duration_sec),
        "--dpi",
        str(dpi),
    ]

    # Pass segmentor JSON only if it exists (optional in the plot script)
    seg_p = Path(paths["segmentor_json"]) if paths.get("segmentor_json") else None
    if seg_p and seg_p.exists():
        cmd.extend(["--segmentor-json", str(seg_p)])

    print("[CMD] ", " ".join(cmd))
    try:
        completed = subprocess.run(cmd, check=False)
        return completed.returncode
    except Exception as e:
        print(f"[ERR] Subprocess failed: {e}")
        return 1


def main():
    ap = argparse.ArgumentParser(
        description="Batch-run plot_energy_comparison over videos and start offsets",
    )
    ap.add_argument(
        "--videos-root",
        type=str,
        default=str(DEFAULT_VIDEOS_ROOT),
        help="Root folder that contains per-video folders under D01_LAPS",
    )
    ap.add_argument(
        "--output-base",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help="Base output directory; a subfolder per video will be created",
    )
    ap.add_argument(
        "--total-sec",
        type=int,
        default=DEFAULT_TOTAL_SEC,
        help="Assumed total video length in seconds (default: 600)",
    )
    ap.add_argument(
        "--duration-sec",
        type=int,
        default=DEFAULT_DURATION_SEC,
        help="Duration per crop in seconds (default: 60)",
    )
    ap.add_argument(
        "--step-sec",
        type=int,
        default=DEFAULT_STEP_SEC,
        help="Step size for start_sec in seconds (default: 30)",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Output DPI forwarded to plotting script (default: 300)",
    )
    ap.add_argument(
        "--video-names",
        type=str,
        default=None,
        help=(
            "Comma-separated list of video names to process; if omitted, scan all"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned commands without running",
    )

    args = ap.parse_args()

    videos_root = Path(args.videos_root)
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    if args.video_names:
        video_names = [v.strip() for v in args.video_names.split(",") if v.strip()]
    else:
        video_names = find_video_names(videos_root)

    if not video_names:
        print("[WARN] No videos found to process.")
        return

    print(f"[INFO] Found {len(video_names)} video(s)")

    total = 0
    failed = 0

    for vid in video_names:
        print(f"\n[INFO] Processing video: {vid}")
        paths = build_paths(vid)
        err = validate_inputs(paths)
        if err:
            print(f"[WARN] Skip {vid}: {err}")
            continue

        # Create per-video output directory
        out_dir = output_base / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        for s in iter_start_secs(args.total_sec, args.duration_sec, args.step_sec):
            total += 1
            if args.dry_run:
                print(
                    f"[DRY] Would run {vid}: start={s}s, duration={args.duration_sec}s -> {out_dir}"
                )
                continue

            rc = run_one(
                paths=paths,
                output_dir=out_dir,
                start_sec=s,
                duration_sec=args.duration_sec,
                dpi=args.dpi,
            )
            if rc != 0:
                failed += 1
                print(f"[WARN] Command failed for {vid} @ {s}s (rc={rc})")

    print(
        f"\n[DONE] Planned runs: {total}. Failures: {failed}. Outputs under: {output_base}"
    )


if __name__ == "__main__":
    main()

