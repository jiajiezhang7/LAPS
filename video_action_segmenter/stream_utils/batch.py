from pathlib import Path
from typing import List, Optional
import os
import sys
import subprocess
import time


def run_batch_over_folder(
    in_dir: Path,
    params_path: str,
    exts: List[str] = None,
    recursive: bool = True,
    enable_parallel: bool = False,
    gpu_ids: Optional[List[int]] = None,
    max_procs_per_gpu: int = 1,
    poll_interval: float = 0.2,
) -> None:
    """Run stream inference over all videos in a folder using a subprocess per file.

    Args:
        in_dir: input directory containing videos
        params_path: path to params.yaml to pass through
        exts: list of allowed extensions (lowercase, with dot). Defaults to common ones
        recursive: whether to recurse subdirectories
        enable_parallel: whether to launch multiple subprocesses concurrently
        gpu_ids: optional list of GPU IDs to bind subprocesses to (requires enable_parallel)
        max_procs_per_gpu: max concurrent subprocesses per GPU (>=1)
        poll_interval: seconds to wait between polling subprocess status in parallel mode
    """
    if exts is None:
        exts = [".mp4", ".mov", ".avi", ".mkv"]
    dir_path = Path(in_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise RuntimeError(f"folder 不存在或不是目录: {dir_path}")

    files: List[Path] = []
    it = dir_path.rglob("*") if recursive else dir_path.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files = sorted(files)
    if not files:
        raise RuntimeError(f"在目录中未找到视频文件: {dir_path}")

    ok_count = 0
    fail_count = 0

    # 并行模式：使用有限并行调度分配到指定 GPU
    if enable_parallel and gpu_ids:
        try:
            gpu_list = [int(g) for g in gpu_ids]
        except Exception:
            gpu_list = []
        gpu_list = [g for g in gpu_list if g >= 0]
        if not gpu_list:
            print("[Batch][WARN] enable_parallel=true 但未提供有效 GPU 列表，退回顺序执行。")
            enable_parallel = False
        else:
            try:
                max_procs_per_gpu = max(1, int(max_procs_per_gpu))
            except Exception:
                max_procs_per_gpu = 1
            try:
                poll_interval = max(0.01, float(poll_interval))
            except Exception:
                poll_interval = 0.2

    if enable_parallel and gpu_ids:
        total_slots = len(gpu_list) * max_procs_per_gpu
        print(f"[Batch] 共发现 {len(files)} 个视频，将并行推理。GPU={gpu_list} slots_per_gpu={max_procs_per_gpu} (总槽位 {total_slots})")
        active = []  # (proc, gpu, path)
        gpu_in_use = {gid: 0 for gid in gpu_list}
        index = 0
        try:
            while index < len(files) or active:
                # 尽可能填满空闲槽位
                while index < len(files):
                    available_gpu = None
                    for gid in gpu_list:
                        if gpu_in_use[gid] < max_procs_per_gpu:
                            available_gpu = gid
                            break
                    if available_gpu is None:
                        break
                    vp = files[index]
                    index += 1
                    print(f"[Batch][GPU{available_gpu}] 开始: {vp}")
                    env = dict(os.environ)
                    env["MT_OVERRIDE_INPUT_PATH"] = str(vp)
                    env["CUDA_VISIBLE_DEVICES"] = str(available_gpu)
                    cmd = [
                        sys.executable,
                        "-m",
                        "video_action_segmenter.stream_inference",
                        "--params",
                        params_path,
                        "--gpu-id",
                        "0",
                    ]
                    try:
                        proc = subprocess.Popen(cmd, env=env)
                    except Exception as e:
                        fail_count += 1
                        print(f"[Batch][ERR] 启动失败: {vp} | {e}")
                        continue
                    active.append((proc, available_gpu, vp))
                    gpu_in_use[available_gpu] += 1

                if not active:
                    # 没有进行中的任务，继续下一轮分配
                    continue

                time.sleep(poll_interval)
                still_active = []
                for proc, gid, path in active:
                    ret = proc.poll()
                    if ret is None:
                        still_active.append((proc, gid, path))
                        continue
                    gpu_in_use[gid] = max(0, gpu_in_use[gid] - 1)
                    if ret == 0:
                        ok_count += 1
                        print(f"[Batch][GPU{gid}] 完成: {path} | exit=0")
                    else:
                        fail_count += 1
                        print(f"[Batch][GPU{gid}][WARN] 退出码 {ret}: {path}")
                active = still_active
        except KeyboardInterrupt:
            print("[Batch] 接收到中断，正在终止所有子任务...")
            for proc, gid, path in active:
                try:
                    proc.terminate()
                except Exception:
                    pass
            for proc, _, _ in active:
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
            raise
    else:
        print(f"[Batch] 共发现 {len(files)} 个视频，将顺序推理。")
        for i, vp in enumerate(files):
            print(f"[Batch] ({i+1}/{len(files)}) 开始: {vp}")
            env = dict(os.environ)
            env["MT_OVERRIDE_INPUT_PATH"] = str(vp)
            cmd = [sys.executable, "-m", "video_action_segmenter.stream_inference", "--params", params_path]
            if gpu_ids:
                try:
                    first_gpu = int(gpu_ids[0])
                    if first_gpu >= 0:
                        env["CUDA_VISIBLE_DEVICES"] = str(first_gpu)
                        cmd.extend(["--gpu-id", "0"])
                except Exception:
                    pass
            try:
                ret = subprocess.run(cmd, env=env)
                if ret.returncode == 0:
                    ok_count += 1
                else:
                    fail_count += 1
                    print(f"[Batch][WARN] 子任务退出码 {ret.returncode}: {vp}")
            except KeyboardInterrupt:
                print("[Batch] 接收到中断，停止后续任务。")
                break
            except Exception as e:
                fail_count += 1
                print(f"[Batch][ERR] 运行失败: {vp} | {e}")

    print(f"[Batch] 完成。成功 {ok_count}，失败 {fail_count}。")
