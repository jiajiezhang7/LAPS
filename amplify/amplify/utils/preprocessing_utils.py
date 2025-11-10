import glob
import os
import sys

import h5py
import numpy as np
import torch

from amplify.utils.cfg_utils import get_device
from cotracker.utils.visualizer import Visualizer
from einops import rearrange, repeat
from tqdm import tqdm
from amplify.utils.kp_utils.query_utils import grid_queries

def inital_save_h5(path, skip_exist, view_names=["agentview", "eye_in_hand"]):
    try:
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                file_processed = True
                for view in view_names:
                    if f"root/{view}" not in f:
                        file_processed = False
                        break
                if file_processed and skip_exist:
                    print(f"File {path} already exists, skipping.")
                    return None
    except Exception as e:
        print(f"Error in opening file {path}, {e}")

    f = h5py.File(path, 'w')
    return f

def tracks_from_video(
    video,
    track_model,
    init_queries='uniform',
    reinit=True,
    horizon=16,
    n_tracks=400,
    batch_size=8,
    dim_order='tchw',
    precision: str = 'fp32',
    auto_batch: bool = True,
    min_batch_size: int = 1,
):
    """
    Use cotracker to generate tracks and vis from a video.
    """
    if dim_order == 'thwc':
        video = rearrange(video, "t h w c -> t c h w")
    T, C, H, W = video.shape
    assert C==3

    device = get_device()
    # Don't allow mps because the border padding mode used by CoTracker
    # is unsupported on mps
    if device == torch.device("mps"):
        device = get_device("cpu")

    with torch.inference_mode():
        # Initialize video and query layout for cotracker
        # NOTE: Use torch.tensor(...) instead of from_numpy to avoid fragile
        # NumPy/PyTorch ABI/type mismatches observed in some conda envs.
        video_t = torch.tensor(video, dtype=torch.float32)
        if device.type == "cuda":
            # Pin host memory to accelerate H2D transfer; use non_blocking copy
            video = video_t.pin_memory().to(device, non_blocking=True)
        else:
            video = video_t.to(device)
        if init_queries == 'uniform':
            queries = grid_queries(views=1, n_tracks=n_tracks, device=device)
        else:
            raise NotImplementedError(f"Query initialization method {init_queries} not implemented.")
        queries = queries.cotracker(H) # (1, n_tracks, 3)

        # Decide AMP dtype
        amp_dtype = None
        if device.type == 'cuda':
            p = (precision or 'fp32').lower()
            if p in ('bf16', 'bfloat16', 'amp_auto'):
                # Prefer bf16 on Ampere+; fallback to fp16 if bf16 unsupported
                try:
                    major, _ = torch.cuda.get_device_capability()
                    if major >= 8:
                        amp_dtype = torch.bfloat16 if p != 'fp16' else torch.float16
                    elif p in ('fp16', 'amp_auto'):
                        amp_dtype = torch.float16
                except Exception:
                    if p == 'fp16':
                        amp_dtype = torch.float16
            elif p in ('fp16', 'half'):
                amp_dtype = torch.float16

        if reinit:
            # From each frame, track next horizon steps with re-initialized queries
            queries = queries.squeeze(0) # (n_tracks, 3) remove dimension to batch later
            tracks = torch.zeros(T, horizon, n_tracks, 2, device=device)
            vis = torch.zeros(T, horizon, n_tracks, device=device)
            # pad video to repeat last frame horizon-1 times, to ensure all videos are same length
            padding = repeat(video[-1], "c h w ->  t c h w", t=horizon-1)
            video = torch.cat([video, padding], dim=0)
            # take windows of horizon-length frame sequences at each time step
            video_windows = video.unfold(0, horizon, 1) # (T, C, H, W, horizon)
            video_windows = rearrange(video_windows, "t c h w tl -> t tl c h w")

            # Auto-batch with AMP and OOM backoff
            bs = int(batch_size) if int(batch_size) > 0 else 1
            start = 0
            pbar = tqdm(total=T, disable=True)
            try:
                while start < T:
                    end = min(start + bs, T)
                    video_batch = video_windows[start:end]
                    q_batch = repeat(queries, "n d -> b n d", b=video_batch.shape[0])
                    try:
                        if amp_dtype is not None:
                            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                                t_out, v_out = track_model(video_batch, q_batch)
                        else:
                            t_out, v_out = track_model(video_batch, q_batch)
                    except RuntimeError as e:
                        msg = str(e).lower()
                        oom = ('out of memory' in msg) or ('cuda error: out of memory' in msg)
                        if oom and auto_batch and bs > max(1, int(min_batch_size)):
                            # Backoff and retry this window
                            torch.cuda.empty_cache()
                            bs = max(1, bs // 2)
                            continue
                        if amp_dtype is not None:
                            # Disable AMP once and retry in fp32
                            amp_dtype = None
                            torch.cuda.empty_cache()
                            continue
                        raise

                    tracks[start:end], vis[start:end] = t_out, v_out
                    pbar.update(end - start)
                    start = end
            finally:
                pbar.close()
        else:
            # Online mode for longer videos
            video = video.unsqueeze(0) # (1, T, C, H, W)
            if amp_dtype is not None:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    track_model(video_chunk=video, is_first_step=True, queries=queries)
            else:
                track_model(video_chunk=video, is_first_step=True, queries=queries)
            for ind in tqdm(range(0, video.shape[1] - track_model.step, track_model.step)):
                if amp_dtype is not None:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        tracks, vis = track_model(
                            video_chunk=video[:, ind : ind + track_model.step * 2]
                        )  # B T N 2,  B T N 1
                else:
                    tracks, vis = track_model(
                        video_chunk=video[:, ind : ind + track_model.step * 2]
                    )  # B T N 2,  B T N 1
    # Ensure float32 CPU outputs
    return tracks.float().cpu().numpy(), vis.float().cpu().numpy()


def _depth_anything_model_configs(metric: bool):
    if metric:
        return {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
    else:
        return {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        }


def load_depth_anything_v2(
    metric_depth: bool = False,
    encoder: str = 'vitl',
    dataset: str = 'hypersim',
    max_depth: int = 20,
    checkpoints_dir: str = 'checkpoints',
    depth_anything_root: str = None,
    device: torch.device = None,
):
    """
    Load Depth-Anything V2 (metric or non-metric) with weights from `checkpoints_dir`.

    Returns a torch.nn.Module in eval mode moved to the specified device (or auto device).
    """
    # Add optional local path for module import
    if depth_anything_root is None:
        depth_anything_root = os.path.join(os.getcwd(), 'Depth-Anything-V2')
    if os.path.isdir(depth_anything_root) and depth_anything_root not in sys.path:
        sys.path.append(depth_anything_root)

    model_configs = _depth_anything_model_configs(metric_depth)
    if encoder not in model_configs:
        raise ValueError(f"Unsupported depth encoder '{encoder}' for Depth-Anything V2")

    if device is None:
        device = get_device()

    try:
        if metric_depth:
            from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
            cfg = {**model_configs[encoder], 'max_depth': max_depth}
            model = DepthAnythingV2(**cfg)
            ckpt_path = os.path.join(checkpoints_dir, f'depth_anything_v2_metric_{dataset}_{encoder}.pth')
        else:
            from depth_anything_v2.dpt import DepthAnythingV2
            cfg = model_configs[encoder]
            model = DepthAnythingV2(**cfg)
            ckpt_path = os.path.join(checkpoints_dir, f'depth_anything_v2_{encoder}.pth')

        state = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state)
        return model.eval().to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load Depth-Anything V2: {e} \n\n Ensure you have cloned https://github.com/DepthAnything/Depth-Anything-V2, installed requirements, and downloaded checkpoints. \n Also ensure the checkpoint you download matches the encoder arg, e.g. 'vitl' and is placed in the folder indicated by checkpoints_dir.") from e


def write_key(f, key, data, dtype=None):
    if key in f:
        del f[key]
    if dtype is not None:
        f.create_dataset(key, data=data, dtype=dtype)
    else:
        f.create_dataset(key, data=data)


def preprocess_datapoint(outfile, cfg, models, **kwargs):
    """
    Writes specified keys (cfg.write_keys) to outfile, doing any preprocessing necessary.
    """
    # Make any parent directories if they don't exist
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Check if file can be opened
    try:
        with h5py.File(outfile, "a") as f:
            pass
    except OSError:
        # if unable to synchronously open file, delete the file and write a new one
        print("WARNING: Error while opening file, writing new one... only keys specified in this run will be written!")
        os.remove(outfile)

    # Write to file
    with h5py.File(outfile, "a") as f:
        # Only write specified keys
        for key in cfg.write_keys:
            if key in f and cfg.skip_exist:
                print(f"Skipping {key} in {outfile}...")
                continue
            print(f"Processing {key} in {outfile}...")
            if key in kwargs:
                write_key(f, key, kwargs[key])
            elif key=="tracks":
                assert models["cotracker"] is not None, "cotracker model must be provided to preprocess tracks!"
                tracks, vis = tracks_from_video(
                    video=kwargs["video"],
                    track_model=models["cotracker"],
                    init_queries=cfg.init_queries,
                    reinit=cfg.reinit,
                    horizon=cfg.horizon,
                    n_tracks=cfg.n_tracks,
                    batch_size=cfg.batch_size,
                    dim_order=cfg.dim_order
                )
                write_key(f, "tracks", tracks)
                write_key(f, "visibility", vis)
            elif key=="tracks_uniform":
                assert models["cotracker"] is not None, "cotracker model must be provided to preprocess tracks_uniform!"
                tracks, vis = tracks_from_video(
                    video=kwargs["video"],
                    track_model=models["cotracker"],
                    init_queries=cfg.init_queries,
                    reinit=cfg.reinit,
                    horizon=cfg.horizon,
                    n_tracks=cfg.n_tracks,
                    batch_size=cfg.batch_size,
                    dim_order=cfg.dim_order
                )
                write_key(f, "tracks_uniform", tracks)
                write_key(f, "visibility_uniform", vis)
            elif key=="agent_env_seg":
                assert models["seg_model"] is not None, "seg_model must be provided to preprocess agent_env_seg!"
                assert "tracks" in f, "tracks must be present in file to preprocess agent_env_seg!"
                try:
                    agent_env_seg = agent_env_seg_from_tracks(models["seg_model"], f["video"][:], f["tracks"][:], cfg.range)
                except Exception:
                    agent_env_seg = agent_env_seg_from_tracks(models["seg_model"], f["video"][:], f["tracks"][:])
                write_key(f, "agent_env_seg", agent_env_seg)
            elif key=="text_emb":
                assert models["text_encoder"] is not None, "text_encoder model must be provided to preprocess text_emb!"
                assert "text" in kwargs, "text must be provided to preprocess text_emb!"
                text_emb = models["text_encoder"](kwargs["text"]).cpu().numpy()
                write_key(f, "text_emb", text_emb, dtype="float32")
            else:
                raise ValueError(f"Key {key} not implemented!")


def check_preprocess_status(dir, file_depth, keys):
    """
    Checks files at a depth `file_depth` in directory `dir` for the existence of keys `keys`.
    Outputs a summary like:
        {key1}: {num_files_with_key}/{num_files}
        {key2}: {num_files_with_key}/{num_files}
        ...
    """
    print(f"Checking files at depth {file_depth} in {dir} for keys {keys}...")
    paths = sorted(glob.glob(os.path.join(dir, *("*" * (file_depth - 1)))))
    num_files = len(paths)
    key_counts = {key: 0 for key in keys}
    missing_paths = []
    for path in tqdm(paths):
        try:
            with h5py.File(path, "r") as f:
                for key in keys:
                    if key in f:
                        key_counts[key] += 1
                    else:
                        # print(f"Key {key} not found in {path}, skipping...")
                        missing_paths.append(path)
        except:
            print(f"Error with while opening {path}, skipping...")
            continue
    for key in keys:
        print(f"{key}: {key_counts[key]}/{num_files}")
        
    # print(missing_paths)


if __name__ == "__main__":
    dir = "./preprocessed_data/bridge_data_v2" # TODO: make this work for libero_demos
    file_depth = 6
    keys = ["tracks"] #["video", "tracks", "visibility", "actions", "states", "text", "text_emb", "agent_env_seg"]
    check_preprocess_status(dir, file_depth, keys)
