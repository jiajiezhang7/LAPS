import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

import wandb
from amplify.loaders import LiberoDataset


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CustomConcatDataset(ConcatDataset):
    """
    Implements a custom version of concat dataset that allows to call the function `dataset.get_full_episode_batch()` on the concatenated datasets, each of which implements this function (check with an assert in the init)
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        for dataset in datasets:
            assert hasattr(dataset, "get_full_episode_batch")
        self.datasets = datasets

    def get_full_episode_batch(self, idx):
        """
        Returns a full episode batch for the given index.
        """
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset.get_full_episode_batch(idx)
            else:
                idx -= len(dataset)
        raise ValueError("Index out of range")


class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset  # Store reference to the original dataset

    def get_full_episode_batch(self, idx):
        return self.dataset.get_full_episode_batch(self.indices[idx])


def batch_to_device(batch, device):
    for key in batch:
        if batch[key] is not None:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
            elif isinstance(batch[key], np.ndarray):
                batch[key] = torch.from_numpy(batch[key]).to(device)
            elif isinstance(batch[key], dict):
                batch[key] = batch_to_device(batch[key], device)

    return batch


# Robust collate to guard against numpy scalar types and odd dtypes
# This wraps default_collate after converting any np.generic scalars to Python scalars
# and leaves ndarrays/tensors unchanged.
def _to_python_scalars(x):
    import numpy as _np
    if isinstance(x, _np.generic):
        return x.item()
    elif isinstance(x, dict):
        return {k: _to_python_scalars(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        t = type(x)
        return t(_to_python_scalars(v) for v in x)
    else:
        return x

def safe_collate(batch):
    # Prefer explicit stacking for numpy arrays to avoid PyTorch dtype inference issues.
    # Assumes each element of batch is a dict with consistent keys and shapes.
    import numpy as _np
    import torch as _torch

    # First, sanitize numpy scalars inside dicts/lists
    batch = [_to_python_scalars(b) for b in batch]

    if isinstance(batch[0], dict):
        out = {}
        keys = batch[0].keys()
        for k in keys:
            vals = [b[k] for b in batch]
            v0 = vals[0]
            if isinstance(v0, _np.ndarray):
                try:
                    arr = _np.stack(vals, axis=0)
                except Exception:
                    # fallback: ensure each is contiguous before stacking
                    vals = [v.copy() for v in vals]
                    arr = _np.stack(vals, axis=0)
                # Use torch.tensor instead of from_numpy to avoid ABI/dtype issues
                if arr.dtype.kind in ('f',):
                    out[k] = _torch.tensor(arr, dtype=_torch.float32)
                elif arr.dtype.kind in ('i','u'):
                    out[k] = _torch.tensor(arr, dtype=_torch.int64)
                elif arr.dtype.kind in ('b',):
                    out[k] = _torch.tensor(arr, dtype=_torch.bool)
                else:
                    out[k] = _torch.tensor(arr)
            elif _torch.is_tensor(v0):
                out[k] = _torch.stack(vals, dim=0)
            elif isinstance(v0, (list, tuple)):
                # recursively collate lists/tuples
                out[k] = default_collate(vals)
            else:
                # numbers/strings/etc.
                out[k] = default_collate(vals)
        return out
    else:
        # Fallback to default behavior
        return default_collate(batch)


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def get_checkpoint_dir(stage, run_name, resume=False):
    """
    Returns checkpoint dir for a given stage (motion_tokenizer, forward_dynamics, inverse_dynamics, inverse_dynamics_finetuned)

    Structure is "checkpoints/{stage}/{run_name}_{id}", where id is a unique int id numbering to avoid overwriting checkpoints, unless resume is True (then last existing dir is used)

    checkpoints/stage/run_name
    checkpoints/stage/run_name_1
    checkpoints/stage/run_name_2
    etc.
    """
    assert stage in ["motion_tokenizer", "forward_dynamics", "inverse_dynamics"]
    run_name = str(run_name)
    stage_dir = os.path.join("checkpoints", stage)
    os.makedirs(stage_dir, exist_ok=True)

    # Identify existing checkpoint directories matching run_name
    matching_runs = [f for f in os.listdir(stage_dir) if f.startswith(run_name)]
    matching_run_ids = []

    for f in matching_runs:
        parts = f.split("_")
        if f == run_name:  # Handle case where base run_name exists
            matching_run_ids.append(0)
        elif len(parts) > 1 and parts[-1].isdigit():
            matching_run_ids.append(int(parts[-1]))

    matching_run_ids.sort()

    # if resume, and matching runs exist, use the last one (highest id)
    # if resume, and no matching runs exist, create new one with id 0
    # if not resume, and matching runs exist, create new one with id + 1
    # if not resume, and no matching runs exist, create new one with id 0
    if matching_run_ids:
        if resume:
            run_id = matching_run_ids[-1]
        else:
            run_id = matching_run_ids[-1] + 1
    else:
        run_id = 0

    checkpoint_dir = os.path.join(stage_dir, f"{run_name}_{run_id}" if run_id > 0 else run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    return checkpoint_dir


def latest_checkpoint_from_dir(checkpoint_dir):
    """
    Returns the path to the latest checkpoint in a given directory. If a "latest.pt" file exists, it is returned, otherwise the latest .pt file is returned.
    """
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if "latest.pt" in checkpoints:
        return os.path.join(checkpoint_dir, "latest.pt")
    elif checkpoints:
        checkpoints.sort()
        return os.path.join(checkpoint_dir, checkpoints[-1])
    else:
        print(f"No checkpoints found in {checkpoint_dir}, nothing to resume")
        return None


def save_checkpoint(
    checkpoint_path,
    epoch,
    cfg,
    model,
    optimizer,
    scaler=None,
    train_loss=None,
    val_loss=None,
    train_global_iter=None,
    val_global_iter=None,
    scheduler=None,
):
    """
    Saves the model and optimizer state.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_global_iter": train_global_iter,
        "val_global_iter": val_global_iter,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "wandb_run_id": wandb.run.id,
    }
    torch.save(checkpoint, checkpoint_path)
    print("Saved checkpoint at ", checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(checkpoint_path)
    checkpoint_cfg = OmegaConf.create(checkpoint["config"])
    if checkpoint_cfg.compile or "_orig_mod.cls_token" in checkpoint["model"].keys():
        checkpoint["model"] = unwrap_compiled_state_dict(checkpoint["model"])

    model.load_state_dict(checkpoint["model"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    checkpoint_info = {
        "epoch": checkpoint["epoch"],
        "train_loss": checkpoint["train_loss"],
        "val_loss": checkpoint["val_loss"],
        "train_global_iter": checkpoint["train_global_iter"],
        "val_global_iter": checkpoint["val_global_iter"],
        "wandb_run_id": checkpoint["wandb_run_id"],
    }

    print(f"Loaded checkpoint from {checkpoint_path}")

    return model, optimizer, scheduler, scaler, checkpoint_cfg, checkpoint_info


def parse_dataset_strings(dataset_strings: List) -> List[Tuple[str, Dict[str, float]]]:
    """
    parses a list of dataset names and returns a list [(dataset, modality_dict)] where modality_dict is a dictionary of the form {"traj": traj_split, "action": action_split}

    E.g. libero_10_demo:traj0.8:action0.04 -> (libero_10_demo, {"traj": 0.8, "action": 0.04})
    """
    dataset_info = []
    for dataset_str in dataset_strings:
        if ":" in dataset_str:
            dataset = dataset_str.split(":")[0]
            modalities = dataset_str.split(":")[1:]

            modality_dict = {}
            for modality in modalities:
                if "traj" in modality:
                    modality_dict["traj"] = float(modality.split("traj")[1])
                elif "action" in modality:
                    modality_dict["action"] = float(modality.split("action")[1])

            dataset_info.append((dataset, modality_dict))

    return dataset_info


def get_datasets(
    root_dir,
    train_datasets,
    val_datasets,
    keys_to_load,
    motion_tokenizer_cfg,
    aug_cfg=None,
    task_names=None, # real only
    normalize_actions=True, # real only
    action_key='abs_actions', # real only
) -> Tuple[Dict[str, CustomConcatDataset], Dict[str, CustomConcatDataset]]:
    """
    Loads train and val datasets.

    Returns a dictionary of concatenated datasets for each modality
    """

    common_cfgs = {
        "root_dir": root_dir,
        "track_method": motion_tokenizer_cfg.track_method,
        "cond_cameraviews": motion_tokenizer_cfg.cond_cameraviews,
        "keys_to_load": keys_to_load,
        "img_shape": motion_tokenizer_cfg.img_shape,
        "true_horizon": motion_tokenizer_cfg.true_horizon,
        "track_pred_horizon": motion_tokenizer_cfg.track_pred_horizon,
        "interp_method": motion_tokenizer_cfg.interp_method,
        "num_tracks": motion_tokenizer_cfg.num_tracks,
        "aug_cfg": aug_cfg,
    }

    # Train datasets
    train_dataset_info = parse_dataset_strings(train_datasets)
    print("train_dataset_info: ", train_dataset_info)

    train_datasets = {}
    for dataset_name, modality_dict in train_dataset_info:
        for modality in modality_dict:
            if "libero" in str(dataset_name):
                dataset = LiberoDataset(
                    dataset_names=[dataset_name],
                    demo_subset=modality_dict[modality],
                    libero_path=motion_tokenizer_cfg.libero_path,
                    **common_cfgs,
                )
            elif "custom_segments" in str(dataset_name):
                from amplify.loaders import CustomSegmentsDataset
                dataset = CustomSegmentsDataset(
                    dataset_names=[dataset_name],
                    video_root=getattr(motion_tokenizer_cfg, "video_root", None),
                    **common_cfgs,
                )
            else:
                raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")

            train_datasets.setdefault(modality, []).append(dataset)

    for modality in train_datasets.keys():
        train_datasets[modality] = CustomConcatDataset(train_datasets[modality])

    if val_datasets is not None:
        # Val dataset
        val_dataset_info = parse_dataset_strings(val_datasets)
        print("val_dataset_info: ", val_dataset_info)

        val_datasets = {}
        for dataset_name, modality_dict in val_dataset_info:
            for modality in modality_dict:
                if "libero" in str(dataset_name):
                    dataset = LiberoDataset(
                        dataset_names=[dataset_name],
                        demo_subset=modality_dict[modality],
                        libero_path=motion_tokenizer_cfg.libero_path,
                        **common_cfgs,
                    )
                elif "custom_segments" in str(dataset_name):
                    from amplify.loaders import CustomSegmentsDataset
                    dataset = CustomSegmentsDataset(
                        dataset_names=[dataset_name],
                        video_root=getattr(motion_tokenizer_cfg, "video_root", None),
                        **common_cfgs,
                    )
                else:
                    raise NotImplementedError(f"Validation dataset {dataset_name} is not implemented.")

                val_datasets.setdefault(modality, []).append(dataset)

        for modality in val_datasets.keys():
            val_datasets[modality] = CustomConcatDataset(val_datasets[modality])

    return train_datasets, val_datasets


def get_dataloaders(
    train_dataset_concat_dict,
    val_dataset_concat_dict,
    gpu_max_bs,
    num_workers,
    epoch_size=None,
    val_epoch_size=None,
    quick=False,
):
    """
    Gets train and val dataloaders
    """
    if quick:
        epoch_size = 2*gpu_max_bs
        val_epoch_size = 2*gpu_max_bs

    train_dataloader_dict = {}
    for key in train_dataset_concat_dict:
        if train_dataset_concat_dict[key] is not None:
            print(f"{key} dataset size: ", len(train_dataset_concat_dict[key]))
            sampler = RandomSampler(
                train_dataset_concat_dict[key], num_samples=epoch_size
            )
            train_dataloader_dict[key] = DataLoader(
                train_dataset_concat_dict[key],
                batch_size=gpu_max_bs,
                sampler=sampler,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                collate_fn=safe_collate,
            )

    if val_dataset_concat_dict is not None:
        val_dataloader_dict = {}
        for key in val_dataset_concat_dict:
            if val_dataset_concat_dict[key] is not None:
                print(f"{key} dataset size: ", len(val_dataset_concat_dict[key]))
                sampler = RandomSampler(
                    val_dataset_concat_dict[key],num_samples=val_epoch_size
                )
                val_dataloader_dict[key] = DataLoader(
                    val_dataset_concat_dict[key],
                    batch_size=gpu_max_bs,
                    sampler=sampler,
                    num_workers=int(num_workers > 0),  # set to 0 if num_workers is 0
                    persistent_workers=num_workers > 0,
                    collate_fn=safe_collate,
                )
    else:
        val_dataloader_dict = None

    return train_dataloader_dict, val_dataloader_dict


def get_vis_dataset(dataloader_dict):
    if "traj" in dataloader_dict.keys():
        if "traj_action" in dataloader_dict.keys():
            vis_dataset = random.choice([dataloader_dict["traj"], dataloader_dict["traj_action"]])
        else:
            vis_dataset = dataloader_dict["traj"]

        if vis_dataset.__class__.__name__ == "BridgeDataset":
            fps = 5
        else:
            fps = 15
        return vis_dataset, fps
    else:
        return None, None


def index_batch(batch, indices):
    ibatch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            ibatch[key] = value[indices]
        else:
            ibatch[key] = [value[i] for i in indices]

    return ibatch

def unwrap_compiled_state_dict(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        # remove leading "_orig_mod." if present
        if k.startswith("_orig_mod."):
            new_sd[k[len("_orig_mod.") :]] = v
        else:
            new_sd[k] = v
    return new_sd


def get_root_dir():
    file_path = Path(__file__).resolve()
    root_dir = file_path.parents[2]
    print(f"Using root_dir: {root_dir}")

    return root_dir

def rsync_copy(source, target, max_retries=3, delay=5):
    """
    Uses rsync to copy a file or directory from source to target.
    This version does not append a trailing slash to the source,
    preventing directory-related errors when copying files.

    Parameters:
    - source (str): The source file or directory path.
    - target (str): The target file or directory path.
    - max_retries (int): Maximum number of retry attempts.
    - delay (int): Delay in seconds between retries.
    """
    os.makedirs(os.path.dirname(target), exist_ok=True)

    attempt = 0
    while attempt <= max_retries:
        try:
            print(f"Attempt {attempt + 1} of {max_retries + 1}:")
            result = subprocess.run(
                ["rsync", "-a", source, target],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print("Rsync succeeded.")
            break  # Exit the loop if rsync is successful

        except subprocess.CalledProcessError as e:
            print(f"Rsync failed on attempt {attempt + 1}.")
            print("Exit Status:", e.returncode)
            print("Rsync Error Output:\n", e.stderr)

            # Increment the attempt counter
            attempt += 1

            if attempt > max_retries:
                print("Max retries exceeded. Aborting rsync operation.")
                raise  # Re-raise the exception after exhausting retries
            else:
                print(f"Retrying in {delay} seconds...\n")
                time.sleep(delay)


def get_paths_to_copy(cfg, root_dir):
    """
    Returns a set of all file paths needed by the training and validation datasets
    as specified by your Hydra config, filtering out non-string or non-existing entries.
    """
    all_datasets = []
    # Build dataset objects
    if hasattr(cfg, "dataset_names") and cfg.dataset_names:
        dataset_names = cfg.dataset_names
        track_method = cfg.track_method
        cond_cameraviews = cfg.cond_cameraviews
        img_shape = cfg.img_shape
        true_horizon = cfg.true_horizon
        track_pred_horizon = cfg.track_pred_horizon
        interp_method = cfg.interp_method
        num_tracks = cfg.num_tracks
    else:
        train_datasets = [
            train_dataset.split(":")[0].replace("_demo", "") for train_dataset in cfg.train_datasets
        ]
        val_datasets = [
            val_dataset.split(":")[0].replace("_demo", "") for val_dataset in cfg.val_datasets
        ]
        dataset_names = train_datasets + val_datasets
        track_method = cfg.loader.track_method
        cond_cameraviews = cfg.loader.cond_cameraviews
        img_shape = cfg.loader.input_image_shape
        true_horizon = cfg.loader.true_horizon
        track_pred_horizon = cfg.forward_dynamics.T_pred
        interp_method = cfg.loader.interp_method
        num_tracks = cfg.forward_dynamics.num_tracks

    print("dataset_names: ", dataset_names)

    for dataset in dataset_names:
        if "libero" in dataset:
            from amplify.loaders.libero_dataset import LiberoDataset

            all_datasets.append(
                LiberoDataset(
                    root_dir=root_dir,
                    dataset_names=[dataset],
                    track_method=track_method,
                    cond_cameraviews=cond_cameraviews,
                    img_shape=img_shape,
                    true_horizon=true_horizon,
                    track_pred_horizon=track_pred_horizon,
                    interp_method=interp_method,
                    num_tracks=num_tracks,
                    use_cached_index_map=False,
                )
            )
        else:
            raise NotImplementedError(f"Dataset {dataset} copying is not implemented.")

    all_files = set()

    for ds in all_datasets:
        if hasattr(ds, "get_index_map"):
            index_map = ds.get_index_map()
            for entry in index_map:
                # For each value in the entry, check if it's a string and a valid path
                for val in entry.values():
                    if isinstance(val, str) and os.path.exists(val):
                        all_files.add(val)
        else:
            raise NotImplementedError(f"{ds} does not implement get_index_map().")

    return all_files


def copy_datasets(cfg, root_dir, target_dir):
    """
    Efficiently copies required dataset files by grouping them by their
    immediate parent directories and copying those directories in one go.
    Preserves the directory structure after the repeated basename of
    root_dir in the apparent file paths, without resolving symlinks.
    """
    all_files = get_paths_to_copy(cfg, root_dir)

    # Derive dynamic pattern from root_dir's basename
    base_name = os.path.basename(os.path.normpath(root_dir))
    pattern = os.sep + base_name + os.sep  # + base_name + os.sep

    print("=== DEBUG: copy_datasets START ===")
    print(f"  target_dir = {target_dir!r}")
    print(f"  total files: {len(all_files)}")
    print("==============================")

    # Map to store unique parent directories to copy:
    # Key: relative directory path after the pattern
    # Value: corresponding absolute source directory
    dirs_to_copy = {}

    for file_path in tqdm(all_files, desc="Grouping directories"):
        # Get absolute path without resolving symlinks
        file_path_abs = os.path.abspath(file_path)
        idx = file_path_abs.find(pattern)

        if idx == -1:
            # Pattern not found; fallback behavior
            rel_path = os.path.relpath(file_path_abs, start=os.sep)
            rel_dir = os.path.dirname(rel_path)
        else:
            # Strip off everything up to and including the pattern
            rel_path = file_path_abs[idx + len(pattern) :]
            rel_dir = os.path.dirname(rel_path)

        # Determine the source directory: parent of the file
        source_dir = os.path.dirname(file_path_abs)

        # Store the mapping if not already present
        if rel_dir not in dirs_to_copy:
            dirs_to_copy[rel_dir] = source_dir

    print("\n=== DEBUG: Directories to copy ===")
    for rel_dir, source_dir in dirs_to_copy.items():
        print(f"  {source_dir} -> {os.path.join(target_dir, rel_dir)}")
    print("===================================")

    # Now perform the actual copying of each unique directory
    for rel_dir, source_dir in tqdm(
        dirs_to_copy.items(), desc="Copying directories", total=len(dirs_to_copy)
    ):
        target_path = os.path.join(target_dir, rel_dir)
        # # Skip copying if the target directory already exists
        # if os.path.exists(target_path):
        #     print(f"Skipping {target_path}, directory already exists.")
        #     continue
        print("Copying from", os.path.join(source_dir, ""))
        print("Copying to", target_path)
        rsync_copy(os.path.join(source_dir, ""), target_path)

    print("\n=== DEBUG: copy_datasets COMPLETE ===")
    print(f"Copied {len(dirs_to_copy)} directories into {target_dir}.")


class DummyGradScaler:
    def scale(self, loss):
        return loss  # simply return the loss tensor

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass  # no operation needed

    def state_dict(self):
        return {}  # returns an empty dict, as there is no state to save

    def load_state_dict(self, state_dict):
        pass  # nothing to load
