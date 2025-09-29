import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from einops import rearrange
from torch.utils.data import Dataset

from amplify.utils.cfg_utils import get_device
from amplify.utils.data_utils import RandomGaussianBlur


class BaseDataset(Dataset, ABC):
    """
    A modular base class that splits the dataset logic into
    small, easy-to-customize functions.

    Expected outputs per sample after `process_data`:
    - `images` (np.float32): shape (V, H, W, C) in [0, 1]. V is the number of
      camera views, C=3 for RGB. Images are vertically flipped (to match viewer
      conventions) and optionally resized to `img_shape`.
    - `actions` (np.float32): shape (T, D). Padded with zeros to fill
      `true_horizon` if the remaining rollout is shorter.
    - `traj` (np.float32): normalized tracks with shape (V, Ht, N, 2), where
      Ht is the track prediction horizon (`track_pred_horizon` if interpolation
      is used, otherwise `true_horizon`). The last dimension stores (row, col)
      image coordinates normalized to [-1, 1].
    - `vis` (optional, np.float32): shape (V, Ht, N, 1), visibility/confidence
      for each point. Interpolated if tracks are interpolated.
    - `proprioception` (optional, np.float32): shape (D_proprio,), single-timestep
      proprioceptive state at `start_t` (e.g., 9-dof for LIBERO: 7 joints + 2 gripper).
    - `text` (optional, str): natural language description of the task.
    - `text_emb` (optional, np.float32): shape (E,), precomputed text embedding.

    Subclasses implement:
    - `get_cache_file()`
    - `create_index_map()`
    - `load_images()` / `load_actions()` / `load_proprioception()` /
      `load_tracks()` / `load_text()`
    - `process_data()` (finalize shapes, normalization, interpolation, renaming)

    Index map entries (returned by `create_index_map`) should be dictionaries
    holding everything needed to load one sample, for example:
    {
      'task_file': '/path/to/task.hdf5',
      'demo_key': 'data/demo_12',
      'track_path': '/path/to/tracks/demo_12.hdf5',  # only if using tracks
      'start_t': 128,  # inclusive
      'end_t': 144,    # exclusive (start_t + true_horizon or end of rollout)
      'rollout_len': 642,
    }
    """

    def __init__(
        self,
        root_dir: str,
        dataset_names: List[str],
        track_method: str = 'uniform_6400_reinit_16',
        cond_cameraviews: List[str] = ('agentview',),
        keys_to_load: List[str] = ('images', 'tracks', 'actions'),
        img_shape: Tuple[int, int] = (128, 128),
        true_horizon: int = 16,
        track_pred_horizon: int = 8,
        interp_method: str = 'linear',
        num_tracks: int = 400,
        use_cached_index_map: bool = False,
        aug_cfg: Dict = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.dataset_names = list(dataset_names)
        self.track_method = track_method
        self.cond_cameraviews = list(cond_cameraviews)
        self.keys_to_load = list(keys_to_load)
        self.img_shape = img_shape
        self.true_horizon = true_horizon
        self.track_pred_horizon = track_pred_horizon
        self.interp_method = interp_method
        self.num_tracks = num_tracks
        self.use_cached_index_map = use_cached_index_map
        self.aug_cfg = aug_cfg

        # Current loaders assume reinit-based track files.
        self.reinit = 'reinit' in track_method

        # Build index
        self.index_map = self.get_index_map()

    @abstractmethod
    def get_cache_file(self) -> str:
        """
        Return an absolute path to a JSON file used to cache the computed
        `index_map`. This is optional, but recommended for large datasets.

        Example: '~/.cache/amplify/index_maps/libero_demo/libero_10_xxx.json'
        """
        raise NotImplementedError

    @abstractmethod
    def create_index_map(self) -> List[Dict]:
        """
        Build and return a list of dictionaries, each describing a single
        dataset sample (see class docstring for a canonical entry).

        This is the only place that should perform filesystem traversal.
        The rest of the load functions can assume this metadata exists.
        """
        raise NotImplementedError

    def get_index_map(self) -> List[Dict]:
        cache_file = self.get_cache_file()
        if self.use_cached_index_map:
            try:
                with open(cache_file, 'r') as f:
                    index_map = json.load(f)
                print(f"Loaded index map from cache: {cache_file}")
                return index_map
            except FileNotFoundError:
                pass

        print("Creating index map...")
        index_map = self.create_index_map()
        print("Index map length: ", len(index_map))
        assert len(index_map) > 0, "Index map is empty"

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(index_map, f)
        print(f"Saved index map to cache: {cache_file}")
        return index_map

    def load_data(self, idx_dict: Dict) -> Dict:
        """
        Orchestrates per-key loaders to assemble a single training sample.
        """
        data: Dict = {}

        if 'images' in self.keys_to_load:
            data.update(self.load_images(idx_dict))
        if 'actions' in self.keys_to_load:
            data.update(self.load_actions(idx_dict))
        if 'proprioception' in self.keys_to_load:
            data.update(self.load_proprioception(idx_dict))
        if 'tracks' in self.keys_to_load or 'vis' in self.keys_to_load:
            data.update(self.load_tracks(idx_dict))
        if 'text' in self.keys_to_load or 'text_emb' in self.keys_to_load:
            data.update(self.load_text(idx_dict))

        return data

    @abstractmethod
    def load_images(self, idx_dict: Dict) -> Dict:
        """
        Load raw images for the selected timestep.

        Returns a dict with:
        - 'images': np.float32 array of shape (V, H, W, C) in [0, 255] or [0, 1].
          Any optional keys can be added if your dataset needs them.
        """
        raise NotImplementedError

    @abstractmethod
    def load_actions(self, idx_dict: Dict) -> Dict:
        """
        Load the action sequence for [start_t, end_t).

        Returns a dict with:
        - 'actions': np.float32 array of shape (T, D).
        """
        raise NotImplementedError

    @abstractmethod
    def load_proprioception(self, idx_dict: Dict) -> Dict:
        """
        Load a single-timestep proprioceptive state at start_t.

        Returns a dict with:
        - 'proprioception': np.float32 array of shape (D_proprio,).
        """
        raise NotImplementedError

    @abstractmethod
    def load_tracks(self, idx_dict: Dict) -> Dict:
        """
        Load raw point tracks (and optional visibility) for the configured views.

        Returns a dict with some subset of:
        - 'tracks': np.float32 array of shape (V, T_raw, N, 2)
        - 'vis':    np.float32 array of shape (V, T_raw, N) or (V, T_raw, N, 1)

        The 'process_data' method is responsible for normalizing coordinates to
        [-1, 1], fixing NaNs/Infs, enforcing time horizon Ht, and renaming
        'tracks' -> 'traj'.
        """
        raise NotImplementedError

    @abstractmethod
    def load_text(self, idx_dict: Dict) -> Dict:
        """
        Load raw text and/or precomputed text embeddings.

        Returns a dict with:
        - 'text': str (optional)
        - 'text_emb': np.float32 array of shape (E,) (optional)
        """
        raise NotImplementedError

    # ---------- Processing / augmentation ----------
    @abstractmethod
    def process_data(self, data: Dict) -> Dict:
        """
        Finalize a sample to the canonical format used downstream.

        Typical operations:
        - images: convert to [0, 1], vertical flip, resize to `img_shape`.
        - actions: pad to `true_horizon` if needed.
        - tracks: reorder coordinates if necessary, sanitize NaNs/Infs,
          normalize to [-1, 1], optionally interpolate to `track_pred_horizon`,
          rename 'tracks' -> 'traj'.
        - vis: match track interpolation shape, enforce final dims (V, Ht, N, 1).
        """
        raise NotImplementedError

    def augment_data(self, data: Dict) -> Dict:
        """
        Optional color and blur augmentations.

        Expects `images` with shape (V, H, W, C). Operates per-view.
        """
        if self.aug_cfg:
            v, h, w, c = data['images'].shape

            if isinstance(data['images'], np.ndarray):
                data['images'] = torch.tensor(data['images'])

            data['images'] = rearrange(data['images'], 'v h w c -> v c h w')

            if self.aug_cfg.get('color_jitter'):
                cj = self.aug_cfg.get('color_jitter_strength', 0.2)
                data['images'] = T.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=cj)(data['images'])
            if self.aug_cfg.get('gaussian_blur'):
                gaussian_blur = RandomGaussianBlur(
                    kernel_sizes=self.aug_cfg.get('gaussian_blur_kernel_size', [3, 5]),
                    sigma_min=self.aug_cfg.get('gaussian_blur_sigma_min', 0.1),
                    sigma_max=self.aug_cfg.get('gaussian_blur_sigma_max', 2.0),
                    p=self.aug_cfg.get('gaussian_blur_p', 0.5),
                )
                data['images'] = gaussian_blur(data['images'])

            data['images'] = rearrange(data['images'], 'v c h w -> v h w c')
            data['images'] = torch.clamp(data['images'], 0, 1)
            data['images'] = data['images'].detach().cpu().numpy().astype(np.float32)
        return data

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx, **kwargs):
        idx_dict = self.index_map[idx]
        data = self.load_data(idx_dict)
        data = self.process_data(data)
        if self.aug_cfg is not None:
            data = self.augment_data(data)
        return data

    def get_full_episode_batch(self, idx: int) -> Dict:
        """
        Build a visualization batch by sweeping the full episode for the
        provided sample index. Returns stacked tensors for each key.
        """
        base_idx_dict = self.index_map[idx].copy()
        rollout_len = base_idx_dict['rollout_len']

        batch = {}
        for start_t in range(rollout_len):
            idx_dict = base_idx_dict.copy()
            idx_dict['start_t'] = start_t
            idx_dict['end_t'] = min(start_t + self.true_horizon, rollout_len)

            # Load using the modified idx_dict instead of using __getitem__(idx)
            data = self.load_data(idx_dict)
            data = self.process_data(data)

            for k, v in data.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)

        for k, v in batch.items():
            device = get_device()
            if isinstance(v[0], np.ndarray):
                batch[k] = torch.stack([torch.tensor(x) for x in v], dim=0).to(device)
            if isinstance(v[0], torch.Tensor):
                batch[k] = torch.stack(v, dim=0).to(device)

        return batch

