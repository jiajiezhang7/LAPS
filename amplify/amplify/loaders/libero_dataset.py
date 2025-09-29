import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import Resize
from tqdm import tqdm

from amplify.loaders.base_dataset import BaseDataset
from amplify.utils.data_utils import (
    interpolate_traj,
    interpolate_traj_spline,
    normalize_traj,
    grab_libero_language_from_filename,
)


class LiberoDataset(BaseDataset):
    """
    Modular libero dataset.

    Output keys per sample (after process_data):
    - 'images': (V, H, W, C) float32 in [0, 1]
    - 'actions': (T, D) float32 (padded to true_horizon if needed)
    - 'traj': (V, Ht, N, 2) float32 in [-1, 1], where Ht is the track
      prediction horizon; coordinates are (row, col)
    - 'vis': optional (V, Ht, N, 1) float32
    - 'proprioception': optional (D_proprio,) float32
    - 'text': optional str
    - 'text_emb': optional (E,) float32
    """

    def __init__(
        self,
        root_dir: str,
        dataset_names: List[str],
        track_method: str = 'uniform_400_reinit_16',
        cond_cameraviews: List[str] = ('agentview', 'eye_in_hand'),
        keys_to_load: List[str] = ('images', 'actions', 'tracks'),
        img_shape: Tuple[int, int] = (128, 128),
        true_horizon: int = 16,
        track_pred_horizon: int = 8,
        interp_method: str = 'linear',
        num_tracks: int = 400,
        use_cached_index_map: bool = False,
        aug_cfg: Dict = None,
        demo_subset: float = 1.0,
        libero_path: Optional[str] = None,
    ):

        self.demo_subset = demo_subset
        self.libero_path = libero_path

        super().__init__(
            root_dir=root_dir,
            dataset_names=dataset_names,
            track_method=track_method,
            cond_cameraviews=list(cond_cameraviews),
            keys_to_load=list(keys_to_load),
            img_shape=img_shape,
            true_horizon=true_horizon,
            track_pred_horizon=track_pred_horizon,
            interp_method=interp_method,
            num_tracks=num_tracks,
            use_cached_index_map=use_cached_index_map,
            aug_cfg=aug_cfg,
        )

        self.track_keys = [k for k in self.keys_to_load if k in ['tracks', 'vis']]

        # Determine native image size from first datapoint
        task_file = self.index_map[0]['task_file']
        demo_key = self.index_map[0]['demo_key']
        with h5py.File(task_file, 'r') as f:
            demo_data = f[demo_key]
            self.data_img_size = demo_data['obs/agentview_rgb'].shape[-3:-1]

        self.image_obs_keys = ['images'] # depth, segmentation etc would also go here
        self.resize_transform = Resize(self.img_shape, antialias=False)


    def get_cache_file(self) -> str:
        """
        Cache path for the computed index map.
        """
        dataset_str = '_'.join(self.dataset_names)
        return os.path.expanduser(
            f'~/.cache/amplify/index_maps/libero_demo/{dataset_str}_{self.track_method}_subset_{self.demo_subset:.2f}.json'
        )

    def create_index_map(self) -> List[Dict]:
        """
        Build the index map by scanning LIBERO task HDF5s and corresponding
        preprocessed track files for the configured track method.
        Each entry includes the H5 task path, demo key, start/end timesteps,
        rollout length, and an optional track path.
        """
        if self.libero_path is None:
            demo_root = os.path.join(self.root_dir, 'LIBERO/libero/datasets')
        else:
            demo_root = os.path.expanduser(self.libero_path)
            if not os.path.isabs(demo_root):
                demo_root = os.path.join(self.root_dir, demo_root)
        if not os.path.exists(demo_root):
            raise ValueError(f"Demo root directory does not exist: {demo_root}")

        track_root = os.path.join(self.root_dir, 'preprocessed_data')

        # Determine demo indices per subset fraction (50 demos per task)
        num_demos = 50
        subset_size = int(num_demos * abs(self.demo_subset))
        if self.demo_subset >= 0:
            start_idx, end_idx = 0, subset_size
        else:
            start_idx, end_idx = num_demos - subset_size, num_demos
        demo_range = range(start_idx, end_idx)
        print(f"demo_range: {demo_range}")

        index_map: List[Dict] = []
        for dataset in self.dataset_names:
            task_files = glob.glob(os.path.join(demo_root, dataset, '*.hdf5'))
            for task_file in tqdm(task_files):
                for demo_no in demo_range:
                    demo_key = f'data/demo_{demo_no}'
                    track_path = os.path.join(
                        track_root, dataset, self.track_method, f'{Path(task_file).stem}', f'demo_{demo_no}.hdf5'
                    )
                    # If tracks are requested but track file doesn't exist, skip this demo
                    if 'tracks' in self.keys_to_load and not os.path.exists(track_path):
                        continue
                    with h5py.File(task_file, 'r') as f:
                        rollout_len = f[demo_key]['actions'].shape[0]

                    for start_t in range(rollout_len):
                        end_t = min(start_t + self.true_horizon, rollout_len)
                        entry = {
                            'task_file': str(task_file),
                            'demo_key': str(demo_key),
                            'start_t': int(start_t),
                            'end_t': int(end_t),
                            'rollout_len': int(rollout_len),
                        }
                        if 'tracks' in self.keys_to_load:
                            entry['track_path'] = str(track_path)
                        index_map.append(entry)

        return index_map

    def _open_demo(self, idx_dict: Dict):
        """Helper to open the task/demo HDF5 in read-only mode."""
        return h5py.File(idx_dict['task_file'], 'r')

    def load_images(self, idx_dict: Dict) -> Dict:
        """
        Returns
        -------
        dict
            'images': np.float32 (V, H, W, C) in [0, 255].
        """
        start_t = idx_dict['start_t']
        images = []
        with self._open_demo(idx_dict) as f:
            demo = f[idx_dict['demo_key']]
            image_keys = [f'obs/{view}_rgb' for view in self.cond_cameraviews]
            for key in image_keys:
                images.append(demo[key][start_t])
        return {'images': np.stack(images, axis=0).astype(np.float32)}  # (V, H, W, C)

    def load_actions(self, idx_dict: Dict) -> Dict:
        """
        Returns
        -------
        dict
            'actions': np.float32 (T, D)
        """
        start_t, end_t = idx_dict['start_t'], idx_dict['end_t']
        with self._open_demo(idx_dict) as f:
            demo = f[idx_dict['demo_key']]
            actions = demo['actions'][start_t:end_t].astype(np.float32)
        return {'actions': actions}  # (T, D)

    def load_proprioception(self, idx_dict: Dict) -> Dict:
        """
        Returns
        -------
        dict
            'proprioception': np.float32 (D_proprio,)
        """
        start_t = idx_dict['start_t']
        with self._open_demo(idx_dict) as f:
            demo = f[idx_dict['demo_key']]
            joint_states = demo['obs/joint_states'][start_t]
            gripper_states = demo['obs/gripper_states'][start_t]
            prop = np.concatenate([joint_states, gripper_states], axis=-1).astype(np.float32)
        return {'proprioception': prop}

    def load_tracks(self, idx_dict: Dict) -> Dict:
        """
        Returns
        -------
        dict
            'tracks': np.float32 (V, T_raw, N, 2)
            'vis':    np.float32 (V, T_raw, N) or (V, T_raw, N, 1)
        Notes
        -----
        For reinit-style tracks, each start_t has a fixed internal horizon in
        the file (e.g., 16). We slice one entry along T and concatenate views.
        """
        start_t, end_t = idx_dict['start_t'], idx_dict['end_t']
        track_path = idx_dict['track_path']
        out: Dict[str, np.ndarray] = {}
        with h5py.File(track_path, 'r') as f:
            for key in [k for k in self.track_keys if k in f['root'][self.cond_cameraviews[0]]]:
                pieces = []
                for camera in self.cond_cameraviews:
                    dset = f[f'root/{camera}/{key}']
                    if self.reinit:
                        pieces.append(dset[[start_t]])
                    else:
                        pieces.append(dset[:, start_t:end_t])
                out[key] = np.concatenate(pieces, axis=0)
        return out  # tracks: (V, T, N, D), vis: (V, T, N)

    def load_text(self, idx_dict: Dict) -> Dict:
        """
        Returns
        -------
        dict
            Optionally 'text' (str) and/or 'text_emb' (np.float32, (E,)).
        """
        out: Dict[str, np.ndarray] = {}
        task_file = idx_dict['task_file']
        # Raw text
        if 'text' in self.keys_to_load:
            out['text'] = grab_libero_language_from_filename(
                Path(task_file).stem.replace('_demo', '.bddl')
            )
        # Embedded text
        if 'text_emb' in self.keys_to_load:
            dataset_dir = Path(task_file).parent.name
            task_name = Path(task_file).stem.replace('_demo', '')
            text_path = os.path.join(self.root_dir, 'preprocessed_data', dataset_dir, 'text', f'{task_name}.hdf5')
            if os.path.exists(text_path):
                with h5py.File(text_path, 'r') as tf:
                    out['text_emb'] = tf['text_emb'][()]
            else:
                raise ValueError(f"Text embedding file not found: {text_path}")
        return out

    def process_data(self, data: Dict) -> Dict:
        """
        Finalize shapes and normalization to match the original loader.
        - Scale images to [0, 1], vertical flip, optional resize.
        - Pad actions to `true_horizon`.
        - Reorder track coords (cr -> rc), sanitize, normalize to [-1, 1].
        - Optional interpolation to `track_pred_horizon` for tracks/vis.
        - Rename 'tracks' to 'traj'.
        """
        # --- ACTIONS ---
        if 'actions' in self.keys_to_load:
            assert data['actions'].ndim == 2
            if data['actions'].shape[0] < self.true_horizon:
                pad_len = self.true_horizon - data['actions'].shape[0]
                data['actions'] = np.pad(data['actions'], ((0, pad_len), (0, 0)))

        # --- IMAGES ---
        assert data['images'].ndim == 4
        if 'images' in self.keys_to_load:
            data['images'] = data['images'] / 255.0
            data['images'] = np.flip(data['images'], axis=-3).copy()

            if self.data_img_size != self.img_shape:
                for key in self.image_obs_keys:
                    img = rearrange(data[key], 'v h w c -> v c h w')
                    img_t = torch.from_numpy(img).contiguous().float() 
                    resized = self.resize_transform(img_t).numpy().astype(np.float32)
                    data[key] = rearrange(resized, 'v c h w -> v h w c')

        # --- TRACKS ---
        if 'tracks' in self.keys_to_load:
            if 'vis' in self.keys_to_load and 'vis' in data:
                data['vis'] = np.expand_dims(data['vis'], axis=-1)
            for key in [k for k in ['tracks', 'vis'] if k in data]:
                assert data[key].ndim == 4

            # if self.true_horizon < 16:
            #     data['tracks'] = data['tracks'][:, : self.true_horizon]

            T_raw = data['tracks'].shape[1]
            if self.true_horizon < T_raw:
                data['tracks'] = data['tracks'][:, :self.true_horizon]

            # (cr -> rc) assuming CoTracker ordering
            data['tracks'] = data['tracks'][..., [1, 0]]

            # fix NaNs and Infs
            data['tracks'] = np.nan_to_num(data['tracks'], nan=0.0, posinf=0.0, neginf=0.0)
            if 'vis' in data:
                data['vis'] = np.nan_to_num(data['vis'], nan=0.0, posinf=0.0, neginf=0.0)

            # normalize to [-1, 1]
            data['tracks'] = normalize_traj(data['tracks'], self.data_img_size)

            # Interpolate to prediction horizon if needed
            if self.track_pred_horizon != self.true_horizon:
                if self.interp_method == 'linear':
                    fn = interpolate_traj
                elif self.interp_method == 'spline':
                    fn = interpolate_traj_spline
                else:
                    raise NotImplementedError
                data['tracks'] = fn(data['tracks'], self.track_pred_horizon)
                if 'vis' in data:
                    data['vis'] = fn(data['vis'], self.track_pred_horizon)

        # Align key name with model expectations
        if 'tracks' in data:
            data['traj'] = data.pop('tracks')

        return data