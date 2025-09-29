"""
CustomDatasetExample
====================

This file shows how to create a new dataset by subclassing
`BaseDataset`. Fill in the TODOs to implement your own indexing and
loading logic. The final output should match the conventions documented in
`BaseDataset` (see `amplify/loaders/base_dataset.py`).

Typical keys and shapes (after `process_data`):
- images: (V, H, W, C) float32 in [0, 1]
- actions: (T, D) float32
- traj: (V, Ht, N, 2) float32 in [-1, 1] with (row, col) coords
- vis: (V, Ht, N, 1) float32 (optional)
- proprioception: (D_proprio,) float32 (optional)
- text: str (optional)
- text_emb: (E,) float32 (optional)
"""

import os
from typing import Dict, List, Tuple

import numpy as np

from .base_dataset import BaseDataset


class CustomDatasetExample(BaseDataset):
    """
    Minimal template you can copy to implement a custom dataset.

    Implement the methods below to suit your data layout. You will typically:
    - Construct an index map that names each sample (paths, time bounds, etc.)
    - Implement per-key loaders (images/actions/...)
    - Finalize shapes and normalization in `process_data`
    """

    def __init__(
        self,
        root_dir: str,
        dataset_names: List[str],
        track_method: str = 'uniform_400_reinit_16',
        cond_cameraviews: List[str] = ('agentview',),
        keys_to_load: List[str] = ('images', 'actions'),
        img_shape: Tuple[int, int] = (128, 128),
        true_horizon: int = 16,
        track_pred_horizon: int = 16,
        interp_method: str = 'linear',
        num_tracks: int = 400,
        use_cached_index_map: bool = False,
        aug_cfg: Dict = None,
    ):
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

        # Optional attributes used in processing
        self.image_obs_keys = ['images']
        self.data_img_size = img_shape  # if different from img_shape, enable resize

    # ---------- Indexing / caching ----------
    def get_cache_file(self) -> str:
        """Place to cache your computed index map as JSON."""
        dataset_str = '_'.join(self.dataset_names)
        return os.path.expanduser(
            f'~/.cache/amplify/index_maps/custom/{dataset_str}_{self.track_method}.json'
        )

    def create_index_map(self) -> List[Dict]:
        """
        Build the index map. Each entry should contain enough information to
        load a single sample. See `BaseDataset` docstring for an example.

        TODO: Replace the content below with your actual indexing logic.
        """
        index: List[Dict] = []
        # Example skeleton that visits some hypothetical directory:
        # for dataset in self.dataset_names:
        #     sample_dir = os.path.join(self.root_dir, 'my_data', dataset)
        #     for sample_id in os.listdir(sample_dir):
        #         path = os.path.join(sample_dir, sample_id)
        #         rollout_len = ...
        #         for start_t in range(rollout_len):
        #             end_t = min(start_t + self.true_horizon, rollout_len)
        #             index.append({
        #                 'task_file': path,
        #                 'demo_key': sample_id,
        #                 'start_t': start_t,
        #                 'end_t': end_t,
        #                 'rollout_len': rollout_len,
        #             })
        raise NotImplementedError('Implement create_index_map for your dataset')

    # ---------- Per-key loaders ----------
    def load_images(self, idx_dict: Dict) -> Dict:
        """
        TODO: Load and return raw images for the sample.
        Must return {'images': np.ndarray of shape (V, H, W, C)}.
        Values can be uint8 [0,255] or float [0,1]; processing will normalize.
        """
        raise NotImplementedError

    def load_actions(self, idx_dict: Dict) -> Dict:
        """
        TODO: Load actions for [start_t, end_t).
        Must return {'actions': np.ndarray of shape (T, D)}.
        """
        raise NotImplementedError

    def load_proprioception(self, idx_dict: Dict) -> Dict:
        """
        TODO: Load proprioception at start_t.
        Must return {'proprioception': np.ndarray of shape (D_proprio,)}.
        """
        raise NotImplementedError

    def load_tracks(self, idx_dict: Dict) -> Dict:
        """
        TODO: Load raw point tracks and optional visibility.
        Must return a dict subset of:
          - 'tracks': np.ndarray (V, T_raw, N, 2)
          - 'vis': np.ndarray (V, T_raw, N) or (V, T_raw, N, 1)
        """
        raise NotImplementedError

    def load_text(self, idx_dict: Dict) -> Dict:
        """
        TODO: Load raw text and/or precomputed text embeddings.
        Return a dict subset of {'text': str, 'text_emb': np.ndarray (E,)}.
        """
        raise NotImplementedError

    # ---------- Processing ----------
    def process_data(self, data: Dict) -> Dict:
        """
        Finalize sample to the canonical output format.
        Implement any resizing, normalization, interpolation, and key renaming
        required by your dataset. See `LiberoDataset` for a concrete example.

        Minimal example for images-only dataset:
            data['images'] = data['images'].astype(np.float32) / 255.0
            # Optional: flip, resize, etc.
            return data
        """
        return data


if __name__ == '__main__':
    # Minimal smoke test: construct the dataset and print its length.
    # Replace with real arguments and a working implementation above.
    try:
        ds = CustomDatasetExample(root_dir='.', dataset_names=['my_dataset'])
        print('Dataset length:', len(ds))
        if len(ds) > 0:
            sample = ds[0]
            print('Keys:', list(sample.keys()))
            for k, v in sample.items():
                if isinstance(v, np.ndarray):
                    print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
                else:
                    print(f'  {k}: type={type(v)}')
    except NotImplementedError as e:
        print('[info] CustomDatasetExample is a template. Fill in the TODOs to use it.')
        print('Raised:', e)
