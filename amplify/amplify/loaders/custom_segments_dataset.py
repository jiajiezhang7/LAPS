import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
from einops import rearrange
import cv2

from amplify.loaders.base_dataset import BaseDataset
from amplify.utils.data_utils import (
    interpolate_traj,
    interpolate_traj_spline,
    normalize_traj,
)


class CustomSegmentsDataset(BaseDataset):
    """
    自定义预处理片段数据集（基于 preprocess_my_segments.py 产出的 HDF5）。

    期望的 HDF5 结构（每个视频一个 HDF5）：
      root/<view>/tracks: float32, 形状 (T, horizon, N, 2)  — 若为 reinit 模式
      root/<view>/vis:    float32, 形状 (T, horizon, N)     — 可选
    其中 <view> 对应 preprocess 配置中的 view_name（默认 'default'）。

    输出给下游（process_data 后）：
      - images: (V, H, W, C) float32 in [0, 1]  — 这里为占位图，用于可视化
      - traj:   (V, Ht, N, 2) float32 in [-1, 1]
      - vis:    (可选) (V, Ht, N, 1) float32
    """

    def __init__(
        self,
        root_dir: str,
        dataset_names: List[str],
        track_method: str = 'uniform_400_reinit_16',
        cond_cameraviews: List[str] = ('default',),
        keys_to_load: List[str] = ('images', 'tracks'),
        img_shape: Tuple[int, int] = (128, 128),
        true_horizon: int = 16,
        track_pred_horizon: int = 16,
        interp_method: str = 'linear',
        num_tracks: int = 400,
        use_cached_index_map: bool = False,
        video_root: Optional[str] = None,
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

        # 与处理逻辑对齐
        self.track_keys = [k for k in self.keys_to_load if k in ['tracks', 'vis']]
        self.image_obs_keys = ['images']
        # 若无法获知原始视频尺寸，则以 cfg.img_shape 作为归一化参考尺寸
        self.data_img_size = img_shape
        # 可选的视频根目录（从配置传入），优先于环境变量
        self.video_root = video_root

    # ---------- 索引/缓存 ----------
    def get_cache_file(self) -> str:
        dataset_str = '_'.join(self.dataset_names)
        return os.path.expanduser(
            f'~/.cache/amplify/index_maps/custom_segments/{dataset_str}_{self.track_method}.json'
        )
    def _find_h5_files(self) -> List[str]:
        # 预处理产物在 root_dir 的嵌套目录下，直接递归查找 .hdf5
        try:
            return sorted(glob.glob(os.path.join(self.root_dir, '**', '*.hdf5'), recursive=True))
        except Exception:
            # glob errors should not cause empty index_map
            return []

    def _find_video_path(self, h5_path: str) -> Optional[str]:
        """
        根据轨迹 HDF5 路径推断对应视频路径。
        规则：
          1) 优先在同目录下寻找同名不同后缀的视频文件（mp4/avi/mov/mkv）。
          2) 若设置环境变量 AMPLIFY_VIDEO_ROOT 或 VIDEO_ROOT，则尝试在该根目录
             用轨迹相对路径替换后缀来定位；若失败，再在该根目录下按文件名递归搜索一次。
        """
        stem = Path(h5_path).stem
        h5_dir = Path(h5_path).parent
        video_exts = [".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV"]

        # 同目录同名
        for ext in video_exts:
            cand = h5_dir / f"{stem}{ext}"
            if cand.exists():
                return str(cand)

        # 外部视频根目录：优先使用配置中的 self.video_root，其次环境变量
        video_root = self.video_root or os.getenv("AMPLIFY_VIDEO_ROOT") or os.getenv("VIDEO_ROOT")
        if video_root:
            try:
                rel = os.path.relpath(h5_path, start=self.root_dir)
            except ValueError:
                rel = Path(h5_path).name

            # 先按相对路径替换后缀直接尝试
            for ext in video_exts:
                cand = Path(video_root) / Path(rel)
                cand = cand.with_suffix(ext)
                if cand.exists():
                    return str(cand)

            # 再退回到按文件名递归搜索（限制为同名）
            pattern = f"**/{stem}*"
            try:
                hits = sorted(Path(video_root).glob(pattern))
            except Exception:
                # 某些文件系统/符号链接可能导致 glob 抛错，此时视为未找到
                hits = []
            for p in hits:
                if p.suffix in video_exts and p.is_file():
                    return str(p)

        return None

    def create_index_map(self) -> List[Dict]:
        index: List[Dict] = []
        h5_paths = self._find_h5_files()
        if len(h5_paths) == 0:
            raise ValueError(f"No .hdf5 found under {self.root_dir}")

        for h5_path in h5_paths:
            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'root' not in f:
                        continue
                    # 该文件中所有可用视角
                    file_views = [k for k in f['root'].keys()]
                    if len(file_views) == 0:
                        continue
                    # 优先用 cfg 指定的视角，否则回退到文件内第一个视角
                    use_views = [v for v in self.cond_cameraviews if v in file_views]
                    if len(use_views) == 0:
                        use_views = [file_views[0]]

                    # 任取一个视角读取时间长度
                    any_view = use_views[0]
                    if 'tracks' not in f[f'root/{any_view}']:
                        continue
                    dset = f[f'root/{any_view}/tracks']
                    # reinit 模式下为 (T, horizon, N, 2)
                    rollout_len = int(dset.shape[0])

                # 为该 h5 计算一次 video_path 并缓存到每个条目，避免训练期重复搜索
                # 注意：视频查找失败不应影响索引构建
                try:
                    video_path = self._find_video_path(h5_path)
                except Exception:
                    video_path = None

                for start_t in range(rollout_len):
                    end_t = min(start_t + self.true_horizon, rollout_len)
                    index.append({
                        'track_path': str(h5_path),
                        'start_t': int(start_t),
                        'end_t': int(end_t),
                        'rollout_len': int(rollout_len),
                        'use_views': use_views,
                        'video_path': video_path,
                    })
            except Exception:
                # 跳过坏文件
                continue

        return index

    # ---------- 各键加载 ----------
    def load_images(self, idx_dict: Dict) -> Dict:
        """
        从原始视频读取 start_t 帧，返回 (V, H, W, C) in [0, 255]。
        若无法定位/解码视频，则回退为占位黑图像。
        """
        v = max(1, len(idx_dict.get('use_views', self.cond_cameraviews)))
        h, w = self.img_shape

        video_path = idx_dict.get('video_path')
        if video_path is None or not os.path.exists(video_path):
            video_path = self._find_video_path(idx_dict.get('track_path', ''))

        frame_rgb = None
        if video_path is not None and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                start_t = int(idx_dict.get('start_t', 0))
                # clamp 到视频帧数范围内
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total > 0:
                    frame_idx = min(start_t, max(0, total - 1))
                else:
                    frame_idx = start_t
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
                ok, frame_bgr = cap.read()
                if not ok:
                    # 尝试回退到第 0 帧
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame_bgr = cap.read()
                cap.release()

                if ok and frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_rgb = frame_rgb.astype(np.float32)  # 保持 [0,255]

        if frame_rgb is None:
            # 回退：占位黑图像（符合接口并保证可视化不报错）
            images = np.zeros((v, h, w, 3), dtype=np.float32)
            return {'images': images}

        # 若视频尺寸与 cfg.img_shape 不同，不在此处 resize；由外部确保 img_shape 与视频分辨率一致
        # 将单视角复制到 V 个视角（若有多视角，同一个底图用于可视化）
        images = np.stack([frame_rgb for _ in range(v)], axis=0)
        return {'images': images}

    def load_actions(self, idx_dict: Dict) -> Dict:
        raise NotImplementedError

    def load_proprioception(self, idx_dict: Dict) -> Dict:
        raise NotImplementedError

    def load_tracks(self, idx_dict: Dict) -> Dict:
        start_t, end_t = idx_dict['start_t'], idx_dict['end_t']
        track_path = idx_dict['track_path']
        use_views = idx_dict.get('use_views', self.cond_cameraviews)

        out: Dict[str, np.ndarray] = {}
        with h5py.File(track_path, 'r') as f:
            view_tracks = []
            view_vis = []
            view_sizes = []  # collect (H, W) per view if available
            for view in use_views:
                grp = f[f'root/{view}']
                if 'tracks' not in grp:
                    continue
                dset_tracks = grp['tracks']
                # reinit: 取 [start_t] 这一窗
                if self.reinit:
                    tracks = dset_tracks[[start_t]]  # (1, horizon, N, 2)
                else:
                    tracks = dset_tracks[:, start_t:end_t]  # (T, ..., N, 2)
                view_tracks.append(tracks)

                # try to read stored resized image size for this view
                try:
                    h_attr = int(grp.attrs.get('height'))
                    w_attr = int(grp.attrs.get('width'))
                    if h_attr > 0 and w_attr > 0:
                        view_sizes.append((h_attr, w_attr))
                except Exception:
                    pass

                if 'vis' in grp:
                    dset_vis = grp['vis']
                    if self.reinit:
                        vis = dset_vis[[start_t]]  # (1, horizon, N)
                    else:
                        vis = dset_vis[:, start_t:end_t]
                    view_vis.append(vis)

            if len(view_tracks) == 0:
                raise RuntimeError(f"No tracks found in {track_path}")

            out_tracks = np.concatenate(view_tracks, axis=0)  # (V, T_raw, N, 2)
            out['tracks'] = out_tracks

            if len(view_vis) == len(view_tracks):
                out_vis = np.concatenate(view_vis, axis=0)
                out['vis'] = out_vis

            # decide per-sample img_size if available (assume all views share same size)
            if len(view_sizes) > 0:
                # pick the first; typically all views are identical
                out['img_size'] = np.array(view_sizes[0], dtype=np.int64)

        return out

    def load_text(self, idx_dict: Dict) -> Dict:
        # 自定义视频数据暂无文本指令，返回空字典满足抽象接口
        return {}

    # ---------- 处理/标准化 ----------
    def process_data(self, data: Dict) -> Dict:
        # images -> [0,1]
        if 'images' in self.keys_to_load and 'images' in data:
            data['images'] = data['images'] / 255.0
            # 注意：移除了 np.flip 操作，因为自定义数据不需要与 LIBERO 对齐
            # 如果需要翻转，应在预处理阶段处理，而不是训练时

        # tracks/vis 规范化
        if 'tracks' in self.keys_to_load and 'tracks' in data:
            if 'vis' in data:
                data['vis'] = np.expand_dims(data['vis'], axis=-1)  # (V, T, N, 1)

            # 截断到 true_horizon（若保存 horizon 更长）
            T_raw = data['tracks'].shape[1]
            if self.true_horizon < T_raw:
                data['tracks'] = data['tracks'][:, : self.true_horizon]
                if 'vis' in data:
                    data['vis'] = data['vis'][:, : self.true_horizon]

            # (cr -> rc) 与 CoTracker 坐标约定对齐
            data['tracks'] = data['tracks'][..., [1, 0]]

            # 数值清理
            data['tracks'] = np.nan_to_num(data['tracks'], nan=0.0, posinf=0.0, neginf=0.0)
            if 'vis' in data:
                data['vis'] = np.nan_to_num(data['vis'], nan=0.0, posinf=0.0, neginf=0.0)

            # 若预处理时尺寸与训练 cfg.img_shape 不一致，先重映射到 cfg.img_shape 的像素坐标，再做归一化
            # data['img_size'] 若存在，应为 (H, W)。
            if 'img_size' in data:
                orig_h, orig_w = int(data['img_size'][0]), int(data['img_size'][1])
                tgt_h, tgt_w = int(self.data_img_size[0]), int(self.data_img_size[1])
                if (orig_h, orig_w) != (tgt_h, tgt_w):
                    # 直接按像素坐标缩放（行、列分别独立缩放），避免依赖 torch 路径
                    scale_r = float(tgt_h) / float(orig_h)
                    scale_c = float(tgt_w) / float(orig_w)
                    data['tracks'][..., 0] = data['tracks'][..., 0] * scale_r
                    data['tracks'][..., 1] = data['tracks'][..., 1] * scale_c

            # 归一化到 [-1, 1]（以 cfg.img_shape 为统一参考尺寸）
            data['tracks'] = normalize_traj(data['tracks'], self.data_img_size)

            # 若需要插值到 track_pred_horizon
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

            # key 对齐
            data['traj'] = data.pop('tracks')

        return data
