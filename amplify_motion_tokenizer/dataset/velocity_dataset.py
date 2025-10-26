from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from amplify_motion_tokenizer.utils.helpers import velocity_to_labels


class VelocityDataset(Dataset):
    """
    读取预处理后的速度张量 (.pt)，并在 __getitem__ 时生成监督标签 ω_t。
    期望每个文件形状为 (T-1, N, 2)，并且速度已归一化到 [-1, 1] 区间。
    """

    def __init__(self, preprocessed_dir: str, window_size: int, static_filter: Optional[Dict] = None):
        self.dir = Path(preprocessed_dir)
        self.data_files = sorted(self.dir.glob("*.pt"))
        self.W = int(window_size)
        # 静态点过滤配置（在数据加载阶段按需产生掩码，而不改变原始数据形状）
        sf = static_filter or {}
        self.sf_enable: bool = bool(sf.get('enable', False))
        # metric: 'max' | 'mean' | 'p95'
        self.sf_metric: str = str(sf.get('metric', 'max'))
        # 阈值在归一化速度尺度下（[-1,1]），建议 0.02~0.10
        self.sf_threshold: float = float(sf.get('threshold', 0.05))
        # 保底至少保留的点数（避免极端情况下全为 False）
        self.sf_min_keep: int = int(sf.get('min_keep', 1))
        if not self.data_files:
            raise FileNotFoundError(
                f"No preprocessed .pt files found in {preprocessed_dir}." \
                f" Please run the preprocessing or dummy data generator."
            )

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # velocities shape: (T-1, N, 2)
        velocities: torch.Tensor = torch.load(self.data_files[idx], map_location="cpu")
        if velocities.ndim != 3 or velocities.shape[-1] != 2:
            raise ValueError(
                f"Invalid velocities shape in {self.data_files[idx].name}: {tuple(velocities.shape)}."
            )
        labels = velocity_to_labels(velocities, self.W)  # shape: (T-1, N)
        # 生成静态点掩码（True 表示参与训练/统计；False 表示忽略）
        Tm1, N, _ = velocities.shape
        if self.sf_enable:
            # 速度模长：(T-1, N)
            vel_norm = torch.linalg.norm(velocities, dim=-1)
            metric = self.sf_metric.lower()
            if metric == 'mean':
                score = vel_norm.mean(dim=0)  # (N,)
            elif metric == 'p95':
                try:
                    score = torch.quantile(vel_norm, q=0.95, dim=0)
                except Exception:
                    # 兼容老版本 PyTorch
                    score, _ = vel_norm.kthvalue(max(1, int(round(0.95 * (Tm1)))), dim=0)
            else:  # 'max'
                score = vel_norm.max(dim=0).values
            mask_point = score >= float(self.sf_threshold)
            # 保底至少保留 sf_min_keep 个点
            if self.sf_min_keep > 0 and int(mask_point.sum().item()) < self.sf_min_keep:
                # 选取得分最高的前 k 个
                k = min(self.sf_min_keep, N)
                topk = torch.topk(score, k=k, largest=True).indices
                mask_point = torch.zeros_like(mask_point)
                mask_point[topk] = True
            mask = mask_point.unsqueeze(0).expand(Tm1, -1)  # (T-1, N)
        else:
            mask = torch.ones((Tm1, N), dtype=torch.bool)
        return velocities, labels, mask


def get_dataloaders(config):
    """
    返回 (train_loader, val_loader)。
    - 若提供 data.val_preprocess_output_dir，则从该目录构造验证集；
    - 否则若 data.val_split 在 (0,1) 内，则按比例随机划分；
    - 否则不返回验证集（val_loader=None）。
    """
    train_batch_size = int(config['training']['batch_size'])
    num_workers = int(config['training']['num_workers'])

    # 静态点过滤配置（可选）
    sf_cfg = config['data'].get('static_filter', {}) if isinstance(config.get('data'), dict) else {}
    train_dataset = VelocityDataset(
        preprocessed_dir=config['data']['preprocess_output_dir'],
        window_size=config['model']['decoder_window_size'],
        static_filter=sf_cfg,
    )

    val_dir = config['data'].get('val_preprocess_output_dir')
    if not val_dir:  # 明确处理 None 和空字符串 ""
        val_dir = None
    val_split = float(config['data'].get('val_split', 0.0) or 0.0)

    val_loader = None
    if val_dir:
        val_dataset = VelocityDataset(
            preprocessed_dir=val_dir,
            window_size=config['model']['decoder_window_size'],
            static_filter=sf_cfg,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    elif 0.0 < val_split < 1.0 and len(train_dataset) > 1:
        val_size = max(1, int(len(train_dataset) * val_split))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(
            train_subset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    # 默认：仅训练集
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_dataloader(config) -> DataLoader:
    """保持兼容的旧接口，仅返回训练 DataLoader。"""
    train_loader, _ = get_dataloaders(config)
    return train_loader
