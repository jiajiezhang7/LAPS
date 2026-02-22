import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from einops import rearrange

from amplify.utils.data_utils import (
    get_autoregressive_indices_efficient,
)
from amplify.utils.vis_utils import vis_pred

# from memory_profiler import profile

def compute_relative_classification_loss(logits, targets, batch_traj, cfg):
    assert logits.dim() == 3
    target_indices = get_autoregressive_indices_efficient(
        batch_traj,
        targets,
        img_size=cfg.cls_img_size,
        rel_img_size=cfg.rel_cls_img_size,
        num_angle_bins=cfg.num_angle_bins,
        num_mag_bins=cfg.num_mag_bins,
        max_polar_mag=cfg.max_polar_mag,
    )
    if cfg.loss_fn == 'relative_ce':
        b, v, t, n = target_indices['relative'].shape
        targets_flat = rearrange(target_indices['relative'], 'b v t n -> (b v t n)')
        logits_flat = logits.reshape(-1, logits.size(-1))
        # Optional radial class weights to counter center-class dominance
        class_weights = None
        try:
            mode = getattr(cfg, 'class_weight_mode', 'none')
        except Exception:
            mode = 'none'
        if mode == 'radial':
            try:
                H, W = int(cfg.rel_cls_img_size[0]), int(cfg.rel_cls_img_size[1])
                cy, cx = H // 2, W // 2
                y = torch.arange(H, device=logits_flat.device, dtype=logits_flat.dtype)
                x = torch.arange(W, device=logits_flat.device, dtype=logits_flat.dtype)
                yy, xx = torch.meshgrid(y, x, indexing='ij')
                r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                r_max = torch.sqrt(torch.tensor(cy**2 + cx**2, device=logits_flat.device, dtype=logits_flat.dtype)).clamp_min(1.0)
                r_norm = (r / r_max).clamp(0, 1)
                strength = float(getattr(cfg, 'class_weight_strength', 2.0) or 2.0)
                power = float(getattr(cfg, 'class_weight_power', 2.0) or 2.0)
                w = 1.0 + strength * (r_norm ** power)
                class_weights = w.reshape(-1).to(logits_flat.dtype)
            except Exception:
                class_weights = None
        use_focal = bool(getattr(cfg, 'use_focal', False) or False)
        if use_focal:
            gamma = float(getattr(cfg, 'focal_gamma', 2.0) or 2.0)
            alpha = float(getattr(cfg, 'focal_alpha', 0.25) or 0.25)
            log_probs = F.log_softmax(logits_flat, dim=-1)
            probs = log_probs.exp()
            mask = targets_flat != -1
            t_idx = targets_flat[mask].unsqueeze(1)
            p_t = probs.gather(1, t_idx).squeeze(1)
            logp_t = log_probs.gather(1, t_idx).squeeze(1)
            focal_weight = (1.0 - p_t).pow(gamma)
            focal_loss_vec = -alpha * focal_weight * logp_t
            # match destination dtype (e.g., Half under AMP)
            focal_loss_vec = focal_loss_vec.to(logits_flat.dtype)
            # apply class weights if provided
            if class_weights is not None:
                cw = class_weights[targets_flat[mask]].to(focal_loss_vec.dtype)
                focal_loss_vec = focal_loss_vec * cw
            loss_full = torch.zeros_like(targets_flat, dtype=logits_flat.dtype)
            loss_full[mask] = focal_loss_vec
            loss = loss_full.view(b, v, t, n)
        else:
            if class_weights is not None:
                # cross_entropy expects weight on same device/dtype and length C
                loss = F.cross_entropy(logits_flat, targets_flat, weight=class_weights, ignore_index=-1, reduction='none')
            else:
                loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1, reduction='none')
            loss = loss.view(b, v, t, n)
    else:
        raise NotImplementedError
    return loss

def get_ce_weight(device, cfg):
    weight = torch.ones(cfg.rel_cls_img_size[0] * cfg.rel_cls_img_size[1]).to(device)
    center_idx = cfg.rel_cls_img_size[0] // 2 * cfg.rel_cls_img_size[1] + cfg.rel_cls_img_size[1] // 2
    weight[center_idx] = cfg.loss_weights.weighted_ce
    return weight

def get_loss_from_loss_dict(loss_dict, cfg):
    loss = 0.0
    for key, value in loss_dict.items():
        scale = cfg.forward_dynamics.loss_weights[key] if key in cfg.forward_dynamics.loss_weights else 1.0
        bias = cfg.forward_dynamics.loss_biases[key] if key in cfg.forward_dynamics.loss_biases else 0.0
        # print(f"scale for {key}: {scale}")
        # print(f"bias for {key}: {bias}")
        loss_component = (value + bias) * scale
        # print(f"weighted loss for {key}: {loss_component}")
        loss += loss_component

    return loss
