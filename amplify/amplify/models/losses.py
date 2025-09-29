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

    # target_indices = get_autoregressive_indices(
    target_indices = get_autoregressive_indices_efficient(
        batch_traj,
        targets,
        img_size=cfg.cls_img_size,
        rel_img_size=cfg.rel_cls_img_size,
        num_angle_bins=cfg.num_angle_bins,
        num_mag_bins=cfg.num_mag_bins,
        max_polar_mag=cfg.max_polar_mag
    )
    if cfg.loss_fn == 'relative_ce':
        b, v, t, n = target_indices['relative'].shape
        targets = rearrange(target_indices['relative'], 'b v t n -> b (v t n)') # (bs, v*t-1*n)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.flatten(), ignore_index=-1, reduction='none')
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
