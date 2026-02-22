import random

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from scipy.interpolate import CubicSpline
from torch import Tensor
from torchvision import transforms


def normalize_traj(traj_input, img_shape):
    """
    transforms the trajectory so that the pixel coordinates are between -1 and 1
    traj: (bs, horizon, num_queries, 2) with (row, col) in (height, width)

    Returns:
    traj: (bs, horizon, num_queries, 2) with (row, col) in (-1, 1)
    """
    initial_shape = traj_input.shape

    # making sure traj is a float numpy array or tensor
    if isinstance(traj_input, np.ndarray):
        traj = traj_input.copy().astype(np.float32)
    elif isinstance(traj_input, torch.Tensor):
        traj = traj_input.clone().float()
    else:
        raise ValueError(f"traj should be a numpy array or tensor, but is of type {type(traj)}")

    traj[..., 0] = (traj[..., 0] - float(img_shape[0] / 2)) / (float(img_shape[0]) / 2)
    traj[..., 1] = (traj[..., 1] - float(img_shape[1] / 2)) / (float(img_shape[1]) / 2)

    # clamping/clipping the values to be between -1 and 1
    if isinstance(traj, np.ndarray):
        traj = np.clip(traj, -1, 1)
    elif isinstance(traj, torch.Tensor):
        traj = torch.clamp(traj, -1, 1)

    return traj.reshape(initial_shape)


def unnormalize_traj(traj_input, img_shape):
    """
    transforms the trajectory so that the pixel coordinates are between height and width of the image
    traj: (bs, num_views, horizon, num_queries, 2) with (row, col) in (-1, 1)

    Returns:
    traj: (bs, num_views, horizon, num_queries, 2) with (row, col) in (height, width)
    """
    traj = traj_input.clone()
    initial_shape = traj.shape

    # making sure traj is a float numpy array or tensor
    if isinstance(traj, np.ndarray):
        traj = traj.astype(np.float32)
    elif isinstance(traj, torch.Tensor):
        traj = traj.float()
    else:
        raise ValueError(f"traj should be a numpy array or tensor, but is of type {type(traj)}")

    traj[..., 0] = (traj[..., 0] * float(img_shape[0]) / 2) + float(img_shape[0]) / 2
    traj[..., 1] = (traj[..., 1] * float(img_shape[1]) / 2) + float(img_shape[1]) / 2

    return traj.reshape(initial_shape)


def idx_to_traj(idx, img_size):
    """
    idx: (bs, views, horizon, num_queries) with 1D indices
    config: config dict

    Returns:
    traj: (bs, views, horizon, num_queries, 2) with 2D coordinates in img_size (row, col)
    """
    b, v, h, n = idx.shape
    flat_indices = idx.clone().float().detach().requires_grad_(True) # TODO: figure out if we need requires_grad

    rearranged = False
    if len(flat_indices.shape) == 4:
        flat_indices = rearrange(flat_indices, 'b v h n -> b (v h n)')  # stacking the queries so we can process them all at once
        rearranged = True

    assert len(flat_indices.shape) == 2, f"flat_indices should have shape (bs, v*h*n), but has shape {flat_indices.shape}"

    # Calculate 2D coordinates from 1D indices
    x = torch.floor(flat_indices / img_size[1])
    y = flat_indices % img_size[1]

    traj = torch.stack([x, y], dim=-1)

    if rearranged:
        traj = rearrange(traj, 'b (v h n) d -> b v h n d', v=v, h=h)

    return traj

def interpolate_traj(traj, new_seq_len):
    """
    Interpolates the time dim to new_seq_len, maintaining the first
    and last points of the trajectory.

    Args:
        traj (torch.Tensor): The input trajectory of shape (..., horizon, num_queries, dim).
        new_seq_len (int): The desired number of points in the output trajectory.

    Returns:
        torch.Tensor: The trajectory of shape (num_views, new_seq_len, num_queries, 2) after interpolation.
    """
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)

    if traj.shape[-3] == new_seq_len:
        return traj

    if traj.ndim == 4:
        v, t, n, d = traj.shape
        traj = rearrange(traj, 'v t n d -> v (n d) t')
        traj = F.interpolate(traj, size=new_seq_len, mode='linear', align_corners=True)
        traj = rearrange(traj, 'v (n d) t -> v t n d', n=n)
    else:
        b, v, t, n, d = traj.shape
        traj = rearrange(traj, 'b v t n d -> b (v n d) t')
        traj = F.interpolate(traj, size=new_seq_len, mode='linear', align_corners=True)
        traj = rearrange(traj, 'b (v n d) t -> b v t n d', v=v, n=n)

    return traj

def interpolate_traj_spline(traj, new_seq_len):
    """
    Interpolates the time dimension to new_seq_len using cubic spline interpolation,
    maintaining the first and last points of the trajectory.

    Args:
        traj (torch.Tensor): The input trajectory of shape (..., horizon, num_queries, dim).
        new_seq_len (int): The desired number of points in the output trajectory.

    Returns:
        torch.Tensor: The trajectory of shape (..., new_seq_len, num_queries, dim) after interpolation.
    """
    if traj.shape[-3] == new_seq_len:
        return traj
    original_seq_len = traj.shape[-3]
    time_orig = np.linspace(0, 1, original_seq_len)
    time_new = np.linspace(0, 1, new_seq_len)

    if isinstance(traj, torch.Tensor):
        traj = traj.numpy()
    spline = CubicSpline(time_orig, traj, axis=-3)
    interpolated = spline(time_new)

    return torch.tensor(interpolated, dtype=torch.float32)

def get_autoregressive_indices_efficient(input_trajs, target_trajs, img_size, rel_img_size,
                                          num_angle_bins=None, num_mag_bins=None, max_polar_mag=None):
    """
    Optimized version that avoids creating the one-hot tensor.

    Returns a dictionary with:
      - 'relative': flattened indices computed as:
            row_index * rel_img_size[1] + col_index,
        where row_index and col_index are derived from the clamped relative positions.
      - 'angle': angle bins (or None)
      - 'magnitude': magnitude bins (or None)
      - 'polar': joint polar index (or None)
    """
    batch_size, views, horizon, num_queries, _ = target_trajs.shape

    # Convert trajectories to pixel space and clip to image bounds
    trajs_pixel_space = unnormalize_traj(target_trajs.clone(), img_shape=img_size)
    trajs_pixel_space[..., 0] = torch.clamp(trajs_pixel_space[..., 0], 0, img_size[0] - 1)
    trajs_pixel_space[..., 1] = torch.clamp(trajs_pixel_space[..., 1], 0, img_size[1] - 1)
    trajs_pixel_space = trajs_pixel_space.round().long()

    input_trajs_pixel_space = unnormalize_traj(input_trajs.clone(), img_shape=img_size)
    input_trajs_pixel_space[..., 0] = torch.clamp(input_trajs_pixel_space[..., 0], 0, img_size[0] - 1)
    input_trajs_pixel_space[..., 1] = torch.clamp(input_trajs_pixel_space[..., 1], 0, img_size[1] - 1)
    input_trajs_pixel_space = input_trajs_pixel_space.round().long()

    # Compute relative positions in pixel space
    vec_kp_pos_relative = trajs_pixel_space - input_trajs_pixel_space  # (bs, views, horizon, num_queries, 2)
    vec_kp_pos_relative = vec_kp_pos_relative.float()

    if num_angle_bins is not None and num_mag_bins is not None and max_polar_mag is not None:
        angle_bins, mag_bins = get_angle_mag_bins(vec_kp_pos_relative, num_angle_bins, num_mag_bins, max_polar_mag)
        polar_bins = get_joint_polar_indices(angle_bins, mag_bins, num_angle_bins, num_mag_bins)
    else:
        angle_bins, mag_bins, polar_bins = None, None, None

    # Clamp relative positions to be within the relative image size
    max_diff_rows, max_diff_cols = rel_img_size[0] // 2, rel_img_size[1] // 2
    vec_kp_pos_relative[..., 0] = torch.clamp(vec_kp_pos_relative[..., 0], -max_diff_rows, max_diff_rows)
    vec_kp_pos_relative[..., 1] = torch.clamp(vec_kp_pos_relative[..., 1], -max_diff_cols, max_diff_cols)
    vec_kp_pos_relative = vec_kp_pos_relative.round().long()

    # Directly compute flattened indices:
    # First, convert relative row and col positions into indices on a [0, rel_img_size-1] grid.
    relative_indices_row = (vec_kp_pos_relative[..., 0] + rel_img_size[0] // 2).long()
    relative_indices_col = (vec_kp_pos_relative[..., 1] + rel_img_size[1] // 2).long()
    vec_kp_indices_relative = relative_indices_row * rel_img_size[1] + relative_indices_col

    return {
        'relative': vec_kp_indices_relative,
        'angle': angle_bins,
        'magnitude': mag_bins,
        'polar': polar_bins
    }

def rel_indices_to_diffs(rel_indices, rel_img_size, global_img_size):
    '''
    rel_indices: (bs, horizon, num_queries) indices of the keypoints relative to the previous timestep.
    rel_img_size: size of the relative image (height, width)
    return: (bs, horizon, num_queries, 2) normalized differences between the keypoints relative to the previous timestep
    '''
    # indices -> pixel coordinates
    pixel_coords = idx_to_traj(idx=rel_indices, img_size=rel_img_size)
    # centering pixel coordinates
    pred_diffs = torch.zeros_like(pixel_coords)
    pred_diffs[..., 0] = pixel_coords[..., 0] - rel_img_size[0] // 2
    pred_diffs[..., 1] = pixel_coords[..., 1] - rel_img_size[1] // 2

    # normalizing pixel coordinates (normalized global image has a width and height of 2)
    pred_diffs[..., 0] = pred_diffs[..., 0] / global_img_size[0] * 2
    pred_diffs[..., 1] = pred_diffs[..., 1] / global_img_size[1] * 2

    return pred_diffs


def rel_cls_logits_to_diffs(
        logits,
        pred_views,
        num_tracks,
        rel_cls_img_size,
        global_img_size,
        zero_pred_idx_multiplier=None,
        get_last_timestep=False
    ):
    """
    Relative classification logits to relative diffs for the last timestep.
    args:
        logits (b, v*t*n, d)
    returns:
        pred_diffs (b, v, 1, n, 2)
    """
    assert logits.dim() == 3
    b, vtn, d = logits.shape
    # old_logits = rearrange(logits, 'b (v t n) d -> b v t n d', n=self.cfg.num_tracks)[:, [-1]] # this is slow for some reason
    logits = logits.view(b, pred_views, vtn//(num_tracks * pred_views), num_tracks, d)

    if get_last_timestep:
        logits = logits[:, :, [-1]]

    assert logits.dim() == 5
    probs = F.softmax(logits, dim=-1) # (b, v, t, n, d)
    if zero_pred_idx_multiplier not in [1.0, None]:
        center_idx = rel_cls_img_size[0] // 2 * rel_cls_img_size[1] + rel_cls_img_size[1] // 2
        probs[:, :, :, center_idx] *= zero_pred_idx_multiplier
    assert probs.dim() == 5
    rel_indices = torch.argmax(probs, dim=-1)# (b, v, t, n)
    assert rel_indices.dim() == 4

    pred_diffs = rel_indices_to_diffs(
        rel_indices=rel_indices,
        rel_img_size=rel_cls_img_size,
        global_img_size=global_img_size,
    )

    return pred_diffs


def round_traj(traj, img_size):
    """
    rounds the trajectory to the nearest pixel. image coords lie between -1 and 1
    """
    traj[..., 0] = torch.round(traj[..., 0] * img_size[0] / 2) * 2 / img_size[0]
    traj[..., 1] = torch.round(traj[..., 1] * img_size[1] / 2) * 2 / img_size[1]

    return traj


def center_crop_video(video, img_size):
    """
    takes a video and crops the center of each frame to img_size
    args:
    video: (t, h, w, 3)
    img_size: (height, width)
    """
    h, w = img_size
    th, tw = video.shape[1], video.shape[2]
    x1 = int(round((th - h) / 2.))
    y1 = int(round((tw - w) / 2.))
    return video[:, x1:x1+h, y1:y1+w]


def center_crop_traj(traj, traj_vis, img_size, video):
    """
    Adjusts the tracks to the cropped video
    args:
    traj: (t, n, 2) normalized traj in (-1, 1)
    img_size: (height, width)
    video: (t, height, width, 3)

    returns:
    traj: (t, n-k, 2) adjusted traj with k cropped out points in (-1, 1)

    """
    initial_traj_shape = traj.shape
    video_height, video_width = video.shape[1], video.shape[2]
    new_height, new_width = img_size

    # center cropping traj to img_size
    x1 = int(round((video_height - new_height) / 2.))
    y1 = int(round((video_width - new_width) / 2.))

    # Adjust the coordinates
    traj[:, :, 0] -= x1
    traj[:, :, 1] -= y1

    # Filtering out points that fall outside the cropped area
    valid_indices = (traj[..., 0] >= 0) & (traj[..., 0] < new_height) & (traj[..., 1] >= 0) & (traj[..., 1] < new_width)
    print("valid_indices.shape:", valid_indices.shape)

    # reshaping the traj back to the original shape, except for the second to last dim. (1, t, n, 2) or (1, t, h, n, 2) if reinit
    new_traj_shape = list(initial_traj_shape)
    new_traj_shape[-2] = valid_indices.sum().item()
    traj = traj[valid_indices].view(new_traj_shape)
    traj_vis = traj_vis[valid_indices].view(new_traj_shape[:-1]).unsqueeze(-1)

    # Normalizing the cropped traj back to (-1, 1)
    traj = normalize_traj(traj, img_shape=(new_height, new_width))

    return traj, traj_vis


def resize_traj(traj, original_img_shape, new_img_shape):
    '''
    Resizes a trajectory to the given parameters
    '''
    assert len(original_img_shape) == 2 and len(new_img_shape) == 2, "Image shapes must be 2D"
    traj = normalize_traj(traj, img_shape=original_img_shape)
    traj = unnormalize_traj(traj, img_shape=new_img_shape)
    traj = torch.round(traj)
    return traj


def resize_crop_traj(traj, params, img_shape):
    '''
    Resizes and crops a trajectory to the given parameters
    args:
        traj: (v, t, n, 2)
        params: (top, left, height, width)
    returns:
        traj: (v, t, n, 2)
    '''
    # kp_new = (kp_old - translation) * scale_factor
    orig_img_center = img_shape[0] / 2, img_shape[1] / 2
    new_img_center = (params[0] + params[2] / 2, params[1] + params[3] / 2)
    translation = (float(new_img_center[0] - orig_img_center[0]) / img_shape[0] * 2, float(new_img_center[1] - orig_img_center[1]) / img_shape[1] * 2) # (new - old) * 2 / old_size
    scale_factor = (float(img_shape[0]) / params[2], float(img_shape[1]) / params[3]) # old / new

    traj[..., 0] = (traj[..., 0] - translation[0]) *  scale_factor[0]
    traj[..., 1] = (traj[..., 1] - translation[1]) *  scale_factor[1]

    return traj

class RandomGaussianBlur(object):
    def __init__(self, kernel_sizes=[3, 5, 7], sigma_min=0.1, sigma_max=2.0, p=0.5):
        self.kernel_sizes = kernel_sizes
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            kernel_size = random.choice(self.kernel_sizes)
            sigma = random.uniform(
                self.sigma_min,
                self.sigma_max
            )
            return transforms.functional.gaussian_blur(img, kernel_size, sigma)
        return img


def resize_everything(new_img_shape, traj, images):
    """
    Resizes trajectories and images
    Args:
     - trajs (..., 2)
     - images (b, c, h, w)
    """
    original_img_shape = images.shape[-2:]
    traj = torch.tensor(traj)
    images = torch.tensor(images)
    traj = resize_traj(traj, original_img_shape, new_img_shape)
    resize = transforms.Resize(new_img_shape)
    images = resize(images)

    resized_dict = {
        'traj': traj,
        'images': images,
    }

    return resized_dict


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    Adapted from: https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html#top_k_top_p_filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def points_to_velocities(points, time_dim):
    return torch.diff(points, dim=time_dim)


def velocities_to_points(velocities, time_dim, init_points):
    """
    There are 1 fewer velocities than points, need to append zeros as the first timestep of the velocities.
    Zeros need to be the same shape as the velocities, need to index time_dim away
    """
    zero_velocity = torch.zeros_like(velocities.select(time_dim, 0).unsqueeze(time_dim))
    velocities_with_zero = torch.cat([zero_velocity, velocities], dim=time_dim)
    points = init_points + torch.cumsum(velocities_with_zero, dim=time_dim)

    return points

def grab_libero_language_from_filename(x):
    if x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]
