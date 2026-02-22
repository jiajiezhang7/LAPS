import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange

from amplify.utils.data_utils import interpolate_traj, unnormalize_traj

def vis_attn_map(q, k, attn_mask):
    # visualize attention map
    attention_map_vis = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    attention_map_vis = attention_map_vis.masked_fill(attn_mask == 0, float('-inf')) # fill the attention weights with -inf where the mask is 0 so that softmax will make them 0
    attention_map_vis = F.softmax(attention_map_vis, dim=-1)

    # averaging across heads and batch
    plt.imshow(attention_map_vis.mean(dim=(0, 1)).detach().cpu().numpy(), cmap='viridis')
    plt.show()


def vis_attn_mask(attn_mask, title=""):
    print("attn_mask shape:", attn_mask.shape)
    plt.imshow(attn_mask.detach().cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()


def compute_moving_indices(traj, threshold=0.05):
    """
    Compute moving indices for the given trajectory tensor.

    Parameters:
    traj (torch.Tensor): Input tensor of shape (b, v, t, n, d) (normalized to [-1, 1]).

    Returns:
    torch.Tensor: Tensor containing moving indices (1 if moving, 0 if not moving) with shape (b, v, n).
    """
    diff = traj[:, :, 1:] - traj[:, :, :-1]
    total_distance = torch.sum(torch.norm(diff.float(), dim=-1), dim=2)
    moving_indices = (total_distance > threshold).float()

    return moving_indices

def vis_pred(images, trajs, cmap='autumn_r', interp_seq_len=64, opacity=0.5, sample_ratio=1.0):
    """
    images (b, v, h, w, c) tensor, (0-1)
    trajs (b, v, t, n, d) tensor, (-1, 1), (row, col)

    Returns:
    vis_imgs (b, h, w, c) tensor, (0-255)
    """
    assert trajs.dim() == 5
    assert images.dim() == 5

    image_shape = images.shape[2:4]  # (h, w)

    # Interpolate for visualization
    vis_traj = interpolate_traj(trajs.clone(), interp_seq_len)

    # Unnormalize for visualization
    vis_images = images.clone() * 255
    vis_traj = unnormalize_traj(vis_traj, image_shape).round().long()
    vis_traj = vis_traj.clamp(0, image_shape[0] - 1)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    horizon = vis_traj.shape[2]
    vis_img_list = []
    batch_size = images.shape[0]

    moving_indices = compute_moving_indices(trajs)  # (b, v, n)

    for v in range(trajs.shape[1]):
        vis_img = vis_images[:, v].clone()  # Shape: [batch_size, h, w, c]
        for t in range(horizon):
            # Set all points at timestep to color
            color = cmap(t / horizon)[:3]
            color = [int(c * 255) for c in color]

            r1 = vis_traj[:, v, t, :, 0]  # Shape: [batch_size, num_points]
            c1 = vis_traj[:, v, t, :, 1]  # Shape: [batch_size, num_points]
            r2 = torch.clamp(r1 + 1, 0, image_shape[0] - 1)
            c2 = torch.clamp(c1 + 1, 0, image_shape[1] - 1)

            num_points = r1.shape[1]
            # Sample indices according to sample_ratio
            if sample_ratio < 1.0:
                target_num_points = max(1, int(num_points * sample_ratio))
                sampled_indices = torch.linspace(0, num_points - 1, target_num_points, dtype=torch.long, device=trajs.device)
            else:
                sampled_indices = torch.arange(0, num_points, device=trajs.device)

            # Index the trajectory coordinates with the sampled indices.
            r1 = r1[:, sampled_indices]
            c1 = c1[:, sampled_indices]
            r2 = r2[:, sampled_indices]
            c2 = c2[:, sampled_indices]
            num_points = r1.shape[1]  # Update number of points after sampling

            batch_indices = torch.arange(batch_size, device=trajs.device).unsqueeze(1).expand(batch_size, num_points)

            color_tensor = torch.tensor(color, device=trajs.device, dtype=vis_images.dtype).view(1, 1, 3).expand(batch_size, num_points, 3)

            # Sample moving indices accordingly.
            opacity_tensor = moving_indices[:, v][:, sampled_indices].unsqueeze(2).expand(batch_size, num_points, 3) * opacity
            inv_opacity_tensor = 1 - opacity_tensor

            vis_img[batch_indices, r1, c1, :] = inv_opacity_tensor * vis_img[batch_indices, r1, c1, :] + opacity_tensor * color_tensor
            vis_img[batch_indices, r2, c1, :] = inv_opacity_tensor * vis_img[batch_indices, r2, c1, :] + opacity_tensor * color_tensor
            vis_img[batch_indices, r1, c2, :] = inv_opacity_tensor * vis_img[batch_indices, r1, c2, :] + opacity_tensor * color_tensor
            vis_img[batch_indices, r2, c2, :] = inv_opacity_tensor * vis_img[batch_indices, r2, c2, :] + opacity_tensor * color_tensor

        vis_img_list.append(vis_img)

    cat_vis_imgs = torch.cat(vis_img_list, dim=-2)
    return cat_vis_imgs.to(dtype=torch.uint8)

def print_dict(dict, name="batch"):
    """
    Pretty print info about each key in the batch
    """
    print(f"----- {name.upper()} -----")
    for key, value in sorted(dict.items()):
        print(f'KEY: {key}')
        print(f'\tTYPE: {type(value)}')
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            print(f'\tSHAPE: {value.shape}')
            print(f'\tDTYPE: {value.dtype}')
            print(f'\tMIN: {value.min()}')
            print(f'\tMAX: {value.max()}')
        elif isinstance(value, list):
            print(f'\tLENGTH: {len(value)}')
        else:
            print(f'\tVALUE: {value}')
    print("----------------------")


def vis_batch(batch, save_path='tests/vis_batch', num_vis=10, view=None, show=False):
    """
    Visualize a batch of data
    Expecting:
    - images    (b, v, h, w, c)              tensor              [0, 1]
    - text      (b)                          list of str
    - actions   (b, T_true, 7)               tensor              [-1, 1]
    - traj      (b, v, T_pred, N_sampled, 2) tensor              [-1, 1]
    - vis       (b, v, T_pred, N_sampled, 1) tensor              [0, 1]
    """
    # Visualize tracks and images
    vis_traj_imgs = vis_pred(batch['images'], batch['traj'])
    num_vis = min(num_vis, batch['images'].shape[0])
    num_views = batch['images'].shape[1]
    plt.figure(figsize=(10, 10))
    for i in range(num_vis):
        for view in range(num_views):
            ax = plt.subplot(2, num_views, view+1)
            ax.imshow(batch['images'][i, view].cpu().detach().numpy())
            ax.set_title('RGB')

        ax = plt.subplot(2, 1, 2)
        ax.imshow(vis_traj_imgs[i].cpu().detach().numpy())
        ax.set_title('Trajectory')

        try:
            plt.suptitle(f'{batch["text"][i]}')
        except KeyError:
            pass
        if show:
            plt.show()
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"batch_vis_{i}.png"))
        plt.close()


def wait_for_key_in_figure(fig):
    """
    Wait until a key is pressed in the given Matplotlib figure.
    Returns the key as a string.
    """
    key_pressed = None

    def on_key(event):
        nonlocal key_pressed
        key_pressed = event.key
        print(f"Key pressed: {key_pressed}")

    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    # Loop until a key is pressed.
    while key_pressed is None:
        plt.pause(0.1)
    fig.canvas.mpl_disconnect(cid)
    return key_pressed


def vis_batch_keyboard(batch, meta, save_path='tests/vis_batch', num_vis=1, view=None, show=False, sample_ratio=0.9):
    """
    Visualize a batch of data interactively and capture a keystroke directly from the figure.

    Expects:
      - batch: a batch of data (ideally with batch_size=1 for interactive inspection)
      - meta: metadata for this sample (e.g., a dict from dataset.index_map)

    Commands:
      - z: Save the current visualization to an image.
           (Filename is built from meta: demo_key with "data/" removed and start_t appended.)
      - n or right arrow: Go to the next sample.
      - left arrow: Go to the previous sample.
      - 5: Skip the next 5 samples.
      - q: Quit inspection.
      - up arrow: Increase the fraction of trajectory points (sample_ratio) drawn.
      - down arrow: Decrease the fraction of trajectory points (sample_ratio) drawn.

    Returns:
      The key (command) entered by the user.
    """
    current_sample_ratio = sample_ratio

    # Loop until a non-adjustment key is pressed.
    while True:
        # Generate visualization using current_sample_ratio.
        vis_traj_imgs = vis_pred(batch['images'], batch['traj'], sample_ratio=current_sample_ratio)
        num_vis_local = min(num_vis, batch['images'].shape[0])
        num_views = batch['images'].shape[1]

        # Create the figure.
        fig = plt.figure(figsize=(10, 10))
        for i in range(num_vis_local):
            # Top row: display each view (assumes images shape: [b, v, h, w, c])
            for view in range(num_views):
                ax = plt.subplot(2, num_views, view + 1)
                ax.imshow(batch['images'][i, view].cpu().detach().numpy())
                ax.set_title('RGB')
            # Bottom row: display the trajectory visualization.
            ax = plt.subplot(2, 1, 2)
            ax.imshow(vis_traj_imgs[i].cpu().detach().numpy())
            ax.set_title('Trajectory')
            # Show metadata text (and current sample ratio)
            try:
                plt.suptitle(f"{batch['text'][i]} (Sample Ratio: {current_sample_ratio:.2f})")
            except KeyError:
                plt.suptitle(f"(Sample Ratio: {current_sample_ratio:.2f})")

        # Show figure non-blocking.
        plt.show(block=False)
        # Wait for a key press.
        key = wait_for_key_in_figure(fig)
        plt.close(fig)

        # If the user pressed up or down arrow, adjust sample ratio and redraw.
        if key == 'up':
            current_sample_ratio = min(1.0, current_sample_ratio + 0.1)
            print(f"Increasing sample_ratio to {current_sample_ratio:.2f}")
            continue  # Redraw with new ratio.
        elif key == 'down':
            current_sample_ratio = max(0.0, current_sample_ratio - 0.1)
            print(f"Decreasing sample_ratio to {current_sample_ratio:.2f}")
            continue  # Redraw with new ratio.
        else:
            break  # Any other key stops the adjustment loop.

    # If the command is 's', then save the current visualization.
    if key == 'z':
        demo_key = meta.get('demo_key', 'unknown').replace("data/", "")
        start_t = meta.get('start_t', 'unknown')
        filename = f"img_{demo_key}_{start_t}.png"
        os.makedirs(save_path, exist_ok=True)
        # Re-generate the final visualization with the chosen sample ratio.
        final_vis = vis_pred(batch['images'], batch['traj'], sample_ratio=current_sample_ratio)
        fig = plt.figure(figsize=(10, 10))
        for i in range(num_vis_local):
            for view in range(num_views):
                ax = plt.subplot(2, num_views, view + 1)
                ax.imshow(batch['images'][i, view].cpu().detach().numpy())
                ax.set_title('RGB')
            ax = plt.subplot(2, 1, 2)
            ax.imshow(final_vis[i].cpu().detach().numpy())
            ax.set_title('Trajectory')
            try:
                plt.suptitle(f"{batch['text'][i]} (Sample Ratio: {current_sample_ratio:.2f})")
            except KeyError:
                plt.suptitle(f"(Sample Ratio: {current_sample_ratio:.2f})")
        fig.savefig(os.path.join(save_path, filename))
        print(f"Saved image as {filename}")
        plt.close(fig)
    elif key in ['i', 'o', 'p']:
        demo_key = meta.get('demo_key', 'unknown').replace("data/", "")
        start_t = meta.get('start_t', 'unknown')
        # Map key to the corresponding view index.
        key_to_index = {'i': 0, 'o': 1, 'p': 2}
        view_index = key_to_index.get(key, 0)
        num_views = batch['images'].shape[1]

        if view_index >= num_views:
            print(f"Requested view index {view_index} not available. Only {num_views} views available.")
        else:
            filename = f"best_quality_view_{view_index}_{demo_key}_{start_t}.png"
            os.makedirs(save_path, exist_ok=True)
            best_quality_img = batch['images'][0, view_index].cpu().detach().numpy()
            # Save the image preserving high quality.
            plt.imsave(os.path.join(save_path, filename), best_quality_img)
            print(f"Saved best quality image from view {view_index} as {filename}")


    return key


def visualize_action_distribution(actions, num_bins=100, separate_plots=True, log_scale=False, nonzero=True):
    """
    Visualizes the action distribution for the first 6 dimensions of a given set of actions
    and prints statistics for each dimension.

    Parameters:
    - actions: np.ndarray of shape (t, 7), where t is the number of timesteps
    - num_bins: int, the number of bins to use for the histogram (default: 100)
    - separate_plots: bool, whether to plot each dimension separately (default: False)
    - log_scale: bool, whether to use a log scale for the histogram (default: False)
    - nonzero: bool, whether to include only nonzero actions (default: True)
    """

    # Ensure actions has the correct shape
    if actions.shape[1] != 7:
        raise ValueError("Input array must have shape (t, 7)")

    # Calculate min and max for the first 6 dimensions
    action_min, action_max = np.min(actions[:, :6]), np.max(actions[:, :6])
    action_range = (action_min, action_max)

    # Calculate probabilities for the last binary dimension
    prob_minus1 = np.mean(actions[:, -1] == -1)
    prob_plus1 = np.mean(actions[:, -1] == 1)
    print(f"Last Dimension - Probability of -1: {prob_minus1:.4f}, Probability of +1: {prob_plus1:.4f}")

    # Initialize histogram for the first 6 dimensions
    action_histograms = np.zeros((6, num_bins))

    # Fill histograms for the first 6 dimensions
    for dim in range(6):
        if nonzero:
            hist, _ = np.histogram(actions[actions[:, dim] != 0, dim], bins=num_bins, range=action_range)
        else:
            hist, _ = np.histogram(actions[:, dim], bins=num_bins, range=action_range)
        action_histograms[dim] = hist

    # Convert action histograms to a log scale to handle large value disparities
    if log_scale:
        action_histograms = np.log1p(action_histograms)
        log_text = "Log "
    else:
        log_text = ""

    if separate_plots:
        # Create separate plots for each dimension
        fig, axs = plt.subplots(3, 2, figsize=(12, 8), constrained_layout=True)
        axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

        for dim in range(6):
            axs[dim].bar(np.linspace(action_min, action_max, num_bins), action_histograms[dim], width=(action_max - action_min) / num_bins)
            axs[dim].set_title(f'{log_text}Action Distribution for Dimension {dim+1}')
            axs[dim].set_xlabel('Action Value Range')
            axs[dim].set_ylabel(f'{log_text}Frequency')

        plt.show()
    else:
        # Plot the 2D histogram heatmap for the first 6 dimensions
        plt.figure(figsize=(10, 6))
        plt.imshow(action_histograms, aspect='auto', cmap='viridis', extent=[action_min, action_max, 6, 0])
        plt.colorbar(label=f'{log_text}Frequency')
        plt.xlabel('Action Value Range')
        plt.ylabel('Action Dimension')
        plt.title(f'{log_text}Action Value Distribution for First 6 Dimensions')
        plt.yticks(range(6), [f'Dim {i+1}' for i in range(6)])
        plt.show()


def visualize_action_time(actions):
    """
    Plot the average magnitude of actions over time, with time dimension normalized 0-1

    Parameters:
    - actions: list of np.ndarray of shape (t, 7), where t is the number of timesteps. Each array may have a different number of timesteps.
    """
    # Calculate the maximum number of timesteps across all actions
    max_timesteps = max([actions[i].shape[0] for i in range(len(actions))])

    # Initialize a list to store the average magnitude of actions at each timestep
    avg_magnitudes = []

    # Iterate over each timestep and calculate the average magnitude of actions
    for t in range(max_timesteps):
        # exclude gripper action from magnitude calculation
        timestep_actions = [actions[i][t][:-1] for i in range(len(actions)) if t < actions[i].shape[0]]
        avg_magnitude = np.mean(np.abs(timestep_actions))
        avg_magnitudes.append(avg_magnitude)

    # Plot the average magnitude of actions over time
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, 1, max_timesteps), avg_magnitudes)
    plt.xlabel('Normalized Timestep')
    plt.ylabel('Average Magnitude of Actions')
    plt.title('Average Magnitude of Actions Over Time')
    plt.show()


def visualize_rel_logits(rel_logits, traj):
    """
    Visualize the relative velocity logits for a given timestep
    """
    b, v, t, n, d = traj.shape

    rel_logits = rearrange(rel_logits, 'b (v t n) d -> b v t n d', v=v, t=t-1, n=n)
    window_size = int(np.sqrt(rel_logits.shape[-1]))
    img_folder = 'tests/vis_logits'
    os.makedirs(img_folder, exist_ok=True)
    num_imgs = len(os.listdir(img_folder))

    # visualize the logits for each timestep
    # reshaping from flat to square
    kp_idx = torch.randint(0, n, (1,))
    rel_logit_sample = rel_logits[0, 0, 0, kp_idx, :]
    # softmax with temperature
    rel_prob_sample = F.softmax(rel_logit_sample / 10, dim=-1).cpu().detach().numpy()
    rel_prob_window = rel_prob_sample.reshape(window_size, window_size)
    if np.argmax(rel_prob_window) != 112 or True:
        plt.imshow(rel_prob_window, cmap='viridis') # (sqrt(d), sqrt(d))
        plt.savefig(os.path.join(img_folder, f'logits_{num_imgs}.png'))
        # plt.show()
        plt.close()
    num_imgs += 1
