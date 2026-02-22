import math

import torch

from amplify.utils.vis_utils import vis_attn_mask


def full_mask(seq_len, device, **kwargs):
    """
    Full attention
    """
    return torch.ones(seq_len, seq_len).to(device).bool()


def causal_mask(seq_len, device, **kwargs):
    """
    Causal attention
    """
    return torch.tril(torch.ones(seq_len, seq_len)).to(device).bool()


def causal_cond_mask(seq_len, num_cond_tokens, device, **kwargs):
    """
    Causal attention 
    + conditioning tokens attend to all other conditioning tokens
    """
    attn_mask = causal_mask(seq_len, device)
    attn_mask[:, :num_cond_tokens] = 1
    return attn_mask.bool()

def diag_cond_mask(seq_len, num_cond_tokens, device, **kwargs):
    """
    identity matrix
    + conditioning tokens attend to all other conditioning tokens
    """
    attn_mask = torch.eye(seq_len).to(device)
    attn_mask[:, :num_cond_tokens] = 1
    return attn_mask.bool()

def block_mask(seq_len, num_cond_tokens, tokens_per_timestep, device,  **kwargs):
    """
    Causal attention
    + conditioning tokens attend to all other conditioning tokens
    + every token can attend to all previous tokens and all tokens in the current timestep
    """
    attn_mask = causal_cond_mask(seq_len, num_cond_tokens, device)
    for i in range(num_cond_tokens, seq_len):
        start_index = 0
        end_index = math.ceil((i - num_cond_tokens + 1) / tokens_per_timestep) * tokens_per_timestep + num_cond_tokens
        attn_mask[i, start_index:end_index] = 1
        # zero out the rest of the row
        attn_mask[i, num_cond_tokens:start_index] = 0
    return attn_mask.bool()


def noimgtext_cls_block_mask(seq_len, num_cond_tokens, tokens_per_timestep, cls, num_img_tokens, num_text_tokens, device, **kwargs):
    """
    Causal attention
    + conditioning tokens attend to all other conditioning tokens
    + every token can attend to all previous tokens and all tokens in the current timestep except cls tokens
    + cls tokens cannot attend to text or image tokens (assuming they come first)
    """
    attn_mask = block_mask(seq_len, num_cond_tokens, tokens_per_timestep, device)
    attn_mask[-cls:, :num_img_tokens + num_text_tokens] = 0
    return attn_mask.bool()


def last_n_timesteps_mask(seq_len, num_cond_tokens, tokens_per_timestep, n, device, cls_token=False, **kwargs):
    """
    Causal attention
    + conditioning tokens attend to all other conditioning tokens
    + every token can attend to its own tokens in the last n timesteps
    + (optinal) cls token can attend to all previous cls tokens. It is assumed that the cls token is the last token in the timestep
    """
    attn_mask = causal_cond_mask(seq_len, num_cond_tokens, device)
    attn_mask[:, num_cond_tokens:] = 0 # zero out non-conditioning tokens
    for i in range(num_cond_tokens, seq_len):
        start_index = i - n * tokens_per_timestep
        end_index = i + 1

        attn_indices = torch.arange(start_index, end_index, tokens_per_timestep)
        attn_indices = attn_indices[attn_indices >= num_cond_tokens]
        attn_mask[i, attn_indices] = 1

        if cls_token and (i - num_cond_tokens) % tokens_per_timestep == 0:
            # allow cls token to attend to all previous cls tokens
            cls_floor = min(i + (tokens_per_timestep * n) + 1, seq_len)
            attn_mask[i:cls_floor, i] = 1
            attn_mask[i, i:i+tokens_per_timestep] = 1

    return attn_mask.bool()


def last_n_tokens_mask(seq_len, num_cond_tokens, tokens_per_timestep, n, device,  **kwargs):
    """
    Causal attention
    + conditioning tokens attend to all other conditioning tokens
    + every token can attend to the last n*tokens_per_timestep tokens
    """
    attn_mask = causal_cond_mask(seq_len, num_cond_tokens, device)
    for i in range(num_cond_tokens, seq_len):
        start_index = max(num_cond_tokens, i + 1 - n * tokens_per_timestep)
        end_index = i + 1
        attn_mask[i, start_index:end_index] = 1
        # zero out the rest of the row
        attn_mask[i, num_cond_tokens:start_index] = 0

    return attn_mask.bool()


def current_token_mask(seq_len, num_cond_tokens, device,  **kwargs):
    """
    Causal attention
    + conditioning tokens attend to all other conditioning tokens
    + every token can only attend to the current token
    """
    attn_mask = causal_cond_mask(seq_len, num_cond_tokens, device)
    for i in range(num_cond_tokens, seq_len):
        attn_mask[i, num_cond_tokens:i] = 0
        attn_mask[i, i] = 1

    return attn_mask.bool()

def bc_mask(seq_len, num_cond_tokens, cls, device, **kwargs):
    """
    cls tokens attend to all conditioning tokens and the cls tokens
    """
    attn_mask = torch.zeros(seq_len, seq_len).to(device)
    attn_mask[:, :num_cond_tokens] = 1
    attn_mask[:, -cls:] = 1
    return attn_mask.bool()

def block_bc_cls_mask(seq_len, num_cond_tokens, tokens_per_timestep, cls, device, **kwargs):
    """
    cls tokens attend to all conditioning tokens and the cls tokens, kp tokens attend to cond tokens and prev kp tokens
    """
    attn_mask = block_mask(seq_len, num_cond_tokens, tokens_per_timestep=tokens_per_timestep, device=device)
    attn_mask[-cls:, -cls:] = 1
    return attn_mask.bool()


# def block_bc_cls_symmetric_mask(seq_len, num_cond_tokens, tokens_per_timestep, cls, device, **kwargs):
#     """
#     cls tokens attend to all conditioning tokens and the cls tokens, kp tokens attend to cond tokens, prev kp tokens, and cls tokens
#     this allows information to leak to future kp tokens, so we should not use this
#     """
#     attn_mask = block_mask(seq_len, num_cond_tokens, tokens_per_timestep=tokens_per_timestep, device=device)
#     attn_mask[-cls:] = 1
#     attn_mask[:, -cls:] = 1
#     return attn_mask.bool()


def block_bc_same_step_cls_mask(seq_len, num_cond_tokens, tokens_per_timestep, cls, device, **kwargs):
    """
    cls tokens attend kps at same timestep and cls tokens
    """
    attn_mask = block_mask(seq_len, num_cond_tokens, tokens_per_timestep=tokens_per_timestep, device=device)
    attn_mask[-cls:] = 0
    attn_mask[:, -cls:] = 0
    attn_mask[-cls:, -cls:] = 1

    for i in range(cls):
        start_index = num_cond_tokens + tokens_per_timestep * i
        end_index = min(start_index + tokens_per_timestep, seq_len)
        attn_mask[-cls + i, start_index:end_index] = 1
    return attn_mask.bool()


def block_bc_same_step_cls_symmetric_mask(seq_len, num_cond_tokens, tokens_per_timestep, cls, device, **kwargs):
    """
    cls tokens attend to kps at same timestep and cls tokens, kp tokens attend to cond tokens, prev kp tokens, and cls tokens at same timestep
    """
    attn_mask = block_mask(seq_len, num_cond_tokens, tokens_per_timestep=tokens_per_timestep, device=device)
    attn_mask[-cls:] = 0
    attn_mask[:, -cls:] = 0
    attn_mask[-cls:, -cls:] = 1

    for i in range(cls):
        start_index = num_cond_tokens + tokens_per_timestep * i
        end_index = min(start_index + tokens_per_timestep, seq_len)
        attn_mask[-cls + i, start_index:end_index] = 1
        attn_mask[start_index:end_index, -cls + i] = 1
    return attn_mask.bool()

if __name__ == "__main__":
    num_cond_tokens = 10
    cls_token = False
    tokens_per_timestep = 2 if cls_token else 3
    num_timesteps = 8
    cls = 8
    seq_len = num_cond_tokens + tokens_per_timestep * num_timesteps + cls
    kp_context = 7
    device = torch.device("cpu")

    full_mask_img = full_mask(seq_len, device)
    causal_mask_img = causal_mask(seq_len, device)
    causal_cond_mask_img = causal_cond_mask(seq_len, num_cond_tokens, device)
    block_mask_img = block_mask(seq_len, num_cond_tokens, tokens_per_timestep, device)
    noimgtext_cls_block_mask_img = noimgtext_cls_block_mask(seq_len, num_cond_tokens, tokens_per_timestep, cls, num_img_tokens=0, num_text_tokens=0, device=device)
    last_n_timesteps_mask_img = last_n_timesteps_mask(seq_len, num_cond_tokens, 64, kp_context, device)
    last_n_timesteps_mask_cls_img = last_n_timesteps_mask(seq_len, num_cond_tokens, 65, kp_context, device, cls_token=True)
    last_n_tokens_mask_img = last_n_tokens_mask(seq_len, num_cond_tokens, tokens_per_timestep, kp_context, device)
    current_token_mask_img = current_token_mask(seq_len, num_cond_tokens, device)
    bc_mask_img = bc_mask(seq_len, num_cond_tokens, cls, device)
    block_bc_mask_img = block_bc_cls_mask(seq_len, num_cond_tokens, tokens_per_timestep, cls, device)
    # block_bc_mask_symmetric_img = block_bc_cls_symmetric_mask(seq_len, num_cond_tokens, tokens_per_timestep, cls, device)
    block_bc_mask_same_step_cls = block_bc_same_step_cls_mask(seq_len, num_cond_tokens, tokens_per_timestep, cls, device)
    block_bc_mask_same_step_cls_symmetric = block_bc_same_step_cls_symmetric_mask(seq_len, num_cond_tokens, tokens_per_timestep, cls, device)

    # vis_attn_mask(full_mask_img, "Full Mask")
    # vis_attn_mask(causal_mask_img, "Causal Mask")
    # vis_attn_mask(causal_cond_mask_img, "Causal CondMask")
    # vis_attn_mask(block_mask_img, "Block Mask")
    # vis_attn_mask(last_n_timesteps_mask_img, "Last N Timesteps Mask")
    # vis_attn_mask(last_n_timesteps_mask_cls_img, "Last N Timesteps Mask w/ CLS Token")
    # vis_attn_mask(last_n_tokens_mask_img, "Last N Tokens Mask")
    # vis_attn_mask(current_token_mask_img, "Current Token Mask")

    # Plot all masks in tiled image
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    axs[0, 0].imshow(full_mask_img.detach().cpu().numpy(), cmap='viridis')
    axs[0, 0].set_title("Full Mask")
    axs[0, 1].imshow(causal_mask_img.detach().cpu().numpy(), cmap='viridis')
    axs[0, 1].set_title("Causal Mask")
    axs[0, 2].imshow(causal_cond_mask_img.detach().cpu().numpy(), cmap='viridis')
    axs[0, 2].set_title("Causal Cond Mask")
    axs[1, 0].imshow(block_mask_img.detach().cpu().numpy(), cmap='viridis')
    axs[1, 0].set_title("Block Mask")
    axs[1, 1].imshow(last_n_timesteps_mask_img.detach().cpu().numpy(), cmap='viridis')
    axs[1, 1].set_title("Last N Timesteps Mask")
    axs[1, 2].imshow(last_n_timesteps_mask_cls_img.detach().cpu().numpy(), cmap='viridis')
    axs[1, 2].set_title("Last N Timesteps Mask w/ CLS Token")
    axs[2, 0].imshow(last_n_tokens_mask_img.detach().cpu().numpy(), cmap='viridis')
    axs[2, 0].set_title("Last N Tokens Mask")
    axs[2, 1].imshow(current_token_mask_img.detach().cpu().numpy(), cmap='viridis')
    axs[2, 1].set_title("Current Token Mask")
    axs[2, 2].imshow(block_bc_mask_same_step_cls.detach().cpu().numpy(), cmap='viridis')
    axs[2, 2].set_title("Block BC Mask w/ Same Step CLS")
    axs[3, 0].imshow(bc_mask_img.detach().cpu().numpy(), cmap='viridis')
    axs[3, 0].set_title("BC Mask")
    axs[3, 1].imshow(block_bc_mask_img.detach().cpu().numpy(), cmap='viridis')
    axs[3, 1].set_title("Block BC Mask")
    # axs[3, 2].imshow(block_bc_mask_symmetric_img.detach().cpu().numpy(), cmap='viridis')
    # axs[3, 2].set_title("Block BC Mask Symmetric")
    axs[3, 2].imshow(noimgtext_cls_block_mask_img.detach().cpu().numpy(), cmap='viridis')
    axs[3, 2].set_title("Block Mask No CLS Cond")
    axs[3, 3].imshow(block_bc_mask_same_step_cls_symmetric.detach().cpu().numpy(), cmap='viridis')
    axs[3, 3].set_title("Block BC Mask Same Step CLS Symmetric")

    plt.show()

