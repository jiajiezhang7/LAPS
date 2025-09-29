import math
import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from omegaconf import OmegaConf
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, Summer
from vector_quantize_pytorch import FSQ

from amplify.models.encoders.vision_encoders import VisionEncoder
from amplify.models.losses import compute_relative_classification_loss
from amplify.models.transformer import TransformerDecoder, TransformerEncoder
from amplify.utils.cfg_utils import merge_checkpoint_config, get_device
from amplify.utils.data_utils import rel_cls_logits_to_diffs
from amplify.utils.model.attn_masks import causal_mask, diag_cond_mask
from amplify.utils.train import get_root_dir, unwrap_compiled_state_dict


def get_fsq_level(codebook_dim):
    power = int(np.log2(codebook_dim))
    if power == 4:
        fsq_level = [5, 3]
    elif power == 6:
        fsq_level = [8, 8]
    elif power == 8:
        fsq_level = [8, 6, 5]
    elif power == 9:
        fsq_level = [8, 8, 8]
    elif power == 10:
        fsq_level = [8, 5, 5, 5]
    elif power == 11:
        fsq_level = [8, 8, 6, 5]
    elif power == 12:
        fsq_level = [7, 5, 5, 5, 5]
    return fsq_level


def get_vae_in_out_dim(motion_tokenizer_cfg):
    '''
    # per view
        # in_dim = num_tracks*point_dim
        # out_dim = rel_cls_img_size[0]*rel_cls_img_size[1]
    # not per view
        # in_dim = views*num_tracks*point_dim
        # out_dim = rel_cls_img_size[0]*rel_cls_img_size[1]
    '''
    in_dim = motion_tokenizer_cfg.num_tracks * motion_tokenizer_cfg.point_dim
    out_dim = motion_tokenizer_cfg.loss.rel_cls_img_size[0] * motion_tokenizer_cfg.loss.rel_cls_img_size[1]

    if not motion_tokenizer_cfg.per_view:
        in_dim *= len(motion_tokenizer_cfg.cond_cameraviews)


    return in_dim, out_dim


class VAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, attn_pdrop, views, num_timesteps, num_tracks, point_dim, per_view, is_causal, num_cond_tokens=0, cond_embed_dim=0) -> None:
        super().__init__()
        self.per_view = per_view
        self.views = views
        self.num_timesteps = num_timesteps
        self.num_tracks = num_tracks
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.is_causal = is_causal
        self.seq_len = num_timesteps * (views if per_view else 1)
        self.cond_seq_len = num_cond_tokens

        self.encoder_proj = nn.Linear(in_dim, hidden_dim)
        device = get_device()

        if is_causal == 'diag':
            attn_mask = diag_cond_mask(self.seq_len, num_cond_tokens, device=device)
        elif is_causal == True:
            attn_mask = causal_mask(self.seq_len, device=device)
        elif is_causal == False:
            attn_mask = None
        else:
            raise ValueError(f"Invalid value for is_causal: {is_causal}")

        if num_cond_tokens > 0:
            cond_embed_dim = cond_embed_dim * views if not per_view else cond_embed_dim
            self.cond_proj = nn.Linear(cond_embed_dim, hidden_dim)
            self.encoder = TransformerDecoder(
                q_seq_len=self.seq_len,
                kv_seq_len=self.cond_seq_len,
                hidden_dim=hidden_dim,
                n_layers=num_layers,
                n_heads=num_heads,
                dropout=attn_pdrop,
                attn_mask=attn_mask
            )
        else:
            self.encoder = TransformerEncoder(
                seq_len=self.seq_len,
                hidden_dim=hidden_dim,
                n_layers=num_layers,
                n_heads=num_heads,
                dropout=attn_pdrop,
                attn_mask=attn_mask
            )


    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    @property
    def device(self):
        return next(iter(self.parameters())).device


    def forward(self, x, cond=None):
        b, v, t, n, d = x.shape
        if self.per_view:
            x = rearrange(x, 'b v t n d -> b (v t) (n d)')
            if cond is not None:
                cond = rearrange(cond, 'b v t d -> b (v t) d')
        else:
            x = rearrange(x, 'b v t n d -> b t (v n d)')
            if cond is not None:
                cond = rearrange(cond, 'b v t d -> b t (v d)')

        x = self.encoder_proj(x)
        if self.cond_seq_len > 0:
            cond = self.cond_proj(cond)
            codes = self.encoder(x, cond)
        else:
            codes = self.encoder(x)
        return codes


class VAEDecoder(nn.Module):
    def __init__(self, hidden_dim, mlp_hidden_dim, num_heads, num_layers, attn_pdrop, views, num_timesteps, num_tracks, out_dim, per_view) -> None:
        super().__init__()
        self.per_view = per_view
        self.views = views
        self.num_timesteps = num_timesteps
        self.num_tracks = num_tracks
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.seq_len = num_timesteps * (views if per_view else 1)

        self.track_pos_emb = nn.Embedding(num_tracks, hidden_dim)
        self.view_emb = nn.Embedding(views, hidden_dim)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(3*hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        self.decoder_proj = nn.Linear(hidden_dim, out_dim)

        self.fixed_positional_emb = PositionalEncoding1D(hidden_dim)
        self.decoder = TransformerDecoder(
            q_seq_len=num_timesteps * (views if per_view else 1),
            kv_seq_len=num_timesteps * (views if per_view else 1),
            hidden_dim=hidden_dim,
            n_layers=num_layers,
            n_heads=num_heads,
            dropout=attn_pdrop,
            attn_mask=None
        )


    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    @property
    def device(self):
        return next(iter(self.parameters())).device


    def forward(self, codes):
        b, seq_len, h = codes.shape
        x_recon = torch.zeros_like(codes, dtype=codes.dtype, device=codes.device) # Positional embeddings are added to this anyway in the decoder
        x_recon = self.decoder(x_recon, codes)

        track_pos = self.track_pos_emb(torch.arange(self.num_tracks, device=codes.device))
        track_pos = repeat(track_pos, 'n h -> 1 1 1 n h')
        view_pos = self.view_emb(torch.arange(self.views, device=codes.device))
        view_pos = repeat(view_pos, 'v h -> 1 v 1 1 h')

        if self.per_view:
            x_recon = rearrange(x_recon, 'b (v t) h -> b v t 1 h', v=self.views)
            x_recon = repeat(x_recon, 'b v t 1 h -> b v t n h', v=self.views, t=self.num_timesteps, n=self.num_tracks)
        else:
            x_recon = repeat(x_recon, 'b t h -> b 1 t 1 h')
            x_recon = repeat(x_recon, 'b 1 t 1 h -> b v t n h', v=self.views, t=self.num_timesteps, n=self.num_tracks)

        track_pos = repeat(track_pos, '1 1 1 n h -> b v t n h', b=b, v=self.views, t=self.num_timesteps)
        view_pos = repeat(view_pos, '1 v 1 1 h -> b v t n h', b=b, v=self.views, t=self.num_timesteps, n=self.num_tracks)
        mlp_out = self.decoder_mlp(torch.cat([x_recon, track_pos, view_pos], dim=-1))
        x_recon = x_recon + mlp_out
        x_recon = self.decoder_proj(x_recon)

        return x_recon


class MotionTokenizer(nn.Module):
    def __init__(self, cfg, load_encoder=True, load_decoder=True):
        super().__init__()
        self.cfg = cfg
        self.views = len(cfg.cond_cameraviews)
        in_dim, out_dim = get_vae_in_out_dim(cfg)
        print(f"VAE in_dim: {in_dim}, out_dim: {out_dim}")
        num_timesteps = cfg.track_pred_horizon - 1 # - 1 because it's velocity

        self.quantize = FSQ(dim=cfg.hidden_dim, levels=get_fsq_level(cfg.codebook_size))

        if cfg.cond_on_img and load_encoder:
            self.vis_enc = VisionEncoder(**cfg.vision_encoder).eval()

        if cfg.type == 'transformer':
            if load_encoder:
                self.encoder = VAEEncoder(
                    in_dim=in_dim,
                    hidden_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    num_layers=cfg.num_layers,
                    attn_pdrop=cfg.attn_pdrop,
                    views=self.views,
                    num_timesteps=num_timesteps,
                    num_tracks=cfg.num_tracks,
                    point_dim=cfg.point_dim,
                    per_view=cfg.per_view,
                    is_causal=cfg.causal_encoder,
                    num_cond_tokens=self.vis_enc.seq_len*len(cfg.cond_cameraviews) if cfg.cond_on_img else 0,
                    cond_embed_dim=self.vis_enc.embed_dim if cfg.cond_on_img else 0,
                )
            if load_decoder:
                self.decoder = VAEDecoder(
                    hidden_dim=cfg.hidden_dim,
                    mlp_hidden_dim=cfg.decoder_mlp_hidden_dim,
                    num_heads=cfg.num_heads,
                    num_layers=int(cfg.num_layers/2),
                    attn_pdrop=cfg.attn_pdrop,
                    views=self.views,
                    num_timesteps=num_timesteps,
                    num_tracks=cfg.num_tracks,
                    out_dim=out_dim,
                    per_view=cfg.per_view,
                )
        else:
            raise ValueError(f"Unknown type: {type}")

        print("motion tokenizer number of parameters: %.2fM" % (self.num_params/1e6,))


    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    @property
    def device(self):
        return next(iter(self.parameters())).device


    def encode(self, x, cond=None):
        b, v, t, n, d = x.shape
        if self.cfg.cond_on_img:
            cond = rearrange(cond, 'b v h w c -> (b v) h w c')
            cond = self.vis_enc(cond)
            cond = rearrange(cond, '(b v) t d -> b v t d', v=self.views)
            z = self.encoder(x, cond)
        else:
            z = self.encoder(x)
        return z


    def decode(self, codes):
        b, t, h = codes.shape
        rel_logits = self.decoder(codes)
        rel_logits = rearrange(rel_logits, 'b v t n d -> b (v t n) d')
        x_recon = rel_cls_logits_to_diffs(
            logits=rel_logits,
            pred_views=self.views,
            num_tracks=self.cfg.num_tracks,
            rel_cls_img_size=self.cfg.loss.rel_cls_img_size,
            global_img_size=self.cfg.img_shape,
            get_last_timestep=False
        )

        return x_recon, rel_logits


    def get_loss(self, x_recon, rel_logits, gt_vel, gt_traj):
        assert self.cfg.loss.loss_fn == 'relative_ce', 'Unsupported loss function'
        unreduced_loss = compute_relative_classification_loss(rel_logits, gt_traj[:, :, 1:], gt_traj[:, :, :-1], self.cfg.loss)
        unreduced_loss = unreduced_loss.unsqueeze(-1)
        view_loss = torch.mean(unreduced_loss, dim=(0, 2, 3, 4)) # (v,)

        loss = 0.
        for i, view in enumerate(self.cfg.cond_cameraviews):
            loss += view_loss[i] * self.cfg.loss.loss_weights[view]

        return loss


    def forward(self, x, cond=None):
        z = self.encode(x, cond)
        codes, codebook_indices = self.quantize(z)
        x_recon, rel_logits = self.decode(codes)

        return x_recon, codebook_indices, rel_logits


def load_vae_encoder(checkpoint_path, frozen=False):
    """
    Loads only the encoder, quantizer, and optionally vis_enc state dict to save memory
    """
    checkpoint = torch.load(checkpoint_path)
    vae_cfg = OmegaConf.create(checkpoint['config'])

    default_vae_cfg = OmegaConf.load('cfg/train_motion_tokenizer.yaml')
    vae_cfg = merge_checkpoint_config(default_vae_cfg, ckpt_cfg=vae_cfg)

    print("================== TRACK ENCODER CONFIG ==================")
    print(OmegaConf.to_yaml(vae_cfg))

    if vae_cfg.compile:
        checkpoint['model'] = unwrap_compiled_state_dict(checkpoint['model'])

    vae_encoder = VAE(vae_cfg, load_encoder=True, load_decoder=False)

    # Load state dict without decoder params (strict=False)
    vae_encoder.load_state_dict(checkpoint['model'], strict=False)

    if frozen:
        for param in vae_encoder.parameters():
            param.requires_grad = False
        vae_encoder.eval()

    return vae_encoder, vae_cfg


def load_vae_decoder(checkpoint_path, frozen=False):
    """
    Loads only the decoder state dict to save memory
    """
    checkpoint = torch.load(checkpoint_path)
    vae_cfg = OmegaConf.create(checkpoint['config'])
    if vae_cfg.compile:
        checkpoint['model'] = unwrap_compiled_state_dict(checkpoint['model'])

    vae_decoder = VAE(vae_cfg, load_encoder=False, load_decoder=True)

    # Load state dict without encoder params (strict=False)
    vae_decoder.load_state_dict(checkpoint['model'], strict=False)

    if frozen:
        for param in vae_decoder.parameters():
            param.requires_grad = False
        vae_decoder.eval()

    return vae_decoder, vae_cfg


def load_motion_tokenizer(checkpoint_path, frozen=False):
    """
    Loads full Motion Tokenizer model
    """
    root_dir = get_root_dir()
    device = get_device()
    checkpoint = torch.load(os.path.join(root_dir, checkpoint_path), map_location=str(device), weights_only=False)

    motion_tokenizer_cfg = OmegaConf.create(checkpoint['config'])

    if motion_tokenizer_cfg.compile:
        checkpoint['model'] = unwrap_compiled_state_dict(checkpoint['model'])

    print("================== FINAL TRACK TOKENIZER CONFIG ==================")
    print(OmegaConf.to_yaml(motion_tokenizer_cfg))

    motion_tokenizer = MotionTokenizer(motion_tokenizer_cfg, load_encoder=True, load_decoder=True)
    motion_tokenizer.load_state_dict(checkpoint['model'])

    if frozen:
        for param in motion_tokenizer.parameters():
            param.requires_grad = False
        motion_tokenizer.eval()

    return motion_tokenizer, motion_tokenizer_cfg


# Test loading in main script
if __name__ == '__main__':
    print("\nTESTING FULL VAE")
    motion_tokenizer_cfg = OmegaConf.load('cfg/train_motion_tokenizer.yaml')
    motion_tokenizer = MotionTokenizer(motion_tokenizer_cfg, load_encoder=True, load_decoder=True)
    motion_tokenizer.to('cuda')

    # fake inputs
    fake_tracks = torch.randn(
        motion_tokenizer_cfg.batch_size,
        len(motion_tokenizer_cfg.cond_cameraviews),
        motion_tokenizer_cfg.track_pred_horizon - 1,
        motion_tokenizer_cfg.num_tracks,
        motion_tokenizer_cfg.point_dim
    ).to('cuda')
    # img between 0 and 1
    fake_img = torch.rand(
        motion_tokenizer_cfg.batch_size,
        len(motion_tokenizer_cfg.cond_cameraviews),
        motion_tokenizer_cfg.img_shape[0],
        motion_tokenizer_cfg.img_shape[1],
        3
    ).to('cuda')

    x_recon, _, _ = motion_tokenizer(fake_tracks, fake_img)
    print(f"x_recon shape {x_recon.shape}")
    del motion_tokenizer

    print("\nTESTING VAE ENCODER")
    vae_encoder, _ = load_vae_encoder(vae_checkpoint)
    vae_encoder.to('cuda')
    codes = vae_encoder.encode(fake_tracks, fake_img)
    codes, _ = vae_encoder.quantize(codes)
    print(f"codes shape {codes.shape}")
    del vae_encoder

    print("\nTESTING VAE DECODER")
    vae_decoder, _ = load_vae_decoder(vae_checkpoint)
    vae_decoder.to('cuda')
    x_recon, _ = vae_decoder.decode(codes)
    print(f"x_recon shape {x_recon.shape}")
    del vae_decoder
    print("Done")
