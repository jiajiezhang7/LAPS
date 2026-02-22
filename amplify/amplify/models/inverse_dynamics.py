import os

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from amplify.models.transformer import TransformerDecoder
from amplify.utils.train import get_root_dir, unwrap_compiled_state_dict


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

    def normalized_log_prob(self, value):
        # Compute log_prob of the original normal distribution
        log_p = super().log_prob(value)

        # Compute the normalization constant: P(low < X < high)
        # CDF of the normal distribution
        cdf_low = 0.5 * (
            1
            + torch.erf(
                (self.low - self.loc)
                / (self.scale * torch.sqrt(torch.tensor(2.0, device=self.loc.device)))
            )
        )
        cdf_high = 0.5 * (
            1
            + torch.erf(
                (self.high - self.loc)
                / (self.scale * torch.sqrt(torch.tensor(2.0, device=self.loc.device)))
            )
        )
        normalization = cdf_high - cdf_low + self.eps  # Add eps to prevent log(0)

        log_normalization = torch.log(normalization)

        # Adjust log_prob for truncation
        return log_p - log_normalization


class GaussianActionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_seq_len: int,
        action_horizon: int,
        action_dim: int,
        cfg,
    ) -> None:
        super().__init__()
        self.latent_seq_len = action_horizon
        self.action_dim = action_dim
        self.cfg = cfg

        std_value = getattr(cfg, "std", 0.1)
        if std_value == "learned":
            self.std = nn.Parameter(torch.tensor(1.0))
        else:
            self.std = std_value if isinstance(std_value, (float, int)) else 0.1

        self.decoder = TransformerDecoder(
            q_seq_len=self.latent_seq_len,
            kv_seq_len=cond_seq_len,
            hidden_dim=hidden_dim,
            n_layers=cfg.num_layers,
            n_heads=cfg.num_heads,
            dropout=cfg.attn_pdrop,
            bias=False,
        )
        self.latents = nn.Parameter(torch.randn(1, self.latent_seq_len, hidden_dim))
        self.action_proj = nn.Linear(hidden_dim, action_dim)
        self.action_squash = nn.Tanh() if cfg.action_squash else nn.Identity()

        squash_scale = getattr(cfg, "action_squash_scale", 1.0)
        if squash_scale == "learned":
            self.action_squash_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.action_squash_scale = squash_scale if isinstance(squash_scale, (float, int)) else 1.0

    def forward(self, cond_tokens: torch.Tensor) -> TruncatedNormal:
        batch = cond_tokens.shape[0]
        latents = repeat(self.latents, "1 n d -> b n d", b=batch)
        action_tokens = self.decoder(latents, cond_tokens)
        action_tokens = self.action_proj(action_tokens)
        mu = self.action_squash(action_tokens) * self.action_squash_scale
        scale = self.std if torch.is_tensor(self.std) else torch.tensor(self.std, device=mu.device, dtype=mu.dtype)
        return TruncatedNormal(mu, scale, low=-self.action_squash_scale, high=self.action_squash_scale)

    def act(self, cond_tokens: torch.Tensor, sample: bool = False) -> torch.Tensor:
        dist = self.forward(cond_tokens)
        return dist.sample() if sample else dist.mean

    def loss(self, dist: TruncatedNormal, target: torch.Tensor) -> torch.Tensor:
        return -dist.normalized_log_prob(target)


class DiffusionActionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_seq_len: int,
        action_horizon: int,
        action_dim: int,
        cfg,
    ) -> None:
        super().__init__()
        try:
            from amplify.models.diffusion_policy import DiffusionPolicy
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Diffusion policy head requires the optional dependency 'diffusers'. "
                "Install it via `pip install diffusers==0.30.0` "
                "or switch cfg.type to 'gaussian' or 'flow'."
            ) from exc
            
        self.policy = DiffusionPolicy(
            obs_dim=hidden_dim,
            act_dim=action_dim,
            obs_horizon=cond_seq_len,
            pred_horizon=action_horizon,
            hidden_dim=hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            attn_pdrop=cfg.attn_pdrop,
            num_diffusion_iters=getattr(cfg, "diffusion_steps", 100),
        )

    def forward(self, cond_tokens: torch.Tensor, action_seq: Optional[torch.Tensor]) -> dict:
        return self.policy(cond_tokens, action_seq)

    def act(self, cond_tokens: torch.Tensor, sample: bool = False) -> torch.Tensor:
        del sample
        return self.policy(cond_tokens, None)

    def loss(self, pred_dict: dict, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        del target
        return F.mse_loss(pred_dict["noise_pred"], pred_dict["noise"])


class FlowActionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_seq_len: int,
        action_horizon: int,
        action_dim: int,
        cfg,
    ) -> None:
        super().__init__()
        try:
            from amplify.models.flow_policy import FlowPolicy
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Flow matching head requires the optional dependency 'flow_matching'. "
                "Install it via `pip install git+https://github.com/facebookresearch/flow_matching.git@main` "
                "or switch cfg.type to 'gaussian' or 'diffusion'."
            ) from exc

        self.policy = FlowPolicy(
            obs_dim=hidden_dim,
            act_dim=action_dim,
            obs_horizon=cond_seq_len,
            pred_horizon=action_horizon,
            hidden_dim=hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            attn_pdrop=cfg.attn_pdrop,
            n_steps=getattr(cfg, "flow_matching_steps", 10),
        )

    def forward(self, cond_tokens: torch.Tensor, action_seq: Optional[torch.Tensor]) -> dict:
        return self.policy(cond_tokens, action_seq)

    def act(self, cond_tokens: torch.Tensor, sample: bool = False) -> torch.Tensor:
        del sample
        return self.policy(cond_tokens, None)

    def loss(self, pred_dict: dict, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        del target
        velocity = pred_dict["velocity"]
        dx_t = pred_dict["dx_t"]
        return F.mse_loss(velocity, dx_t)


class InverseDynamics(nn.Module):
    """Inverse dynamics head with pluggable action decoders.
    Conditioning info:
        - motion tokens
        - proprioception
        - image tokens
        - text tokens
    """

    def __init__(
        self,
        motion_tokenizer_cfg,
        cfg,
    ) -> None:
        super().__init__()
        self.motion_tokenizer_cfg = motion_tokenizer_cfg
        self.cfg = cfg

        head_type = getattr(cfg, "type", None)
        if head_type is None:
            raise ValueError("inverse dynamics cfg.type must be set to 'gaussian', 'diffusion', or 'flow'")
        self.head_type = str(head_type).lower()
        if self.head_type not in {"gaussian", "diffusion", "flow"}:
            raise ValueError(f"Unknown inverse dynamics head type: {self.head_type}")

        code_seq_len = motion_tokenizer_cfg.track_pred_horizon - 1
        if motion_tokenizer_cfg.per_view:
            code_seq_len *= len(motion_tokenizer_cfg.cond_cameraviews)

        cond_seq_len = 0
        if cfg.cond_on_img:
            cond_seq_len += cfg.num_img_tokens
        if cfg.cond_on_text:
            cond_seq_len += 1
        if cfg.cond_on_proprio:
            cond_seq_len += 1
        if cfg.cond_on_tracks:
            cond_seq_len += code_seq_len
        if cond_seq_len == 0:
            raise ValueError("No conditioning info provided for inverse dynamics")

        self.cond_seq_len = cond_seq_len
        self.action_pred_horizon = motion_tokenizer_cfg.true_horizon
        hidden_dim = motion_tokenizer_cfg.hidden_dim

        if cfg.cond_on_img:
            self.img_proj = nn.Linear(cfg.img_embed_dim, hidden_dim)
        else:
            self.img_proj = None
        if cfg.cond_on_text:
            self.text_proj = nn.Linear(cfg.text_embed_dim, hidden_dim)
        else:
            self.text_proj = None
        if cfg.cond_on_proprio:
            self.proprio_proj = nn.Linear(cfg.proprio_dim, hidden_dim)
        else:
            self.proprio_proj = None

        if self.head_type == "gaussian":
            self.head = GaussianActionHead(
                hidden_dim=hidden_dim,
                cond_seq_len=self.cond_seq_len,
                action_horizon=self.action_pred_horizon,
                action_dim=cfg.action_dim,
                cfg=cfg,
            )
        elif self.head_type == "diffusion":
            self.head = DiffusionActionHead(
                hidden_dim=hidden_dim,
                cond_seq_len=self.cond_seq_len,
                action_horizon=self.action_pred_horizon,
                action_dim=cfg.action_dim,
                cfg=cfg,
            )
        else:
            self.head = FlowActionHead(
                hidden_dim=hidden_dim,
                cond_seq_len=self.cond_seq_len,
                action_horizon=self.action_pred_horizon,
                action_dim=cfg.action_dim,
                cfg=cfg,
            )

        self.requires_action_seq = self.head_type in {"diffusion", "flow"}

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def _build_cond_tokens(self, input_dict) -> torch.Tensor:
        cond_seq = []
        if self.cfg.cond_on_img:
            if "img_tokens" not in input_dict:
                raise KeyError("img_tokens missing from input_dict")
            cond_seq.append(self.img_proj(input_dict["img_tokens"]))
        if self.cfg.cond_on_text:
            if "text_tokens" not in input_dict:
                raise KeyError("text_tokens missing from input_dict")
            text_tokens = input_dict["text_tokens"]
            if text_tokens.dim() == 2:
                text_tokens = text_tokens.unsqueeze(1)
            cond_seq.append(self.text_proj(text_tokens))
        if self.cfg.cond_on_proprio:
            if "proprioception" not in input_dict:
                raise KeyError("proprioception missing from input_dict")
            proprio = input_dict["proprioception"]
            if proprio.dim() == 2:
                proprio = proprio.unsqueeze(1)
            cond_seq.append(self.proprio_proj(proprio))
        if self.cfg.cond_on_tracks:
            if "codes" not in input_dict:
                raise KeyError("codes missing from input_dict")
            cond_seq.append(input_dict["codes"])

        if not cond_seq:
            raise RuntimeError("Conditioning sequence is empty")
        return torch.cat(cond_seq, dim=1)

    def forward(self, input_dict, action_seq: Optional[torch.Tensor] = None):
        cond_tokens = self._build_cond_tokens(input_dict)
        if self.requires_action_seq and action_seq is None:
            raise ValueError(f"action_seq required for {self.head_type} head")
        if self.head_type == "gaussian":
            return self.head(cond_tokens)
        return self.head(cond_tokens, action_seq)

    def act(self, input_dict, sample: bool = False):
        cond_tokens = self._build_cond_tokens(input_dict)
        return self.head.act(cond_tokens, sample=sample)

    def loss_fn(self, pred, target):
        if self.head_type == "gaussian":
            loss = self.head.loss(pred, target)
            discount = getattr(self.cfg, "action_loss_discount", 1.0)
            if discount < 1.0:
                discounts = discount ** torch.arange(self.action_pred_horizon, device=loss.device)
                loss = loss * discounts.view(1, -1, 1)
            return loss.mean()
        return self.head.loss(pred, target)


def load_inverse_dynamics(checkpoint_path, motion_tokenizer_cfg, frozen=False):
    root_dir = get_root_dir()
    checkpoint_path = os.path.join(root_dir, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    if cfg.compile:
        checkpoint["model"] = unwrap_compiled_state_dict(checkpoint["model"])

    inverse_dynamics = InverseDynamics(motion_tokenizer_cfg, cfg)
    inverse_dynamics.load_state_dict(checkpoint["model"], strict=True)

    if frozen:
        for param in inverse_dynamics.parameters():
            param.requires_grad = False
        inverse_dynamics.eval()

    return inverse_dynamics, cfg
