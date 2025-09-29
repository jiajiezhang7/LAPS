import math
from typing import Optional, Union

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import repeat


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal positional embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        freq = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)))
        emb = x[:, None] * freq[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ModuleAttrMixin(nn.Module):
    """Mixin that provides convenient accessors for device/dtype."""

    def __init__(self) -> None:
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self) -> torch.device:
        return self._dummy_variable.device

    @property
    def dtype(self) -> torch.dtype:
        return self._dummy_variable.dtype


class TransformerForDiffusion(ModuleAttrMixin):
    """Transformer backbone used for diffusion-based action prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: Optional[int] = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon

        token_count = horizon
        cond_token_count = 1
        if not time_as_cond:
            token_count += 1
            cond_token_count -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            cond_token_count += n_obs_steps

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, token_count, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if cond_token_count > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, cond_token_count, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer, num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb),
                )

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=n_layer
            )
        else:
            encoder_only = True
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_layer
            )

        if causal_attn:
            mask = (torch.triu(torch.ones(token_count, token_count)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, 0.0)
            )
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                s_range = torch.arange(cond_token_count)
                t_range = torch.arange(token_count)
                t, s = torch.meshgrid(t_range, s_range, indexing="ij")
                memory_mask = (t >= (s - 1)).float()
                memory_mask = (
                    memory_mask.masked_fill(memory_mask == 0, float("-inf"))
                    .masked_fill(memory_mask == 1, 0.0)
                )
                self.register_buffer("memory_mask", memory_mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        self.token_count = token_count
        self.cond_token_count = cond_token_count
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        ignore = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            for name in ["in_proj_bias", "bias_k", "bias_v"]:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore):
            return
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        else:
            timesteps = timestep.view(-1).to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        input_emb = self.input_emb(sample)

        if self.encoder_only:
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            pos = self.pos_emb[:, :t, :]
            x = self.drop(token_embeddings + pos)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]
        else:
            cond_embeddings = time_emb
            if self.obs_as_cond and cond is not None:
                cond_obs_emb = self.cond_obs_emb(cond)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            cond_pos = self.cond_pos_emb[:, :tc, :]
            cond_tokens = self.drop(cond_embeddings + cond_pos)
            cond_tokens = self.encoder(cond_tokens)

            t = input_emb.shape[1]
            pos = self.pos_emb[:, :t, :]
            tgt = self.drop(input_emb + pos)
            x = self.decoder(
                tgt=tgt,
                memory=cond_tokens,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask,
            )

        x = self.ln_f(x)
        return self.head(x)


class DiffusionPolicy(nn.Module):
    """Transformer-based diffusion policy that predicts action sequences."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        attn_pdrop: float = 0.1,
        data_act_scale: float = 1.0,
        data_obs_scale: float = 1.0,
        num_diffusion_iters: int = 100,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = act_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.data_act_scale = data_act_scale
        self.data_obs_scale = data_obs_scale
        self.num_diffusion_iters = num_diffusion_iters

        self.noise_pred_net = TransformerForDiffusion(
            input_dim=self.action_dim,
            output_dim=self.action_dim,
            horizon=self.pred_horizon,
            n_obs_steps=self.obs_horizon,
            cond_dim=self.obs_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_emb=hidden_dim,
            p_drop_emb=0.0,
            p_drop_attn=attn_pdrop,
            causal_attn=False,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=0,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
    def _prepare_obs(self, obs_seq: torch.Tensor) -> torch.Tensor:
        if obs_seq.shape[1] >= self.obs_horizon:
            return obs_seq
        pad = self.obs_horizon - obs_seq.shape[1]
        first = obs_seq[:, :1]
        pad_tokens = repeat(first, "b 1 d -> b t d", t=pad)
        return torch.cat([pad_tokens, obs_seq], dim=1)

    def normalize_obs_data(self, data: torch.Tensor) -> torch.Tensor:
        return data / self.data_obs_scale

    def unnormalize_obs_data(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.data_obs_scale

    def normalize_act_data(self, data: torch.Tensor) -> torch.Tensor:
        return data / self.data_act_scale

    def unnormalize_act_data(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.data_act_scale

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, dict]:
        if action_seq is None:
            return self._predict(obs_seq)
        return self._update(obs_seq, action_seq)

    def _update(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> dict:
        obs_seq = self._prepare_obs(obs_seq)
        device = obs_seq.device

        naction = self.normalize_act_data(action_seq).to(device)
        nobs = self.normalize_obs_data(obs_seq).to(device)
        batch = nobs.shape[0]

        obs_cond = nobs[:, : self.obs_horizon, :]

        noise = torch.randn_like(naction, device=device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch,), device=device
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

        noise_pred = self.noise_pred_net(
            noisy_actions,
            timesteps,
            cond=obs_cond,
        )
        return {"noise_pred": noise_pred, "noise": noise}

    @torch.no_grad()
    def _predict(self, obs_seq: torch.Tensor) -> torch.Tensor:
        obs_seq = self._prepare_obs(obs_seq)
        device = obs_seq.device
        batch = obs_seq.shape[0]
        nobs = self.normalize_obs_data(obs_seq)

        naction = torch.randn(
            (batch, self.pred_horizon, self.action_dim), device=device
        )
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            obs_cond = nobs[:, : self.obs_horizon, :]
            noise_pred = self.noise_pred_net(
                sample=naction,
                timestep=k,
                cond=obs_cond,
            )
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample

        return self.unnormalize_act_data(naction)

__all__ = [
    "DiffusionPolicy",
]
