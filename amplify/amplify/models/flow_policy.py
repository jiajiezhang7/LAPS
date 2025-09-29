from typing import Optional

import torch
import torch.nn as nn
from einops import repeat
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from amplify.models.dit import DiT


class FlowPolicy(nn.Module):
    """Flow matching policy that predicts actions via a DiT backbone."""

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
        n_steps: int = 10,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = act_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.data_act_scale = data_act_scale
        self.data_obs_scale = data_obs_scale
        self.n_steps = n_steps

        self.net = DiT(
            num_attention_heads=num_heads,
            attention_head_dim=hidden_dim // num_heads,
            output_dim=act_dim,
            num_layers=num_layers,
            dropout=attn_pdrop,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            upcast_attention=False,
            norm_type="ada_norm",
            norm_elementwise_affine=False,
            norm_eps=1e-5,
            max_num_positional_embeddings=512,
            compute_dtype=torch.float32,
            final_dropout=True,
            positional_embeddings="sinusoidal",
            interleave_self_attention=False,
        )

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
                return self.model(
                    hidden_states=x,
                    encoder_hidden_states=extras["obs_cond"],
                    timestep=t.unsqueeze(0),
                )

        self.solver = ODESolver(velocity_model=WrappedModel(self.net))

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
    ) -> torch.Tensor | dict:
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
        obs_cond = nobs[:, : self.obs_horizon, :]

        x1 = naction
        x0 = torch.randn_like(x1, device=device)
        t = torch.rand(x1.shape[0], device=device)
        path_sample = self.path.sample(t=t, x_0=x0, x_1=x1)

        velocity = self.net(
            hidden_states=path_sample.x_t,
            encoder_hidden_states=obs_cond,
            timestep=path_sample.t,
        )

        return {"velocity": velocity, "dx_t": path_sample.dx_t}

    @torch.no_grad()
    def _predict(self, obs_seq: torch.Tensor) -> torch.Tensor:
        obs_seq = self._prepare_obs(obs_seq)
        device = obs_seq.device
        batch = obs_seq.shape[0]

        nobs = self.normalize_obs_data(obs_seq)
        obs_cond = nobs[:, : self.obs_horizon, :]

        x_init = torch.randn((batch, self.pred_horizon, self.action_dim), device=device)
        naction = self.solver.sample(
            x_init=x_init,
            step_size=1.0 / self.n_steps,
            method="midpoint",
            obs_cond=obs_cond,
        )

        return self.unnormalize_act_data(naction)


__all__ = [
    "FlowPolicy",
]
