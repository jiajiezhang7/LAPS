import sys

import torch
import torch.nn as nn
from einops import repeat
from IPython.core import ultratb

from amplify.models.transformer import TransformerEncoder
from amplify.utils.data_utils import top_k_top_p_filtering
from amplify.utils.model.attn_masks import causal_cond_mask

sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Linux", call_pdb=1)

class ForwardDynamics(nn.Module):
    def __init__(self,
                 trunk_cfg,
                 hidden_dim,
                 img_dim,
                 text_dim,
                 cond_seq_len,
                 pred_seq_len,
                 codebook_size,
                 quantize
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_seq_len = cond_seq_len
        self.pred_seq_len = pred_seq_len
        self.quantize = quantize

        attn_mask = causal_cond_mask(
            self.cond_seq_len + self.pred_seq_len,
            self.cond_seq_len,
            device='cpu'
        )
        self.trunk = TransformerEncoder(
            seq_len=self.cond_seq_len + self.pred_seq_len,
            hidden_dim=hidden_dim,
            n_layers=trunk_cfg.n_layer,
            n_heads=trunk_cfg.n_head,
            dropout=trunk_cfg.dropout,
            bias=trunk_cfg.bias,
            attn_mask=attn_mask,
        )

        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.sos_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.unembed = nn.Linear(hidden_dim, codebook_size)

        print("ForwardDynamics number of parameters: %.2fM" % (self.num_params / 1e6,))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self):
        return next(iter(self.parameters())).device


    def forward(self, obs, goal, targets=None):
        """
        If targets are provided, the model updates using teacher forcing.
        If targets are not provided, the model autoregressively generates code indices.

        Args:
            obs: dictionary of observations (encoded)
            goal: dictionary of goals (encoded)
            targets: ground truth

        Returns:
            pred: predicted code indices
            loss: loss if targets are provided
        """
        cond_tokens = self.get_cond_tokens(obs, goal)

        if targets is not None:
            pred, loss =  self.update(cond_tokens, targets)
        else:
            pred = self.predict(cond_tokens)
            loss = None

        return pred, loss


    def get_cond_tokens(self, obs, goal):
        """
        Project inputs for the model

        Args:
            obs: dictionary of observations (encoded)
            goal: dictionary of goals (encoded)

        Returns:
            cond_tokens: concatenated sequence of conditioning tokens
        """
        # Tokenize observations
        img = obs["image"] # image tokens from vision encoder
        img_tokens = self.img_proj(img)

        # Tokenize goals
        text_emb = goal["text_emb"] # text tokens from text encoder
        text_tokens = self.text_proj(text_emb)

        cond_tokens = torch.cat([img_tokens, text_tokens], dim=1)

        return cond_tokens


    def update(self, cond_tokens, targets):
        """
        Update the model using teacher forcing

        Args:
            cond_tokens: concatenated sequence of conditioning tokens
            targets: ground truth indices
        """
        b, tc, _ = cond_tokens.shape
        sos_token = repeat(self.sos_token, "1 1 d -> b 1 d", b=b)
        target_codes = self.quantize.indices_to_codes(targets)
        input_tokens = torch.cat([cond_tokens, sos_token, target_codes[:, :-1]], dim=1)

        pred_tokens = self.trunk(input_tokens)
        pred_logits = self.unembed(pred_tokens[:, tc:])
        pred_indices = torch.argmax(pred_logits, dim=-1)

        loss = torch.nn.functional.cross_entropy(pred_logits.view(-1, pred_logits.size(-1)), targets.view(-1))

        return pred_indices, loss


    @torch.no_grad()
    def predict(self, cond_tokens, ar_sampling='argmax'):
        """
        Generates predicted indices autoregressively

        Args:
            cond_tokens: concatenated sequence of conditioning tokens

        Returns:
            pred_indices: predicted code indices
        """
        b = cond_tokens.shape[0]
        sos_token = repeat(self.sos_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cond_tokens, sos_token], dim=1)  # Shape: [B, T, D]
        pred_indices = []
        for _ in range(self.pred_seq_len):
            pred_tokens = self.trunk(x)  # Shape: [B, T, D]
            last_token = pred_tokens[:, -1, :]  # Shape: [B, D] (squeeze T=1)
            last_logits = self.unembed(last_token)  # Shape: [B, H]

            # Sample from logits
            if ar_sampling == 'argmax':
                last_idx = torch.argmax(last_logits, dim=-1, keepdim=True)  # Shape: [B, 1]
            elif ar_sampling == 'topk':
                last_topk = top_k_top_p_filtering(last_logits, top_k=10)  # Shape: [B, H]
                last_idx = torch.multinomial(torch.softmax(last_topk, dim=-1), num_samples=1)  # Shape: [B, 1]

            pred_indices.append(last_idx)
            last_code = self.quantize.indices_to_codes(last_idx)  # Shape: [B, D]
            x = torch.cat([x, last_code], dim=1)  # Append along time dimension

        pred_indices = torch.cat(pred_indices, dim=1)  # Shape: [B, pred_seq_len]

        return pred_indices
