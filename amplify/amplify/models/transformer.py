import math

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, Summer
from torch.nn import functional as F

from amplify.utils.cfg_utils import get_device
from amplify.utils.model.attn_masks import full_mask
from amplify.utils.vis_utils import vis_attn_map, vis_attn_mask


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, hidden_dim, dropout, bias):
        super().__init__()
        self.c_fc    = nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self,
            seq_len,
            hidden_dim,
            n_heads,
            dropout,
            bias,
            attn_mask=None
        ):
        super().__init__()
        assert hidden_dim % n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.seq_len = seq_len
        self.device = get_device()

        if attn_mask is None:
            # Default to full attention
            attn_mask = full_mask(seq_len, device=self.device)
        self.register_buffer("attn_mask", attn_mask.view(1, 1, self.seq_len, self.seq_len))

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.hidden_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            attn_mask = self.attn_mask[0, 0, :T, :T]
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
            # vis_attn_map(q, k, attn_mask)
            # vis_attn_mask(attn_mask)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.attn_mask[:,:,:T,:T] == 0, float('-inf')) # fill the attention weights with -inf where the mask is 0 so that softmax will make them 0
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CrossAttention(nn.Module):
    def __init__(self,
                q_seq_len,
                kv_seq_len,
                hidden_dim,
                n_heads,
                dropout,
                bias,
                attn_mask=None):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.device = get_device()
        if attn_mask is None:
            # Default to full attention
            attn_mask = torch.ones(q_seq_len, kv_seq_len).to(self.device).bool()
        self.register_buffer("attn_mask", attn_mask.view(1, 1, q_seq_len, kv_seq_len))

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, cond):

        q = self.q_proj(x)
        k, v = self.kv_proj(cond).split(self.hidden_dim, dim=2)

        B, T_q, C = q.size() # bs, seq_len, hidden_dim
        B_kv, T_kv, C_kv = k.size() # bs, seq_len, hidden_dim
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T_kv, self.n_head, C_kv // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T_kv, self.n_head, C_kv // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            attn_mask = self.attn_mask[0, 0, :T_q, :T_kv]
            # visualize attention mask
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.attn_mask[:,:,:T_q,:T_kv] == 0, float('-inf')) # fill the attention weights with -inf where the mask is 0 so that softmax will make them 0
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class EncoderBlock(nn.Module):
    def __init__(self,
            seq_len,
            hidden_dim,
            n_heads,
            dropout,
            bias,
            attn_mask=None
        ):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim, bias=bias)
        self.attn = SelfAttention(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            attn_mask=attn_mask,
        )
        self.ln_2 = LayerNorm(hidden_dim, bias=bias)
        self.mlp = MLP(
            hidden_dim=hidden_dim,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self,
            q_seq_len,
            kv_seq_len,
            hidden_dim,
            n_heads,
            dropout,
            bias,
            attn_mask
        ):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim, bias=bias)
        self.self_attn = SelfAttention(
            seq_len=q_seq_len,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            attn_mask=attn_mask,
        )
        self.ln_2 = LayerNorm(hidden_dim, bias=bias)
        self.mlp = MLP(
            hidden_dim=hidden_dim,
            dropout=dropout,
            bias=bias,
        )
        self.ln_3 = LayerNorm(hidden_dim, bias=bias)
        self.cross_attn = CrossAttention(
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            attn_mask=None, # no mask for cross attention
        )
        self.ln_4 = LayerNorm(hidden_dim, bias=bias)
        self.mlp_2 = MLP(
            hidden_dim=hidden_dim,
            dropout=dropout,
            bias=bias,
        )


    def forward(self, x, cond):
        x = x + self.self_attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x = x + self.cross_attn(self.ln_3(x), cond)
        x = x + self.mlp_2(self.ln_4(x))

        return x

class TransformerEncoder(nn.Module):
    def __init__(self,
            seq_len,
            hidden_dim,
            n_layers,
            n_heads,
            dropout=0.1,
            bias=False,
            attn_mask=None,
        ):
        super().__init__()
        self.seq_len = seq_len

        self.transformer = nn.ModuleDict(dict(
            pos_emb = nn.Embedding(self.seq_len, hidden_dim),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([EncoderBlock(
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                bias=bias,
                attn_mask=attn_mask,
            ) for _ in range(n_layers)]),
            ln_f = LayerNorm(hidden_dim, bias=bias),
        ))

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))

        # report number of parameters
        print("TransformerEncoder number of parameters: %.2fM" % (self.num_params/1e6,))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x):
        b, t, d = x.size()
        assert t <= self.seq_len, f"Cannot forward sequence of length {t}, block size is only {self.seq_len}"

        pos = torch.arange(0, t, dtype=torch.long, device=x.device) # shape (t)
        pos_emb = self.transformer.pos_emb(pos)
        x = self.transformer.drop(x + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, 
            q_seq_len,
            kv_seq_len,
            hidden_dim,
            n_layers,
            n_heads,
            dropout=0.1,
            bias=False,
            attn_mask=None, 
        ):
        super().__init__()
        self.seq_len = q_seq_len

        self.transformer = nn.ModuleDict(dict(
            pos_emb = nn.Embedding(self.seq_len, hidden_dim),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([DecoderBlock(
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout,
                bias=bias,
                attn_mask=attn_mask,
            ) for _ in range(n_layers)]),
            ln_f = LayerNorm(hidden_dim, bias=bias),
        ))

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))

        # report number of parameters
        print("TransformerDecoder number of parameters: %.2fM" % (self.num_params/1e6,))


    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x, cond):
        b, t, d = x.size()
        assert t <= self.seq_len, f"Cannot forward sequence of length {t}, block size is only {self.seq_len}"

        pos = torch.arange(0, t, dtype=torch.long, device=x.device) # shape (t)
        pos_emb = self.transformer.pos_emb(pos)
        x = self.transformer.drop(x + pos_emb)
        for block in self.transformer.h:
            x = block(x, cond)
        x = self.transformer.ln_f(x)

        return x


class PyTransformerEncoder(nn.Module):
    """
    Same as Transformer class except uses built-in pytorch components.
    """
    def __init__(self,
            seq_len,
            hidden_dim,
            n_layers,
            n_heads,
            dropout=0.1,
            bias=False,
            attn_mask=None,
        ):
        super().__init__()
        self.seq_len = seq_len
        if attn_mask is not None:
            self.register_buffer("attn_mask", ~attn_mask) # Pytorch uses opposite convention for attention masks
        else:
            self.attn_mask = None
        self.positional_emb = Summer(PositionalEncoding1D(hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                    nhead=n_heads,
                                                    dim_feedforward=4*hidden_dim,
                                                    dropout=dropout,
                                                    activation='gelu',
                                                    batch_first=True,
                                                    norm_first=True,
                                                    bias=bias)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        print("PyTransformerEncoder number of parameters: %.2fM" % (self.num_params/1e6,))


    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def forward(self, x):
        b, t, d = x.size()
        assert t <= self.seq_len, f"Cannot forward sequence of length {t}, block size is only {self.seq_len}"
        x = self.positional_emb(x)
        if self.attn_mask is not None:
            x = self.encoder(x, mask=self.attn_mask[:t, :t])
        else:
            x = self.encoder(x)
        return x


class PyTransformerDecoder(nn.Module):
    """
    Conditioning inputs are attended to with cross attention rather than concatenation and self-attention.
    Uses pytorch decoder blocks with cross attention.
    """
    def __init__(self,
            seq_len,
            hidden_dim,
            n_layers,
            n_heads,
            dropout=0.1,
            bias=False,
            attn_mask=None,
        ):
        super().__init__()
        self.seq_len = seq_len
        if attn_mask is not None:
            self.register_buffer("attn_mask", ~attn_mask) # Pytorch uses opposite convention for attention masks
        else:
            self.attn_mask = None
        self.positional_emb = Summer(PositionalEncoding1D(hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim,
                                                    nhead=n_heads,
                                                    dim_feedforward=4*hidden_dim,
                                                    dropout=dropout,
                                                    activation='gelu',
                                                    batch_first=True,
                                                    norm_first=True,
                                                    bias=bias)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        print("PyTransformerDecoder number of parameters: %.2fM" % (self.num_params/1e6,))


    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def forward(self, x, cond):
        b, t, d = x.size()
        assert t <= self.seq_len, f"Cannot forward sequence of length {t}, block size is only {self.seq_len}"
        x = self.positional_emb(x)
        if self.attn_mask is not None:
            x = self.decoder(x, cond, tgt_mask=self.attn_mask[:t, :t])
        else:
            x = self.decoder(x, cond)
        return x
