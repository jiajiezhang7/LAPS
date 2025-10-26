import torch
import torch.nn as nn


class KeypointEncoder(nn.Module):
    """
    Causally-Masked Transformer Encoder
    """

    def __init__(self, num_layers: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D), mask: (S, S)
        return self.transformer_encoder(x, mask=mask)


class KeypointDecoder(nn.Module):
    """
    Unmasked Transformer Decoder with Cross-Attention
    """

    def __init__(self, num_layers: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # tgt: (B, N, D), memory: (B, S, D)
        return self.transformer_decoder(tgt, memory)
