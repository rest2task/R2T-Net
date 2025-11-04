"""Temporal ViT encoder for region-wise fMRI matrices."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _make_transformer_encoder(
    embed_dim: int, depth: int, num_heads: int, dropout: float
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=embed_dim * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth)


@lru_cache(maxsize=32)
def _sinusoidal_position_embedding(length: int, dim: int, device: torch.device) -> Tensor:
    """Return the standard sine–cosine positional embedding."""

    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        * -(math.log(10000.0) / dim)
    )
    embeddings = torch.zeros(length, dim, device=device)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings


class TemporalViTEncoder(nn.Module):
    """Vision Transformer treating time-points as tokens."""

    def __init__(
        self,
        token_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        latent_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim

        self.input_proj = nn.LazyLinear(token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_dropout = nn.Dropout(dropout)
        self.encoder = _make_transformer_encoder(token_dim, depth, num_heads, dropout)
        self.norm = nn.LayerNorm(token_dim)
        self.to_latent = nn.Linear(token_dim, latent_dim)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, modality_embed: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"TemporalViT expects [B,V,T] inputs, got {tuple(x.shape)}")

        b, v, t = x.shape
        tokens = self.input_proj(x.transpose(1, 2))  # [B, T, token_dim]

        pos = _sinusoidal_position_embedding(t, self.token_dim, tokens.device)
        tokens = tokens + pos.unsqueeze(0)
        tokens = tokens + modality_embed.unsqueeze(1)
        tokens = self.pos_dropout(tokens)

        cls = self.cls_token.expand(b, -1, -1) + modality_embed.unsqueeze(1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.encoder(tokens)

        cls_final = self.norm(tokens[:, 0])
        latent = self.to_latent(cls_final)
        return F.normalize(latent, dim=-1)

