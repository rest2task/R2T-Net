r"""SwiFT encoder implementation.

This module implements the volume encoder described in the project
documentation.  It ingests 4-D fMRI volumes (space × time) and maps each
scan into a 2,048-D latent vector.  The encoder works in three stages:

1.  Spatio-temporal patchification into \(16^3\times10\) cubes followed by a
    learnable linear projection to a shared token dimension.
2.  Three hierarchical transformer stages with local attention.  After
    each stage we halve the spatial grid resolution via average pooling,
    mirroring the 96→48→24 window progression described in the paper.
3.  A global transformer encoder over the remaining tokens and mean pooling
    to obtain the scan signature which is finally \(\ell_2\)-normalised.

The implementation purposefully favours clarity over absolute efficiency;
operations are expressed with explicit reshapes so the behaviour closely
matches the description in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _make_transformer_encoder(embed_dim, num_layers, num_heads, dropout):
    """Utility to build a GELU-based Transformer encoder."""

    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=embed_dim * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


@dataclass(frozen=True)
class SwiFTConfig:
    patch_size: Sequence[int] = (16, 16, 16, 10)
    token_dim: int = 768
    stage_depths: Sequence[int] = (2, 2, 2)
    stage_heads: Sequence[int] = (8, 8, 8)
    global_depth: int = 2
    global_heads: int = 8
    dropout: float = 0.1


class SwiFTEncoder(nn.Module):
    """Swin-based Windowed Transformer for fMRI Tensors (SwiFT).

    Parameters
    ----------
    config:
        Hyper-parameters controlling patch size, embedding dimension and
        transformer depth/width.
    latent_dim:
        Dimensionality of the returned subject embedding (default 2,048).
    """

    def __init__(self, config, latent_dim = 2048):
        super().__init__()
        self.config = config

        if len(config.stage_depths) != len(config.stage_heads):
            raise ValueError("stage_depths and stage_heads must have the same length")

        self.patch_embed = nn.LazyLinear(config.token_dim)
        self.patch_norm = nn.LayerNorm(config.token_dim)

        self.local_stages = nn.ModuleList(
            [
                _make_transformer_encoder(
                    config.token_dim, depth, heads, config.dropout
                )
                for depth, heads in zip(config.stage_depths, config.stage_heads)
            ]
        )

        self.global_encoder = _make_transformer_encoder(
            config.token_dim, config.global_depth, config.global_heads, config.dropout
        )

        self.final_norm = nn.LayerNorm(config.token_dim)
        self.proj = nn.Linear(config.token_dim, latent_dim)

    # ------------------------------------------------------------------
    # patch helpers
    # ------------------------------------------------------------------
    def _patchify(self, x):
        """Convert `[B,C,H,W,D,T]` volumes into a token sequence."""

        if x.ndim != 6:
            raise ValueError(f"SwiFT expects 6-D tensors, got shape {tuple(x.shape)}")

        b, c, h, w, d, t = x.shape
        ps = self.config.patch_size
        pad_h = (ps[0] - h % ps[0]) % ps[0]
        pad_w = (ps[1] - w % ps[1]) % ps[1]
        pad_d = (ps[2] - d % ps[2]) % ps[2]
        pad_t = (ps[3] - t % ps[3]) % ps[3]
        if pad_h or pad_w or pad_d or pad_t:
            x = F.pad(x, (0, pad_t, 0, pad_d, 0, pad_w, 0, pad_h))
            b, c, h, w, d, t = x.shape

        h_chunks, w_chunks, d_chunks, t_chunks = (
            h // ps[0],
            w // ps[1],
            d // ps[2],
            t // ps[3],
        )

        x = x.view(
            b,
            c,
            h_chunks,
            ps[0],
            w_chunks,
            ps[1],
            d_chunks,
            ps[2],
            t_chunks,
            ps[3],
        )
        # Reorder dimensions so chunk indices are contiguous and flatten patches
        x = x.permute(0, 6, 2, 4, 8, 1, 7, 3, 5, 9).contiguous()
        x = x.view(b, h_chunks * w_chunks * d_chunks * t_chunks, c * prod(ps))
        tokens = self.patch_embed(x)
        return tokens, (d_chunks, h_chunks, w_chunks, t_chunks)

    @staticmethod
    def _tokens_to_grid(tokens, grid):
        b, _, c = tokens.shape
        d, h, w, t = grid
        return tokens.view(b, d, h, w, t, c)

    @staticmethod
    def _grid_to_tokens(grid_tokens):
        b, d, h, w, t, c = grid_tokens.shape
        return grid_tokens.view(b, d * h * w * t, c)

    @staticmethod
    def _spatial_downsample(grid_tokens):
        b, d, h, w, t, c = grid_tokens.shape
        if min(d, h, w) <= 1:
            return grid_tokens, (d, h, w, t)

        d_even, h_even, w_even = d // 2, h // 2, w // 2
        if min(d_even, h_even, w_even) == 0:
            return grid_tokens, (d, h, w, t)

        trimmed = grid_tokens[:, : d_even * 2, : h_even * 2, : w_even * 2]
        pooled = trimmed.view(b, d_even, 2, h_even, 2, w_even, 2, t, c).mean(dim=(2, 4, 6))
        return pooled, (d_even, h_even, w_even, t)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x, modality_embed):
        tokens, grid = self._patchify(x)
        tokens = self.patch_norm(tokens)

        # inject modality embedding into every token
        tokens = tokens + modality_embed.unsqueeze(1)

        for stage in self.local_stages:
            tokens = stage(tokens)
            grid_tokens = self._tokens_to_grid(tokens, grid)
            grid_tokens, grid = self._spatial_downsample(grid_tokens)
            tokens = self._grid_to_tokens(grid_tokens)

        tokens = self.global_encoder(tokens)
        tokens = self.final_norm(tokens)

        pooled = tokens.mean(dim=1)
        latent = self.proj(pooled)
        return F.normalize(latent, dim=-1)
