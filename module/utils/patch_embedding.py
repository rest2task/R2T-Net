# module/utils/patch_embedding.py
"""
Patch- and ROI-positional embeddings for R2T-Net
------------------------------------------------
* PatchEmbedding3D – splits 4-D+time fMRI volumes into 3-D+T patches,
                     linearly projects each patch, then adds a learnable
                     positional code.
* PositionalEmbedding1D – adds a learnable code along an arbitrary token axis
                          (ROIs, vertices, or time points).
"""
from __future__ import annotations
import math
import torch
from torch import nn
from einops import rearrange


# ------------------------------------------------------------
# 3-D + time patch embedding
# ------------------------------------------------------------
class PatchEmbedding3D(nn.Module):
    """
    Parameters
    ----------
    img_size   : list/tuple of five ints  [H, W, D, C, T]
    patch_size : list/tuple of five ints  [pH, pW, pD, pC, pT]
    embed_dim  : output feature dimension per patch
    """

    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size

        # number of patches per spatial/temporal axis
        self.num_tokens = math.prod(i // p for i, p in zip(img_size, patch_size))

        # flatten patch → embed_dim
        flat_dim = patch_size[0] * patch_size[1] * patch_size[2] * patch_size[4] * img_size[3]
        self.proj = nn.Linear(flat_dim, embed_dim, bias=True)

        # learnable positional code
        self.pos_emb = nn.Parameter(torch.empty(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
        -----
        x : Tensor [B, C, H, W, D, T]

        Output
        ------
        tokens : Tensor [B, N, embed_dim]   (N = num patch tokens)
        """
        pH, pW, pD, _, pT = self.patch_size
        x = rearrange(
            x,
            "b c (h pH) (w pW) (d pD) (t pT) -> b (h w d t) (pH pW pD pT c)",
            pH=pH, pW=pW, pD=pD, pT=pT,
        )
        return self.proj(x) + self.pos_emb


# alias kept for backward-compatibility
PositionalEmbedding3D = PatchEmbedding3D


# ------------------------------------------------------------
# Generic 1-D positional embedding
# ------------------------------------------------------------
class PositionalEmbedding1D(nn.Module):
    """
    Adds a learned positional vector to any 1-D token axis.

    Expected input shapes
    ---------------------
    * ROI path (our default) : [V, B, E] before transformer
    * Time-first variant     : [T, B, E] – just set `axis_len=T`
    """

    def __init__(self, axis_len: int, embed_dim: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.empty(axis_len, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor [seq_len, batch, embed_dim]
        """
        return x + self.pos_emb.unsqueeze(1)
