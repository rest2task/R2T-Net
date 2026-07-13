"""Temporal encoders for region-wise fMRI matrices."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _make_transformer_encoder(embed_dim, depth, num_heads, dropout, norm_first = True):
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=embed_dim * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=norm_first,
    )
    return nn.TransformerEncoder(layer, num_layers=depth)


@lru_cache(maxsize=32)
def _sinusoidal_position_embedding(length, dim, device, dtype):
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        * -(math.log(10000.0) / dim)
    )
    embeddings = torch.zeros(length, dim, dtype=torch.float32, device=device)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term[: embeddings[:, 1::2].size(1)])
    return embeddings.to(dtype=dtype)


class TemporalViTEncoder(nn.Module):
    """Vision Transformer treating time-points as tokens."""

    def __init__(
        self,
        token_dim = 768,
        depth = 12,
        num_heads = 12,
        dropout = 0.1,
        latent_dim = 2048,
        pooling = "cls",
        norm_first = True,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.pooling = str(pooling or "cls").lower()
        if self.pooling not in {"cls", "mean", "cls_mean", "attn"}:
            raise ValueError(f"Bad ViT pooling {pooling!r}; expected cls, mean, cls_mean, or attn")

        self.input_proj = nn.LazyLinear(token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_dropout = nn.Dropout(dropout)
        self.encoder = _make_transformer_encoder(token_dim, depth, num_heads, dropout, norm_first=norm_first)
        self.norm = nn.LayerNorm(token_dim)
        self.cls_mean_pool = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
        ) if self.pooling == "cls_mean" else None
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
        ) if self.pooling == "attn" else None
        self.to_latent = nn.Linear(token_dim, latent_dim)

    # ------------------------------------------------------------------
    def forward(self, x, modality_embed):
        if x.ndim != 3:
            raise ValueError(f"TemporalViT expects [B,V,T] inputs, got {tuple(x.shape)}")

        b, v, t = x.shape
        tokens = self.input_proj(x.transpose(1, 2))  # [B, T, token_dim]

        pos = _sinusoidal_position_embedding(t, self.token_dim, tokens.device, tokens.dtype)
        tokens = tokens + pos.unsqueeze(0)
        tokens = tokens + modality_embed.unsqueeze(1)
        tokens = self.pos_dropout(tokens)

        cls = self.cls_token.expand(b, -1, -1) + modality_embed.unsqueeze(1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.encoder(tokens)

        cls_out = tokens[:, 0]
        patch_out = tokens[:, 1:]
        if self.pooling == "mean":
            pooled = self.norm(patch_out.mean(dim=1))
        elif self.pooling == "cls_mean":
            mean_out = patch_out.mean(dim=1)
            pooled = self.norm(self.cls_mean_pool(torch.cat([cls_out, mean_out], dim=1)))
        elif self.pooling == "attn":
            values = self.norm(patch_out)
            weights = torch.softmax(self.attn_pool(values), dim=1)
            pooled = torch.sum(weights * values, dim=1)
        else:
            pooled = self.norm(cls_out)
        latent = self.to_latent(pooled)
        return F.normalize(latent, dim=-1)


class _TimeSformerMLP(nn.Module):
    def __init__(self, dim, mlp_ratio = 4.0, dropout = 0.0):
        super().__init__()
        hidden_dim = int(dim * float(mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _DividedSpaceTimeBlock(nn.Module):
    """TimeSformer divided attention block: temporal attention, spatial attention, MLP."""

    def __init__(self, dim, num_heads, dropout = 0.0, mlp_ratio = 4.0):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal_fc = nn.Linear(dim, dim)
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.resid_drop = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = _TimeSformerMLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x, num_patches, num_frames):
        b, _, dim = x.shape
        n = int(num_patches)
        t = int(num_frames)
        cls_token = x[:, :1]
        patch_tokens = x[:, 1:]

        xt = patch_tokens.reshape(b, n, t, dim).reshape(b * n, t, dim)
        xt_norm = self.temporal_norm(xt)
        temporal_res, _ = self.temporal_attn(xt_norm, xt_norm, xt_norm, need_weights=False)
        temporal_res = self.temporal_fc(self.resid_drop(temporal_res))
        patch_tokens = patch_tokens + temporal_res.reshape(b, n, t, dim).reshape(b, n * t, dim)

        frame_cls = cls_token.expand(b, t, dim).reshape(b * t, 1, dim)
        xs = patch_tokens.reshape(b, n, t, dim).permute(0, 2, 1, 3).reshape(b * t, n, dim)
        xs = torch.cat((frame_cls, xs), dim=1)
        xs_norm = self.spatial_norm(xs)
        spatial_res, _ = self.spatial_attn(xs_norm, xs_norm, xs_norm, need_weights=False)
        spatial_res = self.resid_drop(spatial_res)

        cls_res = spatial_res[:, 0].reshape(b, t, dim).mean(dim=1, keepdim=True)
        patch_res = spatial_res[:, 1:].reshape(b, t, n, dim).permute(0, 2, 1, 3).reshape(b, n * t, dim)
        x = torch.cat((cls_token, patch_tokens), dim=1) + torch.cat((cls_res, patch_res), dim=1)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TemporalTimeSformerEncoder(nn.Module):
    """TimeSformer-style divided space-time transformer for [B, V, T] fMRI matrices."""

    def __init__(
        self,
        token_dim = 768,
        depth = 12,
        num_heads = 12,
        dropout = 0.1,
        latent_dim = 2048,
        patch_vertices = 8192,
        mlp_ratio = 4.0,
        max_time = 1200,
        max_spatial_patches = 1024,
        pooling = "cls",
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.patch_vertices = max(int(patch_vertices), 1)
        self.max_time = max(int(max_time), 1)
        self.max_spatial_patches = max(int(max_spatial_patches), 1)
        self.pooling = str(pooling or "cls").lower()
        if self.pooling not in {"cls", "mean", "cls_mean", "attn"}:
            raise ValueError(f"Bad TimeSformer pooling {pooling!r}; expected cls, mean, cls_mean, or attn")

        self.patch_embed = nn.Linear(self.patch_vertices, self.token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.max_spatial_patches, self.token_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, self.max_time, self.token_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_DividedSpaceTimeBlock(self.token_dim, num_heads, dropout=dropout, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(self.token_dim)
        self.cls_mean_pool = nn.Sequential(
            nn.LayerNorm(self.token_dim * 2),
            nn.Linear(self.token_dim * 2, self.token_dim),
            nn.GELU(),
        ) if self.pooling == "cls_mean" else None
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 1),
        ) if self.pooling == "attn" else None
        self.to_latent = nn.Linear(self.token_dim, latent_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

    def _resize_pos(self, pos, length):
        if int(length) <= int(pos.shape[1]):
            return pos[:, : int(length)]
        return F.interpolate(pos.transpose(1, 2), size=int(length), mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, x, modality_embed):
        if x.ndim != 3:
            raise ValueError(f"TemporalTimeSformer expects [B,V,T] inputs, got {tuple(x.shape)}")

        b, v, t = x.shape
        n = math.ceil(v / self.patch_vertices)
        padded_vertices = n * self.patch_vertices
        if padded_vertices != v:
            x = F.pad(x, (0, 0, 0, padded_vertices - v))

        patches = x.reshape(b, n, self.patch_vertices, t).permute(0, 1, 3, 2)
        tokens = self.patch_embed(patches)  # [B, N, T, token_dim]
        spatial_pos = self._resize_pos(self.spatial_pos_embed, n).unsqueeze(2)
        time_pos = self._resize_pos(self.time_embed, t).unsqueeze(1)
        tokens = tokens + spatial_pos + time_pos + modality_embed[:, None, None, :]
        tokens = self.pos_drop(tokens).reshape(b, n * t, self.token_dim)

        cls = self.cls_token.expand(b, -1, -1) + modality_embed.unsqueeze(1)
        tokens = torch.cat((cls, tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens, n, t)

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        patch_out = tokens[:, 1:]
        if self.pooling == "mean":
            pooled = patch_out.mean(dim=1)
        elif self.pooling == "cls_mean":
            pooled = self.cls_mean_pool(torch.cat([cls_out, patch_out.mean(dim=1)], dim=1))
        elif self.pooling == "attn":
            weights = torch.softmax(self.attn_pool(patch_out), dim=1)
            pooled = torch.sum(weights * patch_out, dim=1)
        else:
            pooled = cls_out
        latent = self.to_latent(pooled)
        return F.normalize(latent, dim=-1)


class TemporalGRUEncoder(nn.Module):
    def __init__(self, token_dim = 768, depth = 2, dropout = 0.1, latent_dim = 2048, bidirectional = True):
        super().__init__()
        self.input_proj = nn.LazyLinear(token_dim)
        self.gru = nn.GRU(token_dim, token_dim, num_layers=depth, dropout=dropout if depth > 1 else 0.0, batch_first=True, bidirectional=bidirectional)
        self.norm = nn.LayerNorm(token_dim * (2 if bidirectional else 1))
        self.to_latent = nn.Linear(token_dim * (2 if bidirectional else 1), latent_dim)

    def forward(self, x, modality_embed):
        if x.ndim != 3:
            raise ValueError(f"TemporalGRU expects [B,V,T] inputs, got {tuple(x.shape)}")
        tokens = self.input_proj(x.transpose(1, 2)) + modality_embed.unsqueeze(1)
        out, _ = self.gru(tokens)
        latent = self.to_latent(self.norm(out.mean(dim=1)))
        return F.normalize(latent, dim=-1)


class TemporalConvEncoder(nn.Module):
    def __init__(self, token_dim = 768, depth = 4, kernel_size = 7, dropout = 0.1, latent_dim = 2048):
        super().__init__()
        self.input_proj = nn.LazyLinear(token_dim)
        layers = []
        for i in range(depth):
            dilation = 2 ** min(i, 5)
            pad = dilation * (kernel_size // 2)
            layers.extend([nn.Conv1d(token_dim, token_dim, kernel_size, padding=pad, dilation=dilation), nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(token_dim)
        self.to_latent = nn.Linear(token_dim, latent_dim)

    def forward(self, x, modality_embed):
        if x.ndim != 3:
            raise ValueError(f"TemporalConv expects [B,V,T] inputs, got {tuple(x.shape)}")
        tokens = self.input_proj(x.transpose(1, 2)) + modality_embed.unsqueeze(1)
        tokens = self.net(tokens.transpose(1, 2)).transpose(1, 2)
        latent = self.to_latent(self.norm(tokens.mean(dim=1)))
        return F.normalize(latent, dim=-1)


class TemporalMeanEncoder(nn.Module):
    def __init__(self, token_dim = 768, dropout = 0.1, latent_dim = 2048):
        super().__init__()
        self.input_proj = nn.LazyLinear(token_dim)
        self.net = nn.Sequential(nn.LayerNorm(token_dim), nn.Dropout(dropout), nn.Linear(token_dim, latent_dim))

    def forward(self, x, modality_embed):
        if x.ndim != 3:
            raise ValueError(f"TemporalMean expects [B,V,T] inputs, got {tuple(x.shape)}")
        tokens = self.input_proj(x.transpose(1, 2)) + modality_embed.unsqueeze(1)
        return F.normalize(self.net(tokens.mean(dim=1)), dim=-1)
