"""Memory-conscious spatial-temporal regressors for raw 4D fMRI volumes."""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.dropout = nn.Dropout3d(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(self.dropout(F.gelu(self.norm2(x))))
        return x + residual


class FrameEncoder3D(nn.Module):
    """Shared 3D CNN applied independently to each fMRI frame."""

    def __init__(self, base_dim: int = 16, output_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        dims = (base_dim, base_dim * 2, base_dim * 4, output_dim)
        self.stem = nn.Sequential(
            nn.Conv3d(1, dims[0], 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(dims[0]), dims[0]),
            nn.GELU(),
        )
        stages: list[nn.Module] = []
        for left, right in zip(dims[:-1], dims[1:]):
            stages.extend(
                [
                    Residual3DBlock(left, dropout),
                    nn.Conv3d(left, right, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(_group_count(right), right),
                    nn.GELU(),
                ]
            )
        stages.append(Residual3DBlock(dims[-1], dropout))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.stages(self.stem(x))).flatten(1)


def _encode_frame_chunks(encoder: nn.Module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    batch, time = x.shape[:2]
    flat = x.reshape(batch * time, *x.shape[2:])
    if chunk_size <= 0 or flat.shape[0] <= chunk_size:
        encoded = encoder(flat)
    else:
        encoded = torch.cat([encoder(part) for part in flat.split(chunk_size, dim=0)], dim=0)
    return encoded.reshape(batch, time, -1)


class CovariateFusion(nn.Module):
    def __init__(self, dim: int, covariate_dim: int = 0) -> None:
        super().__init__()
        self.projection = (
            nn.Sequential(nn.Linear(covariate_dim, dim), nn.GELU(), nn.Linear(dim, dim))
            if covariate_dim > 0
            else None
        )

    def forward(self, x: torch.Tensor, covariates: torch.Tensor | None = None) -> torch.Tensor:
        if self.projection is None:
            return x
        if covariates is None:
            raise ValueError("covariates are required when covariate_dim is nonzero")
        return x + self.projection(covariates.float()).to(dtype=x.dtype)


class SignatureProjection(nn.Module):
    def __init__(self, in_dim: int, signature_dim: int = 1024, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, signature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    def __init__(
        self,
        dim: int,
        targets: int = 3,
        hidden_dim: int = 512,
        depth: int = 3,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            depth = max(int(depth), 1)
            hidden_dims = tuple(int(hidden_dim) for _ in range(depth - 1))
        else:
            hidden_dims = tuple(int(value) for value in hidden_dims)
        layers: list[nn.Module] = [nn.LayerNorm(dim)]
        current = dim
        for next_dim in hidden_dims:
            layers.extend((nn.Linear(current, next_dim), nn.GELU(), nn.Dropout(dropout)))
            current = next_dim
        layers.append(nn.Linear(current, targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNGRURegressor(nn.Module):
    """Established per-frame 3D CNN plus bidirectional GRU baseline."""

    def __init__(
        self,
        *,
        base_dim: int = 16,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.05,
        frame_chunk_size: int = 16,
        covariate_dim: int = 0,
        target_dim: int = 3,
        signature_dim: int = 1024,
        head_hidden_dim: int = 512,
        head_depth: int = 3,
        head_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = FrameEncoder3D(base_dim, feature_dim, dropout)
        self.gru = nn.GRU(
            feature_dim,
            hidden_dim,
            num_layers=depth,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if depth > 1 else 0.0,
        )
        output_dim = hidden_dim * 2
        self.attention = nn.Sequential(nn.LayerNorm(output_dim), nn.Linear(output_dim, 1))
        self.covariates = CovariateFusion(output_dim, covariate_dim)
        self.signature = SignatureProjection(output_dim, signature_dim, dropout)
        self.head = RegressionHead(
            signature_dim,
            targets=target_dim,
            hidden_dim=head_hidden_dim,
            depth=head_depth,
            hidden_dims=head_dims,
            dropout=dropout,
        )
        self.frame_chunk_size = int(frame_chunk_size)

    def forward(self, image: torch.Tensor, covariates: torch.Tensor | None = None) -> torch.Tensor:
        tokens = _encode_frame_chunks(self.encoder, image, self.frame_chunk_size)
        tokens, _ = self.gru(tokens)
        weights = self.attention(tokens).squeeze(-1).softmax(dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        signature = self.signature(self.covariates(pooled, covariates))
        return self.head(signature)


def _sinusoidal_positions(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class CNNTemporalTransformerRegressor(nn.Module):
    """New compact hybrid: 3D spatial encoder plus temporal Transformer."""

    def __init__(
        self,
        *,
        base_dim: int = 16,
        feature_dim: int = 192,
        depth: int = 4,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
        frame_chunk_size: int = 16,
        covariate_dim: int = 0,
        target_dim: int = 3,
        signature_dim: int = 1024,
        head_hidden_dim: int = 512,
        head_depth: int = 3,
        head_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if feature_dim % heads:
            raise ValueError("feature_dim must be divisible by heads")
        self.encoder = FrameEncoder3D(base_dim, feature_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            feature_dim,
            heads,
            dim_feedforward=int(feature_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(feature_dim))
        self.query = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.pool = nn.MultiheadAttention(feature_dim, heads, dropout=dropout, batch_first=True)
        self.covariates = CovariateFusion(feature_dim, covariate_dim)
        self.signature = SignatureProjection(feature_dim, signature_dim, dropout)
        self.head = RegressionHead(
            signature_dim,
            targets=target_dim,
            hidden_dim=head_hidden_dim,
            depth=head_depth,
            hidden_dims=head_dims,
            dropout=dropout,
        )
        self.frame_chunk_size = int(frame_chunk_size)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, image: torch.Tensor, covariates: torch.Tensor | None = None) -> torch.Tensor:
        tokens = _encode_frame_chunks(self.encoder, image, self.frame_chunk_size)
        tokens = tokens + _sinusoidal_positions(tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype)
        tokens = self.temporal(tokens)
        query = self.query.expand(tokens.shape[0], -1, -1).to(dtype=tokens.dtype)
        pooled = self.pool(query, tokens, tokens, need_weights=False)[0].squeeze(1)
        signature = self.signature(self.covariates(pooled, covariates))
        return self.head(signature)


def _window_partition(x: torch.Tensor, window: Sequence[int]) -> torch.Tensor:
    batch, time, depth, height, width, channels = x.shape
    wt, wd, wh, ww = window
    x = x.view(
        batch,
        time // wt,
        wt,
        depth // wd,
        wd,
        height // wh,
        wh,
        width // ww,
        ww,
        channels,
    )
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9).contiguous()
    return x.view(-1, wt * wd * wh * ww, channels)


def _window_reverse(
    windows: torch.Tensor,
    window: Sequence[int],
    batch: int,
    shape: Sequence[int],
) -> torch.Tensor:
    time, depth, height, width = shape
    wt, wd, wh, ww = window
    channels = windows.shape[-1]
    x = windows.view(
        batch,
        time // wt,
        depth // wd,
        height // wh,
        width // ww,
        wt,
        wd,
        wh,
        ww,
        channels,
    )
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9).contiguous()
    return x.view(batch, time, depth, height, width, channels)


def _shift_slices(size: int, window: int, shift: int) -> tuple[slice, ...]:
    if shift <= 0:
        return (slice(0, size),)
    return (slice(0, -window), slice(-window, -shift), slice(-shift, None))


def _shift_attention_mask(
    padded_shape: Sequence[int],
    window: Sequence[int],
    shift: Sequence[int],
    device: torch.device,
) -> torch.Tensor | None:
    if not any(shift):
        return None
    time, depth, height, width = padded_shape
    mask = torch.zeros((1, time, depth, height, width, 1), device=device)
    counter = 0
    dimensions = zip(padded_shape, window, shift)
    slice_groups = [_shift_slices(size, win, amount) for size, win, amount in dimensions]
    for slices in itertools.product(*slice_groups):
        mask[(slice(None), *slices, slice(None))] = counter
        counter += 1
    windows = _window_partition(mask, window).squeeze(-1)
    difference = windows.unsqueeze(1) - windows.unsqueeze(2)
    return difference.ne(0).to(dtype=torch.float32) * -10000.0


class WindowAttention4D(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.heads, channels // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        if mask is not None:
            window_count = mask.shape[0]
            attention = attention.view(batch_windows // window_count, window_count, self.heads, tokens, tokens)
            attention = attention + mask[None, :, None].to(dtype=attention.dtype)
            attention = attention.view(batch_windows, self.heads, tokens, tokens)
        attention = self.dropout(attention.softmax(dim=-1))
        x = (attention @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.dropout(self.projection(x))


class Swin4DBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        window: Sequence[int],
        shifted: bool,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window = tuple(int(value) for value in window)
        self.shift = tuple(value // 2 if shifted else 0 for value in self.window)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention4D(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        original = x.shape[1:5]
        padding = tuple((win - size % win) % win for size, win in zip(original, self.window))
        pt, pd, ph, pw = padding
        x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd, 0, pt))
        padded = x.shape[1:5]
        effective_shift = tuple(shift if size > win else 0 for shift, size, win in zip(self.shift, padded, self.window))
        if any(effective_shift):
            x = torch.roll(x, shifts=tuple(-value for value in effective_shift), dims=(1, 2, 3, 4))
        mask = _shift_attention_mask(padded, self.window, effective_shift, x.device)
        windows = _window_partition(x, self.window)
        windows = self.attention(windows, mask)
        x = _window_reverse(windows, self.window, residual.shape[0], padded)
        if any(effective_shift):
            x = torch.roll(x, shifts=effective_shift, dims=(1, 2, 3, 4))
        x = x[:, : original[0], : original[1], : original[2], : original[3]]
        x = residual + x
        return x + self.mlp(self.norm2(x))


class PatchMerge4D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.spatial = nn.Conv3d(dim, dim * 2, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, depth, height, width, channels = x.shape
        if time % 2:
            x = torch.cat((x, x[:, -1:]), dim=1)
            time += 1
        x = x.view(batch, time // 2, 2, depth, height, width, channels).mean(dim=2)
        x = x.permute(0, 1, 5, 2, 3, 4).reshape(batch * (time // 2), channels, depth, height, width)
        x = self.spatial(x)
        depth, height, width = x.shape[-3:]
        x = x.view(batch, time // 2, -1, depth, height, width).permute(0, 1, 3, 4, 5, 2)
        return x


class SwiFTRegressor(nn.Module):
    """SwiFT-style hierarchical shifted-window attention over 4D patches."""

    def __init__(
        self,
        *,
        embed_dim: int = 32,
        depths: Sequence[int] = (2, 2),
        heads: Sequence[int] = (4, 8),
        spatial_patch: int = 8,
        temporal_patch: int = 2,
        window: Sequence[int] = (2, 4, 4, 4),
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
        covariate_dim: int = 0,
        target_dim: int = 3,
        signature_dim: int = 1024,
        head_hidden_dim: int = 512,
        head_depth: int = 3,
        head_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if len(depths) != 2 or len(heads) != 2:
            raise ValueError("SwiFT currently expects exactly two hierarchical stages")
        self.spatial_patch = int(spatial_patch)
        self.temporal_patch = int(temporal_patch)
        self.patch = nn.Conv3d(1, embed_dim, self.spatial_patch, stride=self.spatial_patch)
        self.pos_time = nn.Parameter(torch.zeros(1, 256, 1, 1, 1, embed_dim))
        self.pos_depth = nn.Parameter(torch.zeros(1, 1, 32, 1, 1, embed_dim))
        self.pos_height = nn.Parameter(torch.zeros(1, 1, 1, 32, 1, embed_dim))
        self.pos_width = nn.Parameter(torch.zeros(1, 1, 1, 1, 32, embed_dim))
        self.stage1 = nn.Sequential(
            *[
                Swin4DBlock(embed_dim, heads[0], window, bool(index % 2), mlp_ratio, dropout)
                for index in range(depths[0])
            ]
        )
        self.merge = PatchMerge4D(embed_dim)
        second_window = tuple(max(2, value // 2) for value in window)
        self.stage2 = nn.Sequential(
            *[
                Swin4DBlock(embed_dim * 2, heads[1], second_window, bool(index % 2), mlp_ratio, dropout)
                for index in range(depths[1])
            ]
        )
        output_dim = embed_dim * 2
        self.norm = nn.LayerNorm(output_dim)
        self.covariates = CovariateFusion(output_dim, covariate_dim)
        self.signature = SignatureProjection(output_dim, signature_dim, dropout)
        self.head = RegressionHead(
            signature_dim,
            targets=target_dim,
            hidden_dim=head_hidden_dim,
            depth=head_depth,
            hidden_dims=head_dims,
            dropout=dropout,
        )
        for parameter in (self.pos_time, self.pos_depth, self.pos_height, self.pos_width):
            nn.init.trunc_normal_(parameter, std=0.02)

    def _patch_embed(self, image: torch.Tensor) -> torch.Tensor:
        batch, time = image.shape[:2]
        x = self.patch(image.reshape(batch * time, *image.shape[2:]))
        channels, depth, height, width = x.shape[1:]
        x = x.view(batch, time, channels, depth, height, width).permute(0, 2, 1, 3, 4, 5)
        if time % self.temporal_patch:
            padding = self.temporal_patch - time % self.temporal_patch
            x = torch.cat((x, x[:, :, -1:].expand(-1, -1, padding, -1, -1, -1)), dim=2)
            time += padding
        x = x.view(batch, channels, time // self.temporal_patch, self.temporal_patch, depth, height, width).mean(dim=3)
        return x.permute(0, 2, 3, 4, 5, 1)

    def forward(self, image: torch.Tensor, covariates: torch.Tensor | None = None) -> torch.Tensor:
        x = self._patch_embed(image)
        time, depth, height, width = x.shape[1:5]
        if time > 256 or max(depth, height, width) > 32:
            raise ValueError(f"SwiFT token grid {(time, depth, height, width)} exceeds positional limits")
        x = x + self.pos_time[:, :time] + self.pos_depth[:, :, :depth] + self.pos_height[:, :, :, :height] + self.pos_width[:, :, :, :, :width]
        x = self.stage1(x)
        x = self.merge(x)
        x = self.stage2(x)
        pooled = self.norm(x).mean(dim=(1, 2, 3, 4))
        signature = self.signature(self.covariates(pooled, covariates))
        return self.head(signature)


class SpatialGridEncoder(nn.Module):
    def __init__(self, dim: int = 96, grid_size: int = 4, frame_chunk_size: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, dim // 4, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(dim // 4), dim // 4),
            nn.GELU(),
            nn.Conv3d(dim // 4, dim // 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(dim // 2), dim // 2),
            nn.GELU(),
            Residual3DBlock(dim // 2),
            nn.Conv3d(dim // 2, dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(dim), dim),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(grid_size),
        )
        self.grid_size = int(grid_size)
        self.frame_chunk_size = max(int(frame_chunk_size), 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, time = image.shape[:2]
        flat = image.reshape(batch * time, *image.shape[2:])
        x = torch.cat([self.net(chunk) for chunk in flat.split(self.frame_chunk_size)], dim=0)
        x = x.flatten(2).transpose(1, 2)
        return x.view(batch, time, self.grid_size**3, -1)


def _load_mamba_layer(dim: int, state_dim: int, conv_width: int, expand: int) -> nn.Module:
    try:
        from mamba_ssm import Mamba2

        return Mamba2(d_model=dim, d_state=state_dim, d_conv=conv_width, expand=expand)
    except (ImportError, TypeError):
        try:
            from mamba_ssm import Mamba

            return Mamba(d_model=dim, d_state=state_dim, d_conv=conv_width, expand=expand)
        except ImportError as error:
            raise ImportError(
                "BrainMT requires the optional mamba-ssm package. Install a version "
                "compatible with the active PyTorch/CUDA environment; the other 4D "
                "models do not require it."
            ) from error


class BrainMTRegressor(nn.Module):
    """BrainMT-style temporal-first bidirectional Mamba + spatial attention."""

    def __init__(
        self,
        *,
        dim: int = 96,
        grid_size: int = 4,
        mamba_depth: int = 2,
        state_dim: int = 64,
        conv_width: int = 4,
        expand: int = 2,
        spatial_depth: int = 2,
        spatial_heads: int = 6,
        frame_chunk_size: int = 8,
        dropout: float = 0.05,
        covariate_dim: int = 0,
        target_dim: int = 3,
        signature_dim: int = 1024,
        head_hidden_dim: int = 512,
        head_depth: int = 3,
        head_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if dim % spatial_heads:
            raise ValueError("dim must be divisible by spatial_heads")
        self.encoder = SpatialGridEncoder(dim, grid_size, frame_chunk_size)
        self.forward_mamba = nn.ModuleList(
            [_load_mamba_layer(dim, state_dim, conv_width, expand) for _ in range(mamba_depth)]
        )
        self.backward_mamba = nn.ModuleList(
            [_load_mamba_layer(dim, state_dim, conv_width, expand) for _ in range(mamba_depth)]
        )
        self.temporal_norm = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(
            dim,
            spatial_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial = nn.TransformerEncoder(layer, spatial_depth, norm=nn.LayerNorm(dim))
        self.covariates = CovariateFusion(dim, covariate_dim)
        self.signature = SignatureProjection(dim, signature_dim, dropout)
        self.head = RegressionHead(
            signature_dim,
            targets=target_dim,
            hidden_dim=head_hidden_dim,
            depth=head_depth,
            hidden_dims=head_dims,
            dropout=dropout,
        )

    @staticmethod
    def _stack(blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            x = x + block(x)
        return x

    def forward(self, image: torch.Tensor, covariates: torch.Tensor | None = None) -> torch.Tensor:
        grid = self.encoder(image)  # [B,T,S,C]
        batch, time, spatial, channels = grid.shape
        sequence = grid.permute(0, 2, 1, 3).reshape(batch * spatial, time, channels)
        forward = self._stack(self.forward_mamba, sequence)
        backward = torch.flip(self._stack(self.backward_mamba, torch.flip(sequence, dims=(1,))), dims=(1,))
        sequence = self.temporal_norm(forward + backward).mean(dim=1)
        spatial_tokens = sequence.view(batch, spatial, channels)
        pooled = self.spatial(spatial_tokens).mean(dim=1)
        signature = self.signature(self.covariates(pooled, covariates))
        return self.head(signature)


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower().replace("-", "_")
    models = {
        "cnn_gru": CNNGRURegressor,
        "cnn_transformer": CNNTemporalTransformerRegressor,
        "swift": SwiFTRegressor,
        "brainmt": BrainMTRegressor,
    }
    if name not in models:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(models)}")
    return models[name](**kwargs)
