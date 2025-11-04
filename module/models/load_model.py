# module/models/load_model.py
"""
Backbone loader for R2T-Net
---------------------------
Any string `model_name` that you pass on the CLI is mapped to an
encoder returning a tensor shaped **[B, embed_dim]**.

Categories & examples
---------------------
Transformer-4D  : swin4d_ver7 (default) · swin4d_base · vit · timesformer
3-D CNN         : resnet3d18 · resnet3d34 · densenet3d121 · unet3d
Hybrid          : cnn_gru · perceiver_io · temporal_unet

Usage
-----
# 3-D ResNet backbone with 128-d latent vectors
python train.py --model resnet3d18 --embed_dim 128 ...

# Perceiver-IO hybrid encoder
python train.py --model perceiver_io --embed_dim 512 ...

# TimeSformer (ViT with divided temporal attention)
python train.py --model timesformer --embed_dim 768 ...

"""
from __future__ import annotations
import importlib
from types import ModuleType

import torch
from torch import nn
from timm import create_model

# local 4-D transformers
from .swin4d_transformer_ver7 import SwinTransformer4D as SwinTransformer4D_ver7
from .swin_transformer import SwinTransformer as SwinTransformer4D_base


# --------------------------------------------------------------------
# Small adapter that always squeezes spatial / temporal dims and
# optionally projects to the requested embed_dim.
# --------------------------------------------------------------------
class _FeatureAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, in_dim: int, embed_dim: int):
        super().__init__()
        self.backbone = backbone
        self.project = nn.Identity() if in_dim == embed_dim else nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # [B, …] → [B, E]
        feats = self.backbone(x)                                 # model-specific shape
        if feats.ndim > 2:                                       # squeeze H,W,D,T…
            feats = feats.mean(dim=tuple(range(2, feats.ndim)))
        return self.project(feats)


# --------------------------------------------------------------------
# helpers to build each family
# --------------------------------------------------------------------
def _load_timm(name: str, hparams) -> nn.Module:
    timm_model = create_model(name, pretrained=False, in_chans=hparams.in_chans, num_classes=0)
    in_dim = getattr(timm_model, "num_features", None) or timm_model.head.in_features
    return _FeatureAdapter(timm_model, in_dim, hparams.embed_dim)


def _load_torchvision_video(name: str, hparams) -> nn.Module:
    tv = importlib.import_module("torchvision.models.video")          # late import
    base: nn.Module = getattr(tv, name)(pretrained=False, progress=False)
    in_dim = base.fc.in_features
    base.fc = nn.Identity()
    return _FeatureAdapter(base, in_dim, hparams.embed_dim)


def _load_monai(name: str, hparams) -> nn.Module:
    monai: ModuleType = importlib.import_module("monai.networks.nets")
    base: nn.Module = getattr(monai, name)(
        spatial_dims=3,
        in_channels=hparams.in_chans,
        out_channels=1,  # classifier head removed later
    )
    in_dim = base.classification_head.in_features
    base.classification_head = nn.Identity()
    return _FeatureAdapter(base, in_dim, hparams.embed_dim)


# --------------------------------------------------------------------
# public API
# --------------------------------------------------------------------
def load_model(model_name: str, hparams) -> nn.Module:
    """
    Returns a backbone whose forward pass emits [B, embed_dim].

    Parameters
    ----------
    model_name : str
        One of the keys listed in the table above (case-insensitive).
    hparams    : Namespace
        Parsed Lightning/CLI hyper-parameters (needs .in_chans and .embed_dim).

    Raises
    ------
    ValueError if an unknown name is supplied.
    """
    name = model_name.lower()

    # ---------- Transformer 4-D ---------------------------------------
    if name == "swin4d_ver7":
        return SwinTransformer4D_ver7(**vars(hparams))
    if name == "swin4d_base":
        return SwinTransformer4D_base(**vars(hparams))
    if name in {"vit", "timesformer", "transformer2d"}:
        return _load_timm(name, hparams)

    # ---------- 3-D CNN ------------------------------------------------
    if name in {"resnet3d18", "resnet3d34", "r3d_18"}:
        # torchvision video models
        return _load_torchvision_video("r3d_18" if name == "resnet3d18" else "r3d_34", hparams)
    if name in {"densenet3d121", "unet3d"}:
        # MONAI nets (install monai first)
        return _load_monai(name, hparams)

    # ---------- Hybrid models -----------------------------------------
    if name == "cnn_gru":
        from .hybrid.cnn_gru import CNN_GRU                # noqa: local import
        base = CNN_GRU(in_chans=hparams.in_chans)
        return _FeatureAdapter(base, base.out_dim, hparams.embed_dim)

    if name == "perceiver_io":
        from .hybrid.perceiver_io import PerceiverIO       # noqa
        base = PerceiverIO(in_chans=hparams.in_chans)
        return _FeatureAdapter(base, base.out_dim, hparams.embed_dim)

    if name == "temporal_unet":
        from .hybrid.temporal_unet import TemporalUNet     # noqa
        base = TemporalUNet(in_chans=hparams.in_chans)
        return _FeatureAdapter(base, base.out_dim, hparams.embed_dim)

    # ---------- unknown ------------------------------------------------
    raise ValueError(f"[load_model] Unsupported backbone: “{model_name}”")
