"""Plain PyTorch R2T-Net."""

from __future__ import annotations

import random
import argparse
import math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import Tensor, nn

from .backbones.swift import SwiFTConfig, SwiFTEncoder
from .backbones.temporal import TemporalConvEncoder, TemporalGRUEncoder, TemporalMeanEncoder, TemporalViTEncoder


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return {"prediction": self.net(x)}


class TwoTierRegressionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, bin_centers, bin_widths, bin_edges, temperature = 1.0):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.bin_classifier = nn.Linear(hidden_dim, bin_centers.size(0))
        self.residual = nn.Linear(hidden_dim + bin_centers.size(0), bin_centers.numel())
        self.temperature = temperature
        self.target_dim = bin_centers.size(1)
        self.num_bins = bin_centers.size(0)
        self.register_buffer("bin_centers", bin_centers)
        self.register_buffer("bin_widths", bin_widths)
        self.register_buffer("bin_edges", bin_edges)

    def forward(self, x):
        z = self.hidden(x)
        logits = self.bin_classifier(z)
        scaled_logits = logits / self.temperature if self.temperature != 1.0 else logits
        probs = F.softmax(scaled_logits, dim=-1)
        residual = torch.tanh(self.residual(torch.cat([z, probs], dim=-1)))
        residual = residual.view(-1, self.num_bins, self.target_dim)
        refined = self.bin_centers.unsqueeze(0) + self.bin_widths.unsqueeze(0) * residual
        prediction = torch.sum(probs.unsqueeze(-1) * refined, dim=1)
        return {"prediction": prediction, "logits": scaled_logits, "probabilities": probs, "residual": residual}


class MLPRegressionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, depth = 2):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(max(depth - 1, 0)):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return {"prediction": self.net(x)}


@dataclass(frozen=True)
class HeadSpec:
    name: str
    indices: Tuple[int, ...]
    task: str
    weight: float


def _as_target_names(value, dim):
    if isinstance(value, str):
        names = [x.strip() for x in value.split(",") if x.strip()]
    elif value:
        names = [str(x) for x in value]
    else:
        names = []
    return names if len(names) == dim else [f"target_{i}" for i in range(dim)]


def _clean_name(value):
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.strip())
    return out or "head"


def _parse_columns(raw, names, dim):
    if raw in {"*", "all"}:
        return tuple(range(dim))
    idxs = []
    by_name = {name: i for i, name in enumerate(names)}
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if ".." in item:
            lo, hi = item.split("..", 1)
            idxs.extend(range(int(lo), int(hi) + 1))
        elif item.lstrip("-").isdigit():
            idxs.append(int(item))
        else:
            if item not in by_name:
                raise ValueError(f"Unknown target {item!r}; available={list(names)}")
            idxs.append(by_name[item])
    if not idxs:
        raise ValueError("Empty head column list")
    if min(idxs) < 0 or max(idxs) >= dim:
        raise ValueError(f"Head column out of range for target_dim={dim}: {idxs}")
    return tuple(idxs)


def parse_head_specs(raw, target_names, target_dim, default_task):
    if not raw:
        return [HeadSpec("target", tuple(range(target_dim)), default_task, 1.0)]
    specs, seen = [], set()
    for part in [x.strip() for x in raw.split(";") if x.strip()]:
        fields = [x.strip() for x in part.split(":")]
        if len(fields) < 2 or len(fields) > 4:
            raise ValueError("Head spec must be name:cols[:regression|classification[:weight]]")
        name = _clean_name(fields[0])
        indices = _parse_columns(fields[1], target_names, target_dim)
        task = fields[2] if len(fields) >= 3 and fields[2] else default_task
        weight = float(fields[3]) if len(fields) == 4 and fields[3] else 1.0
        if task not in {"regression", "classification"}:
            raise ValueError(f"Bad head task {task!r}")
        if name in seen:
            raise ValueError(f"Duplicate head name {name!r}")
        seen.add(name)
        specs.append(HeadSpec(name, indices, task, weight))
    flat = [i for spec in specs for i in spec.indices]
    if len(flat) != len(set(flat)):
        raise ValueError("Head specs must not overlap")
    if sorted(flat) != list(range(target_dim)):
        raise ValueError("Head specs must cover every target column exactly once")
    return specs


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim, targets_np, target_names, specs, hparams):
        super().__init__()
        self.target_dim = int(targets_np.shape[1])
        self.target_names = list(target_names)
        self.specs = list(specs)
        self.heads = nn.ModuleDict()
        self.scaler_type: Dict[str, str | None] = {}
        self.head_map = {spec.name: list(spec.indices) for spec in self.specs}
        hidden = hparams.get("pred_hidden_dim", 512)
        dropout = hparams.get("head_dropout", 0.1)
        reg_head = hparams.get("reg_head", "yolo")
        for spec in self.specs:
            y = targets_np[:, spec.indices]
            if spec.task == "classification":
                self.heads[spec.name] = ClassificationHead(in_dim, hidden, len(spec.indices), dropout)
                self.scaler_type[spec.name] = None
            else:
                scaled = self._fit_scaler(spec.name, y, hparams.get("label_scaling_method", "standardization"))
                if reg_head == "yolo":
                    bins = self._build_regression_bins(scaled, hparams)
                    self.heads[spec.name] = TwoTierRegressionHead(in_dim, hidden, dropout, bins["centers"], bins["widths"], bins["edges"], hparams.get("reg_temperature", 1.0))
                elif reg_head == "linear":
                    self.heads[spec.name] = MLPRegressionHead(in_dim, hidden, len(spec.indices), dropout, depth=1)
                else:
                    self.heads[spec.name] = MLPRegressionHead(in_dim, hidden, len(spec.indices), dropout, depth=hparams.get("head_depth", 2))

    def _fit_scaler(self, name, y, method):
        if method == "standardization":
            scaler = StandardScaler().fit(y)
            scale = scaler.scale_.astype(np.float32)
            scale[scale == 0] = 1.0
            self.scaler_type[name] = "standard"
            self.register_buffer(f"{name}_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer(f"{name}_scale", torch.tensor(scale, dtype=torch.float32))
            return ((y - scaler.mean_) / scale).astype(np.float32)
        if method == "minmax":
            scaler = MinMaxScaler().fit(y)
            span = (scaler.data_max_ - scaler.data_min_).astype(np.float32)
            span[span == 0] = 1.0
            self.scaler_type[name] = "minmax"
            self.register_buffer(f"{name}_min", torch.tensor(scaler.data_min_, dtype=torch.float32))
            self.register_buffer(f"{name}_range", torch.tensor(span, dtype=torch.float32))
            return ((y - scaler.data_min_) / span).astype(np.float32)
        self.scaler_type[name] = None
        return y.astype(np.float32)

    @staticmethod
    def _build_regression_bins(scaled, hparams):
        num_bins = int(hparams.get("reg_num_bins", 8))
        strategy = hparams.get("reg_binning_strategy", "quantile")
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        centers, widths, edges = [], [], []
        for col in scaled.T:
            e = np.quantile(col, quantiles).astype(np.float32) if strategy == "quantile" else np.linspace(float(col.min()), float(col.max()), num_bins + 1, dtype=np.float32)
            e[0] -= 1e-6
            e[-1] += 1e-6
            w = np.maximum(np.diff(e), 1e-6)
            centers.append(e[:-1] + 0.5 * w)
            widths.append(w)
            edges.append(e)
        return {
            "centers": torch.tensor(np.stack(centers, axis=1), dtype=torch.float32),
            "widths": torch.tensor(np.stack(widths, axis=1), dtype=torch.float32),
            "edges": torch.tensor(np.stack(edges, axis=1), dtype=torch.float32),
        }

    def forward(self, x):
        by_head = {name: head(x) for name, head in self.heads.items()}
        pred = x.new_zeros((x.size(0), self.target_dim))
        for spec in self.specs:
            pred[:, list(spec.indices)] = by_head[spec.name]["prediction"]
        return {"prediction": pred, "heads": by_head}

    def _scale_target(self, spec, y):
        kind = self.scaler_type.get(spec.name)
        if kind is None:
            return y
        if kind == "standard":
            return (y - getattr(self, f"{spec.name}_mean").to(y.device)) / getattr(self, f"{spec.name}_scale").to(y.device)
        return (y - getattr(self, f"{spec.name}_min").to(y.device)) / getattr(self, f"{spec.name}_range").to(y.device)

    def inverse_scale(self, prediction):
        if prediction.ndim == 1:
            prediction = prediction.unsqueeze(0)
        out = prediction.clone()
        for spec in self.specs:
            cols = list(spec.indices)
            vals = prediction[:, cols]
            kind = self.scaler_type.get(spec.name)
            if kind == "standard":
                vals = vals * getattr(self, f"{spec.name}_scale").to(vals.device) + getattr(self, f"{spec.name}_mean").to(vals.device)
            elif kind == "minmax":
                vals = vals * getattr(self, f"{spec.name}_range").to(vals.device) + getattr(self, f"{spec.name}_min").to(vals.device)
            out[:, cols] = vals
        return out.squeeze(0) if out.size(0) == 1 else out

    @staticmethod
    def _assign_bins(head, target):
        interior = head.bin_edges.to(target.device)[1:-1]
        return torch.stack([torch.clamp(torch.bucketize(target[:, j].contiguous(), interior[:, j].contiguous()), max=head.num_bins - 1) for j in range(target.size(1))], dim=1)

    def _regression_loss(self, spec, out, target, hparams):
        if hparams.get("reg_head", "yolo") != "yolo":
            target = self._scale_target(spec, target.float())
            loss_name = hparams.get("reg_loss", "huber")
            if loss_name == "mse":
                return F.mse_loss(out["prediction"], target)
            if loss_name == "l1":
                return F.l1_loss(out["prediction"], target)
            return F.smooth_l1_loss(out["prediction"], target)
        head = self.heads[spec.name]
        target = self._scale_target(spec, target.float())
        logits, probs, residual = out["logits"], out["probabilities"], out["residual"]
        labels = self._assign_bins(head, target)
        logits_rep = logits.unsqueeze(1).expand(-1, target.size(1), -1).reshape(-1, probs.size(1))
        cls_loss = F.cross_entropy(logits_rep, labels.reshape(-1), label_smoothing=hparams.get("reg_label_smoothing", 0.0))
        centers = head.bin_centers.to(target.device).unsqueeze(0)
        widths = head.bin_widths.to(target.device).unsqueeze(0)
        huber = F.smooth_l1_loss(residual, (target.unsqueeze(1) - centers) / widths, reduction="none").mean(dim=2)
        reg_loss = torch.sum(probs * huber, dim=1).mean()
        return hparams.get("reg_alpha", 1.0) * cls_loss + hparams.get("reg_beta", 2.0) * reg_loss

    def _classification_loss(self, spec, out, target, hparams):
        mode = hparams.get("classification_loss", "auto")
        if mode == "auto":
            mode = "cross_entropy" if len(spec.indices) > 1 else "bce"
        if mode == "cross_entropy":
            labels = target.argmax(dim=1) if target.ndim > 1 and target.size(1) > 1 else target.reshape(-1).long()
            return F.cross_entropy(out["prediction"], labels.long(), label_smoothing=hparams.get("classification_label_smoothing", 0.0))
        return F.binary_cross_entropy_with_logits(out["prediction"], target.float())

    def loss(self, head_out, target, hparams):
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        losses, logs = [], {}
        for spec in self.specs:
            cols = list(spec.indices)
            y = target[:, cols]
            out = head_out["heads"][spec.name]
            val = self._classification_loss(spec, out, y, hparams) if spec.task == "classification" else self._regression_loss(spec, out, y, hparams)
            losses.append(float(spec.weight) * val)
            logs[f"head/{spec.name}"] = float(val.detach().cpu())
        total = torch.stack(losses).sum()
        logs["supervised"] = float(total.detach().cpu())
        return total, logs


class SyntheticTaskMapper(nn.Module):
    def __init__(self, signature_dim, depth, heads, dropout, tokens):
        super().__init__()
        self.tokens = int(tokens)
        self.expand = nn.Linear(signature_dim, self.tokens * signature_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, signature_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=signature_dim,
            nhead=heads,
            dim_feedforward=signature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(signature_dim)
        self.proj = nn.Linear(signature_dim, signature_dim)

    def forward(self, rest_sig):
        b = rest_sig.size(0)
        tokens = self.expand(rest_sig).view(b, self.tokens, rest_sig.size(1))
        query = self.query.expand(b, -1, -1)
        out = self.encoder(torch.cat([query, tokens], dim=1))[:, 0]
        return F.normalize(self.proj(self.norm(out)), dim=1)


class R2TNet(nn.Module):
    def __init__(self, targets, **hparams):
        super().__init__()
        self.hparams = hparams
        self.signature_dim = int(hparams.get("signature_dim", 1024))

        targets_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
        if targets_np.ndim == 1:
            targets_np = targets_np.reshape(-1, 1)
        self.target_dim = targets_np.shape[1]
        self.target_names = _as_target_names(hparams.get("target_names"), self.target_dim)
        self.hparams["target_dim"] = self.target_dim
        self.hparams["target_names"] = self.target_names

        swift_config = SwiFTConfig(
            patch_size=tuple(hparams.get("swift_patch", [16, 16, 16, 10])),
            token_dim=hparams.get("token_dim", 768),
            stage_depths=tuple(hparams.get("swift_stage_depths", [2, 2, 2])),
            stage_heads=tuple(hparams.get("swift_stage_heads", [8, 8, 8])),
            global_depth=hparams.get("swift_global_depth", 2),
            global_heads=hparams.get("swift_global_heads", 8),
            dropout=hparams.get("encoder_dropout", 0.1),
        )
        self.swift_encoder = SwiFTEncoder(swift_config, latent_dim=self.signature_dim)
        self.temporal_vit = TemporalViTEncoder(
            token_dim=hparams.get("token_dim", 768),
            depth=hparams.get("vit_depth", 12),
            num_heads=hparams.get("vit_heads", 12),
            dropout=hparams.get("encoder_dropout", 0.1),
            latent_dim=self.signature_dim,
        )
        self.temporal_gru = TemporalGRUEncoder(
            token_dim=hparams.get("token_dim", 768),
            depth=hparams.get("gru_depth", 2),
            dropout=hparams.get("encoder_dropout", 0.1),
            latent_dim=self.signature_dim,
            bidirectional=hparams.get("gru_bidirectional", True),
        )
        self.temporal_conv = TemporalConvEncoder(
            token_dim=hparams.get("token_dim", 768),
            depth=hparams.get("conv_depth", 4),
            kernel_size=hparams.get("conv_kernel", 7),
            dropout=hparams.get("encoder_dropout", 0.1),
            latent_dim=self.signature_dim,
        )
        self.temporal_mean = TemporalMeanEncoder(
            token_dim=hparams.get("token_dim", 768),
            dropout=hparams.get("encoder_dropout", 0.1),
            latent_dim=self.signature_dim,
        )
        self.modality_emb = nn.Embedding(2, hparams.get("token_dim", 768))
        if hparams.get("synthetic_mapper", "cmt") == "mlp":
            self.synthetic_mapper = nn.Sequential(
                nn.LayerNorm(self.signature_dim),
                nn.Linear(self.signature_dim, self.signature_dim),
                nn.GELU(),
                nn.Dropout(hparams.get("synthetic_dropout", 0.1)),
                nn.Linear(self.signature_dim, self.signature_dim),
            )
        else:
            self.synthetic_mapper = SyntheticTaskMapper(
                self.signature_dim,
                hparams.get("synthetic_depth", 2),
                hparams.get("synthetic_heads", 8),
                hparams.get("synthetic_dropout", 0.1),
                hparams.get("synthetic_tokens", 4),
            )

        specs = parse_head_specs(hparams.get("head_spec", ""), self.target_names, self.target_dim, hparams.get("downstream_task_type", "regression"))
        self.head_specs = specs
        self.hparams["head_layout"] = [{"name": s.name, "indices": list(s.indices), "task": s.task, "weight": s.weight} for s in specs]
        self.pred_head = MultiTaskHead(self.signature_dim, targets_np, self.target_names, specs, hparams)
        self.pair_concat = nn.Sequential(nn.LayerNorm(self.signature_dim * 2), nn.Linear(self.signature_dim * 2, self.signature_dim))
        self.pair_gate = nn.Sequential(nn.LayerNorm(self.signature_dim * 2), nn.Linear(self.signature_dim * 2, self.signature_dim), nn.Sigmoid())
        projector = hparams.get("contrastive_projector", "none")
        contrast_dim = int(hparams.get("contrastive_dim", 0) or self.signature_dim)
        hidden_dim = int(hparams.get("contrastive_projector_hidden", 0) or self.signature_dim)
        if projector == "linear":
            self.contrastive_projector = nn.Linear(self.signature_dim, contrast_dim)
            self.contrastive_dim = contrast_dim
        elif projector == "mlp":
            self.contrastive_projector = nn.Sequential(
                nn.LayerNorm(self.signature_dim),
                nn.Linear(self.signature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, contrast_dim),
            )
            self.contrastive_dim = contrast_dim
        else:
            self.contrastive_projector = nn.Identity()
            self.contrastive_dim = self.signature_dim
        queue_size = int(hparams.get("contrastive_queue_size", 0) or 0)
        self.register_buffer("contrastive_queue", torch.empty(queue_size, self.contrastive_dim))
        self.register_buffer("contrastive_queue_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("contrastive_queue_fill", torch.zeros((), dtype=torch.long))

        if hparams.get("freeze_encoder", False):
            for module in (self.swift_encoder, self.temporal_vit, self.temporal_gru, self.temporal_conv, self.temporal_mean, self.modality_emb):
                for param in module.parameters():
                    param.requires_grad = False

    def encode(self, x, modality):
        modality_embed = self.modality_emb(modality.to(x.device, dtype=torch.long))
        if x.ndim == 6:
            return self.swift_encoder(x, modality_embed)
        if x.ndim == 3:
            encoder = self.hparams.get("temporal_encoder", "vit")
            if encoder == "gru":
                return self.temporal_gru(x, modality_embed)
            if encoder == "conv":
                return self.temporal_conv(x, modality_embed)
            if encoder == "mean":
                return self.temporal_mean(x, modality_embed)
            return self.temporal_vit(x, modality_embed)
        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

    def forward(self, x, modality):
        signature = self.encode(x, modality)
        return signature, self.pred_head(signature)["prediction"]

    def inverse_scale(self, prediction):
        return self.pred_head.inverse_scale(prediction)

    def _contrastive_views(self, rest, task):
        return self.contrastive_projector(rest), self.contrastive_projector(task)

    def _contrastive_queue(self):
        fill = int(self.contrastive_queue_fill.item()) if self.contrastive_queue.numel() else 0
        if fill <= 0:
            return None
        return F.normalize(self.contrastive_queue[:fill], dim=1)

    @torch.no_grad()
    def _enqueue_contrastive(self, rest, task):
        if self.contrastive_queue.numel() == 0 or not self.training:
            return
        values = F.normalize(torch.cat([rest, task], dim=0), dim=1).detach()
        size = self.contrastive_queue.size(0)
        if values.size(0) >= size:
            self.contrastive_queue.copy_(values[-size:])
            self.contrastive_queue_ptr.zero_()
            self.contrastive_queue_fill.fill_(size)
            return
        ptr = int(self.contrastive_queue_ptr.item())
        end = ptr + values.size(0)
        if end <= size:
            self.contrastive_queue[ptr:end].copy_(values)
        else:
            first = size - ptr
            self.contrastive_queue[ptr:].copy_(values[:first])
            self.contrastive_queue[: end - size].copy_(values[first:])
        self.contrastive_queue_ptr.fill_(end % size)
        self.contrastive_queue_fill.fill_(min(size, int(self.contrastive_queue_fill.item()) + values.size(0)))

    def _contrastive_labels(self, batch, device):
        return torch.cat([torch.arange(batch, 2 * batch, device=device), torch.arange(batch, device=device)])

    def _target_false_negative_mask(self, target, batch, device):
        quantile = float(self.hparams.get("contrastive_target_mask_quantile", 0.0) or 0.0)
        if target is None or quantile <= 0.0 or batch <= 1:
            return None
        target = target.detach().float().to(device)
        if target.ndim == 1:
            target = target.unsqueeze(1)
        dist = torch.cdist(target, target)
        eye = torch.eye(batch, device=device, dtype=torch.bool)
        values = dist.masked_select(~eye)
        if values.numel() == 0:
            return None
        cutoff = torch.quantile(values, min(max(quantile, 0.0), 1.0))
        similar = (dist <= cutoff) & ~eye
        subject_idx = torch.cat([torch.arange(batch, device=device), torch.arange(batch, device=device)])
        mask = similar[subject_idx[:, None], subject_idx[None, :]]
        labels = self._contrastive_labels(batch, device)
        mask[torch.arange(2 * batch, device=device), labels] = False
        mask.fill_diagonal_(False)
        return mask

    def _apply_target_mask(self, sim, target, batch):
        mask = self._target_false_negative_mask(target, batch, sim.device)
        if mask is not None:
            sim = sim.clone()
            sim[:, : 2 * batch] = sim[:, : 2 * batch].masked_fill(mask, float("-inf"))
        return sim

    def _contrast_candidates(self, embeddings):
        queue = self._contrastive_queue()
        return torch.cat([embeddings, queue], dim=0) if queue is not None else embeddings

    def _info_nce_loss(self, rest, task, target, hard_topk = None):
        batch = rest.size(0)
        embeddings = torch.cat([rest, task], dim=0)
        candidates = self._contrast_candidates(embeddings)
        sim = torch.matmul(embeddings, candidates.t()) / self.hparams.get("temperature", 0.07)
        sim = sim.masked_fill(torch.eye(2 * batch, device=sim.device, dtype=torch.bool), float("-inf"))
        sim = self._apply_target_mask(sim, target, batch)
        labels = self._contrastive_labels(batch, sim.device)
        if hard_topk is None:
            return F.cross_entropy(sim, labels)
        pos = sim[torch.arange(2 * batch, device=sim.device), labels]
        neg = sim.clone()
        neg[torch.arange(2 * batch, device=sim.device), labels] = float("-inf")
        finite = torch.isfinite(neg)
        if not bool(finite.any()):
            return pos.sum() * 0.0
        k = min(max(int(hard_topk), 1), int(finite.sum(dim=1).max().item()))
        hard = neg.topk(k, dim=1).values
        logits = torch.cat([pos.unsqueeze(1), hard], dim=1)
        return F.cross_entropy(logits, torch.zeros(2 * batch, device=sim.device, dtype=torch.long))

    def _symmetric_ce_loss(self, rest, task, target):
        batch = rest.size(0)
        queue = self._contrastive_queue()
        rest_candidates = torch.cat([task, queue], dim=0) if queue is not None else task
        task_candidates = torch.cat([rest, queue], dim=0) if queue is not None else rest
        left = torch.matmul(rest, rest_candidates.t()) / self.hparams.get("temperature", 0.07)
        right = torch.matmul(task, task_candidates.t()) / self.hparams.get("temperature", 0.07)
        mask = self._target_false_negative_mask(target, batch, rest.device)
        if mask is not None:
            left = left.clone()
            right = right.clone()
            left[:, :batch] = left[:, :batch].masked_fill(mask[:batch, batch:], float("-inf"))
            right[:, :batch] = right[:, :batch].masked_fill(mask[batch:, :batch], float("-inf"))
        labels = torch.arange(batch, device=rest.device)
        return 0.5 * (F.cross_entropy(left, labels) + F.cross_entropy(right, labels))

    def _margin_loss(self, rest, task, target):
        batch = rest.size(0)
        embeddings = torch.cat([rest, task], dim=0)
        candidates = self._contrast_candidates(embeddings)
        raw = torch.matmul(embeddings, candidates.t())
        raw = raw.masked_fill(torch.eye(2 * batch, device=raw.device, dtype=torch.bool), float("-inf"))
        raw = self._apply_target_mask(raw, target, batch)
        labels = self._contrastive_labels(batch, raw.device)
        pos = raw[torch.arange(2 * batch, device=raw.device), labels]
        neg = raw.clone()
        neg[torch.arange(2 * batch, device=raw.device), labels] = float("-inf")
        hardest = neg.max(dim=1).values
        valid = torch.isfinite(hardest)
        if not bool(valid.any()):
            return pos.sum() * 0.0
        return F.relu(self.hparams.get("contrastive_margin", 0.2) - pos[valid] + hardest[valid]).mean()

    def _debiased_contrastive_loss(self, rest, task, target):
        batch = rest.size(0)
        embeddings = torch.cat([rest, task], dim=0)
        candidates = self._contrast_candidates(embeddings)
        sim = torch.matmul(embeddings, candidates.t()) / self.hparams.get("temperature", 0.07)
        labels = self._contrastive_labels(batch, sim.device)
        pos_sim = sim[torch.arange(2 * batch, device=sim.device), labels]
        pos = torch.exp(pos_sim)
        neg_sim = sim.masked_fill(torch.eye(2 * batch, device=sim.device, dtype=torch.bool), float("-inf"))
        neg_sim[torch.arange(2 * batch, device=sim.device), labels] = float("-inf")
        neg_sim = self._apply_target_mask(neg_sim, target, batch)
        finite = torch.isfinite(neg_sim)
        neg = torch.exp(neg_sim.masked_fill(~finite, float("-inf"))).masked_fill(~finite, 0.0).sum(dim=1)
        n_neg = finite.sum(dim=1).clamp_min(1).to(neg.dtype)
        tau_plus = float(self.hparams.get("contrastive_tau_plus", 0.1))
        temperature = float(self.hparams.get("temperature", 0.07))
        debiased = (-tau_plus * n_neg * pos + neg) / max(1.0 - tau_plus, 1e-6)
        debiased = debiased.clamp_min(n_neg * math.exp(-1.0 / temperature))
        return (-torch.log(pos / (pos + debiased + 1e-12))).mean()

    def _barlow_twins_loss(self, rest, task):
        batch = rest.size(0)
        rest = (rest - rest.mean(dim=0)) / rest.std(dim=0, unbiased=False).clamp_min(1e-4)
        task = (task - task.mean(dim=0)) / task.std(dim=0, unbiased=False).clamp_min(1e-4)
        corr = torch.matmul(rest.t(), task) / max(batch, 1)
        diag = torch.diagonal(corr).add(-1.0).pow(2).sum()
        off = corr.flatten()[:-1].view(corr.size(0) - 1, corr.size(0) + 1)[:, 1:].flatten().pow(2).sum()
        return diag + self.hparams.get("contrastive_barlow_lambda", 0.005) * off

    def _vicreg_loss(self, rest, task):
        repr_loss = F.mse_loss(rest, task)
        rest_std = torch.sqrt(rest.var(dim=0, unbiased=False) + 1e-4)
        task_std = torch.sqrt(task.var(dim=0, unbiased=False) + 1e-4)
        std_loss = 0.5 * (F.relu(1.0 - rest_std).mean() + F.relu(1.0 - task_std).mean())
        denom = max(rest.size(0) - 1, 1)
        rest = rest - rest.mean(dim=0)
        task = task - task.mean(dim=0)
        rest_cov = torch.matmul(rest.t(), rest) / denom
        task_cov = torch.matmul(task.t(), task) / denom
        rest_off = rest_cov.flatten()[:-1].view(rest_cov.size(0) - 1, rest_cov.size(0) + 1)[:, 1:].flatten()
        task_off = task_cov.flatten()[:-1].view(task_cov.size(0) - 1, task_cov.size(0) + 1)[:, 1:].flatten()
        cov_loss = (rest_off.pow(2).sum() + task_off.pow(2).sum()) / rest.size(1)
        return (
            self.hparams.get("vicreg_sim_coeff", 25.0) * repr_loss
            + self.hparams.get("vicreg_std_coeff", 25.0) * std_loss
            + self.hparams.get("vicreg_cov_coeff", 1.0) * cov_loss
        )

    def _contrastive_loss(self, rest, task, target = None):
        rest_z, task_z = self._contrastive_views(rest, task)
        loss_name = self.hparams.get("contrastive_loss", "ntxent")
        if loss_name == "barlow_twins":
            return self._barlow_twins_loss(rest_z, task_z)
        if loss_name == "vicreg":
            return self._vicreg_loss(rest_z, task_z)

        rest_z = F.normalize(rest_z, dim=1)
        task_z = F.normalize(task_z, dim=1)
        if loss_name == "cosine":
            loss = 1.0 - F.cosine_similarity(rest_z, task_z, dim=1).mean()
        elif loss_name in {"symmetric_ce", "clip"}:
            loss = self._symmetric_ce_loss(rest_z, task_z, target)
        elif loss_name in {"margin", "triplet"}:
            loss = self._margin_loss(rest_z, task_z, target)
        elif loss_name in {"dcl", "debiased"}:
            loss = self._debiased_contrastive_loss(rest_z, task_z, target)
        elif loss_name in {"hard_ntxent", "hard_infonce"}:
            loss = self._info_nce_loss(rest_z, task_z, target, self.hparams.get("hard_negative_topk", 16))
        else:
            loss = self._info_nce_loss(rest_z, task_z, target)
        self._enqueue_contrastive(rest_z, task_z)
        return loss

    def supervised_loss(self, head_out, target):
        loss, _ = self.pred_head.loss(head_out, target, self.hparams)
        return loss

    def supervised_loss_with_logs(self, head_out, target):
        return self.pred_head.loss(head_out, target, self.hparams)

    def fuse_pair(self, rest_sig, task_sig):
        mode = self.hparams.get("pair_fusion", "auto")
        if mode == "auto":
            mode = self.hparams.get("supervised_view", "average")
        if mode == "rest":
            return rest_sig
        if mode == "task":
            return task_sig
        if mode == "sum":
            return F.normalize(rest_sig + task_sig, dim=1)
        if mode == "concat":
            return F.normalize(self.pair_concat(torch.cat([rest_sig, task_sig], dim=1)), dim=1)
        if mode == "gated":
            gate = self.pair_gate(torch.cat([rest_sig, task_sig], dim=1))
            return F.normalize(gate * rest_sig + (1.0 - gate) * task_sig, dim=1)
        return F.normalize((rest_sig + task_sig) * 0.5, dim=1)

    def _augment(self, x):
        if not self.training:
            return x
        ratio = self.hparams.get("temporal_crop_min_ratio", 1.0)
        if ratio < 1.0:
            seq_len = x.size(-1)
            crop_len = random.randint(max(int(seq_len * ratio), 1), seq_len)
            start = random.randint(0, seq_len - crop_len)
            x = F.pad(x[..., start : start + crop_len], (0, seq_len - crop_len))
        std = self.hparams.get("gaussian_noise_std", 0.0)
        prob = self.hparams.get("gaussian_noise_p", 0.0)
        if std > 0 and prob > 0:
            mask = (torch.rand(x.shape[:-1], device=x.device) < prob).unsqueeze(-1).to(x.dtype)
            x = x + torch.randn_like(x) * std * mask
        temporal_mask_p = self.hparams.get("temporal_mask_p", 0.0)
        if temporal_mask_p > 0:
            shape = (x.size(0),) + (1,) * (x.ndim - 2) + (x.size(-1),)
            x = x * (torch.rand(shape, device=x.device) >= temporal_mask_p).to(x.dtype)
        feature_mask_p = self.hparams.get("feature_mask_p", 0.0)
        if feature_mask_p > 0:
            x = x * (torch.rand(x.shape[:-1] + (1,), device=x.device) >= feature_mask_p).to(x.dtype)
        return x

    def training_loss(self, batch):
        if "fmri" in batch:
            x = self._augment(batch["fmri"])
            modality = batch.get("modality", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
            signature = self.encode(x, modality)
            sup, sup_logs = self.supervised_loss_with_logs(self.pred_head(signature), batch["target"])
            return sup, sup_logs

        rest = self._augment(batch["rest"])
        task = self._augment(batch["task"])
        rest_mod = torch.zeros(rest.size(0), dtype=torch.long, device=rest.device)
        task_mod = torch.ones(task.size(0), dtype=torch.long, device=task.device)
        if self.training and self.hparams.get("modality_dropout_p", 0.0) > 0 and random.random() < self.hparams.get("modality_dropout_p", 0.0):
            shared_mod = random.randint(0, 1)
            rest_mod.fill_(shared_mod)
            task_mod.fill_(shared_mod)
        rest_sig = self.encode(rest, rest_mod)
        task_sig = self.encode(task, task_mod)
        zero = rest_sig.new_zeros(())
        con = self._contrastive_loss(rest_sig, task_sig, batch.get("target")) if not self.hparams.get("disable_contrastive", False) else zero

        synth = zero
        if self.hparams.get("training_mode") == "synthetic_task":
            signature = F.normalize(self.synthetic_mapper(rest_sig), dim=1)
            synth = 1.0 - F.cosine_similarity(signature, task_sig.detach(), dim=1).mean()
            synth = synth + self.hparams.get("lambda_synthetic_l2", 0.0) * F.mse_loss(signature, task_sig.detach())
        else:
            signature = self.fuse_pair(rest_sig, task_sig)

        sup_logs = {"supervised": 0.0}
        if self.hparams.get("pretraining", False):
            sup = zero
        else:
            sup, sup_logs = self.supervised_loss_with_logs(self.pred_head(signature), batch["target"])
        total = sup + self.hparams.get("lambda_contrast", 0.5) * con + self.hparams.get("lambda_synthetic", 1.0) * synth
        return total, {
            **sup_logs,
            "contrastive": float(con.detach().cpu()),
            "synthetic": float(synth.detach().cpu()),
        }

    @staticmethod
    def add_model_specific_args(parent):
        p = ArgumentParser(parents=[parent], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        arch = p.add_argument_group("Architecture")
        arch.add_argument("--signature_dim", type=int, default=1024)
        arch.add_argument("--token_dim", type=int, default=768)
        arch.add_argument("--swift_patch", nargs=4, type=int, default=[16, 16, 16, 10])
        arch.add_argument("--swift_stage_depths", nargs=3, type=int, default=[2, 2, 2])
        arch.add_argument("--swift_stage_heads", nargs=3, type=int, default=[8, 8, 8])
        arch.add_argument("--swift_global_depth", type=int, default=2)
        arch.add_argument("--swift_global_heads", type=int, default=8)
        arch.add_argument("--encoder_dropout", type=float, default=0.1)
        arch.add_argument("--vit_depth", type=int, default=12)
        arch.add_argument("--vit_heads", type=int, default=12)
        arch.add_argument("--temporal_encoder", choices=["vit", "gru", "conv", "mean"], default="vit")
        arch.add_argument("--gru_depth", type=int, default=2)
        arch.add_argument("--gru_bidirectional", action=argparse.BooleanOptionalAction, default=True)
        arch.add_argument("--conv_depth", type=int, default=4)
        arch.add_argument("--conv_kernel", type=int, default=7)
        arch.add_argument("--pred_hidden_dim", type=int, default=512)
        arch.add_argument("--head_dropout", type=float, default=0.1)
        arch.add_argument("--head_depth", type=int, default=2)

        train = p.add_argument_group("Training")
        train.add_argument("--learning_rate", type=float, default=3e-4)
        train.add_argument("--weight_decay", type=float, default=1e-2)
        train.add_argument("--max_epochs", type=int, default=100)
        train.add_argument("--pretraining", action="store_true")
        train.add_argument("--disable_contrastive", action="store_true")
        train.add_argument("--freeze_encoder", action="store_true")
        train.add_argument("--supervised_view", choices=["rest", "task", "average"], default="average")
        train.add_argument("--pair_fusion", choices=["auto", "rest", "task", "average", "sum", "concat", "gated"], default="auto")
        train.add_argument("--lambda_contrast", type=float, default=0.5)
        train.add_argument("--lambda_synthetic", type=float, default=1.0)
        train.add_argument("--lambda_synthetic_l2", type=float, default=0.0)
        train.add_argument("--synthetic_mapper", choices=["cmt", "mlp"], default="cmt")
        train.add_argument("--synthetic_depth", type=int, default=2)
        train.add_argument("--synthetic_heads", type=int, default=8)
        train.add_argument("--synthetic_tokens", type=int, default=4)
        train.add_argument("--synthetic_dropout", type=float, default=0.1)
        train.add_argument("--temperature", type=float, default=0.07)
        train.add_argument(
            "--contrastive_loss",
            choices=[
                "ntxent",
                "infonce",
                "simclr",
                "symmetric_ce",
                "clip",
                "cosine",
                "margin",
                "triplet",
                "hard_ntxent",
                "hard_infonce",
                "dcl",
                "debiased",
                "barlow_twins",
                "vicreg",
            ],
            default="ntxent",
        )
        train.add_argument("--contrastive_margin", type=float, default=0.2)
        train.add_argument("--contrastive_projector", choices=["none", "linear", "mlp"], default="none")
        train.add_argument("--contrastive_dim", type=int, default=0, help="Projection dimension; 0 keeps signature_dim")
        train.add_argument("--contrastive_projector_hidden", type=int, default=0, help="MLP projector hidden dim; 0 keeps signature_dim")
        train.add_argument("--contrastive_queue_size", type=int, default=0, help="Cross-batch memory queue size for negative-based losses")
        train.add_argument("--hard_negative_topk", type=int, default=16)
        train.add_argument("--contrastive_tau_plus", type=float, default=0.1, help="Class-prior estimate for debiased contrastive loss")
        train.add_argument("--contrastive_target_mask_quantile", type=float, default=0.0, help="Mask closest target-neighbor negatives within a batch")
        train.add_argument("--contrastive_barlow_lambda", type=float, default=0.005)
        train.add_argument("--vicreg_sim_coeff", type=float, default=25.0)
        train.add_argument("--vicreg_std_coeff", type=float, default=25.0)
        train.add_argument("--vicreg_cov_coeff", type=float, default=1.0)
        train.add_argument("--temporal_crop_min_ratio", type=float, default=0.8)
        train.add_argument("--gaussian_noise_std", type=float, default=0.01)
        train.add_argument("--gaussian_noise_p", type=float, default=0.1)
        train.add_argument("--temporal_mask_p", type=float, default=0.0)
        train.add_argument("--feature_mask_p", type=float, default=0.0)
        train.add_argument("--modality_dropout_p", type=float, default=0.0)
        train.add_argument("--grad_clip_norm", type=float, default=1.0)

        downstream = p.add_argument_group("Downstream")
        downstream.add_argument("--downstream_task_type", choices=["regression", "classification"], default="regression")
        downstream.add_argument("--head_spec", default="", help="name:cols[:regression|classification[:weight]]; e.g. score:wm_0bk,wm_2bk:regression:1;rt:rt_wm,rt_rel:regression:0.3")
        downstream.add_argument("--classification_loss", choices=["auto", "bce", "cross_entropy"], default="auto")
        downstream.add_argument("--classification_label_smoothing", type=float, default=0.0)
        downstream.add_argument("--label_scaling_method", choices=["standardization", "minmax", "none"], default="standardization")
        downstream.add_argument("--reg_head", choices=["yolo", "mlp", "linear"], default="yolo")
        downstream.add_argument("--reg_loss", choices=["huber", "mse", "l1"], default="huber")
        downstream.add_argument("--reg_num_bins", type=int, default=8)
        downstream.add_argument("--reg_binning_strategy", choices=["quantile", "uniform"], default="quantile")
        downstream.add_argument("--reg_alpha", type=float, default=1.0)
        downstream.add_argument("--reg_beta", type=float, default=2.0)
        downstream.add_argument("--reg_temperature", type=float, default=1.0)
        downstream.add_argument("--reg_label_smoothing", type=float, default=0.0)
        return p
