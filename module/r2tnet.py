"""Lightning implementation of R2T-Net."""

from __future__ import annotations

import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import Tensor, nn

import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy, MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef

from .models.swift_encoder import SwiFTConfig, SwiFTEncoder
from .models.temporal_vit import TemporalViTEncoder
from .utils.lr_scheduler import CosineAnnealingWarmUpRestarts


class PredictionHead(nn.Module):
    """Two-layer MLP used for behavioural predictions."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class R2TNet(pl.LightningModule):
    """Contrastive R2T-Net with SwiFT + Temporal ViT encoders."""

    signature_dim = 2048

    def __init__(self, data_module, **hparams) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["data_module"])

        # ---------------------------------------------------------------
        # Target scaling (regression only)
        # ---------------------------------------------------------------
        targets = data_module.train_dataset.target_values
        if isinstance(targets, torch.Tensor):
            targets_np = targets.cpu().numpy()
        else:
            targets_np = targets
        if targets_np.ndim == 1:
            targets_np = targets_np.reshape(-1, 1)
        self.target_dim = targets_np.shape[1]

        self.scaler_type = None
        if self.hparams.downstream_task_type == "regression":
            if self.hparams.label_scaling_method == "standardization":
                scaler = StandardScaler().fit(targets_np)
                self.scaler_type = "standard"
                self.register_buffer("target_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
                self.register_buffer("target_scale", torch.tensor(scaler.scale_, dtype=torch.float32))
            else:
                scaler = MinMaxScaler().fit(targets_np)
                self.scaler_type = "minmax"
                self.register_buffer("target_min", torch.tensor(scaler.data_min_, dtype=torch.float32))
                self.register_buffer("target_range", torch.tensor(scaler.data_max_ - scaler.data_min_, dtype=torch.float32))

        # ---------------------------------------------------------------
        # Encoders
        # ---------------------------------------------------------------
        swift_config = SwiFTConfig(
            patch_size=tuple(self.hparams.swift_patch),
            token_dim=self.hparams.token_dim,
            stage_depths=tuple(self.hparams.swift_stage_depths),
            stage_heads=tuple(self.hparams.swift_stage_heads),
            global_depth=self.hparams.swift_global_depth,
            global_heads=self.hparams.swift_global_heads,
            dropout=self.hparams.encoder_dropout,
        )
        self.swift_encoder = SwiFTEncoder(swift_config, latent_dim=self.signature_dim)
        self.vit_encoder = TemporalViTEncoder(
            token_dim=self.hparams.token_dim,
            depth=self.hparams.vit_depth,
            num_heads=self.hparams.vit_heads,
            dropout=self.hparams.encoder_dropout,
            latent_dim=self.signature_dim,
        )

        self.modality_emb = nn.Embedding(2, self.hparams.token_dim)

        # ---------------------------------------------------------------
        # Prediction head
        # ---------------------------------------------------------------
        self.pred_head = PredictionHead(
            in_dim=self.signature_dim,
            hidden_dim=self.hparams.pred_hidden_dim,
            out_dim=self.target_dim,
            dropout=self.hparams.head_dropout,
        )

        # Metrics --------------------------------------------------------
        self.metrics = self._init_metrics()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _init_metrics(self) -> Dict[str, pl.metrics.Metric]:
        if self.hparams.downstream_task_type == "classification":
            return {
                "bal_acc": Accuracy(task="binary", average="macro", threshold=0.5),
                "auroc": AUROC(task="binary"),
            }
        return {
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "r": PearsonCorrCoef(),
        }

    def encode(self, x: Tensor, modality: Tensor) -> Tensor:
        modality = modality.to(x.device, dtype=torch.long)
        modality_embed = self.modality_emb(modality)
        if x.ndim == 6:
            return self.swift_encoder(x, modality_embed)
        if x.ndim == 3:
            return self.vit_encoder(x, modality_embed)
        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

    def forward(self, x: Tensor, modality: Tensor) -> Tuple[Tensor, Tensor]:
        signature = self.encode(x, modality)
        prediction = self.pred_head(signature)
        return signature, prediction

    # ------------------------------------------------------------------
    # data augmentation helpers
    # ------------------------------------------------------------------
    def _temporal_crop(self, x: Tensor) -> Tensor:
        """Randomly crop 80–100% of the time dimension and zero-pad back."""

        if not self.training or self.hparams.temporal_crop_min_ratio >= 1.0:
            return x

        seq_len = x.size(-1)
        min_len = max(int(seq_len * self.hparams.temporal_crop_min_ratio), 1)
        if min_len >= seq_len:
            return x

        crop_len = random.randint(min_len, seq_len)
        start = random.randint(0, seq_len - crop_len)
        cropped = x[..., start : start + crop_len]
        if crop_len == seq_len:
            return cropped

        pad = seq_len - crop_len
        return F.pad(cropped, (0, pad))

    def _gaussian_noise(self, x: Tensor) -> Tensor:
        """Inject Gaussian noise on a random subset of voxels/tokens."""

        if not self.training or self.hparams.gaussian_noise_std <= 0:
            return x

        if self.hparams.gaussian_noise_p <= 0:
            return x

        noise = torch.randn_like(x) * self.hparams.gaussian_noise_std
        mask_shape = x.shape[:-1]
        noise_mask = torch.rand(mask_shape, device=x.device) < self.hparams.gaussian_noise_p
        noise_mask = noise_mask.unsqueeze(-1).to(x.dtype)
        return x + noise * noise_mask

    def _augment(self, x: Tensor) -> Tensor:
        x = self._temporal_crop(x)
        x = self._gaussian_noise(x)
        return x

    def _maybe_modality_dropout(
        self,
        rest_x: Tensor,
        rest_mod: Tensor,
        task_x: Tensor,
        task_mod: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Randomly replace one modality to encourage invariance."""

        if not self.training:
            return rest_x, rest_mod, task_x, task_mod

        if random.random() >= self.hparams.modality_dropout_prob:
            return rest_x, rest_mod, task_x, task_mod

        if random.random() < 0.5:
            task_x = rest_x.clone()
            task_mod = rest_mod.clone()
        else:
            rest_x = task_x.clone()
            rest_mod = task_mod.clone()
        return rest_x, rest_mod, task_x, task_mod

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------
    def _contrastive_loss(self, rest: Tensor, task: Tensor) -> Tensor:
        rest = F.normalize(rest, dim=1)
        task = F.normalize(task, dim=1)
        embeddings = torch.cat([rest, task], dim=0)

        sim = torch.matmul(embeddings, embeddings.t()) / self.hparams.temperature
        batch = rest.size(0)
        mask = torch.eye(2 * batch, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))
        positives = torch.cat(
            [torch.arange(batch, 2 * batch, device=sim.device), torch.arange(batch, device=sim.device)]
        )
        loss = F.cross_entropy(sim, positives)
        return loss

    def _scale_targets(self, target: Tensor) -> Tensor:
        if self.scaler_type is None:
            return target
        if self.scaler_type == "standard":
            mean = self.target_mean.to(target.device)
            scale = self.target_scale.to(target.device)
            return (target - mean) / scale
        min_ = self.target_min.to(target.device)
        range_ = self.target_range.to(target.device)
        return (target - min_) / range_

    def _supervised_loss(self, preds: Tensor, target: Tensor) -> Tensor:
        if self.hparams.downstream_task_type == "classification":
            return F.binary_cross_entropy_with_logits(preds, target)
        return F.mse_loss(preds, target)

    # ------------------------------------------------------------------
    # training / validation
    # ------------------------------------------------------------------
    def _extract_views(self, batch) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if {"rest", "task"}.issubset(batch.keys()):
            rest_x = batch["rest"]
            task_x = batch["task"]
        else:
            rest_x = batch["fmri1"]
            task_x = batch["fmri2"]

        device = rest_x.device
        rest_mod = torch.zeros(rest_x.size(0), dtype=torch.long, device=device)
        task_mod = torch.ones(task_x.size(0), dtype=torch.long, device=device)
        return rest_x, rest_mod, task_x, task_mod

    def training_step(self, batch, _):
        rest_x, rest_mod, task_x, task_mod = self._extract_views(batch)
        rest_x = self._augment(rest_x)
        task_x = self._augment(task_x)
        rest_x, rest_mod, task_x, task_mod = self._maybe_modality_dropout(rest_x, rest_mod, task_x, task_mod)
        rest_sig = self.encode(rest_x, rest_mod)
        task_sig = self.encode(task_x, task_mod)

        contrastive_loss = self._contrastive_loss(rest_sig, task_sig)
        self.log("train_contrastive_loss", contrastive_loss, prog_bar=True, batch_size=rest_x.size(0))

        if self.hparams.pretraining:
            return contrastive_loss

        subject_sig = F.normalize((rest_sig + task_sig) * 0.5, dim=1)
        preds = self.pred_head(subject_sig)

        target = batch["target"].float()
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        if self.hparams.downstream_task_type != "classification":
            target = self._scale_targets(target)

        sup_loss = self._supervised_loss(preds, target)
        self.log("train_supervised_loss", sup_loss, prog_bar=True, batch_size=rest_x.size(0))

        total_loss = sup_loss + self.hparams.lambda_contrast * contrastive_loss
        return total_loss

    def _run_supervised(self, batch) -> Tuple[Tensor, Tensor]:
        x = batch["fmri"]
        modality = batch.get("modality")
        if modality is None:
            modality = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        signature = self.encode(x, modality)
        preds = self.pred_head(signature)
        return preds, signature

    def validation_step(self, batch, _):
        preds, _ = self._run_supervised(batch)
        target = batch["target"].float()
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        if self.hparams.downstream_task_type != "classification":
            target = self._scale_targets(target)

        loss = self._supervised_loss(preds, target)
        self.log("valid_loss", loss, prog_bar=False, batch_size=target.size(0))

        if self.hparams.downstream_task_type == "classification":
            probs = torch.sigmoid(preds)
            self.metrics["bal_acc"].update(probs, target.int())
            self.metrics["auroc"].update(probs, target.int())
        else:
            self.metrics["mae"].update(preds, target)
            self.metrics["mse"].update(preds, target)
            self.metrics["r"].update(preds, target)

    def on_validation_epoch_end(self):
        for name, metric in self.metrics.items():
            self.log(f"valid_{name}", metric.compute(), prog_bar=True)
            metric.reset()

    test_step = validation_step
    on_test_epoch_end = on_validation_epoch_end

    # ------------------------------------------------------------------
    # optimisers / schedulers
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        encoder_params: Iterable[Tensor] = []
        head_params: Iterable[Tensor] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("pred_head"):
                head_params.append(param)
            else:
                encoder_params.append(param)

        encoder_params = list(encoder_params)
        head_params = list(head_params)

        param_groups = [{"params": encoder_params, "weight_decay": 0.0}]
        if head_params:
            param_groups.append({"params": head_params, "weight_decay": self.hparams.weight_decay})

        opt = torch.optim.AdamW(
            param_groups,
            lr=self.hparams.learning_rate,
        )
        if not self.hparams.use_scheduler:
            return opt

        total_steps = self.hparams.total_steps
        warmup_steps = int(total_steps * self.hparams.warmup_pct)
        sched = CosineAnnealingWarmUpRestarts(
            opt,
            first_cycle_steps=total_steps,
            max_lr=self.hparams.learning_rate,
            min_lr=self.hparams.min_lr,
            warmup_steps=warmup_steps,
            gamma=self.hparams.gamma,
        )
        return [opt], [{"scheduler": sched, "interval": "step"}]

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.grad_clip_norm)

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    @staticmethod
    def add_model_specific_args(parent: ArgumentParser):
        p = ArgumentParser(
            parents=[parent], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter
        )

        arch = p.add_argument_group("Architecture")
        arch.add_argument("--token_dim", type=int, default=768)
        arch.add_argument("--swift_patch", nargs=4, type=int, default=[16, 16, 16, 10])
        arch.add_argument("--swift_stage_depths", nargs=3, type=int, default=[2, 2, 2])
        arch.add_argument("--swift_stage_heads", nargs=3, type=int, default=[8, 8, 8])
        arch.add_argument("--swift_global_depth", type=int, default=2)
        arch.add_argument("--swift_global_heads", type=int, default=8)
        arch.add_argument("--encoder_dropout", type=float, default=0.1)
        arch.add_argument("--vit_depth", type=int, default=12)
        arch.add_argument("--vit_heads", type=int, default=12)
        arch.add_argument("--pred_hidden_dim", type=int, default=512)
        arch.add_argument("--head_dropout", type=float, default=0.1)

        train = p.add_argument_group("Training")
        train.add_argument("--learning_rate", type=float, default=3e-4)
        train.add_argument("--weight_decay", type=float, default=1e-2)
        train.add_argument("--use_scheduler", action="store_true")
        train.add_argument("--warmup_pct", type=float, default=50 / 200)
        train.add_argument("--total_steps", type=int, default=5000)
        train.add_argument("--min_lr", type=float, default=1e-5)
        train.add_argument("--gamma", type=float, default=0.99)
        train.add_argument("--pretraining", action="store_true")
        train.add_argument("--lambda_contrast", type=float, default=0.5)
        train.add_argument("--temperature", type=float, default=0.07)
        train.add_argument("--temporal_crop_min_ratio", type=float, default=0.8)
        train.add_argument("--gaussian_noise_std", type=float, default=0.01)
        train.add_argument("--gaussian_noise_p", type=float, default=0.1)
        train.add_argument("--modality_dropout_prob", type=float, default=0.2)
        train.add_argument("--grad_clip_norm", type=float, default=1.0)

        downstream = p.add_argument_group("Downstream")
        downstream.add_argument(
            "--downstream_task_type",
            choices=["regression", "classification"],
            default="regression",
        )
        downstream.add_argument(
            "--label_scaling_method",
            choices=["standardization", "minmax"],
            default="standardization",
        )

        return p

