"""Plain PyTorch trainer for R2T-Net."""

from __future__ import annotations

import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parameter import UninitializedParameter

from r2tnet.model import R2TNet
from r2tnet.data import fMRIDataModule


def build_parser():
    p = ArgumentParser(description="Train/test R2T-Net", formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out_dir", default="runs/r2tnet")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", choices=["fp32", "amp"], default="amp")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", default="")
    p.add_argument("--init_from", default="", help="Initialize compatible weights from a checkpoint without resuming optimizer or epoch")
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--optimizer", choices=["adamw", "adam", "sgd", "rmsprop"], default="adamw")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--scheduler", choices=["none", "cosine", "step", "onecycle"], default="none")
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--step_size", type=int, default=50)
    p.add_argument("--step_gamma", type=float, default=0.5)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--selection_metric", choices=["auto", "loss", "alignment"], default="auto")
    p = R2TNet.add_model_specific_args(p)
    p = fMRIDataModule.add_data_specific_args(p)
    return p


def ddp_setup(args):
    if not args.ddp:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    return rank, world_size, local_rank


def move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _subjects(value, n):
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] * n


def _corr(pred, target, prefix, names = None):
    pred = pred.float()
    target = target.float()
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)
    out = {}
    vals = []
    for j in range(pred.size(1)):
        mask = torch.isfinite(pred[:, j]) & torch.isfinite(target[:, j])
        if int(mask.sum()) < 3:
            continue
        p = pred[mask, j]
        y = target[mask, j]
        if float(p.std()) == 0.0 or float(y.std()) == 0.0:
            continue
        val = float(torch.corrcoef(torch.stack([p, y]))[0, 1])
        name = names[j] if names and j < len(names) else str(j)
        out[f"{prefix}_r_{name}"] = val
        vals.append(val)
    if vals:
        out[f"{prefix}_r"] = sum(vals) / len(vals)
    return out


def _classification_metrics(pred, target, specs, prefix):
    out = {}
    for spec in specs or []:
        if getattr(spec, "task", None) != "classification":
            continue
        cols = list(spec.indices)
        logits = pred[:, cols]
        y = target[:, cols]
        if len(cols) > 1:
            truth = y.argmax(dim=1)
            guess = logits.argmax(dim=1)
            n_classes = len(cols)
        else:
            truth = y.reshape(-1).long()
            guess = (logits.reshape(-1) > 0).long()
            n_classes = 2
        valid = torch.isfinite(truth.float())
        if int(valid.sum()) == 0:
            continue
        truth = truth[valid]
        guess = guess[valid]
        out[f"{prefix}_acc_{spec.name}"] = float((guess == truth).float().mean())
        recalls, f1s = [], []
        for cls in range(n_classes):
            tp = ((guess == cls) & (truth == cls)).sum().float()
            fp = ((guess == cls) & (truth != cls)).sum().float()
            fn = ((guess != cls) & (truth == cls)).sum().float()
            if int((truth == cls).sum()) > 0:
                recalls.append(tp / (tp + fn).clamp_min(1.0))
            if int(tp + fp + fn) > 0:
                f1s.append((2 * tp) / (2 * tp + fp + fn).clamp_min(1.0))
        if recalls:
            out[f"{prefix}_balanced_acc_{spec.name}"] = float(torch.stack(recalls).mean())
        if f1s:
            out[f"{prefix}_macro_f1_{spec.name}"] = float(torch.stack(f1s).mean())
    return out


def _regression_view(pred, target, specs, names):
    if not specs:
        return pred, target, names
    indices = [i for spec in specs if getattr(spec, "task", None) == "regression" for i in spec.indices]
    if not indices:
        return None, None, None
    return pred[:, indices], target[:, indices], [names[i] for i in indices] if names else None


def _subject_average(pred, target, subject_ids):
    buckets = defaultdict(list)
    target_ref = {}
    for sid, p, y in zip(subject_ids, pred, target):
        buckets[sid].append(p)
        target_ref[sid] = y
    keys = sorted(buckets)
    pred_avg = torch.stack([torch.stack(buckets[k]).mean(0) for k in keys])
    target_avg = torch.stack([target_ref[k] for k in keys])
    return pred_avg, target_avg


@torch.no_grad()
def evaluate_alignment(model, loader, device, amp):
    if loader is None:
        return {}
    model.eval()
    pos, neg, losses = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            rest = batch["rest"]
            task = batch["task"]
            rest_sig = torch.nn.functional.normalize(unwrap(model).encode(rest, torch.zeros(rest.size(0), dtype=torch.long, device=device)), dim=1)
            task_sig = torch.nn.functional.normalize(unwrap(model).encode(task, torch.ones(task.size(0), dtype=torch.long, device=device)), dim=1)
            sim = rest_sig @ task_sig.t()
            losses.append(unwrap(model)._contrastive_loss(rest_sig, task_sig, batch.get("target")).detach())
        diag = sim.diag()
        mask = ~torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        pos.append(diag.detach().float().cpu())
        if int(mask.sum()) > 0:
            neg.append(sim[mask].detach().float().cpu())
    if not pos:
        return {}
    pos_mean = float(torch.cat(pos).mean())
    neg_mean = float(torch.cat(neg).mean()) if neg else 0.0
    margin = pos_mean - neg_mean
    ratio = pos_mean / (neg_mean + 1e-6) if abs(neg_mean) > 1e-6 else margin
    return {
        "alignment_pos": pos_mean,
        "alignment_neg": neg_mean,
        "alignment_margin": margin,
        "alignment_score": float(ratio),
        "alignment_loss": float(torch.stack(losses).mean().cpu()) if losses else 0.0,
    }


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    losses = []
    preds, targets, subjects = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            if "fmri" in batch:
                signature = unwrap(model).encode(batch["fmri"], batch["modality"])
                head_out = unwrap(model).pred_head(signature)
                loss = unwrap(model).supervised_loss(head_out, batch["target"])
                pred = unwrap(model).inverse_scale(head_out["prediction"])
            else:
                loss, _ = unwrap(model).training_loss(batch)
                continue
        losses.append(loss.detach())
        preds.append(pred.detach().float().cpu())
        targets.append(batch["target"].detach().float().cpu())
        subjects.extend(_subjects(batch.get("subject_id", "NA"), pred.size(0)))
    if not losses:
        return {}
    out = {"loss": float(torch.stack(losses).mean().cpu())}
    if not preds:
        return out
    pred_all = torch.cat(preds)
    target_all = torch.cat(targets)
    names = getattr(unwrap(model), "target_names", None)
    specs = getattr(unwrap(model), "head_specs", None)
    out.update(_classification_metrics(pred_all, target_all, specs, "window"))
    reg_pred, reg_target, reg_names = _regression_view(pred_all, target_all, specs, names)
    if reg_pred is not None:
        out.update(_corr(reg_pred, reg_target, "window", reg_names))
    pred_subj, target_subj = _subject_average(pred_all, target_all, subjects)
    out.update(_classification_metrics(pred_subj, target_subj, specs, "subject"))
    reg_pred, reg_target, reg_names = _regression_view(pred_subj, target_subj, specs, names)
    if reg_pred is not None:
        out.update(_corr(reg_pred, reg_target, "subject", reg_names))
    return out


def selection_value(val, metric):
    if metric in {"alignment", "auto"} and "alignment_score" in val:
        return float(val.get("alignment_score", float("-inf"))), "alignment"
    return -float(val.get("loss", float("inf"))), "loss"


def save_checkpoint(path, model, optimizer, epoch, args, scheduler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    hparams = vars(args).copy()
    hparams["target_dim"] = unwrap(model).target_dim
    hparams["target_names"] = unwrap(model).target_names
    hparams["head_layout"] = unwrap(model).hparams.get("head_layout", [])
    state_dict = {
        key: value
        for key, value in unwrap(model).state_dict().items()
        if not isinstance(value, UninitializedParameter)
    }
    payload = {"epoch": epoch, "state_dict": state_dict, "optimizer": optimizer.state_dict(), "hparams": hparams}
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def build_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=args.nesterov)
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    return torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args, steps_per_epoch = 1):
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(int(args.step_size), 1), gamma=float(args.step_gamma))
    if args.scheduler == "onecycle":
        pct_start = min(max(float(args.warmup_epochs) / max(float(args.max_epochs), 1.0), 0.01), 0.9)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(args.learning_rate),
            epochs=max(int(args.max_epochs), 1),
            steps_per_epoch=max(int(steps_per_epoch), 1),
            pct_start=pct_start,
            div_factor=25.0,
            final_div_factor=max(float(args.learning_rate) / max(float(args.lr_min), 1e-12), 1.0),
        )
    warmup = max(int(args.warmup_epochs), 0)
    total = max(int(args.max_epochs), 1)
    base_lr = float(args.learning_rate)
    min_ratio = float(args.lr_min) / max(base_lr, 1e-12)

    def scale(epoch):
        if warmup and epoch < warmup:
            return max((epoch + 1) / warmup, min_ratio)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def load_compatible_state(model, state):
    current = unwrap(model).state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current
        and not isinstance(current[key], UninitializedParameter)
        and hasattr(value, "shape")
        and tuple(value.shape) == tuple(current[key].shape)
    }
    missing, unexpected = unwrap(model).load_state_dict(compatible, strict=False)
    return len(compatible), len(missing), len(unexpected)


def main():
    args = build_parser().parse_args()
    rank, _, local_rank = ddp_setup(args)
    torch.manual_seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if args.ddp else args.device)
    out_dir = Path(args.out_dir)
    if rank == 0 and not args.test_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    data = fMRIDataModule(**vars(args))
    data.prepare_data()
    data.setup("test" if args.test_only else None)
    args.target_names = data.target_names
    if rank == 0 and not args.test_only:
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    target_values = data.train_dataset.target_values if hasattr(data, "train_dataset") else torch.from_numpy(np.stack([rec.target for rec in data.subjects.values()])).float()
    model = R2TNet(target_values, **vars(args)).to(device)
    if args.compile:
        model = torch.compile(model)
    if args.ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = build_optimizer(model, args)
    steps_per_epoch = max(len(data.train_dataloader()), 1) if not args.test_only else 1
    scheduler = build_scheduler(optimizer, args, steps_per_epoch)
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(ckpt["state_dict"], strict=False)
        if not args.test_only and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if not args.test_only and scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        loaded, missing, unexpected = load_compatible_state(model, state)
        if rank == 0:
            print(json.dumps({"init_from": args.init_from, "loaded_tensors": loaded, "missing_tensors": missing, "unexpected_tensors": unexpected}))

    amp = args.precision == "amp"
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    if args.test_only:
        metrics = evaluate(model, data.test_dataloader(), device, amp)
        if rank == 0:
            print(json.dumps({"test": metrics}, indent=2))
        return

    best_value = float("-inf")
    stale = 0
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        train_loader = data.train_dataloader()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        running = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                loss, logs = unwrap(model).training_loss(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and args.scheduler == "onecycle":
                scheduler.step()
            running.append(float(loss.detach().cpu()))

        val = evaluate(model, data.val_dataloader(), device, amp)
        val.update(evaluate_alignment(model, data.val_pair_dataloader(), device, amp))
        if scheduler is not None and args.scheduler != "onecycle":
            scheduler.step()
        selected, selected_name = selection_value(val, args.selection_metric)
        improved = selected > best_value
        if rank == 0:
            print(json.dumps({"epoch": epoch, "train_loss": sum(running) / max(len(running), 1), "selected": selected_name, "val": val}))
            save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, args, scheduler)
            if improved:
                save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, args, scheduler)
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                save_checkpoint(out_dir / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, args, scheduler)
        stale = 0 if improved else stale + 1
        best_value = max(best_value, selected)
        if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
            break

    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
