from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from r2tnet.model import R2TNet
from r2tnet.datasets import SingleFMRIWindowDataset
from r2tnet.manifest import load_scans, load_subjects


def _target_cols(raw):
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    return cols or None


def _load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hparams", ckpt.get("hyper_parameters", {}))
    model = R2TNet(torch.zeros(4, int(hparams.get("target_dim", 1))), **hparams)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def _ids(value, n):
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)] * n


@torch.no_grad()
def predict(args):
    device = torch.device(args.device)
    subjects = load_subjects(args.data_dir, _target_cols(args.target_cols))
    scans = load_scans(args.data_dir)
    dataset = SingleFMRIWindowDataset(subjects, scans, args.split, args.role, args.sequence_length, args.stride_between_seq, args.stride_within_seq)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    model = _load_model(args.ckpt, device)

    pred_buckets = defaultdict(list)
    sig_buckets = defaultdict(list)
    target_ref = {}
    for batch in loader:
        x = batch["fmri"].to(device, non_blocking=True)
        modality = batch["modality"].to(device, non_blocking=True)
        signature = model.encode(x, modality)
        pred = model.inverse_scale(model.pred_head(signature)["prediction"]).cpu()
        for sid, p, z, y in zip(_ids(batch["subject_id"], pred.size(0)), pred, signature.cpu(), batch["target"]):
            pred_buckets[sid].append(p)
            sig_buckets[sid].append(z)
            target_ref[sid] = y.float()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred_dim = next(iter(pred_buckets.values()))[0].numel()
    names = getattr(model, "target_names", None) or [f"target_{i}" for i in range(pred_dim)]
    with args.output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject_id", *[f"pred_{name}" for name in names], *[f"target_{name}" for name in names]])
        for sid in sorted(pred_buckets):
            pred = torch.stack(pred_buckets[sid]).mean(0)
            target = target_ref.get(sid, torch.full_like(pred, float("nan")))
            writer.writerow([sid, *pred.tolist(), *target.tolist()])

    if args.signatures:
        args.signatures.parent.mkdir(parents=True, exist_ok=True)
        torch.save({sid: torch.stack(sig_buckets[sid]).mean(0) for sid in sorted(sig_buckets)}, args.signatures)


def parse_args():
    parser = argparse.ArgumentParser(description="Run frozen R2T-Net over a prepared manifest")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--role", choices=["rest", "task"], default="rest")
    parser.add_argument("--target-cols", default="")
    parser.add_argument("--sequence-length", type=int, default=300)
    parser.add_argument("--stride-between-seq", type=int, default=1)
    parser.add_argument("--stride-within-seq", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--signatures", type=Path)
    return parser.parse_args()


def main():
    predict(parse_args())


if __name__ == "__main__":
    main()
