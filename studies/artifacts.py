from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from r2tnet.model import R2TNet
from r2tnet.datasets import PairedFMRIWindowDataset, SingleFMRIWindowDataset
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


def manifest(args):
    subjects = load_subjects(args.data_dir, _target_cols(args.target_cols))
    scans = load_scans(args.data_dir)
    split_counts = Counter(rec.split for rec in subjects.values())
    scan_counts = Counter((s.role, s.input_kind) for rows in scans.values() for s in rows)
    frame_counts = Counter(s.role for rows in scans.values() for s in rows for _ in range(s.n_frames))
    print("subjects", dict(sorted(split_counts.items())))
    print("scans", {f"{role}:{kind}": n for (role, kind), n in sorted(scan_counts.items())})
    print("frames", dict(sorted(frame_counts.items())))


@torch.no_grad()
def alignment(args):
    device = torch.device(args.device)
    subjects = load_subjects(args.data_dir, _target_cols(args.target_cols))
    scans = load_scans(args.data_dir)
    dataset = PairedFMRIWindowDataset(subjects, scans, args.split, args.sequence_length, args.stride_within_seq)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    model = _load_model(args.ckpt, device)
    rows = []
    for i, batch in enumerate(loader):
        rest = batch["rest"].to(device)
        task = batch["task"].to(device)
        rest_sig = model.encode(rest, torch.zeros(rest.size(0), dtype=torch.long, device=device))
        task_sig = model.encode(task, torch.ones(task.size(0), dtype=torch.long, device=device))
        sim = rest_sig @ task_sig.t()
        for j, sid in enumerate(_ids(batch["subject_id"], rest.size(0))):
            neg = torch.cat([sim[j, :j], sim[j, j + 1 :]])
            rows.append([sid, float(sim[j, j].cpu()), float(neg.mean().cpu()) if neg.numel() else float("nan")])
        if args.max_batches and i + 1 >= args.max_batches:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject_id", "positive_cosine", "batch_negative_mean"])
        writer.writerows(rows)


def _spatial_score(x, grad):
    score = (x.detach() * grad.detach()).abs().squeeze(0).cpu()
    if score.ndim >= 2:
        score = score.mean(dim=-1)
    return score


def saliency(args):
    device = torch.device(args.device)
    subjects = load_subjects(args.data_dir, _target_cols(args.target_cols))
    scans = load_scans(args.data_dir)
    dataset = SingleFMRIWindowDataset(subjects, scans, args.split, args.role, args.sequence_length, args.stride_between_seq, args.stride_within_seq)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    model = _load_model(args.ckpt, device)
    buckets = defaultdict(list)
    for i, batch in enumerate(loader):
        x = batch["fmri"].to(device).requires_grad_(True)
        modality = batch["modality"].to(device)
        pred = model.pred_head(model.encode(x, modality))["prediction"]
        value = pred[:, args.target_index].sum()
        model.zero_grad(set_to_none=True)
        value.backward()
        sid = _ids(batch["subject_id"], 1)[0]
        buckets[sid].append(_spatial_score(x, x.grad))
        if args.max_windows and i + 1 >= args.max_windows:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({sid: torch.stack(vals).mean(0) for sid, vals in buckets.items()}, args.output)


def signatures(args):
    device = torch.device(args.device)
    subjects = load_subjects(args.data_dir, _target_cols(args.target_cols))
    scans = load_scans(args.data_dir)
    dataset = SingleFMRIWindowDataset(subjects, scans, args.split, args.role, args.sequence_length, args.stride_between_seq, args.stride_within_seq)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    model = _load_model(args.ckpt, device)
    buckets = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            x = batch["fmri"].to(device)
            z = F.normalize(model.encode(x, batch["modality"].to(device)), dim=1).cpu()
            for sid, sig in zip(_ids(batch["subject_id"], z.size(0)), z):
                buckets[sid].append(sig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({sid: torch.stack(vals).mean(0) for sid, vals in buckets.items()}, args.output)


def add_common(parser):
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target-cols", default="")


def add_model_common(parser):
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--role", choices=["rest", "task"], default="rest")
    parser.add_argument("--sequence-length", type=int, default=300)
    parser.add_argument("--stride-between-seq", type=int, default=1)
    parser.add_argument("--stride-within-seq", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Raw model-analysis artifacts; no inference statistics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("manifest")
    add_common(p)

    p = sub.add_parser("alignment")
    add_common(p)
    add_model_common(p)
    p.add_argument("--max-batches", type=int, default=0)

    p = sub.add_parser("saliency")
    add_common(p)
    add_model_common(p)
    p.add_argument("--target-index", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=0)

    p = sub.add_parser("signatures")
    add_common(p)
    add_model_common(p)
    return parser.parse_args()


def main():
    args = parse_args()
    {"manifest": manifest, "alignment": alignment, "saliency": saliency, "signatures": signatures}[args.cmd](args)


if __name__ == "__main__":
    main()
