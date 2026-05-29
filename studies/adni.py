from __future__ import annotations

import argparse
from pathlib import Path

from .plan import manifest_cmd, transfer_cmd, train_cmd
from .specs import STUDIES


def _resolve_ckpt(ckpt):
    resolved = ckpt or STUDIES["adni"].default_ckpt
    if resolved is None:
        raise ValueError("adni has no default checkpoint; provide --ckpt")
    # Probe mode requires a checkpoint, so resolve now and fail fast if nothing is available.
    return resolved


def _probe_args(ckpt):
    ckpt = _resolve_ckpt(ckpt)
    return [
        "--resume",
        ckpt,
        "--freeze_encoder",
        "--disable_contrastive",
        "--learning_rate",
        "1e-3",
        "--scheduler",
        "cosine",
        "--batch_size",
        "8",
        "--max_epochs",
        "100",
        "--selection_metric",
        "loss",
    ]


def parse_args():
    p = argparse.ArgumentParser(description="ADNI commands")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--participants", type=Path)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--manifest-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(manifest_cmd("adni", args.data_dir, args.source_root, args.participants))
    if args.manifest_only:
        return
    study_data_dir = args.data_dir or STUDIES["adni"].data_dir
    if args.probe:
        # Probe fine-tunes on top of an existing checkpoint while freezing the encoder.
        print(train_cmd("adni", "rs_only", "grayord", study_data_dir, args.out_dir or Path("runs/adni_probe"), extra=_probe_args(args.ckpt)))
    else:
        # Default mode keeps the older behavior: run transfer and dump predictions/signatures.
        print(transfer_cmd("adni", study_data_dir, args.ckpt, args.out_dir))


if __name__ == "__main__":
    main()
