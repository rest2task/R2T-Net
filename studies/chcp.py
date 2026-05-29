from __future__ import annotations

import argparse
from pathlib import Path

from .plan import manifest_cmd, transfer_cmd, train_cmd
from .specs import STUDIES


def _resolve_ckpt(ckpt):
    resolved = ckpt or STUDIES["chcp"].default_ckpt
    if resolved is None:
        raise ValueError("chcp has no default checkpoint; provide --ckpt")
    # For fine-tuning, always resolve once and reuse a hard failure if the checkpoint is unavailable.
    return resolved


def _finetune_args(ckpt):
    ckpt = _resolve_ckpt(ckpt)
    return [
        "--resume",
        ckpt,
        "--learning_rate",
        "2e-4",
        "--scheduler",
        "cosine",
        "--warmup_epochs",
        "20",
        "--batch_size",
        "8",
        "--max_epochs",
        "300",
        "--selection_metric",
        "loss",
        "--lambda_contrast",
        "0.25",
        "--supervised_view",
        "average",
    ]


def parse_args():
    p = argparse.ArgumentParser(description="CHCP commands")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--participants", type=Path)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--manifest-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(manifest_cmd("chcp", args.data_dir, args.source_root, args.participants))
    if args.manifest_only:
        return
    study_data_dir = args.data_dir or STUDIES["chcp"].data_dir
    if args.finetune:
        # Fine-tune starts from an existing checkpoint and writes to a dedicated output tree.
        print(train_cmd("chcp", "r2t", "grayord", study_data_dir, args.out_dir or Path("runs/chcp_finetune"), extra=_finetune_args(args.ckpt)))
    else:
        # Transfer uses the default checkpoint and emits prediction/signature artifacts.
        print(transfer_cmd("chcp", study_data_dir, args.ckpt, args.out_dir))


if __name__ == "__main__":
    main()
