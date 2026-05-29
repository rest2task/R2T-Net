from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from .specs import REPRESENTATIONS, STUDIES, TARGETS, TRAINING_MODES


def _cmd(parts):
    return shlex.join([str(p) for p in parts if p is not None and p != ""])


def train_cmd(study, mode, rep, data_dir, out_dir):
    spec = STUDIES[study]
    mode_cfg = TRAINING_MODES[mode]
    rep_cfg = REPRESENTATIONS[rep]
    data_dir = data_dir or Path(rep_cfg.get("data_dir", spec.data_dir))
    out_dir = out_dir or Path("runs") / f"{study}_{rep}_{mode}"
    parts = [
        "python", "train.py",
        "--data_dir", data_dir,
        "--target_cols", ",".join(spec.target_cols),
        "--training_mode", mode_cfg["training_mode"],
        "--eval_role", mode_cfg["eval_role"],
        "--sequence_length", rep_cfg["sequence_length"],
        "--signature_dim", rep_cfg["signature_dim"],
        "--learning_rate", "1e-3",
        "--weight_decay", "1e-2",
        "--scheduler", "cosine",
        "--warmup_epochs", "50",
        "--lr_min", "1e-5",
        "--batch_size", "16",
        "--max_epochs", "1600",
        "--save_every", "50",
        "--early_stop_patience", "5",
        "--selection_metric", "alignment",
        "--out_dir", out_dir,
        "--modality_dropout_p", "0.2",
    ]
    if mode_cfg.get("disable_contrastive"):
        parts.append("--disable_contrastive")
    for key in ("lambda_contrast", "lambda_synthetic", "supervised_view"):
        if key in mode_cfg:
            parts.extend([f"--{key}", mode_cfg[key]])
    return _cmd(parts)


def transfer_cmd(study, data_dir, ckpt, out_dir):
    spec = STUDIES[study]
    ckpt = ckpt or spec.default_ckpt
    if ckpt is None:
        raise ValueError(f"{study} has no default checkpoint")
    data_dir = data_dir or spec.data_dir
    out_dir = out_dir or Path("runs") / f"{study}_transfer"
    return _cmd([
        "python", "-m", "studies.transfer",
        "--ckpt", ckpt,
        "--data-dir", data_dir,
        "--target-cols", ",".join(spec.target_cols),
        "--split", "test",
        "--role", "rest",
        "--sequence-length", 300,
        "--output", out_dir / "predictions.csv",
        "--signatures", out_dir / "signatures.pt",
    ])


def manifest_cmd(study, data_dir, source_root, participants):
    spec = STUDIES[study]
    parts = ["python", "-m", "studies.manifest", study, "--source-root", source_root or spec.source_root, "--data-dir", data_dir or spec.data_dir]
    if participants:
        parts.extend(["--participants", participants])
    return _cmd(parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Print manuscript study commands")
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument("--mode", choices=sorted(TRAINING_MODES))
    parser.add_argument("--representation", choices=sorted(REPRESENTATIONS), default="grayord")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--participants", type=Path)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    print(manifest_cmd(args.study, args.data_dir, args.source_root, args.participants))
    if args.mode:
        print(train_cmd(args.study, args.mode, args.representation, args.data_dir, args.out_dir))
    if args.study in {"chcp", "adni"} and not args.mode:
        print(transfer_cmd(args.study, args.data_dir, args.ckpt, args.out_dir))
    if args.study == "hcp" and not args.mode:
        for mode in ("rs_only", "t_only", "synthetic_t", "r2t"):
            print(train_cmd("hcp", mode, args.representation, args.data_dir, args.out_dir))
        print("# representation checks: " + ", ".join(REPRESENTATIONS))
        print("# targets: " + ", ".join(TARGETS))


if __name__ == "__main__":
    main()
