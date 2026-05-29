from __future__ import annotations

import argparse
import csv
import shlex
from dataclasses import dataclass
from pathlib import Path

from .specs import REPRESENTATIONS, SCAN_LENGTHS, SEEDS, SIGNATURE_DIMS, STUDIES, TRAINING_MODES


@dataclass(frozen=True)
class Experiment:
    suite: str
    name: str
    study: str
    mode: str
    representation: str
    data_dir: Path
    out_dir: Path
    seed: int
    sequence_length: int
    signature_dim: int
    batch_size: int = 16
    learning_rate: str = "1e-3"
    weight_decay: str = "1e-2"
    scheduler: str = "cosine"
    warmup_epochs: int = 50
    lr_min: str = "1e-5"
    max_epochs: int = 1600
    save_every: int = 50
    early_stop_patience: int = 5
    selection_metric: str = "alignment"


def _cmd(parts):
    return shlex.join([str(p) for p in parts if p is not None and p != ""])


def _train_cmd(exp):
    spec = STUDIES[exp.study]
    mode = TRAINING_MODES[exp.mode]
    parts = [
        "python", "train.py",
        "--seed", exp.seed,
        "--data_dir", exp.data_dir,
        "--target_cols", ",".join(spec.target_cols),
        "--training_mode", mode["training_mode"],
        "--eval_role", mode["eval_role"],
        "--sequence_length", exp.sequence_length,
        "--signature_dim", exp.signature_dim,
        "--learning_rate", exp.learning_rate,
        "--weight_decay", exp.weight_decay,
        "--scheduler", exp.scheduler,
        "--warmup_epochs", exp.warmup_epochs,
        "--lr_min", exp.lr_min,
        "--batch_size", exp.batch_size,
        "--max_epochs", exp.max_epochs,
        "--save_every", exp.save_every,
        "--early_stop_patience", exp.early_stop_patience,
        "--selection_metric", exp.selection_metric,
        "--out_dir", exp.out_dir,
        "--modality_dropout_p", 0.2,
    ]
    if mode.get("disable_contrastive"):
        parts.append("--disable_contrastive")
    for key in ("lambda_contrast", "lambda_synthetic", "supervised_view"):
        if key in mode:
            parts.extend([f"--{key}", mode[key]])
    return _cmd(parts)


def _test_cmd(exp):
    mode = TRAINING_MODES[exp.mode]
    parts = [
        "python", "train.py",
        "--data_dir", exp.data_dir,
        "--target_cols", ",".join(STUDIES[exp.study].target_cols),
        "--training_mode", mode["training_mode"],
        "--eval_role", mode["eval_role"],
        "--sequence_length", exp.sequence_length,
        "--signature_dim", exp.signature_dim,
        "--resume", exp.out_dir / "last.pt",
        "--test_only",
    ]
    if mode.get("disable_contrastive"):
        parts.append("--disable_contrastive")
    return _cmd(parts)


def modality_suite(seed):
    rep = REPRESENTATIONS["grayord"]
    return [
        Experiment("modality", mode, "hcp", mode, "grayord", Path(rep["data_dir"]), Path("runs") / "cmp_modality" / f"{mode}_seed{seed}", seed, 300, 1024)
        for mode in ("rs_only", "t_only", "synthetic_t", "r2t")
    ]


def representation_suite(seed):
    out = []
    for rep_name, rep in REPRESENTATIONS.items():
        out.append(Experiment("representation", rep_name, "hcp", "r2t", rep_name, Path(rep["data_dir"]), Path("runs") / "cmp_representation" / f"{rep_name}_seed{seed}", seed, int(rep["sequence_length"]), int(rep["signature_dim"])))
    return out


def length_dim_suite(seed):
    rep = REPRESENTATIONS["grayord"]
    out = []
    for length in SCAN_LENGTHS:
        for dim in SIGNATURE_DIMS:
            name = f"t{length}_d{dim}"
            out.append(Experiment("length_dim", name, "hcp", "r2t", "grayord", Path(rep["data_dir"]), Path("runs") / "cmp_length_dim" / f"{name}_seed{seed}", seed, length, dim))
    return out


def seed_suite():
    out = []
    for seed in SEEDS:
        out.extend(modality_suite(seed))
        out.extend(representation_suite(seed))
    return out


def experiments(suite, seed):
    if suite == "modality":
        return modality_suite(seed)
    if suite == "representation":
        return representation_suite(seed)
    if suite == "length_dim":
        return length_dim_suite(seed)
    if suite == "seeds":
        return seed_suite()
    if suite == "all":
        return modality_suite(seed) + representation_suite(seed) + length_dim_suite(seed)
    raise ValueError(suite)


def write_csv(exps, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["suite", "name", "study", "mode", "representation", "seed", "sequence_length", "signature_dim", "data_dir", "out_dir", "train_cmd", "test_cmd"])
        for exp in exps:
            writer.writerow([exp.suite, exp.name, exp.study, exp.mode, exp.representation, exp.seed, exp.sequence_length, exp.signature_dim, exp.data_dir, exp.out_dir, _train_cmd(exp), _test_cmd(exp)])


def parse_args():
    parser = argparse.ArgumentParser(description="Print manuscript model-comparison commands")
    parser.add_argument("--suite", choices=["modality", "representation", "length_dim", "seeds", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--tests", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    exps = experiments(args.suite, args.seed)
    if args.csv:
        write_csv(exps, args.csv)
    for exp in exps:
        print(_train_cmd(exp))
        if args.tests:
            print(_test_cmd(exp))


if __name__ == "__main__":
    main()
