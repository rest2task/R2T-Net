from __future__ import annotations

import argparse
import csv
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .specs import STUDIES


def _cmd(parts):
    return shlex.join([str(p) for p in parts if p is not None and p != ""])


def _overlay(base, override):
    keys = {x for x in override if x.startswith("--")}
    out, i = [], 0
    while i < len(base):
        if base[i] in keys:
            i += 2 if i + 1 < len(base) and not base[i + 1].startswith("--") else 1
        else:
            out.append(base[i])
            i += 1
    return out + override


@dataclass(frozen=True)
class Run:
    suite: str
    name: str
    out_dir: Path
    args: list[str] = field(default_factory=list)


BASE = [
    "--data_dir", "data/hcp_grayord",
    "--target_cols", ",".join(STUDIES["hcp"].target_cols),
    "--training_mode", "r2t",
    "--eval_role", "rest",
    "--sequence_length", "300",
    "--signature_dim", "1024",
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
    "--modality_dropout_p", "0.2",
    "--lambda_contrast", "0.5",
    "--supervised_view", "average",
]


def encoder_suite():
    return [
        Run("encoder", "vit_d12_h12", Path("runs/space_encoder/vit_d12_h12"), ["--temporal_encoder", "vit", "--vit_depth", "12", "--vit_heads", "12"]),
        Run("encoder", "vit_d8_h8", Path("runs/space_encoder/vit_d8_h8"), ["--temporal_encoder", "vit", "--vit_depth", "8", "--vit_heads", "8"]),
        Run("encoder", "gru_bi2", Path("runs/space_encoder/gru_bi2"), ["--temporal_encoder", "gru", "--gru_depth", "2"]),
        Run("encoder", "conv_d6_k7", Path("runs/space_encoder/conv_d6_k7"), ["--temporal_encoder", "conv", "--conv_depth", "6", "--conv_kernel", "7"]),
        Run("encoder", "mean_probe", Path("runs/space_encoder/mean_probe"), ["--temporal_encoder", "mean"]),
    ]


def objective_suite():
    return [
        Run("objective", "ntxent", Path("runs/space_objective/ntxent"), ["--contrastive_loss", "ntxent"]),
        Run("objective", "sym_ce", Path("runs/space_objective/sym_ce"), ["--contrastive_loss", "symmetric_ce"]),
        Run("objective", "cosine", Path("runs/space_objective/cosine"), ["--contrastive_loss", "cosine"]),
        Run("objective", "margin_02", Path("runs/space_objective/margin_02"), ["--contrastive_loss", "margin", "--contrastive_margin", "0.2"]),
        Run("objective", "no_contrast", Path("runs/space_objective/no_contrast"), ["--disable_contrastive"]),
    ]


def head_suite():
    return [
        Run("head", "yolo_q8", Path("runs/space_head/yolo_q8"), ["--reg_head", "yolo", "--reg_num_bins", "8", "--reg_binning_strategy", "quantile"]),
        Run("head", "yolo_u12", Path("runs/space_head/yolo_u12"), ["--reg_head", "yolo", "--reg_num_bins", "12", "--reg_binning_strategy", "uniform"]),
        Run("head", "mlp_huber", Path("runs/space_head/mlp_huber"), ["--reg_head", "mlp", "--reg_loss", "huber", "--head_depth", "3"]),
        Run("head", "mlp_mse", Path("runs/space_head/mlp_mse"), ["--reg_head", "mlp", "--reg_loss", "mse", "--head_depth", "2"]),
        Run("head", "linear", Path("runs/space_head/linear"), ["--reg_head", "linear", "--label_scaling_method", "standardization"]),
    ]


def fusion_suite():
    return [
        Run("fusion", "avg", Path("runs/space_fusion/avg"), ["--pair_fusion", "average"]),
        Run("fusion", "rest", Path("runs/space_fusion/rest"), ["--pair_fusion", "rest"]),
        Run("fusion", "task", Path("runs/space_fusion/task"), ["--pair_fusion", "task"]),
        Run("fusion", "sum", Path("runs/space_fusion/sum"), ["--pair_fusion", "sum"]),
        Run("fusion", "concat", Path("runs/space_fusion/concat"), ["--pair_fusion", "concat"]),
        Run("fusion", "gated", Path("runs/space_fusion/gated"), ["--pair_fusion", "gated"]),
    ]


def optim_suite():
    return [
        Run("optim", "adamw_cosine", Path("runs/space_optim/adamw_cosine"), ["--optimizer", "adamw", "--scheduler", "cosine"]),
        Run("optim", "adamw_onecycle", Path("runs/space_optim/adamw_onecycle"), ["--optimizer", "adamw", "--scheduler", "onecycle", "--warmup_epochs", "80"]),
        Run("optim", "adam", Path("runs/space_optim/adam"), ["--optimizer", "adam", "--scheduler", "cosine"]),
        Run("optim", "sgd_step", Path("runs/space_optim/sgd_step"), ["--optimizer", "sgd", "--scheduler", "step", "--step_size", "200", "--step_gamma", "0.3", "--learning_rate", "0.02"]),
        Run("optim", "rmsprop", Path("runs/space_optim/rmsprop"), ["--optimizer", "rmsprop", "--scheduler", "cosine", "--learning_rate", "0.0005"]),
    ]


def regularization_suite():
    return [
        Run("regularization", "light", Path("runs/space_regularization/light"), ["--encoder_dropout", "0.05", "--head_dropout", "0.05", "--gaussian_noise_p", "0.05"]),
        Run("regularization", "mask_time", Path("runs/space_regularization/mask_time"), ["--temporal_mask_p", "0.08", "--temporal_crop_min_ratio", "0.7"]),
        Run("regularization", "mask_feature", Path("runs/space_regularization/mask_feature"), ["--feature_mask_p", "0.05", "--gaussian_noise_std", "0.02"]),
        Run("regularization", "modality_drop", Path("runs/space_regularization/modality_drop"), ["--modality_dropout_p", "0.4"]),
    ]


def synthetic_suite():
    return [
        Run("synthetic", "cmt_d2", Path("runs/space_synthetic/cmt_d2"), ["--training_mode", "synthetic_task", "--synthetic_mapper", "cmt", "--synthetic_depth", "2", "--synthetic_heads", "8"]),
        Run("synthetic", "cmt_d4", Path("runs/space_synthetic/cmt_d4"), ["--training_mode", "synthetic_task", "--synthetic_mapper", "cmt", "--synthetic_depth", "4", "--synthetic_heads", "8", "--lambda_synthetic_l2", "0.05"]),
        Run("synthetic", "mlp", Path("runs/space_synthetic/mlp"), ["--training_mode", "synthetic_task", "--synthetic_mapper", "mlp"]),
    ]


def size_suite():
    out = []
    for length in (100, 200, 300, 600):
        for dim in (256, 512, 1024, 2048):
            name = f"t{length}_d{dim}"
            out.append(Run("size", name, Path("runs/space_size") / name, ["--sequence_length", str(length), "--signature_dim", str(dim)]))
    return out


def runs(suite):
    suites = {
        "encoder": encoder_suite,
        "objective": objective_suite,
        "head": head_suite,
        "fusion": fusion_suite,
        "optim": optim_suite,
        "regularization": regularization_suite,
        "synthetic": synthetic_suite,
        "size": size_suite,
    }
    if suite == "all":
        out = []
        for fn in suites.values():
            out.extend(fn())
        return out
    return suites[suite]()


def train_cmd(run, extra):
    args = _overlay(BASE, run.args) + ["--out_dir", run.out_dir] + extra
    return _cmd(["python", "train.py", *args])


def parse_args():
    parser = argparse.ArgumentParser(description="Print model-space trial commands")
    parser.add_argument("--suite", choices=["encoder", "objective", "head", "fusion", "optim", "regularization", "synthetic", "size", "all"], default="all")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    items = runs(args.suite)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["suite", "name", "out_dir", "cmd"])
            for item in items:
                writer.writerow([item.suite, item.name, item.out_dir, train_cmd(item, args.extra)])
    for item in items:
        print(train_cmd(item, args.extra))


if __name__ == "__main__":
    main()
