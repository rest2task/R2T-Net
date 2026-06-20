from __future__ import annotations

import argparse
import csv
import random
import shlex
from collections import Counter, defaultdict
from pathlib import Path

from .plan import manifest_cmd
from .specs import ADNI_CLASS_TARGETS, STUDIES


DEFAULT_CSV = Path("3T_HCP_Proc_6_12_2025.csv")
DEFAULT_PARTICIPANTS = Path("data/adni_classification_participants.csv")


def _cmd(parts):
    return shlex.join([str(p) for p in parts if p is not None and p != ""])


def _label(group, mci_policy):
    group = (group or "").strip().upper()
    if group in {"AD", "CN", "MCI"}:
        return group
    if mci_policy == "collapse_subtypes" and group in {"EMCI", "LMCI"}:
        return "MCI"
    return None


def _sex(value):
    value = (value or "").strip().upper()
    if value in {"M", "MALE", "1"}:
        return 1
    if value in {"F", "FEMALE", "0"}:
        return 0
    return 0


def _subject_rows(csv_path, source_root, mci_policy):
    by_subject = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("Modality") or "").strip().lower() != "fmri":
                continue
            sid = (row.get("Subject") or "").strip()
            label = _label(row.get("Group"), mci_policy)
            if not sid or label is None:
                continue
            if source_root and source_root.exists() and not (source_root / sid).exists():
                continue
            by_subject.setdefault(sid, row | {"diagnosis": label})
    return by_subject


def _splits(subjects, val_fraction, test_fraction, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for sid, row in subjects.items():
        by_label[row["diagnosis"]].append(sid)

    out = {}
    for label, ids in by_label.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * test_fraction)) if n >= 3 and test_fraction > 0 else 0
        n_val = max(1, round(n * val_fraction)) if n - n_test >= 3 and val_fraction > 0 else 0
        for sid in ids[:n_test]:
            out[sid] = "test"
        for sid in ids[n_test : n_test + n_val]:
            out[sid] = "val"
        for sid in ids[n_test + n_val :]:
            out[sid] = "train"
    return out


def write_participants(args):
    subjects = _subject_rows(args.csv, args.source_root, args.mci_policy)
    if not subjects:
        raise ValueError(f"No AD/CN/MCI fMRI subjects found in {args.csv}")
    splits = _splits(subjects, args.val_fraction, args.test_fraction, args.seed)
    args.participants_out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()

    with args.participants_out.open("w", newline="") as fh:
        fieldnames = ["subject_id", "split", "sex", *ADNI_CLASS_TARGETS, "diagnosis", "source_group", "age", "visit"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for sid in sorted(subjects):
            row = subjects[sid]
            diagnosis = row["diagnosis"]
            counts[(splits[sid], diagnosis)] += 1
            writer.writerow(
                {
                    "subject_id": sid,
                    "split": splits[sid],
                    "sex": _sex(row.get("Sex")),
                    "AD": int(diagnosis == "AD"),
                    "CN": int(diagnosis == "CN"),
                    "MCI": int(diagnosis == "MCI"),
                    "diagnosis": diagnosis,
                    "source_group": row.get("Group", ""),
                    "age": row.get("Age", ""),
                    "visit": row.get("Visit", ""),
                }
            )

    print(f"wrote {args.participants_out}")
    for split in ("train", "val", "test"):
        values = {label: counts[(split, label)] for label in ADNI_CLASS_TARGETS}
        print(f"{split}: {values}")


def train_cmd(args):
    spec = STUDIES["adni_classification"]
    out_dir = args.out_dir or Path("runs/adni_classification_finetune")
    parts = [
        "python",
        "train.py",
        "--data_dir",
        args.data_dir or spec.data_dir,
        "--target_cols",
        ",".join(spec.target_cols),
        "--training_mode",
        "rest_only",
        "--eval_role",
        "rest",
        "--downstream_task_type",
        "classification",
        "--head_spec",
        "diagnosis:AD,CN,MCI:classification:1",
        "--classification_loss",
        "cross_entropy",
        "--sequence_length",
        args.sequence_length,
        "--signature_dim",
        args.signature_dim,
        "--learning_rate",
        args.learning_rate,
        "--weight_decay",
        args.weight_decay,
        "--scheduler",
        "cosine",
        "--warmup_epochs",
        args.warmup_epochs,
        "--lr_min",
        args.lr_min,
        "--batch_size",
        args.batch_size,
        "--max_epochs",
        args.max_epochs,
        "--save_every",
        args.save_every,
        "--early_stop_patience",
        args.early_stop_patience,
        "--selection_metric",
        "loss",
        "--out_dir",
        out_dir,
    ]
    ckpt = args.ckpt or spec.default_ckpt
    if not args.scratch and ckpt:
        parts.extend(["--init_from", ckpt])
    if args.freeze_encoder:
        parts.append("--freeze_encoder")
    return _cmd(parts)


def test_cmd(args):
    spec = STUDIES["adni_classification"]
    out_dir = args.out_dir or Path("runs/adni_classification_finetune")
    return _cmd(
        [
            "python",
            "train.py",
            "--data_dir",
            args.data_dir or spec.data_dir,
            "--target_cols",
            ",".join(spec.target_cols),
            "--training_mode",
            "rest_only",
            "--eval_role",
            "rest",
            "--downstream_task_type",
            "classification",
            "--head_spec",
            "diagnosis:AD,CN,MCI:classification:1",
            "--classification_loss",
            "cross_entropy",
            "--sequence_length",
            args.sequence_length,
            "--signature_dim",
            args.signature_dim,
            "--batch_size",
            args.batch_size,
            "--resume",
            out_dir / "best.pt",
            "--test_only",
        ]
    )


def parse_args():
    p = argparse.ArgumentParser(description="ADNI AD/CN/MCI classification study")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--participants-out", type=Path, default=DEFAULT_PARTICIPANTS)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--write-participants", action="store_true")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--scratch", action="store_true", help="Train from random initialization instead of --init_from")
    p.add_argument("--freeze-encoder", action="store_true", help="Use the pretrained encoder as a frozen feature extractor")
    p.add_argument("--mci-policy", choices=["exact", "collapse_subtypes"], default="collapse_subtypes")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--sequence-length", type=int, default=300)
    p.add_argument("--signature-dim", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--early-stop-patience", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    spec = STUDIES["adni_classification"]
    args.source_root = args.source_root or spec.source_root
    args.data_dir = args.data_dir or spec.data_dir
    if args.write_participants:
        write_participants(args)
        return

    print(
        _cmd(
            [
                "python",
                "-m",
                "studies.adni_classification",
                "--write-participants",
                "--csv",
                args.csv,
                "--participants-out",
                args.participants_out,
                "--source-root",
                args.source_root,
                "--mci-policy",
                args.mci_policy,
                "--val-fraction",
                args.val_fraction,
                "--test-fraction",
                args.test_fraction,
                "--seed",
                args.seed,
            ]
        )
    )
    print(manifest_cmd("adni_classification", args.data_dir, args.source_root, args.participants_out))
    if args.manifest_only:
        return
    print(_cmd(["python", "prepare_data.py", "convert", "--data-dir", args.data_dir, "--normalise", "zscore", "--dtype", "float16"]))
    print(train_cmd(args))
    print(test_cmd(args))


if __name__ == "__main__":
    main()
