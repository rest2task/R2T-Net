"""Manifest-based data module for plain PyTorch training."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .datasets import PairedFMRIWindowDataset, SingleFMRIWindowDataset
from .manifest import ScanRecord, SubjectRecord, load_scans, load_subjects, target_keys


class fMRIDataModule:
    @staticmethod
    def add_data_specific_args(parent_parser):
        p = parent_parser.add_argument_group("Data")
        p.add_argument("--data_dir", required=True, type=str)
        p.add_argument("--batch_size", default=8, type=int)
        p.add_argument("--num_workers", default=4, type=int)
        p.add_argument("--sequence_length", type=int, default=300)
        p.add_argument("--stride_between_seq", type=int, default=1)
        p.add_argument("--stride_within_seq", type=int, default=1)
        p.add_argument("--pin_memory", action="store_true")
        p.add_argument(
            "--training_mode",
            choices=["r2t", "rest_only", "task_only", "synthetic_task"],
            default="r2t",
        )
        p.add_argument("--eval_role", choices=["rest", "task"], default="rest")
        p.add_argument("--target_cols", default="", help="Comma-separated target keys in subjects.jsonl")
        return parent_parser

    def __init__(self, **hparams):
        self.hparams = SimpleNamespace(**hparams)
        self.data_dir = Path(hparams["data_dir"])
        self.subjects: Dict[str, SubjectRecord] = {}
        self.scans: Dict[str, List[ScanRecord]] = {}
        self.target_names: List[str] = []

    def _is_distributed(self):
        return int(getattr(self.hparams, "world_size", 1)) > 1

    def _target_cols(self):
        raw = getattr(self.hparams, "target_cols", "") or ""
        cols = [c.strip() for c in raw.split(",") if c.strip()]
        return cols or None

    def prepare_data(self):
        meta = self.data_dir / "meta"
        if not (meta / "subjects.jsonl").exists() and not (meta / "subjects.csv").exists():
            raise FileNotFoundError(meta / "subjects.jsonl")
        if not (meta / "scans.jsonl").exists() and not (meta / "scans.csv").exists():
            raise FileNotFoundError(meta / "scans.jsonl")

    def _load_metadata(self):
        if self.subjects and self.scans:
            return
        cols = self._target_cols()
        self.target_names = target_keys(self.data_dir, cols)
        self.subjects = load_subjects(self.data_dir, self.target_names)
        self.scans = load_scans(self.data_dir)

    def _single(self, split, role):
        return SingleFMRIWindowDataset(
            subjects=self.subjects,
            scans=self.scans,
            split=split,
            role=role,
            sequence_length=self.hparams.sequence_length,
            stride_between_seq=self.hparams.stride_between_seq,
            stride_within_seq=self.hparams.stride_within_seq,
        )

    def _paired(self, split):
        return PairedFMRIWindowDataset(
            subjects=self.subjects,
            scans=self.scans,
            split=split,
            sequence_length=self.hparams.sequence_length,
            stride_within_seq=self.hparams.stride_within_seq,
        )

    def _train_dataset(self):
        mode = self.hparams.training_mode
        if mode in {"r2t", "synthetic_task"}:
            return self._paired("train")
        if mode == "rest_only":
            return self._single("train", "rest")
        if mode == "task_only":
            return self._single("train", "task")
        raise ValueError(mode)

    def setup(self, stage=None):
        self._load_metadata()

        if stage in (None, "fit", "validate"):
            self.train_dataset = self._train_dataset()
            self.val_dataset = self._single("val", self.hparams.eval_role)
            self.val_pair_dataset = None
            if self.hparams.training_mode in {"r2t", "synthetic_task"}:
                try:
                    self.val_pair_dataset = self._paired("val")
                except ValueError:
                    self.val_pair_dataset = None

        if stage in (None, "fit", "test"):
            self.test_dataset = self._single("test", self.hparams.eval_role)

    def _sampler(self, dataset, shuffle):
        if self._is_distributed():
            return DistributedSampler(
                dataset,
                num_replicas=int(self.hparams.world_size),
                rank=int(self.hparams.rank),
                shuffle=shuffle,
                drop_last=False,
            )
        return RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self._sampler(self.train_dataset, True),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self._sampler(self.val_dataset, False),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_pair_dataloader(self):
        if self.val_pair_dataset is None:
            return None
        return DataLoader(
            self.val_pair_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self._sampler(self.val_pair_dataset, False),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self._sampler(self.test_dataset, False),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.hparams.num_workers > 0,
        )
