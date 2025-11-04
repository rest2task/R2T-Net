# module/utils/data_module.py
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

from .datasets import S1200        # your existing dataset class

# ---------------------------------------------------------------- #
# Lightning DataModule
# ---------------------------------------------------------------- #
class fMRIDataModule(pl.LightningDataModule):
    # ------------- CLI flags ----------------------------
    @staticmethod
    def add_data_specific_args(parent_parser):
        p = parent_parser.add_argument_group("Data")
        p.add_argument("--data_dir", default=".", type=str)
        p.add_argument("--batch_size", default=8,  type=int)
        p.add_argument("--num_workers", default=4, type=int)

        p.add_argument("--dataset_type", choices=["rest", "task"], default="rest")
        p.add_argument("--input_kind", choices=["vol", "roi", "grayord"], default="vol")
        p.add_argument("--sequence_length",    type=int, default=20)
        p.add_argument("--stride_between_seq", type=int, default=1)
        p.add_argument("--stride_within_seq",  type=int, default=1)

        p.add_argument("--contrastive", action="store_true",
                       help="Return two augmented clips for NT-Xent")
        p.add_argument("--with_voxel_norm",     action="store_true")
        p.add_argument("--shuffle_time_sequence", action="store_true")

        # NEW ---------------------------------------------------------------
        p.add_argument("--grayordinates", action="store_true",
                       help="Treat ROI axis as 91 282 gray-ordinates")
        p.add_argument("--label_scaling_method",
                       choices=["standardization", "minmax", "none"],
                       default="standardization")
        p.add_argument("--balanced_sampling", action="store_true",
                       help="Resample subjects so each appears equally in a epoch")
        return parent_parser

    # ------------- ctor ---------------------------------
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_kwargs = dict(
            task_type=hparams["dataset_type"],
            input_kind=hparams.get("input_kind", "vol"),
            sequence_length=hparams["sequence_length"],
            stride_between_seq=hparams["stride_between_seq"],
            stride_within_seq=hparams["stride_within_seq"],
            contrastive=hparams["contrastive"],
            with_voxel_norm=hparams["with_voxel_norm"],
            shuffle_time_sequence=hparams["shuffle_time_sequence"],
        )

        # decide number of ROIs automatically
        self.num_rois = 91_282 if hparams["grayordinates"] else None
        self.scaler = None                     # set in setup()
        self._subject_dict: Dict[str, Tuple[int, float]] | None = None
        self._splits: Dict[str, List[str]] | None = None

    # ------------- setup --------------------------------
    def _load_metadata(self) -> None:
        if self._subject_dict is not None and self._splits is not None:
            return

        data_dir = Path(self.hparams["data_dir"])
        meta_dir = data_dir / "meta"
        subject_dict_path = meta_dir / "subject_dict.json"
        splits_path = meta_dir / "splits.json"

        if not subject_dict_path.exists():
            raise FileNotFoundError(f"Missing subject metadata: {subject_dict_path}")
        if not splits_path.exists():
            raise FileNotFoundError(f"Missing split metadata: {splits_path}")

        with subject_dict_path.open("r") as f:
            raw_subjects = json.load(f)
        with splits_path.open("r") as f:
            self._splits = json.load(f)

        self._subject_dict = {
            str(subj): tuple(values) for subj, values in raw_subjects.items()
        }
        self.data_root = data_dir

    def _build_dataset(self, split: str) -> S1200:
        if self._subject_dict is None or self._splits is None:
            self._load_metadata()

        subjects = self._splits.get(split, [])
        if not subjects:
            raise ValueError(f"Split '{split}' has no subjects in splits.json")

        missing = [s for s in subjects if s not in self._subject_dict]
        if missing:
            raise KeyError(f"Subjects {missing} from split '{split}' not found in subject_dict.json")

        subject_dict = {s: self._subject_dict[s] for s in subjects}

        return S1200(
            root=str(self.data_root),
            subject_dict=subject_dict,
            **self.dataset_kwargs,
        )

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            self.train_dataset = self._build_dataset("train")
            self.val_dataset = self._build_dataset("val")

        if stage in (None, "fit", "test"):
            self.test_dataset = self._build_dataset("test")

        # -------- label scaling (regression only)
        if (
            self.hparams["label_scaling_method"] != "none"
            and self.hparams["downstream_task_type"] == "regression"
        ):
            y = np.concatenate([self.train_dataset.targets, self.val_dataset.targets])
            scaler_cls = (
                StandardScaler
                if self.hparams["label_scaling_method"] == "standardization"
                else MinMaxScaler
            )
            self.scaler = scaler_cls().fit(y.reshape(-1, 1))
            for ds in (self.train_dataset, self.val_dataset, self.test_dataset):
                ds.targets = self.scaler.transform(ds.targets.reshape(-1, 1)).ravel()

        # expose for the model (needed for inverse-transform at test time)
        self.train_dataset.target_values = torch.tensor(
            self.train_dataset.targets, dtype=torch.float
        )

    # ------------- loaders ------------------------------
    def train_dataloader(self):
        if self.hparams["balanced_sampling"]:
            counts = Counter(self.train_dataset.subject_ids)
            weights = [1.0 / counts[sid] for sid in self.train_dataset.subject_ids]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        else:
            sampler = RandomSampler(self.train_dataset)

        persistent_workers = self.hparams["num_workers"] > 0

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size"],
            sampler=sampler,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            persistent_workers=persistent_workers,
        )

    def val_dataloader(self):
        persistent_workers = self.hparams["num_workers"] > 0

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams["batch_size"],
            sampler=SequentialSampler(self.val_dataset),
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            persistent_workers=persistent_workers,
        )

    def test_dataloader(self):
        persistent_workers = self.hparams["num_workers"] > 0

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams["batch_size"],
            sampler=SequentialSampler(self.test_dataset),
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            persistent_workers=persistent_workers,
        )
