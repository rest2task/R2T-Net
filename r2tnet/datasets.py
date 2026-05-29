"""Datasets for prepared fMRI frame stores."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import ROLE_ID, ScanRecord, SubjectRecord, frame_index


def _load_frame(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


class PreparedScanMixin:
    sequence_length: int
    stride_within_seq: int

    def _read_clip(self, scan, start):
        frames = frame_index(scan.path)
        idxs = range(start, start + self.sequence_length * self.stride_within_seq, self.stride_within_seq)
        tensors = [_load_frame(frames[i]) for i in idxs]
        return torch.stack(tensors, dim=-1).float()

    def _random_start(self, scan):
        max_start = scan.n_frames - self.sequence_length * self.stride_within_seq
        if max_start < 0:
            raise ValueError(f"{scan.path} has {scan.n_frames} frames; need {self.sequence_length}")
        return random.randint(0, max_start)


class PairedFMRIWindowDataset(PreparedScanMixin, Dataset):
    def __init__(self, subjects, scans, split, sequence_length, stride_within_seq = 1):
        self.subjects = {
            sid: rec
            for sid, rec in subjects.items()
            if rec.split == split
            and any(s.role == "rest" for s in scans.get(sid, []))
            and any(s.role == "task" for s in scans.get(sid, []))
        }
        self.scans = scans
        self.sequence_length = sequence_length
        self.stride_within_seq = stride_within_seq
        self.subject_ids = sorted(self.subjects)
        if not self.subject_ids:
            raise ValueError(f"No paired rest/task subjects for split {split!r}")
        self.targets = np.stack([self.subjects[sid].target for sid in self.subject_ids])

    @property
    def target_values(self):
        return torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        subject = self.subjects[sid]
        rest_scan = random.choice([s for s in self.scans[sid] if s.role == "rest"])
        task_scan = random.choice([s for s in self.scans[sid] if s.role == "task"])
        rest = self._read_clip(rest_scan, self._random_start(rest_scan))
        task = self._read_clip(task_scan, self._random_start(task_scan))
        return {
            "rest": rest,
            "task": task,
            "target": torch.tensor(subject.target, dtype=torch.float32),
            "sex": subject.sex,
            "subject_id": sid,
        }


class SingleFMRIWindowDataset(PreparedScanMixin, Dataset):
    def __init__(self, subjects, scans, split, role, sequence_length, stride_between_seq = 1, stride_within_seq = 1):
        if role not in ROLE_ID:
            raise ValueError(f"Bad role {role!r}")
        self.subjects = {sid: rec for sid, rec in subjects.items() if rec.split == split}
        self.scans = scans
        self.role = role
        self.sequence_length = sequence_length
        self.stride_within_seq = stride_within_seq
        sample_duration = sequence_length * stride_within_seq
        step = max(int(stride_between_seq * sample_duration), 1)

        self.samples: List[tuple[str, ScanRecord, int]] = []
        for sid, subject in sorted(self.subjects.items()):
            for scan in scans.get(sid, []):
                if scan.role != role:
                    continue
                max_start = scan.n_frames - sample_duration
                for start in range(0, max_start + 1, step):
                    self.samples.append((sid, scan, start))
        if not self.samples:
            raise ValueError(f"No {role} samples for split {split!r}")
        self.targets = np.stack([self.subjects[sid].target for sid, _, _ in self.samples])
        self.subject_ids = [sid for sid, _, _ in self.samples]

    @property
    def target_values(self):
        return torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, scan, start = self.samples[idx]
        subject = self.subjects[sid]
        return {
            "fmri": self._read_clip(scan, start),
            "modality": torch.tensor(ROLE_ID[scan.role], dtype=torch.long),
            "target": torch.tensor(subject.target, dtype=torch.float32),
            "sex": subject.sex,
            "subject_id": sid,
            "scan_id": scan.scan_id,
        }
