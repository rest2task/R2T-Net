"""Datasets for prepared fMRI frame stores."""

from __future__ import annotations

from functools import lru_cache
import random
from math import prod
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import ROLE_ID, ScanRecord, SubjectRecord, frame_index


BF16_BYTES = 2


def _load_frame(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


class PreparedScanMixin:
    sequence_length: int
    stride_within_seq: int

    def _read_clip(self, scan, start):
        if _is_int10_volume_scan(scan):
            return _read_int10_volume_clip(scan, start, self.sequence_length, self.stride_within_seq)
        if _is_bf16_volume_scan(scan):
            return _read_bf16_volume_clip(scan, start, self.sequence_length, self.stride_within_seq)
        frames = frame_index(scan.path)
        idxs = range(start, start + self.sequence_length * self.stride_within_seq, self.stride_within_seq)
        tensors = [_load_frame(frames[i]) for i in idxs]
        return torch.stack(tensors, dim=-1).float()

    def _random_start(self, scan):
        max_start = scan.n_frames - self.sequence_length * self.stride_within_seq
        if max_start < 0:
            raise ValueError(f"{scan.path} has {scan.n_frames} frames; need {self.sequence_length}")
        return random.randint(0, max_start)


def _is_bf16_volume_scan(scan: ScanRecord) -> bool:
    return (
        scan.input_kind == "vol"
        and (
            scan.storage in {"bf16_volume", "raw_bf16", "volume_bf16", "bf16_masked_volume", "masked_bf16_volume"}
            or scan.path.suffix == ".bf16"
        )
    )


def _is_int10_volume_scan(scan: ScanRecord) -> bool:
    return scan.input_kind == "vol" and (scan.storage == "int10_masked_volume" or scan.path.suffix == ".i10")


def _read_exact_into(path: Path, offset: int, nbytes: int, out: torch.Tensor):
    view = memoryview(out.view(torch.uint8).numpy().reshape(-1))
    done = 0
    with path.open("rb", buffering=0) as fh:
        fh.seek(offset)
        while done < nbytes:
            n = fh.readinto(view[done:])
            if n is None:
                continue
            if n == 0:
                raise EOFError(f"Short read from {path}: got {done} bytes, expected {nbytes}")
            done += n


def _read_bf16_volume_clip(scan: ScanRecord, start: int, length: int, stride: int) -> torch.Tensor:
    if not scan.shape:
        raise ValueError(f"{scan.path} is a BF16 volume scan but has no spatial shape in the manifest")
    spatial = tuple(int(x) for x in scan.shape)
    span = (int(length) - 1) * int(stride) + 1
    masked = scan.storage in {"bf16_masked_volume", "masked_bf16_volume"}
    if masked:
        if scan.mask_path is None:
            raise ValueError(f"{scan.path} is a masked BF16 volume scan but has no mask_path in the manifest")
        mask_indices = _load_mask_indices(scan.mask_path)
        frame_values = int(scan.mask_voxels) or int(mask_indices.numel())
        if frame_values != int(mask_indices.numel()):
            raise ValueError(f"{scan.path} mask_voxels={frame_values} but {scan.mask_path} has {mask_indices.numel()} indices")
    else:
        mask_indices = None
        frame_values = prod(spatial)
    nbytes = span * frame_values * BF16_BYTES
    offset = int(start) * frame_values * BF16_BYTES
    data = torch.empty((span, frame_values), dtype=torch.bfloat16) if masked else torch.empty((span, *spatial), dtype=torch.bfloat16)
    _read_exact_into(scan.path, offset, nbytes, data)
    if int(stride) != 1:
        data = data[:: int(stride)].contiguous()
    data = data[: int(length)]
    if masked:
        dense = torch.zeros((int(length), prod(spatial)), dtype=torch.bfloat16)
        dense[:, mask_indices] = data
        data = dense.reshape((int(length), *spatial))
    return data.movedim(0, -1).unsqueeze(0).float()


def _read_int10_volume_clip(scan: ScanRecord, start: int, length: int, stride: int) -> torch.Tensor:
    if not scan.shape:
        raise ValueError(f"{scan.path} is an INT10 volume scan but has no spatial shape in the manifest")
    if scan.mask_path is None:
        raise ValueError(f"{scan.path} is an INT10 volume scan but has no mask_path in the manifest")
    spatial = tuple(int(x) for x in scan.shape)
    if len(spatial) != 3:
        raise ValueError(f"Expected a 3-D INT10 spatial shape, got {spatial} for {scan.path}")
    mask_indices = _load_mask_indices(scan.mask_path)
    frame_values = int(scan.mask_voxels) or int(mask_indices.numel())
    if frame_values != int(mask_indices.numel()):
        raise ValueError(f"{scan.path} mask_voxels={frame_values} but {scan.mask_path} has {mask_indices.numel()} indices")

    from experiments_4D.code.recover_i10 import read_i10_masked

    frame_indices = range(int(start), int(start) + int(length) * int(stride), int(stride))
    values = read_i10_masked(
        scan.path,
        scan.mask_path,
        frame_indices,
        n_values=frame_values,
        workers=1,
    )
    dense = torch.zeros((int(length), prod(spatial)), dtype=torch.float32)
    dense[:, mask_indices] = torch.from_numpy(values)
    return dense.reshape((int(length), *spatial)).movedim(0, -1).unsqueeze(0)


@lru_cache(maxsize=4096)
def _load_mask_indices(path: Path) -> torch.Tensor:
    arr = np.load(path)
    return torch.from_numpy(np.asarray(arr, dtype=np.int64))


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
