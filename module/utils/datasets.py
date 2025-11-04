# module/utils/datasets.py
# ----------------------------------------------------------
# A minimal, shape-agnostic dataset wrapper for HCP-S1200.
#
#  ▸ Supports three input kinds
#      "vol"       : 4-D volumes  [C, H, W, D]  per frame
#      "roi"       : ROI vectors  [V]            per frame
#      "grayord"   : grayordinates [91 282]      per frame
#  ▸ Optional NT-Xent pairs (contrastive mode)
#  ▸ Optional voxel-wise z-norm maps (for volumes only)
#  ▸ Returns ‘modality’ id ∈ {0,1,2} so the network can
#    add a learnable embedding.
# ----------------------------------------------------------
import os, random
from typing import List, Tuple

import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# ----------------------------------------------------------
# helpers
# ----------------------------------------------------------
_MODALITY_ID = {"vol": 0, "roi": 1, "grayord": 2}


def _load_pt(path: str) -> torch.Tensor:
    """
    Small wrapper so that a missing file throws a clearer error.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


# ----------------------------------------------------------
# Base class
# ----------------------------------------------------------
class BaseDataset(Dataset):
    """
    Generic time-series dataset for fMRI volumes / ROI vectors.
    A ‘frame’ is a single TR stored as <frame_#.pt>.
    """
    def __init__(
        self,
        root: str,
        subject_dict: dict,
        task_type: str = "rest",           # labels only: 'rest' or 'task'
        input_kind: str = "vol",           # 'vol', 'roi', or 'grayord'
        sequence_length: int = 20,
        stride_between_seq: int = 1,
        stride_within_seq: int = 1,
        contrastive: bool = False,
        with_voxel_norm: bool = False,
        shuffle_time_sequence: bool = False,
    ):
        assert task_type in ("rest", "task")
        assert input_kind in _MODALITY_ID, f"bad input_kind {input_kind}"
        self.root = root
        self.subject_dict = subject_dict          # subj-id → (sex, target)
        self.task_type = task_type
        self.input_kind = input_kind
        self.modality_id = _MODALITY_ID[input_kind]

        self.sequence_length = sequence_length
        self.stride_within_seq = stride_within_seq
        self.sample_duration = sequence_length * stride_within_seq

        # distance between consecutive *starting* indices
        self.stride_between_seq = max(int(stride_between_seq * self.sample_duration), 1)

        self.contrastive = contrastive
        self.with_voxel_norm = with_voxel_norm and input_kind == "vol"
        self.shuffle_time_sequence = shuffle_time_sequence and not contrastive

        # Pre-compute index tuples so __getitem__ is fast
        self.samples = self._index_subjects()
        self.targets = np.array(self._targets, dtype=np.float32)
        self.subject_ids = self._subject_ids

        # Free auxiliary lists – we no longer need them after NumPy conversion.
        del self._targets
        del self._subject_ids

    # ---------- index all clips -----------------------------------------
    def _index_subjects(self) -> List[Tuple[str, int, int, int, str]]:
        """
        Returns a list like:
            (subject_path, start_frame, num_frames, sex, subject_id)
        That list is stored in self.samples while per-sample targets and
        subject IDs are tracked separately for fast lookups.
        """
        out: List[Tuple[str, int, int, int, str]] = []
        self._targets: List[float] = []
        self._subject_ids: List[str] = []
        img_root = os.path.join(self.root, "img")
        for subj, (sex, target) in self.subject_dict.items():
            s_path = os.path.join(img_root, subj)
            n_frames = len([f for f in os.listdir(s_path) if f.startswith("frame_")])
            max_start = n_frames - self.sample_duration + 1
            for st in range(0, max_start, self.stride_between_seq):
                out.append((s_path, st, n_frames, sex, subj))
                self._targets.append(float(target))
                self._subject_ids.append(subj)
        return out

    # ---------- IO helpers ----------------------------------------------
    def _read_frames(self, subj_path: str, start: int) -> torch.Tensor:
        """
        Load a clip of length `self.sequence_length` starting at `start`.
        We always step by `self.stride_within_seq`.
        Returns:
            - vol:  Tensor [C,H,W,D,T]
            - roi:  Tensor [V,T]
        """
        idxs = (
            random.sample(
                range(start, start + self.sample_duration, self.stride_within_seq),
                self.sequence_length,
            )
            if self.shuffle_time_sequence
            else range(start, start + self.sample_duration, self.stride_within_seq)
        )

        frames = [_load_pt(os.path.join(subj_path, f"frame_{i}.pt")) for i in idxs]
        clip = torch.stack(frames, dim=-1)  # time dim last

        if self.with_voxel_norm:
            μ = _load_pt(os.path.join(subj_path, "voxel_mean.pt")).unsqueeze(-1)
            σ = _load_pt(os.path.join(subj_path, "voxel_std.pt")).unsqueeze(-1)
            clip = torch.cat([clip, μ, σ], dim=0)  # aux chans on channel axis

        return clip

    # ---------- NT-Xent pair builder ------------------------------------
    def _make_pair(self, subj_path: str, start: int, num_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build positive pair centred at `start` and an *out-of-window* negative
        starting somewhere else in the scan.
        """
        seq1 = self._read_frames(subj_path, start)

        margin = self.sample_duration
        bad = set(range(start - margin, start + margin))
        candidates = [s for s in range(num_frames - self.sample_duration + 1) if s not in bad]
        seq2_start = random.choice(candidates)
        seq2 = self._read_frames(subj_path, seq2_start)
        return seq1, seq2

    # ---------- standard interface --------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        subj_path, start, n_frames, sex, subj = self.samples[idx]
        target = self.targets[idx]

        if self.contrastive:
            seq1, seq2 = self._make_pair(subj_path, start, n_frames)
            return {
                "fmri1": seq1,                     # shape depends on input_kind
                "modality1": torch.tensor(self.modality_id, dtype=torch.long),
                "fmri2": seq2,
                "modality2": torch.tensor(self.modality_id, dtype=torch.long),
                "target": torch.tensor(target, dtype=torch.float32),
                "sex": sex,
                "subject_id": subj,
            }

        seq = self._read_frames(subj_path, start)
        return {
            "fmri": seq,
            "modality": torch.tensor(self.modality_id, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
            "sex": sex,
            "subject_id": subj,
        }


# ----------------------------------------------------------
# S1200 specialisation
# ----------------------------------------------------------
class S1200(BaseDataset):
    """
    Thin wrapper for the HCP-S1200 release.  All logic already lives
    in BaseDataset; we only override init to plug default paths.
    """
    def __init__(self, root: str, subject_dict: dict, **kwargs):
        super().__init__(root=root, subject_dict=subject_dict, **kwargs)
