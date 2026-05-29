"""Source-to-frame conversion helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch


SourceKind = Literal["raw4d", "grayord", "parcel"]


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for source conversion") from exc
    return nib


def _load_raw4d(path):
    nib = _require_nibabel()
    arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    if arr.ndim != 4:
        raise ValueError(f"RAW 4D input must be [X,Y,Z,T], got {arr.shape} from {path}")
    return arr


def _load_cifti(path):
    nib = _require_nibabel()
    arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"CIFTI series must be [T,V], got {arr.shape} from {path}")
    return arr


def _parcellate_cifti(source, atlas, wb_command):
    if shutil.which(wb_command) is None:
        raise FileNotFoundError(f"{wb_command} not found")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "parcellated.ptseries.nii"
        subprocess.run(
            [
                wb_command,
                "-cifti-parcellate",
                str(source),
                str(atlas),
                "COLUMN",
                str(out),
                "-method",
                "MEAN",
            ],
            check=True,
        )
        return _load_cifti(out)


def _normalise(data, method):
    if method == "none":
        return data
    if method == "zscore":
        mean = np.nanmean(data)
        std = np.nanstd(data)
        return (data - mean) / max(float(std), 1e-6)
    if method == "minmax":
        mn = np.nanmin(data)
        mx = np.nanmax(data)
        return (data - mn) / max(float(mx - mn), 1e-6)
    raise ValueError(f"Unknown normalisation method {method}")


def _iter_frames(data, kind):
    if kind == "raw4d":
        for t in range(data.shape[-1]):
            yield torch.from_numpy(data[..., t]).unsqueeze(0)
    else:
        for t in range(data.shape[0]):
            yield torch.from_numpy(data[t])


def convert_source_to_frames(source, output_dir, kind, normalise = "none", dtype = torch.float16, parcellation_atlas = None, wb_command = "wb_command"):
    source = source.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if kind == "raw4d":
        data = _load_raw4d(source)
    elif kind == "grayord":
        data = _load_cifti(source)
    elif kind == "parcel":
        if parcellation_atlas is None:
            raise ValueError("parcellation_atlas is required for parcel conversion")
        data = _parcellate_cifti(source, parcellation_atlas.expanduser().resolve(), wb_command)
    else:
        raise ValueError(kind)

    data = _normalise(data, normalise).astype(np.float32, copy=False)
    count = 0
    for count, frame in enumerate(_iter_frames(data, kind), start=1):
        torch.save(frame.to(dtype), output_dir / f"frame_{count - 1}.pt")
    return count
