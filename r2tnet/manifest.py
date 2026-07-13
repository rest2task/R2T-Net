"""Manifest helpers for prepared fMRI datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


ROLE_ID = {"rest": 0, "task": 1}
INPUT_KINDS = {"vol", "grayord", "roi"}


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    split: str
    sex: int
    target: np.ndarray


@dataclass(frozen=True)
class ScanRecord:
    subject_id: str
    scan_id: str
    role: str
    path: Path
    input_kind: str
    n_frames: int
    storage: str = "frames"
    dtype: str = ""
    shape: tuple[int, ...] = ()
    mask_path: Path | None = None
    mask_voxels: int = 0


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _subject_rows(data_dir):
    path = data_dir / "meta" / "subjects.jsonl"
    if path.exists():
        return "jsonl", read_jsonl(path)
    return "csv", read_csv(data_dir / "meta" / "subjects.csv")


def scan_rows(data_dir):
    path = data_dir / "meta" / "scans.jsonl"
    if path.exists():
        return "jsonl", read_jsonl(path)
    return "csv", read_csv(data_dir / "meta" / "scans.csv")


def resolve_path(data_dir, value):
    path = Path(value)
    return path if path.is_absolute() else data_dir / path


def _parse_shape(value):
    if value in (None, ""):
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    text = str(value).strip()
    if not text:
        return ()
    if text.startswith("["):
        return tuple(int(x) for x in json.loads(text))
    return tuple(int(x) for x in text.replace("x", ",").split(",") if x.strip())


def frame_index(path):
    frames = sorted(path.glob("frame_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not frames:
        raise FileNotFoundError(f"No frame_*.pt files in {path}")
    return frames


def _target_keys(rows, target_cols):
    if target_cols:
        return target_cols
    rows = list(rows)
    if not rows:
        raise ValueError("No subject rows found")
    if isinstance(rows[0].get("targets"), dict):
        return list(rows[0]["targets"].keys())
    cols = [c for c in rows[0].keys() if c.startswith("target")]
    if not cols and "target" in rows[0]:
        cols = ["target"]
    if not cols:
        raise ValueError("No target columns found. Provide --target_cols or targets{}.")
    return cols


def target_keys(data_dir, target_cols = None):
    _, rows = _subject_rows(data_dir)
    return list(_target_keys(rows, target_cols))


def load_subjects(data_dir, target_cols = None):
    kind, rows = _subject_rows(data_dir)
    out: Dict[str, SubjectRecord] = {}
    target_cols = _target_keys(rows, target_cols)

    for row in rows:
        subject_id = str(row.get("id") or row.get("subject_id"))
        target_src = row.get("targets") if kind == "jsonl" else row
        target = np.asarray([float(target_src[c]) for c in target_cols], dtype=np.float32)
        sex = int(float(row.get("sex", 0) or 0))
        out[subject_id] = SubjectRecord(subject_id=subject_id, split=str(row["split"]), sex=sex, target=target)
    return out


def load_scans(data_dir):
    kind, rows = scan_rows(data_dir)
    out: Dict[str, List[ScanRecord]] = {}
    for row in rows:
        subject_id = str(row.get("subject") or row.get("subject_id"))
        role = row["role"].strip().lower()
        input_kind = (row.get("kind") or row.get("input_kind")).strip().lower()
        if role not in ROLE_ID:
            raise ValueError(f"Bad scan role {role!r}; expected rest or task")
        if input_kind not in INPUT_KINDS:
            raise ValueError(f"Bad input_kind {input_kind!r}; expected {sorted(INPUT_KINDS)}")
        path = resolve_path(data_dir, row.get("frames") or row["path"])
        mask_value = row.get("mask_path") or row.get("mask")
        n_frames = int(row.get("n_frames") or len(frame_index(path)))
        scan_id = str(row.get("scan") or row.get("scan_id") or row["id"])
        if kind == "jsonl" and ":" in scan_id:
            scan_id = scan_id.split(":", 1)[1]
        rec = ScanRecord(
            subject_id=subject_id,
            scan_id=scan_id,
            role=role,
            path=path,
            input_kind=input_kind,
            n_frames=n_frames,
            storage=str(row.get("storage") or row.get("format") or "frames").strip().lower(),
            dtype=str(row.get("dtype") or row.get("storage_dtype") or "").strip().lower(),
            shape=_parse_shape(row.get("shape") or row.get("spatial_shape") or row.get("volume_shape")),
            mask_path=resolve_path(data_dir, mask_value) if mask_value else None,
            mask_voxels=int(row.get("mask_voxels") or row.get("n_mask_voxels") or 0),
        )
        out.setdefault(rec.subject_id, []).append(rec)
    return out


def write_templates(data_dir):
    meta = data_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    dataset = meta / "dataset.json"
    subjects = meta / "subjects.jsonl"
    scans = meta / "scans.jsonl"
    if not dataset.exists():
        dataset.write_text(json.dumps({"schema": "r2t.manifest.v2", "frame_pattern": "frame_{index}.pt"}, indent=2) + "\n", encoding="utf-8")
    if not subjects.exists():
        subjects.write_text("", encoding="utf-8")
    if not scans.exists():
        scans.write_text("", encoding="utf-8")
