from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .specs import STUDIES, StudySpec


def _read_table(path):
    if path is None:
        return {}
    with path.open(newline="") as fh:
        return {str(row["subject_id"]): row for row in csv.DictReader(fh)}


def _subjects(root, table):
    if table:
        return sorted(table)
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _target(row, cols):
    return [row.get(col, "nan") if row else "nan" for col in cols]


def _scan_id(path):
    for suffix in (".dtseries.nii", ".nii.gz", ".nii"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def write_manifest(spec, participants, source_root, data_dir):
    table = _read_table(participants)
    subjects = _subjects(source_root, table)
    meta = data_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "dataset.json").write_text(json.dumps({"schema": "r2t.manifest.v2", "study": spec.name, "source_root": str(source_root), "targets": list(spec.target_cols)}, indent=2) + "\n", encoding="utf-8")

    with (meta / "subjects.jsonl").open("w", encoding="utf-8") as fh:
        for sid in subjects:
            row = table.get(sid, {})
            targets = dict(zip(spec.target_cols, _target(row, spec.target_cols)))
            fh.write(json.dumps({"id": sid, "split": row.get("split", "test"), "sex": int(float(row.get("sex", 0) or 0)), "targets": targets}, separators=(",", ":")) + "\n")

    with (meta / "scans.jsonl").open("w", encoding="utf-8") as fh:
        for sid in subjects:
            subject_dir = source_root / sid
            if not subject_dir.exists():
                continue
            for item in spec.scans:
                for source in sorted(subject_dir.glob(item.pattern)):
                    scan_id = _scan_id(source)
                    rec = {"id": f"{sid}:{scan_id}", "subject": sid, "scan": scan_id, "role": item.role, "kind": item.input_kind, "frames": f"blocks/{sid}/{scan_id}", "n_frames": item.n_frames, "source": str(source)}
                    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Write study manifests from source trees")
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument("--participants", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = STUDIES[args.study]
    write_manifest(spec, args.participants, args.source_root or spec.source_root, args.data_dir or spec.data_dir)


if __name__ == "__main__":
    main()
