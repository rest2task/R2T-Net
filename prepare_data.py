"""Prepare R2T-Net frame stores."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from r2tnet.manifest import scan_rows, write_templates
from r2tnet.preprocess import convert_source_to_frames


FORMAT_NOTE = """meta/subjects.jsonl: {"id","split","sex","targets":{...}}
meta/scans.jsonl: {"id","subject","scan","role","kind","frames","n_frames","source"}
kind: vol=.nii/.nii.gz -> [1,X,Y,Z], grayord=.dtseries.nii -> [V], roi=.dtseries.nii+atlas -> [V]"""


def _dtype(name):
    return {"float16": torch.float16, "float32": torch.float32}[name]


def _kind(input_kind, atlas):
    if input_kind == "vol":
        return "raw4d"
    if input_kind == "grayord":
        return "grayord"
    if input_kind == "roi":
        if atlas is None:
            raise ValueError("--parcellation-atlas is required for input_kind=roi")
        return "parcel"
    raise ValueError(input_kind)


def convert_manifest(args):
    data_dir = args.data_dir.expanduser().resolve()
    _, rows = scan_rows(data_dir)

    written = 0
    for row in rows:
        source_value = row.get("source") or row.get("source_path")
        if not source_value:
            continue
        out_dir = Path(row.get("frames") or row["path"])
        if not out_dir.is_absolute():
            out_dir = data_dir / out_dir
        source = Path(source_value)
        kind = _kind(row.get("kind") or row["input_kind"], args.parcellation_atlas)
        n_frames = convert_source_to_frames(
            source=source,
            output_dir=out_dir,
            kind=kind,
            normalise=args.normalise,
            dtype=_dtype(args.dtype),
            parcellation_atlas=args.parcellation_atlas,
            wb_command=args.wb_command,
        )
        written += 1
        print(f"{row.get('subject') or row.get('subject_id')} {row.get('scan') or row.get('scan_id')}: {n_frames} frames")

    print(f"converted {written} scans")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare R2T-Net frame data", epilog=FORMAT_NOTE)
    sub = parser.add_subparsers(dest="cmd", required=True)

    tmpl = sub.add_parser("init", help="create empty manifest templates")
    tmpl.add_argument("--data-dir", type=Path, required=True)

    conv = sub.add_parser("convert", help="convert scans listed in meta/scans.jsonl")
    conv.add_argument("--data-dir", type=Path, required=True)
    conv.add_argument("--normalise", choices=["none", "zscore", "minmax"], default="none")
    conv.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    conv.add_argument("--parcellation-atlas", type=Path)
    conv.add_argument("--wb-command", default="wb_command")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "init":
        write_templates(args.data_dir.expanduser().resolve())
        print(f"created metadata templates under {args.data_dir / 'meta'}")
    elif args.cmd == "convert":
        convert_manifest(args)


if __name__ == "__main__":
    main()
