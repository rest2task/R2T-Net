"""Utility script to convert HCP resting-state NIfTI volumes into fp16 tensors."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import torch
from monai.transforms import LoadImage


def process_subject(
    subj_name: str,
    load_root: Path,
    save_img_root: Path,
    nifti_name: str,
    scaling_method: str = "minmax",
    fill_zeroback: bool = True,
) -> bool:
    """Load a single subject's NIfTI file and dump each TR as ``frame_X.pt``.

    Returns ``True`` if the subject was processed successfully and ``False`` otherwise.
    """

    src_path = load_root / subj_name / nifti_name
    if not src_path.exists():
        print(f"[skip] {subj_name}: missing {nifti_name}")
        return False

    print(f"### Processing: {subj_name}, scaling_method={scaling_method}", flush=True)

    try:
        data, _ = LoadImage()(str(src_path))
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[error] {subj_name}: failed to load ({exc})")
        return False

    # Dataset-specific crop (adjust as needed for alternative alignments)
    data = data[:, 14:-7, :, :]

    background = data == 0
    if background.all():
        print(f"[warn] {subj_name}: volume appears empty after crop; skipping")
        return False

    if scaling_method == "z-norm":
        global_mean = data[~background].mean()
        global_std = data[~background].std()
        scaled = (data - global_mean) / global_std
        data_global = torch.empty_like(data)
        fill_value = scaled[~background].min() if not fill_zeroback else 0
        data_global[background] = fill_value
        data_global[~background] = scaled[~background]
    else:
        non_zero = data[~background]
        if non_zero.numel() == 0:
            print(f"[warn] {subj_name}: no foreground voxels detected; skipping")
            return False
        denom = non_zero.max() - non_zero.min()
        if denom.abs() < 1e-8:
            scaled_values = torch.zeros_like(non_zero)
        else:
            scaled_values = (non_zero - non_zero.min()) / denom
        data_global = torch.empty_like(data)
        data_global[background] = 0 if fill_zeroback else non_zero.min()
        data_global[~background] = scaled_values

    save_dir = save_img_root / subj_name
    save_dir.mkdir(parents=True, exist_ok=True)

    data_global = data_global.to(torch.float16)
    for idx, tr_volume in enumerate(torch.split(data_global, 1, dim=3)):
        torch.save(tr_volume.clone(), save_dir / f"frame_{idx}.pt")

    print(f"    -> saved {idx + 1} frames to {save_dir}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess HCP resting-state runs into tensors")
    parser.add_argument("--load-root", type=Path, required=True, help="Folder containing subject sub-directories with NIfTI files")
    parser.add_argument("--save-root", type=Path, required=True, help="Destination folder for fp16 tensors")
    parser.add_argument(
        "--nifti-name",
        type=str,
        default="rfMRI_REST1_LR_hp2000_clean.nii.gz",
        help="Name of the resting-state NIfTI file inside each subject folder",
    )
    parser.add_argument(
        "--scaling-method",
        choices=["minmax", "z-norm"],
        default="minmax",
        help="Intensity normalisation strategy",
    )
    parser.add_argument(
        "--expected-length",
        type=int,
        default=1200,
        help="Number of TRs expected per subject; used to skip already processed folders",
    )
    parser.add_argument(
        "--fill-zero-background",
        dest="fill_zero_background",
        action="store_true",
        help="Fill background voxels with zero after scaling",
    )
    parser.add_argument(
        "--keep-min-background",
        dest="fill_zero_background",
        action="store_false",
        help="Reuse the minimum foreground value instead of zero for background voxels",
    )
    parser.set_defaults(fill_zero_background=True)
    return parser.parse_args()


def iter_subjects(load_root: Path) -> Iterable[str]:
    for path in sorted(load_root.iterdir()):
        if path.is_dir():
            yield path.name


def main() -> None:
    args = parse_args()
    load_root = args.load_root.expanduser().resolve()
    save_root = args.save_root.expanduser().resolve()

    img_root = save_root / "img"
    meta_root = save_root / "meta"
    img_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    processed = 0

    for subj_name in iter_subjects(load_root):
        dest_dir = img_root / subj_name
        existing = len(list(dest_dir.glob("frame_*.pt")))
        if existing >= args.expected_length:
            print(f"[skip] {subj_name}: found {existing} frames (>= {args.expected_length})")
            continue

        ok = process_subject(
            subj_name=subj_name,
            load_root=load_root,
            save_img_root=img_root,
            nifti_name=args.nifti_name,
            scaling_method=args.scaling_method,
            fill_zeroback=args.fill_zero_background,
        )
        if ok:
            processed += 1

    minutes = (time.time() - start_time) / 60
    print(f"Finished preprocessing {processed} subjects in {minutes:.1f} min")


if __name__ == "__main__":
    main()
