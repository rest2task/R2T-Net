"""Export 2,048-D dynamic activity signatures from a checkpointed R2T-Net.

The utility mirrors :mod:`inference.py` but, instead of running the
prediction head, it collects the latent embeddings produced by the
resting-state and task encoders. Each modality independently yields a
2,048-D descriptor; when both are available the script averages them to
obtain the dynamic activity signature described in the paper.

Examples
--------
# Rest + task pairs → 2,048-D dynamic signatures (default)
python extract.py \
    --ckpt logs/epoch03-valid_loss=0.2100.ckpt \
    --rest_dir data/S1200/rest \
    --task_dir data/S1200/task \
    --output dynamic_signatures.pt

# Rest-only fallback → 2,048-D signatures from rs-fMRI only
python extract.py \
    --ckpt logs/epoch03-valid_loss=0.2100.ckpt \
    --rest_dir data/S1200/rest \
    --rest_only \
    --output rest_signatures.pt
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import OrderedDict
from typing import Dict

import torch
from tqdm import tqdm

from module.r2tnet import R2TNet


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _infer_modality_id(x: torch.Tensor) -> int:
    """Infer the modality id expected by :class:`R2TNet` from tensor shape."""

    if x.ndim == 5:  # volumetric clip [C, H, W, D, T]
        return 0
    if x.ndim == 2:  # ROI / grayordinate matrix [V, T]
        return 1 if x.shape[0] < 91_282 else 2
    raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}")


def _instantiate_from_ckpt(ckpt_path: str, num_rois: int, device: torch.device) -> R2TNet:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    if num_rois > 0:
        hparams["num_rois"] = num_rois

    dummy_dm = type("DummyDM", (), {"train_dataset": type("T", (), {"target_values": [[0.0]]})})()
    model = R2TNet(data_module=dummy_dm, **hparams)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def _encode_folder(
    model: R2TNet,
    folder: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Return an ordered mapping filename → latent signature."""

    signatures: Dict[str, torch.Tensor] = OrderedDict()
    pt_files = sorted(glob.glob(os.path.join(folder, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {folder}")

    for path in tqdm(pt_files, desc=f"encoding {os.path.basename(folder)}"):
        tensor = torch.load(path, map_location="cpu")
        modality_id = _infer_modality_id(tensor)
        x = tensor.unsqueeze(0).to(device)
        modality = torch.tensor([modality_id], device=device)
        signature = model.encode(x, modality).squeeze(0).cpu()
        signatures[os.path.basename(path)] = signature
    return signatures


def _merge_pairs(
    rest: Dict[str, torch.Tensor],
    task: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Average rest + task signatures, ensuring 1:1 filename matches."""

    rest_keys = set(rest.keys())
    task_keys = set(task.keys())
    if rest_keys != task_keys:
        missing_in_task = sorted(rest_keys - task_keys)
        missing_in_rest = sorted(task_keys - rest_keys)
        raise ValueError(
            "Mismatch between rest and task files. "
            f"Missing in task: {missing_in_task[:5]} "
            f"Missing in rest: {missing_in_rest[:5]}"
        )

    merged: Dict[str, torch.Tensor] = OrderedDict()
    for key in rest.keys():
        merged[key] = 0.5 * (rest[key] + task[key])
    return merged


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export dynamic activity signatures from an R2T-Net checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", required=True, help="Path to .ckpt or .pth checkpoint")
    ap.add_argument("--rest_dir", required=True, help="Directory with resting-state tensors (*.pt)")
    ap.add_argument("--task_dir", help="Directory with task fMRI tensors (*.pt)")
    ap.add_argument(
        "--rest_only",
        action="store_true",
        help="Allow exporting rest signatures without task pairs",
    )
    ap.add_argument("--output", default="signatures.pt", help="Path to the output .pt file")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="cuda:N or cpu")
    ap.add_argument("--num_rois", type=int, default=0, help="Override ROI size when using ROI inputs")
    args = ap.parse_args()

    if not args.rest_only and not args.task_dir:
        ap.error("--task_dir is required unless --rest_only is set")

    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = _instantiate_from_ckpt(args.ckpt, args.num_rois, device)

    with torch.no_grad():
        rest_signatures = _encode_folder(model, args.rest_dir, device)
        if args.task_dir:
            task_signatures = _encode_folder(model, args.task_dir, device)
            signatures = _merge_pairs(rest_signatures, task_signatures)
        else:
            signatures = rest_signatures

    torch.save(signatures, args.output)
    example_shape = next(iter(signatures.values())).shape
    print(f"✅  Saved {len(signatures)} signatures (dim={example_shape[-1]}) → {args.output}")


if __name__ == "__main__":
    main()
