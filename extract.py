from __future__ import annotations

import argparse
import glob
import os
from collections import OrderedDict
from typing import Dict

import torch
from tqdm import tqdm

from r2tnet.model import R2TNet


def _instantiate_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hparams", ckpt.get("hyper_parameters", {}))
    model = R2TNet(torch.zeros(4, int(hparams.get("target_dim", 1))), **hparams)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def _encode_folder(model, folder, device, role):
    signatures: Dict[str, torch.Tensor] = OrderedDict()
    files = sorted(glob.glob(os.path.join(folder, "*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {folder}")

    modality_id = 0 if role == "rest" else 1
    for path in tqdm(files, desc=f"encoding {os.path.basename(folder)}"):
        x = torch.load(path, map_location="cpu").unsqueeze(0).to(device)
        modality = torch.tensor([modality_id], device=device)
        signatures[os.path.basename(path)] = model.encode(x, modality).squeeze(0).cpu()
    return signatures


def _merge_pairs(rest, task):
    if set(rest) != set(task):
        missing_task = sorted(set(rest) - set(task))
        missing_rest = sorted(set(task) - set(rest))
        raise ValueError(f"Mismatched rest/task files; missing_task={missing_task[:5]}, missing_rest={missing_rest[:5]}")
    return OrderedDict((key, 0.5 * (rest[key] + task[key])) for key in rest)


def parse_args():
    parser = argparse.ArgumentParser(description="Export R2T-Net signatures", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--rest_dir", required=True)
    parser.add_argument("--task_dir")
    parser.add_argument("--rest_only", action="store_true")
    parser.add_argument("--output", default="signatures.pt")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.rest_only and not args.task_dir:
        parser.error("--task_dir is required unless --rest_only is set")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = _instantiate_from_ckpt(args.ckpt, device)
    with torch.no_grad():
        rest = _encode_folder(model, args.rest_dir, device, "rest")
        signatures = _merge_pairs(rest, _encode_folder(model, args.task_dir, device, "task")) if args.task_dir else rest
    torch.save(signatures, args.output)
    dim = next(iter(signatures.values())).shape[-1]
    print(f"wrote {len(signatures)} signatures, dim={dim}, to {args.output}")


if __name__ == "__main__":
    main()
