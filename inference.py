from __future__ import annotations

import csv
import glob
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

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


@torch.no_grad()
def _predict_tensor(model, x, modality_id, device):
    x = x.unsqueeze(0).to(device)
    modality = torch.tensor([modality_id], device=device)
    _, pred = model(x, modality)
    values = model.inverse_scale(pred).squeeze(0).to("cpu").float()
    if values.ndim == 0:
        values = values.unsqueeze(0)
    return values.tolist()


def predict_folder(model, input_dir, device, role):
    rows = []
    modality_id = 0 if role == "rest" else 1
    for path in tqdm(sorted(glob.glob(os.path.join(input_dir, "*.pt"))), desc="inference"):
        x = torch.load(path, map_location="cpu")
        if x.ndim not in (2, 5):
            raise ValueError(f"{path} has unexpected shape {tuple(x.shape)}")
        rows.append((os.path.basename(path), _predict_tensor(model, x, modality_id, device)))
    return rows


def main():
    parser = ArgumentParser(description="R2T-Net inference", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--role", choices=["rest", "task"], default="rest")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = _instantiate_from_ckpt(args.ckpt, torch.device(args.device))
    predictions = predict_folder(model, args.input_dir, torch.device(args.device), args.role)

    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        names = getattr(model, "target_names", None) or [f"target_{i}" for i in range(len(predictions[0][1]))]
        writer.writerow(["file", *[f"pred_{name}" for name in names]])
        writer.writerows([(name, *pred) for name, pred in predictions])
    print(f"wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
