# inference.py
# ------------------------------------------------------------
# Predict cognitive scores (or class logits) from fMRI tensors
# with a checkpointed R2T-Net.
#
# Example
#   python inference.py \
#       --ckpt logs/epoch03-valid_loss=0.2100.ckpt \
#       --input_dir demo_data/rest \
#       --output predictions.csv
#
# Inputs: *.pt files inside --input_dir
#   ▸ volumetric  : [C, H, W, D, T]
#   ▸ ROI / gray  : [V, T]
# ------------------------------------------------------------
import csv
import glob
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from tqdm import tqdm

from module.r2tnet import R2TNet
from module.models.load_model import load_model   # just for type hints


# ------------------------------------------------------------#
# build model skeleton + load weights
# ------------------------------------------------------------#
def _instantiate_from_ckpt(ckpt_path: str, num_rois: int, device: torch.device) -> R2TNet:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Hyper-parameters are saved by Lightning under this key
    hparams = ckpt.get("hyper_parameters", {})
    if num_rois > 0:
        hparams["num_rois"] = num_rois           # user override

    # dummy DataModule to satisfy ctor (targets never used at inference)
    dummy_dm = type("DummyDM", (), {"train_dataset": type("T", (), {"target_values": [[0.]]})})()

    model = R2TNet(data_module=dummy_dm, **hparams)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    model.eval().to(device)
    return model


# ------------------------------------------------------------#
# single-file forward pass
# ------------------------------------------------------------#
@torch.no_grad()
def _predict_tensor(model: R2TNet, x: torch.Tensor, modality_id: int, device: torch.device) -> float:
    """
    Returns a **scalar** prediction.  For classification you’ll get the raw
    logit (apply sigmoid/softmax yourself if needed).
    """
    x = x.unsqueeze(0).to(device)          # add batch dim
    mod = torch.tensor([modality_id], device=device)

    _, pred = model(x, mod)                # forward
    score = pred.squeeze().to("cpu").float().item()

    # inverse-scale if model carries a fitted scaler (regression only)
    if hasattr(model, "scaler") and hasattr(model.scaler, "inverse_transform"):
        score = model.scaler.inverse_transform([[score]])[0][0]
    return score


# ------------------------------------------------------------#
# directory-level inference
# ------------------------------------------------------------#
def predict_folder(model: R2TNet, input_dir: str, device: torch.device):
    rows = []
    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    for p in tqdm(pt_files, desc="inference"):
        x = torch.load(p, map_location="cpu")

        if x.ndim == 5:                    # [C,H,W,D,T] volume
            modality_id = 0
        elif x.ndim == 2:                  # [V,T] ROI / grayord
            modality_id = 1 if x.shape[0] < 91_282 else 2
        else:
            raise ValueError(f"{p} has unexpected shape {tuple(x.shape)}")

        score = _predict_tensor(model, x, modality_id, device)
        rows.append((os.path.basename(p), score))
    return rows


# ------------------------------------------------------------#
def main():
    ap = ArgumentParser(description="R2T-Net inference script",
                        formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="Path to .ckpt or .pth checkpoint")
    ap.add_argument("--input_dir", required=True, help="Folder with *.pt tensors")
    ap.add_argument("--output", default="predictions.csv", help="CSV to write")
    ap.add_argument("--num_rois", type=int, default=0, help="Force ROI size if != 0")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="cuda:N or cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = _instantiate_from_ckpt(args.ckpt, args.num_rois, device)

    predictions = predict_folder(model, args.input_dir, device)

    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "prediction"])
        writer.writerows(predictions)

    print(f"✅  {len(predictions)} predictions written → {args.output}")


if __name__ == "__main__":
    main()
