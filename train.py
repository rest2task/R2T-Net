# train.py
# Entry‑point for R2T‑Net training / evaluation
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from module.utils.data_module import fMRIDataModule
from module.r2tnet import R2TNet


# --------------------------------------------------------------------- #
# 1. CLI parsing
# --------------------------------------------------------------------- #

def build_parser() -> ArgumentParser:
    p = ArgumentParser(
        description="Train / evaluate R2T‑Net (contrastive + supervised)",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # generic
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument("--logger_name", choices=["tensorboard"], default="tensorboard")
    p.add_argument("--project_name", default="R2TNet")
    p.add_argument("--dirpath", default="logs", help="Where checkpoints & logs go")

    # pre‑trained / evaluation
    p.add_argument("--load_model", default=None, help="Path to .ckpt or .pth to load")
    p.add_argument("--test_only", action="store_true", help="Skip training and only run test")
    p.add_argument("--test_ckpt", default=None, help="Checkpoint to load before test")

    # model / data / trainer flags
    p = R2TNet.add_model_specific_args(p)
    p = fMRIDataModule.add_data_specific_args(p)
    p = pl.Trainer.add_argparse_args(p)

    return p


# --------------------------------------------------------------------- #
# 2. small helpers
# --------------------------------------------------------------------- #

def make_logger(args) -> TensorBoardLogger:
    return TensorBoardLogger(save_dir=args.dirpath, name=args.project_name)


def make_ckpt_cb(args) -> ModelCheckpoint:
    monitor, mode = (
        ("valid_balacc", "max") if args.downstream_task_type == "classification" else ("valid_loss", "min")
    )
    return ModelCheckpoint(
        dirpath=args.dirpath,
        monitor=monitor,
        filename=f"epoch{{epoch:02d}}-{{{monitor}:.4f}}",
        save_last=True,
        mode=mode,
    )


# --------------------------------------------------------------------- #
# 3. main
# --------------------------------------------------------------------- #

def main():
    args = build_parser().parse_args()

    # grayordinates shortcut
    if hasattr(args, "grayordinates") and args.grayordinates:
        args.num_rois = 91282

    pl.seed_everything(args.seed, workers=True)

    data = fMRIDataModule(**vars(args))
    model = R2TNet(data_module=data, **vars(args))

    # optional partial weight‑loading (strict=False lets us ignore head size mismatches)
    if args.load_model:
        ckpt = torch.load(args.load_model, map_location="cpu")
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print(f"✔  Loaded weights from {args.load_model}")

    logger = make_logger(args)
    ckpt_callback = make_ckpt_cb(args)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[ckpt_callback, lr_monitor],
    )

    if args.test_only:
        trainer.test(model, datamodule=data, ckpt_path=args.test_ckpt)
    else:
        trainer.fit(model, datamodule=data)
        trainer.test(model, datamodule=data)  # evaluate best / last ckpt


if __name__ == "__main__":
    main()
