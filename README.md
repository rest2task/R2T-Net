# R2T‑Net

**R2T-Net** is a PyTorch-Lightning framework that transforms paired 4-D fMRI samples—**resting-state (rs-fMRI)** and **task-based (t-fMRI)**—into a unified **2048-dimensional dynamic brain activity signature**.

For each subject, the model encodes both rs-fMRI and t-fMRI separately into their own 2048-D latent vectors. During training, it uses a **contrastive learning objective** to make the two representations of the *same subject* align, while ensuring they remain distinct from those of *other subjects*.

A Transformer encoder generates the signatures (**Step 1**), and an **NT-Xent contrastive loss** enforces the alignment and separation (**Step 2**).
A lightweight supervised head can optionally be added to predict cognitive or behavioural traits from these signatures.

A test script, `extract.py`, allows direct extraction of these signatures from a trained checkpoint.

---

## Motivation

Traditional pipelines handle rs‑fMRI and t‑fMRI separately, often compressing t‑fMRI into static contrast maps—losing temporal dynamics and personalization.  
**R2T‑Net** instead:

* Learns a **modality‑invariant** embedding (rest ⇄ task).  
* Produces **person‑specific** vectors (positive pairs = same subject).  
* Shows good **test–retest stability**.  
* Lets you predict behaviour from *resting scans only*, cutting scan time.

---

## Backbone Flexibility

Edit one line in `module/models/load_model.py` to plug in any encoder that outputs a `[B, embed_dim]` feature:

| Category           | Examples |
|--------------------|----------|
| Transformer‑4D     | `swin4d_ver7` (default) ·  ViT · TimeSformer |
| 3‑D CNN            | 3‑D ResNet, 3‑D DenseNet, UNet‑3D |
| Hybrid             | CNN + GRU, Perceiver IO, Temporal‑U‑Net |

---

## Key Features

| Feature | Details |
|---------|---------|
| **Input‑agnostic** | Raw 4‑D volumes `[C,H,W,D,T]`, grayordinates `[91 282,T]`, or ROI series `[V,T]` |
| **Always contrastive** | NT‑Xent runs in every mode; you choose to freeze or unfreeze the prediction head |
| **Cosine‑warmup scheduler** | Enable with `--use_scheduler` |
| **Built-in regularisation** | Temporal crops, Gaussian noise, modality dropout, and gradient clipping (\|g\|=1) |
| **Metrics out‑of‑the‑box** | Pearson / MSE / MAE (regression), Balanced‑Acc / AUROC (classification) |
| **Multi-GPU ready** | DDP support with distributed samplers and sampler replacement disabled |

---

## Directory Layout

```

R2TNet/
├── train.py                # train / validate / test
├── inference.py            # batch inference on rs‑fMRI
│
├── module/
│   ├── r2tnet.py           # LightningModule (encoder + heads + losses)
│   ├── models/
│   │   ├── load_model.py
│   │   ├── swin4d_transformer_ver7.py
│   │   └── swin_transformer.py
│   └── utils/
│       ├── data_module.py
│       ├── datasets.py
│       ├── patch_embedding.py
│       └── lr_scheduler.py
│
└── logs/                   # auto‑generated (TensorBoard & checkpoints)

````

---


## Quick Start

Train and evaluate on 4D fMRI, ROI series, or grayordinates — all from one CLI.


### 1 · Install

```bash
# PyTorch 2.x with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install pytorch-lightning timm einops torchmetrics scikit-learn

# Optional extras
pip install monai pandas matplotlib
```


### 2 · Prepare your data

```
data/S1200/
├── img/
│   ├── 100307/
│   │   ├── frame_0.pt
│   │   ├── frame_1.pt
│   │   └── ...
│   └── 103414/
└── meta/
    ├── subject_dict.json        # {"100307": [0, 83.2], ...}
    └── splits.json              # {"train": [...], "val": [...], "test": [...]}
```

* Each `frame_*.pt` is a tensor shaped `[C, H, W, D]` (for 4D fMRI).
* ROIs or grayordinates should be saved as `[V]` tensors per frame.
* Optional: `voxel_mean.pt` and `voxel_std.pt` for normalization.


#### 2a · Convert NIfTI volumes (HCP example)

The repository ships two helper scripts that dump each fMRI time point into the layout above. Both scripts normalise the data, crop empty borders (adjust inside the script if your acquisition differs), and skip subjects that are already processed.

```bash
# Resting-state window (default file: rfMRI_REST1_LR_hp2000_clean.nii.gz)
python preprocessing_HCP_Rest.py \
  --load-root /path/to/HCP_1200 \
  --save-root data/S1200 \
  --expected-length 1200

# Task run (default file: tfMRI_WM_LR.nii.gz)
python preprocessing_HCP_Task.py \
  --load-root /path/to/HCP_1200 \
  --save-root data/S1200 \
  --expected-length 405
```

Key flags:

* `--nifti-name` — override the file name inside each subject folder (e.g. use `tfMRI_REL_LR.nii.gz` for relational reasoning).
* `--scaling-method {minmax,z-norm}` — choose intensity scaling.
* `--keep-min-background` — fill background voxels with the minimum foreground value instead of zero.

The scripts populate `data/S1200/img/<subject>/frame_*.pt` in fp16 format and create an empty `data/S1200/meta/` directory for metadata files (`subject_dict.json`, `splits.json`).


### 3 · Training Paradigms

| Mode                | NT-Xent | Supervised | CLI Flags                                                |
| ------------------- | ------- | ---------- | -------------------------------------------------------- |
| **Self-supervised** | ✅       | ❌ (frozen) | `--contrastive --pretraining --freeze_head`              |
| **Full fine-tune**  | ✅       | ✅          | `--contrastive` *(default)*                              |

> ⚠️ Omitting `--contrastive` disables NT-Xent loss (core to R2T‑Net).


### 4 · Example Commands

#### A. Self-supervised Pre-training

```bash
python train.py \
  --data_dir data/S1200 \
  --dataset_type rest \
  --contrastive --pretraining --freeze_head \
  --model swin4d_ver7 \
  --batch_size 8 --max_epochs 50 \
  --use_scheduler --total_steps 20000
```

#### Multi-GPU (DDP) Training

Enable distributed training by requesting multiple devices. The trainer defaults to `ddp` when more than one device is set, and
the data module automatically switches to distributed samplers.

```bash
python train.py \
  --data_dir data/S1200 \
  --dataset_type rest \
  --contrastive \
  --model swin4d_ver7 \
  --batch_size 8 --max_epochs 50 \
  --accelerator gpu --devices 4 --precision 16
```

#### B. Fine-tune with Labels (Regression)

```bash
python train.py \
  --data_dir data/S1200 \
  --dataset_type rest \
  --contrastive \
  --load_model logs/last.ckpt \
  --downstream_task_type regression \
  --label_scaling_method standardization \
  --model swin4d_ver7 \
  --batch_size 4 --max_epochs 30 --use_scheduler \
  --temporal_crop_min_ratio 0.8 --gaussian_noise_std 0.01 \
  --gaussian_noise_p 0.1 --modality_dropout_prob 0.2
```

#### C. Evaluate/Test Only

```bash
python train.py \
  --test_only --test_ckpt logs/epoch02-valid_loss=0.2100.ckpt \
  --data_dir data/S1200
```


### 5 · Inference (e.g., rs-fMRI → prediction)

```bash
python inference.py \
  --ckpt logs/epoch03-valid_loss=0.2100.ckpt \
  --input_dir data/S1200/img \
  --input_kind vol \
  --output rs_predictions.csv
```

Output: CSV file with columns `subject_id,prediction`.


### 6 · Recommended Flags

| Flag                              | Purpose                                        |
| --------------------------------- | ---------------------------------------------- |
| `--precision 16`                  | Mixed precision — saves memory                 |
| `--accumulate_grad_batches 2`     | Gradient accumulation (for small GPUs)         |
| `--resume_from_checkpoint ...`    | Resume training from last/best                 |
| `--balanced_sampling`             | Ensures equal subject exposure per epoch       |
| `--temporal_crop_min_ratio 0.9`   | Tighten temporal cropping during training      |
| `--gaussian_noise_std 0.0`        | Disable stochastic noise injection             |
| `--modality_dropout_prob 0.0`     | Train without modality dropout                 |
| `--num_rois 360 --input_kind roi` | Switch to ROI-based input (instead of volumes) |
| `--grayordinates`                 | Use 91,282-dim grayordinate inputs             |

