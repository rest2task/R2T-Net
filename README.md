# R2T-Net

R2T-Net maps rest/task fMRI windows into a 1024-D subject signature and trains
that signature against cognitive scores.  The repo uses plain PyTorch.  No
Lightning wrapper.

## Layout

```text
R2T-Net/
├── train.py
├── prepare_data.py
├── inference.py
├── extract.py
├── studies/
├── scripts/
├── r2tnet/
│   └── backbones/
└── manuscript/
```

## Data Contract

Training never reads HCP/CHCP/ADNI source files directly.  Convert sources to
per-TR tensors first.  The current manifest format is line-delimited JSON; CSV
manifests from older runs are still readable, but new data should use this:

```text
data/<experiment>/
├── blocks/
│   └── <subject>/<scan_id>/frame_0.pt
└── meta/
    ├── dataset.json
    ├── subjects.jsonl
    └── scans.jsonl
```

`meta/dataset.json`

```json
{
  "schema": "r2t.manifest.v2",
  "study": "hcp",
  "source_root": "/nfshdd/y2jiang/HCP_1200",
  "targets": ["wm_0bk", "wm_2bk", "wm_diff", "rel"]
}
```

`meta/subjects.jsonl`

```json
{"id":"100206","split":"train","sex":0,"targets":{"wm_0bk":0.91,"wm_2bk":0.83,"wm_diff":-0.08,"rel":0.88,"rt_wm":721.4,"rt_rel":812.0}}
```

`meta/scans.jsonl`

```json
{"id":"100206:REST1_LR","subject":"100206","scan":"REST1_LR","role":"rest","kind":"grayord","frames":"blocks/100206/REST1_LR","n_frames":1200,"source":"/nfshdd/.../rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii"}
{"id":"100206:WM_LR","subject":"100206","scan":"WM_LR","role":"task","kind":"grayord","frames":"blocks/100206/WM_LR","n_frames":405,"source":"/nfshdd/.../tfMRI_WM_LR_Atlas_MSMAll.dtseries.nii"}
```

`kind`

```text
vol      RAW 4D NIfTI        frame_i.pt = [1, X, Y, Z]
grayord  CIFTI dtseries      frame_i.pt = [91282]
roi      CIFTI + parcellation frame_i.pt = [V]
```

## Prepare

Create empty manifests:

```bash
python prepare_data.py init --data-dir data/hcp_grayord
```

Convert rows in `meta/scans.jsonl`:

```bash
python prepare_data.py convert \
  --data-dir data/hcp_grayord \
  --normalise zscore \
  --dtype float16
```

Parcellated mode:

```bash
python prepare_data.py convert \
  --data-dir data/hcp_mmp \
  --parcellation-atlas atlas/MMP.dlabel.nii \
  --wb-command wb_command
```

## Modes

```text
--training_mode r2t             paired rest/task, contrastive + supervised
--training_mode rest_only       rest windows only
--training_mode task_only       task windows only
--training_mode synthetic_task  rest signature mapped toward task signature

--pretraining                   contrastive only
--freeze_encoder                train prediction head only
--supervised_view rest|task|average
--pair_fusion auto|rest|task|average|sum|concat|gated
--disable_contrastive
--optimizer adamw|adam|sgd|rmsprop
--scheduler none|cosine|step|onecycle --warmup_epochs 50 --lr_min 1e-5
--modality_dropout_p 0.2
--selection_metric alignment --early_stop_patience 5
--synthetic_mapper cmt --synthetic_depth 2 --synthetic_heads 8
```

Backbone/objective switches:

```text
--temporal_encoder vit|gru|conv|mean
--contrastive_loss ntxent|symmetric_ce|cosine|margin
--reg_head yolo|mlp|linear
--reg_loss huber|mse|l1
--temporal_mask_p 0.05 --feature_mask_p 0.05
```

Named heads share the same encoder and 1024-D signature.  The grammar is:

```text
--head_spec name:target_a,target_b[:regression|classification[:weight]];...
```

Examples:

```bash
--target_cols wm_0bk,wm_2bk,wm_diff,rel \
--head_spec 'score:wm_0bk,wm_2bk:regression:1.0;contrast:wm_diff,rel:regression:0.5'

--target_cols wm_0bk,wm_2bk,rt_wm,rt_rel \
--head_spec 'score:wm_0bk,wm_2bk:regression:1.0;rt:rt_wm,rt_rel:regression:0.25'
```

## Manuscript Studies

The model-side study entry points are split by cohort.

```bash
python -m studies.hcp
python -m studies.hcp --mode r2t --representation grayord
python -m studies.chcp --ckpt runs/hcp_grayord_r2t/last.pt
python -m studies.adni --ckpt runs/hcp_raw4d_r2t/last.pt
```

Model-comparison grids have direct launchers:

```bash
python -m studies.hcp_modality
python -m studies.hcp_representation
python -m studies.hcp_length_dim
python -m studies.hcp_seed_grid --tests
python -m studies.model_space --suite all --csv runs/model_space.csv
python -m studies.hcp_encoder_space
python -m studies.hcp_objective_space
python -m studies.hcp_head_space
python -m studies.hcp_fusion_space
python -m studies.hcp_optim_space
python -m studies.hcp_regularization_space
python -m studies.hcp_synthetic_space
python -m studies.hcp_size_space
```

Raw model artifacts:

```bash
python -m studies.artifacts manifest --data-dir data/hcp_grayord --target-cols wm_0bk,wm_2bk,wm_diff,rel
python -m studies.artifacts alignment --ckpt runs/hcp_grayord_r2t/last.pt --data-dir data/hcp_grayord --target-cols wm_0bk,wm_2bk,wm_diff,rel --output runs/artifacts/alignment.csv
python -m studies.artifacts saliency --ckpt runs/hcp_grayord_r2t/last.pt --data-dir data/hcp_grayord --target-cols wm_0bk,wm_2bk,wm_diff,rel --target-index 0 --output runs/artifacts/saliency.pt
```

Script shortcuts:

```bash
bash scripts/hcp_manifest.sh
bash scripts/hcp_study.sh --representation all
bash scripts/chcp_study.sh
bash scripts/adni_study.sh
bash scripts/hcp_train_grayord.sh
bash scripts/hcp_modality.sh --tests
bash scripts/hcp_encoder_space.sh
bash scripts/hcp_artifacts.sh
```

HCP source patterns expected under `/nfshdd/y2jiang/HCP_1200/<subject>/`:

```text
rfMRI_REST{1,2}_{LR,RL}_Atlas_MSMAll_hp2000_clean.dtseries.nii
tfMRI_WM_{LR,RL}_Atlas_MSMAll.dtseries.nii
tfMRI_RELATIONAL_{LR,RL}_Atlas_MSMAll.dtseries.nii
```

CHCP source patterns expected under `/nfshdd/y2jiang/CHCP_DB/<subject>/`:

```text
rfMRI_REST{1,2}_{AP,PA}_Atlas_hp2000_clean.dtseries.nii
tfMRI_WM_*Atlas*.dtseries.nii
tfMRI_RELATIONAL_*Atlas*.dtseries.nii
```

ADNI is treated as rest-only transfer, RAW 4D windows.  Use a prepared manifest
with dummy target columns if only clinical labels are available.

## Train

R2T grayordinate:

```bash
python train.py \
  --data_dir data/hcp_grayord \
  --training_mode r2t \
  --eval_role rest \
  --sequence_length 300 \
  --signature_dim 1024 \
  --batch_size 16 \
  --max_epochs 1600 \
  --out_dir runs/hcp_r2t
```

Rest-only baseline:

```bash
python train.py \
  --data_dir data/hcp_grayord \
  --training_mode rest_only \
  --eval_role rest \
  --disable_contrastive \
  --sequence_length 300 \
  --out_dir runs/rest_only
```

Frozen-signature fine-tune:

```bash
python train.py \
  --data_dir data/hcp_grayord \
  --training_mode r2t \
  --resume runs/pretrain/last.pt \
  --freeze_encoder \
  --supervised_view rest \
  --max_epochs 100 \
  --out_dir runs/finetune
```

DDP:

```bash
torchrun --nproc_per_node=4 train.py \
  --ddp \
  --data_dir data/hcp_grayord \
  --training_mode r2t \
  --batch_size 8 \
  --out_dir runs/ddp_r2t
```

## Test

```bash
python train.py \
  --data_dir data/hcp_grayord \
  --resume runs/hcp_r2t/last.pt \
  --test_only \
  --eval_role rest
```

Test output reports window correlations and subject-averaged correlations.

## Extract / Infer

`inference.py` expects complete clips, not frame folders:

```text
clip.pt = [V,T] or [1,X,Y,Z,T]
```

```bash
python inference.py --ckpt runs/hcp_r2t/last.pt --input_dir clips/rest --role rest
python extract.py --ckpt runs/hcp_r2t/last.pt --rest_dir clips/rest --rest_only
```
