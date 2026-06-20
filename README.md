# R2T-Net

R2T-Net (Rest-to-Task Network) is a transformer-based framework for integrating resting-state and task fMRI into a shared 1,024-dimensional neural activity signature that preserves individual-specific brain patterns while aligning rest and task representations from the same person. It is trained on paired rest/task scans to learn this shared signature, then used to predict individual cognitive performance from resting-state data alone at test time, which makes it useful for settings where task scans are unavailable or difficult to collect. The implementation in this repository supports the full workflow, including data preparation, paired and single-modality training, transfer to independent datasets, inference on complete clips, signature extraction, and the analysis scripts used for HCP, CHCP, and ADNI.

## Repository Layout

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
```

## Data Format

Training reads prepared manifests, not raw HCP, CHCP, or ADNI files directly. Source data should first be converted into per-TR or per-window tensors.

Recommended manifest layout:

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
  "source_root": "/path/to/HCP_1200",
  "targets": ["wm_0bk", "wm_2bk", "wm_diff", "rel"]
}
```

`meta/subjects.jsonl`

```json
{"id":"100206","split":"train","sex":0,"targets":{"wm_0bk":0.91,"wm_2bk":0.83,"wm_diff":-0.08,"rel":0.88,"rt_wm":721.4,"rt_rel":812.0}}
```

`meta/scans.jsonl`

```json
{"id":"100206:REST1_LR","subject":"100206","scan":"REST1_LR","role":"rest","kind":"grayord","frames":"blocks/100206/REST1_LR","n_frames":1200,"source":"/path/to/HCP_1200/100206/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii"}
{"id":"100206:WM_LR","subject":"100206","scan":"WM_LR","role":"task","kind":"grayord","frames":"blocks/100206/WM_LR","n_frames":405,"source":"/path/to/HCP_1200/100206/tfMRI_WM_LR_Atlas_MSMAll.dtseries.nii"}
```

Supported `kind` values:

```text
vol      RAW 4D NIfTI        frame_i.pt = [1, X, Y, Z]
grayord  CIFTI dtseries      frame_i.pt = [91282]
roi      CIFTI + parcellation frame_i.pt = [V]
```

## Supported At A Glance

- Input types: raw 4D NIfTI volumes, grayordinate CIFTI time series, and ROI/parcellated CIFTI time series.
- Study cohorts: HCP, CHCP, and ADNI.
- Training modes: paired rest/task (`r2t`), rest-only, task-only, and synthetic-task transfer.
- Encoders: SwiFT for 6D volume clips, temporal ViT, temporal GRU, temporal convolution, and temporal mean pooling.
- Heads: YOLO-style regression head, MLP regression head, linear regression head, and classification head.
- Fusion: rest, task, average, sum, concat, and gated.

## Supported Manifest Fields

- `subjects.jsonl` and `subjects.csv` support `id` or `subject_id`, `split`, `sex`, and `targets` for JSONL or `target*` columns for CSV.
- `scans.jsonl` and `scans.csv` support `subject` or `subject_id`, `scan` or `scan_id`, `role` as `rest` or `task`, `kind` or `input_kind`, `frames` or `path`, `n_frames`, and `source` or `source_path`.
- If `n_frames` is missing, the loader infers it from `frame_*.pt` files.

## Supported Preprocessing

Use `prepare_data.py init` to create empty manifest templates:

```bash
python prepare_data.py init --data-dir data/hcp_grayord
```

Use `prepare_data.py convert` to write per-frame tensors from the scan list:

```bash
python prepare_data.py convert \
  --data-dir data/hcp_grayord \
  --normalise zscore \
  --dtype float16
```

Supported conversion options:

- `--normalise none|zscore|minmax`
- `--dtype float16|float32`
- `--parcellation-atlas` for ROI input
- `--wb-command` to choose the Workbench binary

## Supported Training Flags

- Core runtime: `--seed`, `--out_dir`, `--device`, `--precision fp32|amp`, `--save_every`, `--resume`, `--test_only`, `--compile`, and `--ddp`.
- Optimization: `--optimizer adamw|adam|sgd|rmsprop`, `--learning_rate`, `--weight_decay`, `--momentum`, `--nesterov`, `--scheduler none|cosine|step|onecycle`, `--warmup_epochs`, `--lr_min`, `--step_size`, `--step_gamma`, and `--grad_clip_norm`.
- Training control: `--training_mode r2t|rest_only|task_only|synthetic_task`, `--eval_role rest|task`, `--selection_metric auto|loss|alignment`, `--early_stop_patience`, `--batch_size`, `--num_workers`, `--sequence_length`, `--stride_between_seq`, `--stride_within_seq`, `--pin_memory`, and `--target_cols`.

## Supported Model Options

- Architecture: `--signature_dim`, `--token_dim`, `--temporal_encoder vit|gru|conv|mean`, `--swift_patch`, `--swift_stage_depths`, `--swift_stage_heads`, `--swift_global_depth`, `--swift_global_heads`, `--vit_depth`, `--vit_heads`, `--gru_depth`, `--gru_bidirectional`, `--conv_depth`, `--conv_kernel`, and `--encoder_dropout`.
- Training and objective: `--pretraining`, `--freeze_encoder`, `--supervised_view rest|task|average`, `--pair_fusion auto|rest|task|average|sum|concat|gated`, `--disable_contrastive`, `--lambda_contrast`, `--lambda_synthetic`, `--lambda_synthetic_l2`, `--synthetic_mapper cmt|mlp`, `--synthetic_depth`, `--synthetic_heads`, `--synthetic_tokens`, `--synthetic_dropout`, `--temperature`, `--contrastive_loss ntxent|infonce|simclr|symmetric_ce|clip|cosine|margin|triplet|hard_ntxent|hard_infonce|dcl|debiased|barlow_twins|vicreg`, `--contrastive_margin`, `--contrastive_projector none|linear|mlp`, `--contrastive_dim`, `--contrastive_projector_hidden`, `--contrastive_queue_size`, `--hard_negative_topk`, `--contrastive_tau_plus`, `--contrastive_target_mask_quantile`, `--contrastive_barlow_lambda`, `--vicreg_sim_coeff`, `--vicreg_std_coeff`, and `--vicreg_cov_coeff`.
- Downstream heads: `--downstream_task_type regression|classification`, `--head_spec`, `--label_scaling_method standardization|minmax|none`, `--reg_head yolo|mlp|linear`, `--reg_loss huber|mse|l1`, `--reg_num_bins`, `--reg_binning_strategy quantile|uniform`, `--reg_alpha`, `--reg_beta`, `--reg_temperature`, `--reg_label_smoothing`, `--pred_hidden_dim`, `--head_dropout`, and `--head_depth`.
- Classification heads: `--classification_loss auto|bce|cross_entropy` and `--classification_label_smoothing`. `auto` uses binary BCE for one target column and multiclass cross-entropy for multi-column one-hot targets.
- Regularization and augmentation: `--temporal_crop_min_ratio`, `--gaussian_noise_std`, `--gaussian_noise_p`, `--temporal_mask_p`, `--feature_mask_p`, and `--modality_dropout_p`.

Named heads use the same encoder and signature. The grammar is:

```text
--head_spec name:target_a,target_b[:regression|classification[:weight]];...
```

Examples:

```bash
--target_cols wm_0bk,wm_2bk,wm_diff,rel \
--head_spec 'score:wm_0bk,wm_2bk:regression:1.0;contrast:wm_diff,rel:regression:0.5'

--target_cols wm_0bk,wm_2bk,rt_wm,rt_rel \
--head_spec 'score:wm_0bk,wm_2bk:regression:1.0;rt:rt_wm,rt_rel:regression:0.25'

--target_cols AD,CN,MCI \
--downstream_task_type classification \
--head_spec 'diagnosis:AD,CN,MCI:classification:1.0' \
--classification_loss cross_entropy
```

## Studies

Study entry points are split by cohort:

```bash
python -m studies.hcp
python -m studies.hcp --mode r2t --representation grayord
python -m studies.chcp --ckpt runs/hcp_grayord_r2t/last.pt
python -m studies.adni --ckpt runs/hcp_raw4d_r2t/last.pt
python -m studies.adni_classification
```

Model-comparison grids:

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

Supported study wrappers:

- `python -m studies.hcp`
- `python -m studies.chcp`
- `python -m studies.adni`
- `python -m studies.adni_classification`

Supported study command generators:

- `python -m studies.plan`
- `python -m studies.grid`
- `python -m studies.model_space`

Supported study suites:

- `modality`
- `representation`
- `length_dim`
- `seeds`
- `all`

Supported model-space suites:

- `encoder`
- `objective`
- `head`
- `fusion`
- `optim`
- `regularization`
- `synthetic`
- `size`
- `all`

Raw analysis artifacts:

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

These scripts are thin wrappers around the Python entry points above and mainly exist for repeatable shell-based runs.

Expected source file patterns:

HCP under `/path/to/HCP_1200/<subject>/`

```text
rfMRI_REST{1,2}_{LR,RL}_Atlas_MSMAll_hp2000_clean.dtseries.nii
tfMRI_WM_{LR,RL}_Atlas_MSMAll.dtseries.nii
tfMRI_RELATIONAL_{LR,RL}_Atlas_MSMAll.dtseries.nii
```

CHCP under `/path/to/CHCP_DB/<subject>/`

```text
rfMRI_REST{1,2}_{AP,PA}_Atlas_hp2000_clean.dtseries.nii
tfMRI_WM_*Atlas*.dtseries.nii
tfMRI_RELATIONAL_*Atlas*.dtseries.nii
```

ADNI is handled as a rest-only transfer setup with RAW 4D windows. If only clinical labels are available, use a prepared manifest with placeholder target columns.

## Training Examples

R2T grayordinate training:

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

Frozen-signature fine-tuning:

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

## Testing

```bash
python train.py \
  --data_dir data/hcp_grayord \
  --resume runs/hcp_r2t/last.pt \
  --test_only \
  --eval_role rest
```

Test output reports window-level correlations and subject-averaged correlations.

## Inference And Extraction

`inference.py` expects complete clips, not frame folders:

```text
clip.pt = [V,T] or [1,X,Y,Z,T]
```

```bash
python inference.py --ckpt runs/hcp_r2t/last.pt --input_dir clips/rest --role rest
python extract.py --ckpt runs/hcp_r2t/last.pt --rest_dir clips/rest --rest_only
```
