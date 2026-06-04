# Studies

Model-side study entry points.  No supplementary stats live here.

```bash
python -m studies.hcp
python -m studies.hcp --mode r2t --representation grayord
python -m studies.hcp --representation all
python -m studies.chcp --ckpt runs/hcp_grayord_r2t/last.pt
python -m studies.chcp --finetune
python -m studies.adni --ckpt runs/hcp_raw4d_r2t/last.pt
python -m studies.adni --probe
```

Model comparisons:

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

Suites:

```text
modality        rs-only, task-only, synthetic-task, full R2T
representation  raw4d/SwiFT, grayord/TimeSformer, CA718, MMP379
length_dim      T in 100..1200, signature dim in 256..2048
seeds           manuscript optimizer seeds for modality + representation
```

`grid.py`, `model_space.py`, and `plan.py` remain importable for old notebooks,
but the named files above are the scripts used for new runs.

Model-space suites:

```text
encoder         temporal ViT, GRU, TCN, mean pooling
objective       contrastive losses and contrastive ablation
head            YOLO-style, MLP, linear regression heads
fusion          rest/task signature fusion rules
optim           optimizer and scheduler choices
regularization  crop, noise, mask, modality dropout
synthetic       CMT and MLP synthetic-task mappers
size            T and signature-dimension scan
```

Raw analysis artifacts:

```bash
python -m studies.artifacts manifest --data-dir data/hcp_grayord --target-cols wm_0bk,wm_2bk,wm_diff,rel
python -m studies.artifacts alignment --ckpt runs/hcp_grayord_r2t/last.pt --data-dir data/hcp_grayord --target-cols wm_0bk,wm_2bk,wm_diff,rel --output runs/artifacts/alignment.csv
python -m studies.artifacts signatures --ckpt runs/hcp_grayord_r2t/last.pt --data-dir data/hcp_grayord --target-cols wm_0bk,wm_2bk,wm_diff,rel --output runs/artifacts/signatures.pt
python -m studies.artifacts saliency --ckpt runs/hcp_grayord_r2t/last.pt --data-dir data/hcp_grayord --target-cols wm_0bk,wm_2bk,wm_diff,rel --target-index 0 --output runs/artifacts/saliency.pt
```

These commands emit counts, cosine rows, signatures, or gradient-input maps.
They do not run bootstrap, ANOVA, permutation, group t-tests, or FDR correction.

ADNI structural controls:

```bash
python -m studies.adni_structural_control
python -m studies.adni_structural_control --control-predictions temporal_mean=adni_temporal_mean_predictions.csv --control-predictions time_shuffled=adni_time_shuffled_predictions.csv
python -m studies.adni_structural_control --write-temporal-controls
```

The default ADNI structural-control command expects relative CSVs in the working
directory: `adni_r2tnet_predictions.csv`, `adni_covariates.csv`, and
`adni_structural_features.csv`. It writes ANCOVA, Digit Span, classification,
bootstrap, and optional relevance/atrophy-overlap outputs under
`runs/adni_structural_control`.

`manifest.py` writes the JSONL manifests from source trees.  Pass a participant
table when labels or fixed train/val/test splits are known.

Participant table columns:

```text
subject_id,split,sex,wm_0bk,wm_2bk,wm_diff,rel
```

ADNI can use dummy score columns for inference; diagnosis and Digit Span are
kept outside the model path.

Source patterns are in `specs.py`.
