#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --data_dir "${HCP_DATA_DIR:-data/hcp_grayord}" \
  --target_cols "${R2T_TARGETS:-wm_0bk,wm_2bk,wm_diff,rel}" \
  --training_mode r2t \
  --eval_role rest \
  --sequence_length "${R2T_SEQUENCE_LENGTH:-300}" \
  --signature_dim "${R2T_SIGNATURE_DIM:-1024}" \
  --learning_rate "${R2T_LR:-1e-3}" \
  --weight_decay "${R2T_WEIGHT_DECAY:-1e-2}" \
  --scheduler cosine \
  --warmup_epochs "${R2T_WARMUP_EPOCHS:-50}" \
  --lr_min "${R2T_LR_MIN:-1e-5}" \
  --batch_size "${R2T_BATCH_SIZE:-16}" \
  --max_epochs "${R2T_MAX_EPOCHS:-1600}" \
  --save_every "${R2T_SAVE_EVERY:-50}" \
  --early_stop_patience "${R2T_EARLY_STOP_PATIENCE:-5}" \
  --selection_metric "${R2T_SELECTION_METRIC:-alignment}" \
  --out_dir "${R2T_OUT_DIR:-runs/hcp_grayord_r2t}" \
  --lambda_contrast "${R2T_LAMBDA_CONTRAST:-0.5}" \
  --modality_dropout_p "${R2T_MODALITY_DROPOUT:-0.2}" \
  --supervised_view "${R2T_SUPERVISED_VIEW:-average}" \
  "$@"
