#!/usr/bin/env bash
set -euo pipefail

# Defaults point to the CHCP study's recommended checkpoint and output artifacts, but every flag can still be overridden.
python -m studies.transfer \
  --ckpt "${CHCP_CKPT:-runs/hcp_grayord_r2t/last.pt}" \
  --data-dir "${CHCP_DATA_DIR:-data/chcp_grayord}" \
  --target-cols "${R2T_TARGETS:-wm_0bk,wm_2bk,wm_diff,rel}" \
  --split test \
  --role rest \
  --sequence-length "${R2T_SEQUENCE_LENGTH:-300}" \
  --output "${CHCP_OUT:-runs/chcp_transfer/predictions.csv}" \
  --signatures "${CHCP_SIGNATURES:-runs/chcp_transfer/signatures.pt}" \
  "$@"
