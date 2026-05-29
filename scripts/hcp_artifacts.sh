#!/usr/bin/env bash
set -euo pipefail

python -m studies.artifacts signatures \
  --ckpt "${R2T_CKPT:-runs/hcp_grayord_r2t/last.pt}" \
  --data-dir "${HCP_DATA_DIR:-data/hcp_grayord}" \
  --target-cols "${R2T_TARGETS:-wm_0bk,wm_2bk,wm_diff,rel}" \
  --output "${R2T_SIGNATURES:-runs/artifacts/signatures.pt}" \
  "$@"

python -m studies.artifacts alignment \
  --ckpt "${R2T_CKPT:-runs/hcp_grayord_r2t/last.pt}" \
  --data-dir "${HCP_DATA_DIR:-data/hcp_grayord}" \
  --target-cols "${R2T_TARGETS:-wm_0bk,wm_2bk,wm_diff,rel}" \
  --split "${R2T_ALIGNMENT_SPLIT:-train}" \
  --output "${R2T_ALIGNMENT:-runs/artifacts/alignment.csv}"
