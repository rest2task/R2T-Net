#!/usr/bin/env bash
set -euo pipefail

# Defaults target the ADNI transfer baseline while keeping every CLI option overridable for local experiments.
python -m studies.transfer \
  --ckpt "${ADNI_CKPT:-runs/hcp_raw4d_r2t/last.pt}" \
  --data-dir "${ADNI_DATA_DIR:-data/adni_raw4d}" \
  --target-cols "${ADNI_TARGETS:-wm_0bk,wm_2bk,rel}" \
  --split test \
  --role rest \
  --sequence-length "${R2T_SEQUENCE_LENGTH:-300}" \
  --output "${ADNI_OUT:-runs/adni_transfer/predictions.csv}" \
  --signatures "${ADNI_SIGNATURES:-runs/adni_transfer/signatures.pt}" \
  "$@"
