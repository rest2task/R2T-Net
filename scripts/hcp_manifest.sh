#!/usr/bin/env bash
set -euo pipefail

python -m studies.manifest hcp \
  --source-root "${HCP_SOURCE_ROOT:-/nfshdd/y2jiang/HCP_1200}" \
  --data-dir "${HCP_DATA_DIR:-data/hcp_grayord}" \
  "$@"
