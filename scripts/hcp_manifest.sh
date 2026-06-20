#!/usr/bin/env bash
set -euo pipefail

python -m studies.manifest hcp \
  --source-root "${HCP_SOURCE_ROOT:-/path/to/HCP_1200}" \
  --data-dir "${HCP_DATA_DIR:-data/hcp_grayord}" \
  "$@"
