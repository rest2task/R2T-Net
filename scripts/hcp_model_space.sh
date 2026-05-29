#!/usr/bin/env bash
set -euo pipefail

# Keep the model-space suite overridable while defaulting to the "all" sweep.
python -m studies.model_space --suite "${R2T_MODEL_SPACE:-all}" "$@"
