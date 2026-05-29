#!/usr/bin/env bash
set -euo pipefail

# Use repo defaults unless environment overrides are provided, then pass through extra CLI args.
python -m studies.grid --suite "${R2T_COMPARISON_SUITE:-all}" --seed "${R2T_SEED:-42}" "$@"
