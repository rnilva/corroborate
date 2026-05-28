#!/usr/bin/env bash
# Reproduce every figure in this paper end-to-end.
#
# Inputs (read-only): `experiments/data/cache/hasselt_clean.parquet`
#                     `experiments/data/cache/hasselt_clean.sources.json`
# Outputs: `papers/g099_mediation/figures/0[1-5]_*.{png,csv}`
#
# Usage: from repo root,
#   bash papers/g099_mediation/run_all.sh
#
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

SCRIPTS_DIR="papers/g099_mediation/scripts"

for s in 01_mech_per_env 02_outcome_per_env 03_static_mediation \
         04_aggregation_danger 05_dynamic_mediation; do
    echo "─── ${s} ───"
    uv run python "${SCRIPTS_DIR}/${s}.py"
done

echo
echo "Done. See papers/g099_mediation/figures/ for output."
