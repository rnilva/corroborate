#!/usr/bin/env bash
# Reproduce every figure in the γ=0.999 DDQN-harm case study end-to-end.
#
# Inputs (read-only):
#   experiments/data/cache/hasselt_clean_gpanel.parquet   (binary V-vs-D panel)
#   papers/g999_harm/data/alpha_dose_cells.csv            (frozen α dose panel)
#   papers/g999_harm/data/deep2010_sym_cells.csv          (frozen deep-2010 arm)
# Outputs:
#   papers/g999_harm/figures/0[1-6]*.{png,csv}
#
# Fully offline — no AWS / cloud restore needed. The α-dose corpora are
# cloud-evicted, so their per-cell scalars are frozen in data/ (see README).
#
# Usage: from repo root,
#   bash papers/g999_harm/run_all.sh
#
set -euo pipefail

cd "$(dirname "$0")/../.."

SCRIPTS_DIR="papers/g999_harm/scripts"
# Scripts import `_common` from their own dir; expose repo root too.
export PYTHONPATH="$(pwd):${SCRIPTS_DIR}:${PYTHONPATH:-}"

for s in 01_alpha_dose_response \
         02_powered_learning_curves \
         03_powered_mediation \
         04_redq_vd_trajectories \
         05_redq_full_panel \
         06_better_ddqn_deep2010 \
         06b_learning_curve \
         06c_bias_trajectory; do
    echo "─── ${s} ───"
    uv run python "${SCRIPTS_DIR}/${s}.py"
done

echo
echo "Done. See papers/g999_harm/figures/ for output."
