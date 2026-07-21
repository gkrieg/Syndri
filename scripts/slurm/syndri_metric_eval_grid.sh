#!/bin/bash
set -euo pipefail

SBATCH=/home/jwinchel/slurmfiles/metrics_eval.sbatch
BASE_OUT=/compute/oven-0-13/jwinchel/spatial_adata/metrics_eval_syndri_grid
FRACTIONS="0.01,0.02,0.05,0.1,0.2"

for cell_specific in 0 1; do
  for mode in zero probability_scale; do
    if [[ "$cell_specific" == "1" ]]; then
      group_label="celltype_specific"
    else
      group_label="global"
    fi

    if [[ "$mode" == "zero" ]]; then
      mode_label="zero"
    else
      mode_label="nonlinear_power"
    fi

    outdir="${BASE_OUT}/${group_label}_${mode_label}"

    CELL_TYPE_SPECIFIC_REMOVAL="$cell_specific" \
    CLEANING_MODE="$mode" \
    PROBABILITY_TRANSFORM="power" \
    PROBABILITY_POWER="2.0" \
    TARGET_REMOVED_FRACTIONS="$FRACTIONS" \
    OUTPUT_DIR="$outdir" \
    sbatch "$SBATCH"
  done
done