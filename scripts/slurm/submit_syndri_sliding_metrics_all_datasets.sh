#!/usr/bin/env bash
# Recalculate only the adaptive Syndri rolling-cosine metric case.
set -euo pipefail

repo=/home/jwinchel/Syndri
export SPLIT_ONLY=0
export SYNDRI_REFERENCE_COSINE=1
export SYNDRI_COSINE_ONLY=1
export SYNDRI_COSINE_WINDOW=${SYNDRI_COSINE_WINDOW:-5}

echo "Submitting adaptive Syndri non-T-cell metrics for all four datasets (window=$SYNDRI_COSINE_WINDOW)"
bash "$repo/scripts/slurm/submit_metrics_all_datasets.sh" \
  agingbrain_merfish brain_merfish nsclc_cosmx breast_xenium

echo "Submitting adaptive Syndri T-cell metrics for their configured datasets (window=$SYNDRI_COSINE_WINDOW)"
bash "$repo/scripts/slurm/submit_tcell_metrics_datasets.sh" \
  breast_xenium nsclc_cosmx
