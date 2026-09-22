#!/usr/bin/env bash
# Recalculate only corrected SPLIT metric rows after remerging SPLIT layers.
set -euo pipefail

repo=/home/jwinchel/Syndri
export SPLIT_ONLY=1
export SYNDRI_REFERENCE_COSINE=0

echo "Submitting SPLIT-only non-T-cell metrics for all four datasets"
bash "$repo/scripts/slurm/submit_metrics_all_datasets.sh" \
  agingbrain_merfish brain_merfish nsclc_cosmx breast_xenium

echo "Submitting SPLIT-only T-cell metrics for their configured datasets"
bash "$repo/scripts/slurm/submit_tcell_metrics_datasets.sh" \
  breast_xenium nsclc_cosmx
