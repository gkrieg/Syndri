#!/usr/bin/env bash
set -euo pipefail

mkdir -p /home/jwinchel/Syndri/logs

root=/work/magroup/skrieger/Syndri/data/datasets/agingbrain_merfish

for donor in 5 6 7 8; do
  sbatch \
    --job-name="rctd-aging-donor${donor}" \
    --array=0-2%2 \
    --export="ALL,REFERENCE_FILE=${root}/interim/single-cell/rctd_donor_references/reference_donor_${donor}.h5ad,SPATIAL_DIR=${root}/interim/spatial,INPUT_GLOB=st_integrated_MsBrainAgingSpatialDonor_${donor}_[0-9].h5ad,INPUT_LIST_FILE=,INPUT_FILES=,OUTPUT_DIR=${root}/results/spatial,LAYER_PATTERN=^X$,CELL_TYPE_COL=cell_type,MODE=doublet,FORCE=1" \
    /home/jwinchel/Syndri/scripts/slurm/rctd_annotation.sbatch
done