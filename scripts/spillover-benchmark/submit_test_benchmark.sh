#!/bin/bash
# Submit test-slice RCTD/SPLIT, ResolVI, and Syndri correction jobs.
set -euo pipefail

root=/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/spillover-benchmark/test_admixture
inputs="$root/slices"
reference="$root/train_val_sc_reference_unique.h5ad"
model_reference=/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/spillover-benchmark/train_val.h5ad
mkdir -p /home/jwinchel/Syndri/logs "$root/manifests"
shopt -s nullglob
files=("$inputs"/Lung*.h5ad)
shopt -u nullglob
(( ${#files[@]} == 4 )) || { echo "Expected four test-slice files in $inputs; found ${#files[@]}" >&2; exit 1; }
printf '%s\n' "${files[@]}" > "$root/manifests/test_slices.txt"
manifest="$root/manifests/test_slices.txt"

rctd_job=$(sbatch --parsable --array=0-3%2 \
  --export=ALL,SPILLOVER_TEST_MANIFEST="$manifest",SPILLOVER_SC_REFERENCE="$reference",SPILLOVER_RCTD_OUTPUT_DIR="$root/rctd" \
  /home/jwinchel/Syndri/scripts/spillover-benchmark/run_rctd_test.sbatch)

split_job=$(sbatch --parsable --dependency="afterok:$rctd_job" --array=0-3%2 \
  --export=ALL,INPUT_DIR="$root/rctd",INPUT_GLOB='*.h5ad',INPUT_LAYERS=X,REFERENCE_H5AD="$reference",REFERENCE_LAYER=X,REFERENCE_CELL_TYPE_COL=cell_type,OUTPUT_ROOT="$root/split",INTERMEDIATE_ROOT="$root/split/intermediate",SPLIT_OUTPUT_ROOT="$root/split/split_output",MERGED_OUTPUT_DIR="$root/split/merged_h5ad",RESULT_LAYER_PREFIX=split_purified_counts,MERGE_INPLACE=0 \
  /home/jwinchel/Syndri/scripts/slurm/run_split_h5ad_layers.sbatch)

resolvi_job=$(sbatch --parsable --array=0-3%2 \
  --export=ALL,SPILLOVER_TEST_MANIFEST="$manifest",SPILLOVER_RESOLVI_OUTPUT_ROOT="$root" \
  /home/jwinchel/Syndri/scripts/spillover-benchmark/run_resolvi_test.sbatch)

syndri_job=$(sbatch --parsable --array=0-3%2 \
  --export=ALL,SPILLOVER_TEST_MANIFEST="$manifest",SPILLOVER_MODEL_REFERENCE="$model_reference",SPILLOVER_SYNDRI_OUTPUT_DIR="$root/syndri" \
  /home/jwinchel/Syndri/scripts/spillover-benchmark/run_syndri_test.sbatch)

printf 'RCTD job: %s\nSPLIT job: %s (after RCTD)\nResolVI job: %s\nSyndri job: %s\n' \
  "$rctd_job" "$split_job" "$resolvi_job" "$syndri_job"
