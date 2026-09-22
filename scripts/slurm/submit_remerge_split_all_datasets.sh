#!/usr/bin/env bash
# Remerge completed SPLIT Matrix Market exports into final h5ad files.
# No SPLIT purification or reference preparation is rerun.
set -euo pipefail

base=${DATASET_ROOT:-/work/magroup/skrieger/Syndri/data/datasets}
repo=/home/jwinchel/Syndri
script="$repo/scripts/slurm/remerge_split_outputs.sbatch"
MAX_CONCURRENT=${MAX_CONCURRENT:-2}
DRY_RUN=${DRY_RUN:-0}

if (( $# )); then
  datasets=("$@")
else
  datasets=(agingbrain_merfish brain_merfish nsclc_cosmx breast_xenium)
fi

[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_CONCURRENT must be a positive integer." >&2
  exit 1
}
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1." >&2; exit 1 ;; esac

mkdir -p "$repo/logs"
if [[ "$DRY_RUN" == 0 ]]; then
  manifest_dir=$(mktemp -d "$repo/logs/remerge-split-inputs.XXXXXXXX")
fi
shopt -s nullglob

# Validate all inputs and exports before submitting any jobs.
for dataset in "${datasets[@]}"; do
  case "$dataset" in
    agingbrain_merfish|brain_merfish|nsclc_cosmx|breast_xenium) ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 1 ;;
  esac
  final_dir="$base/$dataset/results/final"
  split_root="$base/$dataset/results/split/split_output"
  files=("$final_dir"/*.h5ad)
  (( ${#files[@]} )) || { echo "No final h5ad files in $final_dir" >&2; exit 1; }
  [[ -d "$split_root" ]] || { echo "Missing SPLIT output root: $split_root" >&2; exit 1; }

  for file in "${files[@]}"; do
    sample=$(basename "$file" .h5ad)
    exports=("$split_root/$sample"/*__X/split_purified_counts.mtx)
    (( ${#exports[@]} == 1 )) || {
      echo "$file: expected one X-layer SPLIT export, found ${#exports[@]}" >&2
      exit 1
    }
    sample_dir=$(dirname "${exports[0]}")
    for required in split_purified_counts_genes.txt split_purified_counts_cells.txt; do
      [[ -f "$sample_dir/$required" ]] || { echo "Missing $sample_dir/$required" >&2; exit 1; }
    done
  done
  echo "$dataset: validated ${#files[@]} final files and SPLIT exports"
done

for dataset in "${datasets[@]}"; do
  final_dir="$base/$dataset/results/final"
  split_root="$base/$dataset/results/split/split_output"
  files=("$final_dir"/*.h5ad)
  array="0-$((${#files[@]} - 1))%$MAX_CONCURRENT"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY RUN: $dataset array=$array"
    continue
  fi

  manifest="$manifest_dir/$dataset.txt"
  printf '%s\n' "${files[@]}" > "$manifest"
  job_id=$(sbatch --parsable \
    --job-name="remerge-$dataset" \
    --array="$array" \
    --export="ALL,INPUT_LIST_FILE=$manifest,SPLIT_OUTPUT_ROOT=$split_root,INPUT_LAYERS=X,RESULT_LAYER_PREFIX=split_purified_counts,FILL_MISSING=original" \
    "$script")
  printf '%s\t%s\t%s\n' "$dataset" "$job_id" "$manifest"
done
