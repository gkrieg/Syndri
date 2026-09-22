#!/usr/bin/env bash
set -euo pipefail

base=${DATASET_ROOT:-/work/magroup/skrieger/Syndri/data/datasets}
repo=/home/jwinchel/Syndri
script="$repo/scripts/slurm/split_sparsity.sbatch"
if (( $# )); then
  datasets=("$@")
else
  datasets=(agingbrain_merfish brain_merfish nsclc_cosmx breast_xenium)
fi
MAX_CONCURRENT=${MAX_CONCURRENT:-2}
[[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_CONCURRENT must be positive.' >&2; exit 1; }
breast_reference=${BREAST_REFERENCE_FILE:-$base/breast_xenium/interim/single-cell/breast_cancer_chromium_genesubset.h5ad}
shopt -s nullglob

# Validate every dataset before submitting anything.
for dataset in "${datasets[@]}"; do
  case "$dataset" in
    agingbrain_merfish|brain_merfish|nsclc_cosmx) ;;
    breast_xenium)
      [[ -f "$breast_reference" ]] || { echo "Missing reference: $breast_reference" >&2; exit 1; }
      ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 1 ;;
  esac
  files=("$base/$dataset/results/final/"*.h5ad)
  (( ${#files[@]} )) || { echo "No final H5AD files for $dataset" >&2; exit 1; }
done

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  mkdir -p "$repo/logs"
  manifest_dir=$(mktemp -d "$repo/logs/split-sparsity-inputs.XXXXXXXX")
fi

for dataset in "${datasets[@]}"; do
  (
    export SPATIAL_DIR="$base/$dataset/results/final"
    files=("$SPATIAL_DIR/"*.h5ad)
    export INPUT_FILES="" INPUT_LAYERS=X RESULT_LAYER_PREFIX=split_purified_counts
    export REFERENCE_FILE="" SPATIAL_CELL_TYPE_COL=""
    case "$dataset" in
      brain_merfish) export CELL_TYPE_COL=class ;;
      nsclc_cosmx) export CELL_TYPE_COL=cell_type_harmonized ;;
      agingbrain_merfish) export CELL_TYPE_COL=cell_type ;;
      breast_xenium)
        export CELL_TYPE_COL=cell_type REFERENCE_FILE="$breast_reference"
        ;;
    esac
    export CACHE_DIR="${CACHE_DIR:-$repo/data/cache/rctd_profiles}"
    if [[ -n "${SPARSITY_OUTPUT_ROOT:-}" ]]; then
      export OUTPUT_DIR="$SPARSITY_OUTPUT_ROOT/$dataset"
    else
      export OUTPUT_DIR="$base/$dataset/results/split/sparsity_comparison"
    fi
    echo "$dataset: ${#files[@]} files; reference labels=$CELL_TYPE_COL; output=$OUTPUT_DIR"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      echo "  array=0-$((${#files[@]} - 1))%$MAX_CONCURRENT; reference=${REFERENCE_FILE:-from per-file RCTD metadata}"
      exit 0
    fi
    export INPUT_LIST_FILE="$manifest_dir/$dataset.txt"
    printf '%s\n' "${files[@]}" > "$INPUT_LIST_FILE"
    sbatch --job-name="sparsity-$dataset" \
      --array="0-$((${#files[@]} - 1))%$MAX_CONCURRENT" --export=ALL "$script"
  )
done
