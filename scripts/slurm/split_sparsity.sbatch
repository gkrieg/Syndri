#!/bin/bash
#SBATCH -p cpu,gpu-small,gpu-large
#SBATCH --mem=16Gb
#SBATCH --job-name=split-sparsity
#SBATCH --output=/home/jwinchel/Syndri/logs/split-sparsity-%A_%a.txt
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=4
set -euo pipefail

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${PY_CONDA_ENV:-rctd}"

# Inputs are merged outputs of run_split_h5ad_layers.sbatch.
SPATIAL_DIR=${SPATIAL_DIR:-/home/jwinchel/Syndri/data}
INPUT_GLOB=${INPUT_GLOB:-*.h5ad}
if [[ -n "${INPUT_LIST_FILE:-}" ]]; then
  mapfile -t FILES < "$INPUT_LIST_FILE"
elif [[ -n "${INPUT_FILES:-}" ]]; then
  IFS=',' read -r -a FILES <<< "$INPUT_FILES"
else
  shopt -s nullglob
  FILES=("$SPATIAL_DIR"/$INPUT_GLOB)
  shopt -u nullglob
fi
INPUTS=()
for file in "${FILES[@]}"; do
  [[ -z "$file" || "$file" == \#* ]] || INPUTS+=("$file")
done
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#INPUTS[@]} )); then
  echo "Task $TASK_ID is out of range for ${#INPUTS[@]} input files." >&2
  exit 1
fi

# Use ':' between layers in sbatch --export, whose separator is a comma.
INPUT_LAYERS=${INPUT_LAYERS:-X}
ARGS=(
  --spatial "${INPUTS[$TASK_ID]}"
  --layers "${INPUT_LAYERS//:/,}"
  --result-layer-prefix "${RESULT_LAYER_PREFIX:-split_purified_counts}"
  --cell-type-col "${CELL_TYPE_COL:-cell_type}"
  --cell-min "${CELL_MIN:-25}"
  --n-max-cells "${N_MAX_CELLS:-10000}"
  --reference-min-umi "${REFERENCE_MIN_UMI:-10}"
  --reference-max-umi "${REFERENCE_MAX_UMI:-2000}"
  --cache-dir "${CACHE_DIR:-/home/jwinchel/Syndri/data/cache/rctd_profiles}"
  --output-dir "${OUTPUT_DIR:-/home/jwinchel/Syndri/eval/split_sparsity}"
)
if [[ -n "${REFERENCE_FILE:-}" ]]; then
  ARGS+=(--reference "$REFERENCE_FILE")
fi
if [[ -n "${SPATIAL_CELL_TYPE_COL:-}" ]]; then
  ARGS+=(--spatial-cell-type-col "$SPATIAL_CELL_TYPE_COL")
fi
python /home/jwinchel/Syndri/eval/split_sparsity.py "${ARGS[@]}"

# Example (one merged h5ad per array task):
# sbatch --array=0-7%2 --export=ALL,SPATIAL_DIR=/path/to/merged_h5ad,INPUT_LAYERS=X,REFERENCE_FILE=/path/to/sc_reference.h5ad scripts/slurm/split_sparsity.sbatch
