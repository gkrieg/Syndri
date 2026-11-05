#!/bin/bash
#SBATCH -p cpu,gpu-large
#SBATCH --mem=32Gb
#SBATCH --job-name=split-h5ad-layers
#SBATCH --output=/home/jwinchel/Syndri/logs/split-h5ad-layers-%A_%a.txt
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh

# File selection, one input file per array task.
INPUT_DIR=${INPUT_DIR:-/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/rctd_annotations}
INPUT_GLOB=${INPUT_GLOB:-*.h5ad}
INPUT_LIST_FILE=${INPUT_LIST_FILE:-}
INPUT_FILES=${INPUT_FILES:-}

# Layer selection. If INPUT_LAYERS is unset, uses
# ${INPUT_LAYER_PREFIX}_${LAYER_START} through ${INPUT_LAYER_PREFIX}_${LAYER_START + N_LAYERS - 1}.
INPUT_LAYERS=${INPUT_LAYERS:-}
INPUT_LAYERS_SEP=${INPUT_LAYERS_SEP:-,}
N_LAYERS=${N_LAYERS:-3}
LAYER_START=${LAYER_START:-0}
INPUT_LAYER_PREFIX=${INPUT_LAYER_PREFIX:-trial}
INPUT_LAYER_TEMPLATE=${INPUT_LAYER_TEMPLATE:-${INPUT_LAYER_PREFIX}_%d}
INCLUDE_X=${INCLUDE_X:-0}

# Reference settings for SPLIT reference profile construction.
# If REFERENCE_H5AD is unset, the Python prep step uses rctd_run_metadata_<layer>$reference_file.
REFERENCE_H5AD=${REFERENCE_H5AD:-}
REFERENCE_LAYER=${REFERENCE_LAYER:-X}
REFERENCE_CELL_TYPE_COL=${REFERENCE_CELL_TYPE_COL:-cell_type}

OUTPUT_ROOT=${OUTPUT_ROOT:-/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/split_h5ad_layers}
INTERMEDIATE_ROOT=${INTERMEDIATE_ROOT:-$OUTPUT_ROOT/intermediate}
SPLIT_OUTPUT_ROOT=${SPLIT_OUTPUT_ROOT:-$OUTPUT_ROOT/split_output}
MERGED_OUTPUT_DIR=${MERGED_OUTPUT_DIR:-$OUTPUT_ROOT/merged_h5ad}

PY_CONDA_ENV=${PY_CONDA_ENV:-rctd}
SPLIT_R_ENV=${SPLIT_R_ENV:-/home/jwinchel/micromamba/envs/split-r}
SPLIT_CHUNK_SIZE=${SPLIT_CHUNK_SIZE:-50000}
OVERWRITE_INTERMEDIATE=${OVERWRITE_INTERMEDIATE:-0}
SPLIT_OVERWRITE=${SPLIT_OVERWRITE:-0}
SPLIT_EXPORT_OVERWRITE=${SPLIT_EXPORT_OVERWRITE:-0}
OVERWRITE_MERGED_LAYERS=${OVERWRITE_MERGED_LAYERS:-0}
MERGE_INPLACE=${MERGE_INPLACE:-0}
FILL_MISSING=${FILL_MISSING:-original}
RESULT_LAYER_PREFIX=${RESULT_LAYER_PREFIX:-split_purified_counts}
KEEP_ZERO_WEIGHT_CELLS=${KEEP_ZERO_WEIGHT_CELLS:-0}

if [[ -n "$INPUT_LIST_FILE" ]]; then
  mapfile -t ALL_INPUT_FILES < "$INPUT_LIST_FILE"
elif [[ -n "$INPUT_FILES" ]]; then
  IFS=',' read -r -a ALL_INPUT_FILES <<< "$INPUT_FILES"
else
  shopt -s nullglob
  ALL_INPUT_FILES=("$INPUT_DIR"/$INPUT_GLOB)
  shopt -u nullglob
fi

FILTERED_INPUT_FILES=()
for file in "${ALL_INPUT_FILES[@]:-}"; do
  if [[ -n "$file" && "${file:0:1}" != "#" ]]; then
    FILTERED_INPUT_FILES+=("$file")
  fi
done
ALL_INPUT_FILES=("${FILTERED_INPUT_FILES[@]}")

if (( ${#ALL_INPUT_FILES[@]} == 0 )); then
  echo "No input h5ad files found." >&2
  echo "  INPUT_DIR=$INPUT_DIR" >&2
  echo "  INPUT_GLOB=$INPUT_GLOB" >&2
  echo "  INPUT_LIST_FILE=${INPUT_LIST_FILE:-<unset>}" >&2
  echo "  INPUT_FILES=${INPUT_FILES:-<unset>}" >&2
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#ALL_INPUT_FILES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=$TASK_ID is out of range for ${#ALL_INPUT_FILES[@]} input file(s)." >&2
  echo "Submit with --array=0-$(( ${#ALL_INPUT_FILES[@]} - 1 ))%N, or edit the array range." >&2
  exit 1
fi

ST_DATA=${ALL_INPUT_FILES[$TASK_ID]}
SAMPLE_NAME=$(basename "$ST_DATA")
SAMPLE_NAME=${SAMPLE_NAME%.h5ad}
SAMPLE_NAME=${SAMPLE_NAME%.h5}
SAFE_SAMPLE_NAME=${SAMPLE_NAME//[^A-Za-z0-9_.-]/_}

if [[ -n "$INPUT_LAYERS" ]]; then
  LAYER_CSV=${INPUT_LAYERS//${INPUT_LAYERS_SEP}/,}
else
  INPUT_LAYER_NAMES=()
  if [[ "$INCLUDE_X" == "1" || "$INCLUDE_X" == "true" || "$INCLUDE_X" == "TRUE" ]]; then
    INPUT_LAYER_NAMES+=("X")
  fi
  for ((i = 0; i < N_LAYERS; i++)); do
    layer_idx=$((LAYER_START + i))
    printf -v layer_name "$INPUT_LAYER_TEMPLATE" "$layer_idx"
    INPUT_LAYER_NAMES+=("$layer_name")
  done
  LAYER_CSV=$(IFS=,; echo "${INPUT_LAYER_NAMES[*]}")
fi

TASK_INTERMEDIATE_ROOT="$INTERMEDIATE_ROOT/$SAFE_SAMPLE_NAME"
TASK_SPLIT_OUTPUT_ROOT="$SPLIT_OUTPUT_ROOT/$SAFE_SAMPLE_NAME"
mkdir -p "$TASK_INTERMEDIATE_ROOT" "$TASK_SPLIT_OUTPUT_ROOT" "$MERGED_OUTPUT_DIR" /home/jwinchel/Syndri/logs

PREP_ARGS=(
  prepare
  --h5ad "$ST_DATA"
  --output-root "$TASK_INTERMEDIATE_ROOT"
  --layers "$LAYER_CSV"
  --reference-layer "$REFERENCE_LAYER"
  --reference-cell-type-col "$REFERENCE_CELL_TYPE_COL"
)
if [[ -n "$REFERENCE_H5AD" ]]; then
  PREP_ARGS+=(--reference-h5ad "$REFERENCE_H5AD")
fi
if [[ "$KEEP_ZERO_WEIGHT_CELLS" == "1" || "$KEEP_ZERO_WEIGHT_CELLS" == "true" || "$KEEP_ZERO_WEIGHT_CELLS" == "TRUE" ]]; then
  PREP_ARGS+=(--include-zero-weight-cells)
fi
if [[ "$OVERWRITE_INTERMEDIATE" == "1" || "$OVERWRITE_INTERMEDIATE" == "true" || "$OVERWRITE_INTERMEDIATE" == "TRUE" ]]; then
  PREP_ARGS+=(--overwrite)
fi

echo "Preparing SPLIT inputs"
echo "  input: $ST_DATA"
echo "  layers: $LAYER_CSV"
echo "  intermediate: $TASK_INTERMEDIATE_ROOT"
conda activate "$PY_CONDA_ENV"
python /home/jwinchel/Syndri/data/split_h5ad_io.py "${PREP_ARGS[@]}"

echo "Converting intermediates to RDS"
conda activate "$SPLIT_R_ENV"
Rscript /home/jwinchel/SPLIT/convert_to_rds.R "$TASK_INTERMEDIATE_ROOT" "" "$OVERWRITE_INTERMEDIATE"

echo "Running SPLIT"
SPLIT_CHUNK_SIZE="$SPLIT_CHUNK_SIZE" Rscript /home/jwinchel/SPLIT/run_split.R \
  "$TASK_INTERMEDIATE_ROOT" \
  "$TASK_SPLIT_OUTPUT_ROOT" \
  "" \
  "$SPLIT_OVERWRITE"

echo "Exporting SPLIT purified counts"
Rscript /home/jwinchel/Syndri/data/export_split_purified_counts_mtx.R \
  "$TASK_SPLIT_OUTPUT_ROOT" \
  "$SPLIT_EXPORT_OVERWRITE"

MERGE_ARGS=(
  merge
  --h5ad "$ST_DATA"
  --split-output-root "$TASK_SPLIT_OUTPUT_ROOT"
  --layers "$LAYER_CSV"
  --result-layer-prefix "$RESULT_LAYER_PREFIX"
  --fill-missing "$FILL_MISSING"
)
if [[ "$OVERWRITE_MERGED_LAYERS" == "1" || "$OVERWRITE_MERGED_LAYERS" == "true" || "$OVERWRITE_MERGED_LAYERS" == "TRUE" ]]; then
  MERGE_ARGS+=(--overwrite)
fi
if [[ "$MERGE_INPLACE" != "1" && "$MERGE_INPLACE" != "true" && "$MERGE_INPLACE" != "TRUE" ]]; then
  MERGE_ARGS+=(--output-h5ad "$MERGED_OUTPUT_DIR/$SAMPLE_NAME.h5ad")
fi

echo "Merging SPLIT outputs into AnnData"
conda activate "$PY_CONDA_ENV"
python /home/jwinchel/Syndri/data/split_h5ad_io.py "${MERGE_ARGS[@]}"

echo "Done"

# Example submit for the xenium noise experiment RCTD layers already present as of 2026-07-28:
# sbatch --array=0-75%4 --export=ALL,INPUT_LAYERS=trial_0:trial_1:trial_2,INPUT_LAYERS_SEP=: /home/jwinchel/Syndri/scripts/slurm/run_split_h5ad_layers.sbatch
#
# Include adata.X too:
# sbatch --array=0-76%4 --export=ALL,INPUT_LAYERS=X:trial_0:trial_1:trial_2,INPUT_LAYERS_SEP=: /home/jwinchel/Syndri/scripts/slurm/run_split_h5ad_layers.sbatch

# sbatch --dependency=afterok:56783 --array=0-7%2 \
#   --export=ALL,INPUT_DIR=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/,INPUT_GLOB='cosmx_nsclc*1pct.h5ad',INPUT_LAYERS=X,REFERENCE_H5AD=/compute/oven-0-13/jwinchel/scRNA-seq/adata/GSE127465_tumor_annotated.h5ad,REFERENCE_LAYER=X,REFERENCE_CELL_TYPE_COL=cell_type \
#   /home/jwinchel/Syndri/scripts/slurm/run_split_h5ad_layers.sbatch