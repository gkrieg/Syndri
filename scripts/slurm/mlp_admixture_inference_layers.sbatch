#!/bin/bash
#SBATCH -p gpu-small
#SBATCH --mem=16Gb
#SBATCH --job-name mlp-infer-layers
#SBATCH --output=/home/jwinchel/Syndri/logs/mlp-infer-layers-%A_%a.txt
#SBATCH -t 0-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail

module load cuda-12.1

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-heimdall}"

REFERENCE_DATA=${REFERENCE_DATA:-/compute/oven-0-13/jwinchel/scRNA-seq/adata/chromium_GSM9509174_breast_integrated.h5ad}
MLP_CHECKPOINT=${MLP_CHECKPOINT:-/home/jwinchel/Syndri/admixture_prediction_results/chromium_breast_GSM95019174_augmented_mlp/mlp_baseline/best_model.pt}
CONFIG_SOURCE=${CONFIG_SOURCE:-checkpoint}
EXPERIMENT=${EXPERIMENT:-admixture_prediction}
BATCH_SIZE=${BATCH_SIZE:-256}
SPLIT_VALUE=${SPLIT_VALUE:-val}
SPLIT_COL=${SPLIT_COL:-split}
GENE_LIST=${GENE_LIST:-}
CELL_TYPE_COL=${CELL_TYPE_COL:-harmony_cell_type}
PRESERVE_SPLIT_COLUMN=${PRESERVE_SPLIT_COLUMN:-0}
COORD_X=${COORD_X:-x_centroid}
COORD_Y=${COORD_Y:-y_centroid}
COORD_COLS=${COORD_COLS:-${COORD_X},${COORD_Y}}
TISSUE_COL=${TISSUE_COL:-}
FAST_INFERENCE=${FAST_INFERENCE:-1}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}
MLP_HIDDEN_DIMS=${MLP_HIDDEN_DIMS:-1024,512}
MLP_DROPOUT=${MLP_DROPOUT:-0.1}
DEVICE=${DEVICE:-cuda}

# File selection, one input file per array task.
ST_DATA_LIST_FILE=${ST_DATA_LIST_FILE:-}
ST_DATA_LIST=${ST_DATA_LIST:-}
ST_DATA_DIR=${ST_DATA_DIR:-}
ST_DATA_GLOB=${ST_DATA_GLOB:-*.h5ad}

# Layer selection. If INPUT_LAYERS is unset, predicts layers
# ${INPUT_LAYER_PREFIX}_${LAYER_START} through ${INPUT_LAYER_PREFIX}_${LAYER_START + N_LAYERS - 1}.
N_LAYERS=${N_LAYERS:-10}
LAYER_START=${LAYER_START:-0}
INPUT_LAYERS=${INPUT_LAYERS:-}
INPUT_LAYER_PREFIX=${INPUT_LAYER_PREFIX:-trial}
INPUT_LAYER_TEMPLATE=${INPUT_LAYER_TEMPLATE:-${INPUT_LAYER_PREFIX}_%d}
PRED_LAYERS=${PRED_LAYERS:-}
PRED_LAYER_PREFIX=${PRED_LAYER_PREFIX:-mlp_admix_probs_trial}
PRED_LAYER_TEMPLATE=${PRED_LAYER_TEMPLATE:-${PRED_LAYER_PREFIX}_%d}
OVERWRITE_LAYERS=${OVERWRITE_LAYERS:-0}

TEMP_ROOT=${TEMP_ROOT:-${SLURM_TMPDIR:-/work/magroup/skrieger/Syndri/mlp_layer_tmp}}
KEEP_TEMP=${KEEP_TEMP:-0}

if [[ -n "$ST_DATA_LIST_FILE" ]]; then
  mapfile -t ALL_ST_FILES < "$ST_DATA_LIST_FILE"
elif [[ -n "$ST_DATA_LIST" ]]; then
  IFS=',' read -r -a ALL_ST_FILES <<< "$ST_DATA_LIST"
elif [[ -n "$ST_DATA_DIR" ]]; then
  shopt -s nullglob
  ALL_ST_FILES=("$ST_DATA_DIR"/$ST_DATA_GLOB)
  shopt -u nullglob
else
  ALL_ST_FILES=("${ST_DATA:-/compute/oven-0-13/jwinchel/spatial_adata/cosmx_nsclc_Lung5-Rep1_adata_annotated.h5ad}")
fi

FILTERED_ST_FILES=()
for st_file in "${ALL_ST_FILES[@]}"; do
  if [[ -n "$st_file" && "${st_file:0:1}" != "#" ]]; then
    FILTERED_ST_FILES+=("$st_file")
  fi
done
ALL_ST_FILES=("${FILTERED_ST_FILES[@]}")

if (( ${#ALL_ST_FILES[@]} == 0 )); then
  echo "No ST input files found." >&2
  echo "  ST_DATA_LIST_FILE=${ST_DATA_LIST_FILE:-<unset>}" >&2
  echo "  ST_DATA_LIST=${ST_DATA_LIST:-<unset>}" >&2
  echo "  ST_DATA_DIR=${ST_DATA_DIR:-<unset>}" >&2
  echo "  ST_DATA_GLOB=$ST_DATA_GLOB" >&2
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#ALL_ST_FILES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=$TASK_ID is out of range for ${#ALL_ST_FILES[@]} input file(s)." >&2
  echo "Submit with --array=0-$(( ${#ALL_ST_FILES[@]} - 1 ))%N, or edit the array range." >&2
  exit 1
fi

ST_DATA=${ALL_ST_FILES[$TASK_ID]}
SAMPLE_NAME=$(basename "$ST_DATA")
SAMPLE_NAME=${SAMPLE_NAME%.h5ad}
SAMPLE_NAME=${SAMPLE_NAME%.h5}
SAFE_SAMPLE_NAME=${SAMPLE_NAME//[^A-Za-z0-9_.-]/_}

if [[ -n "$INPUT_LAYERS" ]]; then
  IFS=',' read -r -a INPUT_LAYER_NAMES <<< "$INPUT_LAYERS"
else
  INPUT_LAYER_NAMES=()
  for ((i = 0; i < N_LAYERS; i++)); do
    layer_idx=$((LAYER_START + i))
    printf -v layer_name "$INPUT_LAYER_TEMPLATE" "$layer_idx"
    INPUT_LAYER_NAMES+=("$layer_name")
  done
fi

if [[ -n "$PRED_LAYERS" ]]; then
  IFS=',' read -r -a PRED_LAYER_NAMES <<< "$PRED_LAYERS"
else
  PRED_LAYER_NAMES=()
  for ((i = 0; i < ${#INPUT_LAYER_NAMES[@]}; i++)); do
    layer_idx=$((LAYER_START + i))
    printf -v layer_name "$PRED_LAYER_TEMPLATE" "$layer_idx"
    PRED_LAYER_NAMES+=("$layer_name")
  done
fi

if (( ${#INPUT_LAYER_NAMES[@]} != ${#PRED_LAYER_NAMES[@]} )); then
  echo "INPUT_LAYERS and PRED_LAYERS must contain the same number of layer names." >&2
  echo "  input layers: ${#INPUT_LAYER_NAMES[@]}" >&2
  echo "  pred layers: ${#PRED_LAYER_NAMES[@]}" >&2
  exit 1
fi

PRESERVE_SPLIT_ARG=--no-preserve-split-column
if [[ "$PRESERVE_SPLIT_COLUMN" == "1" || "$PRESERVE_SPLIT_COLUMN" == "true" || "$PRESERVE_SPLIT_COLUMN" == "TRUE" ]]; then
  PRESERVE_SPLIT_ARG=--preserve-split-column
fi

FAST_INFERENCE_ARG=--no-fast-inference
if [[ "$FAST_INFERENCE" == "1" || "$FAST_INFERENCE" == "true" || "$FAST_INFERENCE" == "TRUE" ]]; then
  FAST_INFERENCE_ARG=--fast-inference
fi

GENE_LIST_ARGS=()
if [[ -n "$GENE_LIST" ]]; then
  GENE_LIST_ARGS=(--gene-list "$GENE_LIST")
fi

TISSUE_COL_ARGS=()
if [[ -n "$TISSUE_COL" ]]; then
  TISSUE_COL_ARGS=(--tissue-col "$TISSUE_COL")
fi

RUN_DIR="$TEMP_ROOT/${SLURM_JOB_ID:-manual}_${TASK_ID}_${SAFE_SAMPLE_NAME}"
mkdir -p "$RUN_DIR"

cleanup() {
  if [[ "$KEEP_TEMP" != "1" && "$KEEP_TEMP" != "true" && "$KEEP_TEMP" != "TRUE" ]]; then
    rm -rf "$RUN_DIR"
  fi
}
trap cleanup EXIT

merge_prediction_layer() {
  local target_file=$1
  local prediction_file=$2
  local layer_name=$3

  python - "$target_file" "$prediction_file" "$layer_name" "$OVERWRITE_LAYERS" <<'PY'
import os
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, issparse

target_path = Path(sys.argv[1])
prediction_path = Path(sys.argv[2])
layer_name = sys.argv[3]
overwrite = sys.argv[4].lower() in {"1", "true", "yes"}

target = ad.read_h5ad(target_path)
prediction = ad.read_h5ad(prediction_path)

if layer_name in target.layers and not overwrite:
    raise ValueError(f"Layer {layer_name!r} already exists in {target_path}; set OVERWRITE_LAYERS=1 to replace it.")
if layer_name not in prediction.layers:
    raise KeyError(f"Layer {layer_name!r} not found in prediction output {prediction_path}")

pred_matrix = prediction.layers[layer_name]
if not issparse(pred_matrix):
    pred_matrix = csr_matrix(pred_matrix)
else:
    pred_matrix = pred_matrix.tocsr()

target_obs = pd.Index(target.obs_names.astype(str))
target_var = pd.Index(target.var_names.astype(str))
pred_obs = pd.Index(prediction.obs_names.astype(str))
pred_var = pd.Index(prediction.var_names.astype(str))

row_index = target_obs.get_indexer(pred_obs)
col_index = target_var.get_indexer(pred_var)
if (row_index < 0).any():
    missing = pred_obs[row_index < 0].tolist()[:20]
    raise ValueError(f"Prediction rows are missing from target AnnData. First 20: {missing}")
if (col_index < 0).any():
    missing = pred_var[col_index < 0].tolist()[:20]
    raise ValueError(f"Prediction genes are missing from target AnnData. First 20: {missing}")

if pred_matrix.shape == target.shape and pred_obs.equals(target_obs) and pred_var.equals(target_var):
    target.layers[layer_name] = pred_matrix
else:
    pred_coo = pred_matrix.tocoo()
    target.layers[layer_name] = coo_matrix(
        (pred_coo.data, (row_index[pred_coo.row], col_index[pred_coo.col])),
        shape=target.shape,
        dtype=pred_matrix.dtype,
    ).tocsr()

tmp_path = target_path.with_name(f".{target_path.name}.tmp-{os.getpid()}")
target.write_h5ad(tmp_path)
tmp_path.replace(target_path)
print(f"> Merged {layer_name!r} into {target_path}")
PY
}

echo "Running layered MLP admixture inference"
echo "  array task: $TASK_ID / $(( ${#ALL_ST_FILES[@]} - 1 ))"
echo "  st data: $ST_DATA"
echo "  input layers: ${INPUT_LAYER_NAMES[*]}"
echo "  prediction layers: ${PRED_LAYER_NAMES[*]}"
echo "  mlp checkpoint: $MLP_CHECKPOINT"
echo "  config source: $CONFIG_SOURCE"
echo "  experiment: $EXPERIMENT"
echo "  split column: $SPLIT_COL"
echo "  split value: $SPLIT_VALUE"
echo "  preserve split column: $PRESERVE_SPLIT_COLUMN"
echo "  cell type column: $CELL_TYPE_COL"
echo "  gene list: ${GENE_LIST:-<all spatial AnnData genes>}"
echo "  coord cols: $COORD_COLS"
echo "  tissue col: ${TISSUE_COL:-<default>}"
echo "  fast inference: $FAST_INFERENCE"
echo "  overwrite layers: $OVERWRITE_LAYERS"
echo "  run dir: $RUN_DIR"

for ((i = 0; i < ${#INPUT_LAYER_NAMES[@]}; i++)); do
  INPUT_LAYER=${INPUT_LAYER_NAMES[$i]}
  PRED_LAYER=${PRED_LAYER_NAMES[$i]}
  SAFE_INPUT_LAYER=${INPUT_LAYER//[^A-Za-z0-9_.-]/_}
  SAFE_PRED_LAYER=${PRED_LAYER//[^A-Za-z0-9_.-]/_}
  PREPARED_NAME="${SAFE_SAMPLE_NAME}_${SAFE_INPUT_LAYER}_mlp_prepared.h5ad"
  OUTPUT_NAME="${SAFE_SAMPLE_NAME}_${SAFE_PRED_LAYER}_mlp_prediction.h5ad"

  EXPRESSION_LAYER_ARGS=()
  if [[ "$INPUT_LAYER" != "X" && "$INPUT_LAYER" != "__X__" ]]; then
    EXPRESSION_LAYER_ARGS=(--expression-layer "$INPUT_LAYER")
  fi

  echo "Predicting layer $((i + 1))/${#INPUT_LAYER_NAMES[@]}"
  echo "  input expression layer: $INPUT_LAYER"
  echo "  output probability layer: $PRED_LAYER"

  python admixture_mlp_baseline/infer.py \
    --st-data "$ST_DATA" \
    --reference-data "$REFERENCE_DATA" \
    --output-dir "$RUN_DIR" \
    --prepared-name "$PREPARED_NAME" \
    --output-name "$OUTPUT_NAME" \
    --config-source "$CONFIG_SOURCE" \
    --experiment "$EXPERIMENT" \
    --checkpoint-path "$MLP_CHECKPOINT" \
    --cell-type-col "$CELL_TYPE_COL" \
    --prob-layer "$PRED_LAYER" \
    --batch-size "$BATCH_SIZE" \
    --split-col "$SPLIT_COL" \
    --split-value "$SPLIT_VALUE" \
    --coord-cols "$COORD_COLS" \
    --extra-overrides "$EXTRA_OVERRIDES" \
    --hidden-dims "$MLP_HIDDEN_DIMS" \
    --dropout "$MLP_DROPOUT" \
    --device "$DEVICE" \
    "$PRESERVE_SPLIT_ARG" \
    "$FAST_INFERENCE_ARG" \
    --no-apply-admixture \
    "${GENE_LIST_ARGS[@]}" \
    "${EXPRESSION_LAYER_ARGS[@]}" \
    "${TISSUE_COL_ARGS[@]}"

  merge_prediction_layer "$ST_DATA" "$RUN_DIR/$OUTPUT_NAME" "$PRED_LAYER"
done

echo "Finished layered MLP predictions for $ST_DATA"

# Examples:
# sbatch --array=0-18%2 --export=ALL,ST_DATA_LIST_FILE=/path/to/files.txt,MLP_CHECKPOINT=/path/to/best_model.pt scripts/slurm/mlp_admixture_inference_layers.sbatch
#
# Custom input/output layer names:
# sbatch --array=0-4%1 --export=ALL,ST_DATA_LIST_FILE=/path/to/files.txt,INPUT_LAYERS=counts_noise_low,counts_noise_high,PRED_LAYERS=mlp_probs_low,mlp_probs_high scripts/slurm/mlp_admixture_inference_layers.sbatch
#
# To predict from .X, use INPUT_LAYERS=__X__ and explicit PRED_LAYERS.


# sbatch --array=0-76%4 --export=ALL,ST_DATA_DIR,=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/,ST_DATA_GLOB=GSM95*pct,MLP_CHECKPOINT=/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/chromium_breast_GSM95019174_augmented_mlp/mlp_baseline/best_model.pt,N_LAYERS=3,INPUT_LAYER_TEMPLATE='trial_%d',PRED_LAYER_TEMPLATE='mlp_admix_probs_trial_%d' /home/jwinchel/Syndri/scripts/slurm/mlp_admixture_inference_layers.sbatch

# sbatch --array=0-76%4 --export=ALL,ST_DATA_DIR=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/,ST_DATA_GLOB='GSM95*pct.h5ad',MLP_CHECKPOINT=/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/chromium_breast_GSM95019174_augmented_mlp/mlp_baseline/best_model.pt,N_LAYERS=3,INPUT_LAYER_TEMPLATE='trial_%d',PRED_LAYER_TEMPLATE='mlp_admix_probs_trial_%d' /home/jwinchel/Syndri/scripts/slurm/mlp_admixture_inference_layers.sbatch