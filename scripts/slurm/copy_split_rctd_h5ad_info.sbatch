#!/bin/bash
#SBATCH -p cpu,gpu-small,gpu-large
#SBATCH --mem=16Gb
#SBATCH --job-name=copy-split-rctd
#SBATCH --output=/home/jwinchel/Syndri/logs/copy-split-rctd-%A_%a.txt
#SBATCH -t 0-08:00:00
#SBATCH --cpus-per-task=2

set -euo pipefail

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${PY_CONDA_ENV:-rctd}"

SOURCE_DIR=${SOURCE_DIR:-/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/split_h5ad_layers/merged_h5ad}
TARGET_DIR=${TARGET_DIR:-/work/magroup/skrieger/Syndri/data/xenium_noise_experiment}
SOURCE_GLOB=${SOURCE_GLOB:-*.h5ad}
OUTPUT_DIR=${OUTPUT_DIR:-}
OVERWRITE=${OVERWRITE:-1}
DRY_RUN=${DRY_RUN:-0}

LAYER_PREFIXES=${LAYER_PREFIXES:-split_purified_counts}
OBS_PREFIXES=${OBS_PREFIXES:-split_,rctd_}
OBSM_PREFIXES=${OBSM_PREFIXES:-rctd_,X_pca_harmony_}
UNS_PREFIXES=${UNS_PREFIXES:-split_,rctd_,source_h5_files,simulated_dropout}
MANIFEST_DIR=${MANIFEST_DIR:-$TARGET_DIR/copy_split_rctd_manifests}

shopt -s nullglob
ALL_SOURCE_FILES=("$SOURCE_DIR"/$SOURCE_GLOB)
shopt -u nullglob

if (( ${#ALL_SOURCE_FILES[@]} == 0 )); then
  echo "No source h5ad files found." >&2
  echo "  SOURCE_DIR=$SOURCE_DIR" >&2
  echo "  SOURCE_GLOB=$SOURCE_GLOB" >&2
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#ALL_SOURCE_FILES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=$TASK_ID is out of range for ${#ALL_SOURCE_FILES[@]} source file(s)." >&2
  echo "Submit with --array=0-$(( ${#ALL_SOURCE_FILES[@]} - 1 ))%N, or edit the array range." >&2
  exit 1
fi

SOURCE_FILE=${ALL_SOURCE_FILES[$TASK_ID]}
BASENAME=$(basename "$SOURCE_FILE")
mkdir -p "$MANIFEST_DIR" /home/jwinchel/Syndri/logs

ARGS=(
  --source-dir "$SOURCE_DIR"
  --target-dir "$TARGET_DIR"
  --glob "$BASENAME"
  --layer-prefixes "$LAYER_PREFIXES"
  --obs-prefixes "$OBS_PREFIXES"
  --obsm-prefixes "$OBSM_PREFIXES"
  --uns-prefixes "$UNS_PREFIXES"
  --manifest "$MANIFEST_DIR/${BASENAME%.h5ad}.csv"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
  ARGS+=(--dry-run)
fi

python /home/jwinchel/Syndri/data/copy_split_rctd_h5ad_info.py "${ARGS[@]}"

# Example dry-run:
# sbatch --array=0-75%8 --export=ALL,DRY_RUN=1 /home/jwinchel/Syndri/scripts/slurm/copy_split_rctd_h5ad_info.sbatch
#
# Example in-place copy:
# sbatch --array=0-75%8 --export=ALL,OVERWRITE=1 /home/jwinchel/Syndri/scripts/slurm/copy_split_rctd_h5ad_info.sbatch
