#!/bin/bash
#SBATCH -p cpu,gpu-small
#SBATCH --mem=24Gb
#SBATCH --job-name=rctd-annot
#SBATCH --output=/home/jwinchel/Syndri/logs/rctd-annotation-%A_%a.txt
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=8
set -euo pipefail

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate rctd

REFERENCE_FILE=${REFERENCE_FILE:-}
SPATIAL_DIR=${SPATIAL_DIR:-/home/jwinchel/Syndri/data}
INPUT_GLOB=${INPUT_GLOB:-*.h5ad}
INPUT_LIST_FILE=${INPUT_LIST_FILE:-}
INPUT_FILES=${INPUT_FILES:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
LAYER_PATTERN=${LAYER_PATTERN:-counts?$|raw|X}
CELL_TYPE_COL=${CELL_TYPE_COL:-cell_type}
MODE=${MODE:-doublet}
FORCE=${FORCE:-0}
BATCH_SIZE=${BATCH_SIZE:-auto}
SIGMA_OVERRIDE=${SIGMA_OVERRIDE:-}
CLASS_DF=${CLASS_DF:-}
SIMULATED_DROPOUT=${SIMULATED_DROPOUT:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [[ -z "$REFERENCE_FILE" ]]; then
  echo "REFERENCE_FILE must be set." >&2
  exit 1
fi

if [[ -n "$INPUT_LIST_FILE" ]]; then
  mapfile -t ALL_INPUT_FILES < "$INPUT_LIST_FILE"
elif [[ -n "$INPUT_FILES" ]]; then
  IFS=',' read -r -a ALL_INPUT_FILES <<< "$INPUT_FILES"
else
  shopt -s nullglob
  ALL_INPUT_FILES=("$SPATIAL_DIR"/$INPUT_GLOB)
  shopt -u nullglob
fi

FILTERED_INPUT_FILES=()
for file in "${ALL_INPUT_FILES[@]:-}"; do
  if [[ -n "$file" ]]; then
    FILTERED_INPUT_FILES+=("$file")
  fi
done
ALL_INPUT_FILES=("${FILTERED_INPUT_FILES[@]}")

if (( ${#ALL_INPUT_FILES[@]} == 0 )); then
  echo "No input files found." >&2
  echo "  SPATIAL_DIR=$SPATIAL_DIR" >&2
  echo "  INPUT_GLOB=$INPUT_GLOB" >&2
  echo "  INPUT_LIST_FILE=${INPUT_LIST_FILE:-<unset>}" >&2
  echo "  INPUT_FILES=${INPUT_FILES:-<unset>}" >&2
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#ALL_INPUT_FILES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=$TASK_ID is out of range for ${#ALL_INPUT_FILES[@]} input file(s)." >&2
  exit 1
fi

TASK_INPUT_FILE=${ALL_INPUT_FILES[$TASK_ID]}
TASK_BASENAME=$(basename "$TASK_INPUT_FILE")

if [[ -n "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

PYTHON_ARGS=(
  --reference "$REFERENCE_FILE"
  --spatial "$TASK_INPUT_FILE"
  --layer-pattern "$LAYER_PATTERN"
  --cell-type-col "$CELL_TYPE_COL"
  --mode "$MODE"
  --batch-size "$BATCH_SIZE"
  --reference-min-umi "${REFERENCE_MIN_UMI:-10}"
  --reference-max-umi "${REFERENCE_MAX_UMI:-2000}"
  --umi-min-sigma "${UMI_MIN_SIGMA:-100}"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  PYTHON_ARGS+=(--output-dir "$OUTPUT_DIR")
fi

if [[ "$FORCE" == "1" || "$FORCE" == "true" || "$FORCE" == "TRUE" ]]; then
  PYTHON_ARGS+=(--force)
fi

if [[ -n "$SIGMA_OVERRIDE" ]]; then
  PYTHON_ARGS+=(--sigma-override "$SIGMA_OVERRIDE")
fi

if [[ -n "$CLASS_DF" ]]; then
  PYTHON_ARGS+=(--class-df "$CLASS_DF")
fi

if [[ -n "$SIMULATED_DROPOUT" ]]; then
  PYTHON_ARGS+=(--simulated-dropout "$SIMULATED_DROPOUT")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARG_ARRAY=($EXTRA_ARGS)
  PYTHON_ARGS+=("${EXTRA_ARG_ARRAY[@]}")
fi

echo "Running RCTD annotation"
echo "  task id: $TASK_ID"
echo "  input: $TASK_INPUT_FILE"
echo "  reference: $REFERENCE_FILE"
echo "  layer pattern: $LAYER_PATTERN"
echo "  output dir: ${OUTPUT_DIR:-<inplace>}"
echo "  mode: $MODE"
echo "  batch size: $BATCH_SIZE"
echo "  force: $FORCE"

python /home/jwinchel/Syndri/code/rctd_annotation.py "${PYTHON_ARGS[@]}"

# Example submit:
# sbatch --array=0-9 --export=ALL,REFERENCE_FILE=/path/to/ref.h5ad,SPATIAL_DIR=/home/jwinchel/Syndri/data,INPUT_GLOB='*rctd*.h5ad',OUTPUT_DIR=/home/jwinchel/Syndri/data/rctd_out,LAYER_PATTERN='counts?$|raw|X',MODE=doublet /home/jwinchel/Syndri/scripts/slurm/rctd_annotation.sbatch


# 07/26/2026: xenium noise experiment
# sbatch --array=0-76%4 --export=ALL,REFERENCE_FILE=/compute/oven-0-13/jwinchel/scRNA-seq/adata/chromium_GSM9509174_breast_integrated.h5ad,SPATIAL_DIR=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment,INPUT_GLOB='GSM95*pct.h5ad',OUTPUT_DIR=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/rctd_annotations/,LAYER_PATTERN='^(X|trial_(0|1|2))$',MODE=doublet /home/jwinchel/Syndri/scripts/slurm/rctd_annotation.sbatch

# sbatch --array=0-7%2 \
  # --export=ALL,REFERENCE_FILE=/compute/oven-0-13/jwinchel/scRNA-seq/adata/GSE127465_tumor_integrated.h5ad,SPATIAL_DIR=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/,INPUT_GLOB='cosmx_nsclc*1pct.h5ad',OUTPUT_DIR=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/,LAYER_PATTERN='^X$',CELL_TYPE_COL=cell_type,MODE=doublet \
  # /home/jwinchel/Syndri/scripts/slurm/rctd_annotation.sbatch

# 09/10/2026: WHOLE MOUSE BRAIN MERFISH
# sbatch --job-name=rctd-brain --array=0-2%2 \
#   --export=ALL,REFERENCE_FILE=/work/magroup/skrieger/Syndri/data/datasets/brain_merfish/interim/single-cell/10Xv3_subsampled.h5ad,SPATIAL_DIR=/work/magroup/skrieger/Syndri/data/datasets/brain_merfish/interim/spatial,INPUT_GLOB='st_integrated_slice-Zhuang-ABCA-4.00[0-9].h5ad',INPUT_LIST_FILE=,INPUT_FILES=,OUTPUT_DIR=/work/magroup/skrieger/Syndri/data/datasets/brain_merfish/results/spatial,LAYER_PATTERN='^X$',CELL_TYPE_COL=class,MODE=doublet \
#   /home/jwinchel/Syndri/scripts/slurm/rctd_annotation.sbatch


# 09/10/2026: AGING MOUSE BRAIN MERFISH
# sbatch --array=0-2%2 \
  # --export=ALL,REFERENCE_FILE=/work/magroup/skrieger/Syndri/data/datasets/agingbrain_merfish/interim/single-cell/10Xv3_subsampled.h5ad,SPATIAL_DIR=/work/magroup/skrieger/Syndri/data/datasets/brain_merfish/interim,INPUT_GLOB='',OUTPUT_DIR=/work/magroup/skrieger/Syndri/data/datasets/brain_merfish/results/spatial,LAYER_PATTERN='^X$',CELL_TYPE_COL=class,MODE=doublet \
  # /home/jwinchel/Syndri/scripts/slurm/rctd_annotation.sbatch

# 9/11/2026: NSCLC COSMX
# sbatch --job-name=rctd-cosmx --array=0-7%2 \
#   --export=ALL,REFERENCE_FILE=/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/interim/single-cell/sc_integrated.h5ad,SPATIAL_DIR=/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/interim/spatial,INPUT_GLOB='*annotated.h5ad',INPUT_LIST_FILE=,INPUT_FILES=,OUTPUT_DIR=,LAYER_PATTERN='^X$',CELL_TYPE_COL=cell_type,MODE=doublet,FORCE=0 \
#   /home/jwinchel/Syndri/scripts/slurm/rctd_annotation.sbatch
