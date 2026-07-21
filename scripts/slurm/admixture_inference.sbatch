#!/bin/bash
#SBATCH -p gpu-small
#SBATCH --mem=64Gb
#SBATCH --job-name syndri-infer
#SBATCH --output Syndri/logs/syndri-infer-%A_%a.txt
#SBATCH -t 0-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module load cuda-12.1

cd ~/Syndri/Heimdall
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall

REFERENCE_DATA=${REFERENCE_DATA:-/compute/oven-0-13/jwinchel/scRNA-seq/adata/GSE127465_tumor_integrated_10pct.h5ad}
NOTEBOOK=${NOTEBOOK:-/home/jwinchel/Syndri/seg_cleaner/InferenceTest.ipynb}
OUTPUT_DIR=${OUTPUT_DIR:-/compute/oven-0-13/jwinchel/spatial_adata/heimdall_inference}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/nsclc_augmented_hardcap}
CONFIG_SOURCE=${CONFIG_SOURCE:-checkpoint}
MILESTONE=${MILESTONE:-999}
BATCH_SIZE=${BATCH_SIZE:-256}
SPLIT_VALUE=${SPLIT_VALUE:-val}
SPLIT_COL=${SPLIT_COL:-split}
PRESERVE_SPLIT_COLUMN=${PRESERVE_SPLIT_COLUMN:-0}
COORD_X=${COORD_X:-CenterX_global_px}
COORD_Y=${COORD_Y:-CenterY_global_px}
COORD_COLS=${COORD_COLS:-${COORD_X},${COORD_Y}}
TISSUE_COL=${TISSUE_COL:-}
APPLY_ADMIXTURE=${APPLY_ADMIXTURE:-0}
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

# Input selection:
# - ST_DATA: one file, current single-sample behavior.
# - ST_DATA_LIST: comma-separated list of files, selected by SLURM_ARRAY_TASK_ID.
# - ST_DATA_LIST_FILE: newline-delimited file list, selected by SLURM_ARRAY_TASK_ID.
if [[ -n "${ST_DATA_LIST_FILE:-}" ]]; then
  mapfile -t ST_FILES < "$ST_DATA_LIST_FILE"
elif [[ -n "${ST_DATA_LIST:-}" ]]; then
  IFS=',' read -r -a ST_FILES <<< "$ST_DATA_LIST"
else
  ST_FILES=("${ST_DATA:-/compute/oven-0-13/jwinchel/spatial_adata/cosmx_nsclc_Lung5-Rep1_adata_annotated.h5ad}")
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#ST_FILES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is out of range for ${#ST_FILES[@]} ST file(s)." >&2
  exit 1
fi

ST_DATA=${ST_FILES[$TASK_ID]}
SAMPLE_NAME=$(basename "$ST_DATA")
SAMPLE_NAME=${SAMPLE_NAME%.h5ad}
SAMPLE_NAME=${SAMPLE_NAME%.h5}
PREPARED_NAME=${PREPARED_NAME:-${SAMPLE_NAME}_prepared.h5ad}
OUTPUT_NAME=${OUTPUT_NAME:-${SAMPLE_NAME}_${SPLIT_VALUE}_med_with_admix_probs.h5ad}

PRESERVE_SPLIT_ARG=--no-preserve-split-column
if [[ "$PRESERVE_SPLIT_COLUMN" == "1" || "$PRESERVE_SPLIT_COLUMN" == "true" || "$PRESERVE_SPLIT_COLUMN" == "TRUE" ]]; then
  PRESERVE_SPLIT_ARG=--preserve-split-column
fi

APPLY_ADMIXTURE_ARG=--no-apply-admixture
if [[ "$APPLY_ADMIXTURE" == "1" || "$APPLY_ADMIXTURE" == "true" || "$APPLY_ADMIXTURE" == "TRUE" ]]; then
  APPLY_ADMIXTURE_ARG=--apply-admixture
fi

TISSUE_COL_ARGS=()
if [[ -n "$TISSUE_COL" ]]; then
  TISSUE_COL_ARGS=(--tissue-col "$TISSUE_COL")
fi

echo "Running Heimdall admixture inference"
echo "  task id: $TASK_ID"
echo "  st data: $ST_DATA"
echo "  config source: $CONFIG_SOURCE"
echo "  checkpoint dir: $CHECKPOINT_DIR"
echo "  split column: $SPLIT_COL"
echo "  split value: $SPLIT_VALUE"
echo "  preserve split column: $PRESERVE_SPLIT_COLUMN"
echo "  coord cols: $COORD_COLS"
echo "  tissue col: ${TISSUE_COL:-<default>}"
echo "  apply admixture: $APPLY_ADMIXTURE"
echo "  extra overrides: ${EXTRA_OVERRIDES:-<none>}"
echo "  prepared name: $PREPARED_NAME"
echo "  output name: $OUTPUT_NAME"

python scripts/run_admixture_inference.py \
  --st-data "$ST_DATA" \
  --reference-data "$REFERENCE_DATA" \
  --notebook "$NOTEBOOK" \
  --output-dir "$OUTPUT_DIR" \
  --prepared-name "$PREPARED_NAME" \
  --output-name "$OUTPUT_NAME" \
  --config-source "$CONFIG_SOURCE" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --milestone "$MILESTONE" \
  --batch-size "$BATCH_SIZE" \
  --split-col "$SPLIT_COL" \
  --split-value "$SPLIT_VALUE" \
  --coord-cols "$COORD_COLS" \
  --extra-overrides "$EXTRA_OVERRIDES" \
  "$PRESERVE_SPLIT_ARG" \
  "$APPLY_ADMIXTURE_ARG" \
  "${TISSUE_COL_ARGS[@]}"

# Example:
# sbatch --export=ALL,MILESTONE=5,CHECKPOINT_DIR=/home/jwinchel/Heimdall/admixture_prediction_results/posweight_4p0 /home/jwinchel/slurmfiles/admixture_inference.sbatch
# sbatch --export=ALL,MILESTONE=99,CHECKPOINT_DIR=/home/jwinchel/Heimdall/admixture_prediction_results/posweight_4p0 /home/jwinchel/slurmfiles/admixture_inference.sbatch
# Comma-separated list. Set --array to 0 through N-1:
# sbatch --array=0-2 --export=ALL,ST_DATA_LIST=/path/a.h5ad,/path/b.h5ad,/path/c.h5ad,MILESTONE=99 /home/jwinchel/slurmfiles/admixture_inference.sbatch
# Newline-delimited list file:
# sbatch --array=0-7 --export=ALL,ST_DATA_LIST_FILE=/compute/oven-0-13/jwinchel/spatial_adata/inf_files.txt,MILESTONE=99 /home/jwinchel/slurmfiles/admixture_inference.sbatch
# sbatch --array=0-7 --export=ALL,ST_DATA_LIST_FILE=/compute/oven-0-13/jwinchel/spatial_adata/inf_files.txt,MILESTONE=49,CHECKPOINT_DIR=/home/jwinchel/Heimdall/admixture_prediction_results/identity_transformer_syndri_medium_posweight_1p0 /home/jwinchel/slurmfiles/admixture_inference.sbatch
# Original augmented training/validation data. Preserves obs[SPLIT_COL] and writes only matching rows:
# sbatch --export=ALL,ST_DATA=/compute/oven-0-13/jwinchel/scRNA-seq/adata/GSE127465_tumor_integrated.h5ad,SPLIT_VALUE=val,PRESERVE_SPLIT_COLUMN=1,COORD_X=sampled_st_x,COORD_Y=sampled_st_y,APPLY_ADMIXTURE=1,MILESTONE=49 /home/jwinchel/slurmfiles/admixture_inference.sbatch
# sbatch --export=ALL,ST_DATA=/compute/oven-0-13/jwinchel/scRNA-seq/adata/GSE127465_tumor_integrated.h5ad,SPLIT_VALUE=train,PRESERVE_SPLIT_COLUMN=1,COORD_X=sampled_st_x,COORD_Y=sampled_st_y,APPLY_ADMIXTURE=1,MILESTONE=49 /home/jwinchel/slurmfiles/admixture_inference.sbatch

# 06/03/2026
# sbatch --export=ALL,ST_DATA=/compute/oven-0-13/jwinchel/scRNA-seq/adata/GSE127465_tumor_integrated_hardcap_100th.h5ad,SPLIT_VALUE=train,PRESERVE_SPLIT_COLUMN=1,COORD_X=sampled_st_x,COORD_Y=sampled_st_y,APPLY_ADMIXTURE=1,MILESTONE=49 /home/jwinchel/slurmfiles/admixture_inference.sbatch
# sbatch --export=ALL,ST_DATA=/compute/oven-0-13/jwinchel/scRNA-seq/adata/GSE127465_tumor_integrated_hardcap_100th.h5ad,SPLIT_VALUE=val,PRESERVE_SPLIT_COLUMN=1,COORD_X=sampled_st_x,COORD_Y=sampled_st_y,APPLY_ADMIXTURE=1,MILESTONE=49 /home/jwinchel/slurmfiles/admixture_inference.sbatch


# 06/05/2026
# sbatch --export=ALL,ST_DATA=/compute/oven-0-13/jwinchel/scRNA-seq/adata/GSE127465_tumor_integrated_hardcap.h5ad,SPLIT_VALUE=val,PRESERVE_SPLIT_COLUMN=1,COORD_X=sampled_st_x,COORD_Y=sampled_st_y,APPLY_ADMIXTURE=1,MILESTONE=23 /home/jwinchel/slurmfiles/admixture_inference.sbatch
# sbatch --array=0-7%2 --export=ALL,ST_DATA_LIST_FILE=/compute/oven-0-13/jwinchel/spatial_adata/inf_files.txt,MILESTONE=23,CHECKPOINT_DIR=/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/nsclc_augmented_hardcap/ /home/jwinchel/slurmfiles/admixture_inference.sbatch
