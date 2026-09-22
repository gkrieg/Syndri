#!/bin/bash
#SBATCH -p cpu,gpu-small,gpu-large
#SBATCH --mem=24Gb
#SBATCH --job-name syndri-metrics
#SBATCH --output=/home/jwinchel/Syndri/logs/syndri-metrics-%A_%a.txt
#SBATCH -t 1-23:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-18%2

set -euo pipefail

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-heimdall}"

SPATIAL_DIR=${SPATIAL_DIR:-/compute/oven-0-13/jwinchel/spatial_adata/heimdall_inference}
INPUT_GLOB=${INPUT_GLOB:-GSM*rctd_annotated_val_med_with_admix_probs.h5ad}
INPUT_LIST_FILE=${INPUT_LIST_FILE:-}
INPUT_FILES=${INPUT_FILES:-}
OUTPUT_DIR=${OUTPUT_DIR:-/compute/oven-0-13/jwinchel/spatial_adata/metrics_eval}
OUTPUT_SUBDIR_MODE=${OUTPUT_SUBDIR_MODE:-sample}

SAMPLE_FRAC=${SAMPLE_FRAC:-1.0}
LABEL_KEY=${LABEL_KEY:-harmony_cell_type}
EXCLUDE_LABELS=${EXCLUDE_LABELS:-malignant cell}
TCELL_CELL_TYPE_KEY=${TCELL_CELL_TYPE_KEY:-harmony_cell_type}
TCELL_SAMPLE_KEY=${TCELL_SAMPLE_KEY:-batch}
COORD_X=${COORD_X:-x_centroid}
COORD_Y=${COORD_Y:-y_centroid}

ADMIX_PROBS_LAYER=${ADMIX_PROBS_LAYER:-admix_probs}
TARGET_REMOVED_FRACTIONS=${TARGET_REMOVED_FRACTIONS:-0.01,0.02,0.05,0.1,0.2}
CLEAN_PROB_THRESHOLD=${CLEAN_PROB_THRESHOLD:-0.5}
CLEANING_MODE=${CLEANING_MODE:-zero}
CELL_TYPE_SPECIFIC_REMOVAL=${CELL_TYPE_SPECIFIC_REMOVAL:-0}
REMOVAL_GROUP_KEY=${REMOVAL_GROUP_KEY:-}
PROBABILITY_TRANSFORM=${PROBABILITY_TRANSFORM:-power}
PROBABILITY_POWER=${PROBABILITY_POWER:-2.0}
PROBABILITY_SIGMOID_MIDPOINT=${PROBABILITY_SIGMOID_MIDPOINT:-0.5}
PROBABILITY_SIGMOID_STEEPNESS=${PROBABILITY_SIGMOID_STEEPNESS:-10.0}
TCELL_PROBABILITY_REMOVAL_SCALE=${TCELL_PROBABILITY_REMOVAL_SCALE:-1.0}
ADDITIONAL_COUNTS_LAYER=${ADDITIONAL_COUNTS_LAYER:-split_purified_counts}
ADDITIONAL_COUNTS_CASE_LABEL=${ADDITIONAL_COUNTS_CASE_LABEL:-SPLIT}

LEIDEN_RESOLUTION=${LEIDEN_RESOLUTION:-1.0}
LEIDEN_OPTIMIZE_RESOLUTION=${LEIDEN_OPTIMIZE_RESOLUTION:-false}
LEIDEN_N_NEIGHBORS=${LEIDEN_N_NEIGHBORS:-15}
N_PCS=${N_PCS:-50}
LIBRARY_SIZE_TARGET_SUM=${LIBRARY_SIZE_TARGET_SUM:-10000}
RANDOM_STATE=${RANDOM_STATE:-0}

TCELL_RADIUS_UM=${TCELL_RADIUS_UM:-15.0}
TCELL_TOP_N=${TCELL_TOP_N:-20}
TCELL_MIN_CELLS_PER_LABEL=${TCELL_MIN_CELLS_PER_LABEL:-100}
TCELL_DOWNSAMPLE_IF_N_CELLS=${TCELL_DOWNSAMPLE_IF_N_CELLS:-50000}
TCELL_MAX_CELLS_PER_LABEL=${TCELL_MAX_CELLS_PER_LABEL:-25000}
TCELL_EXACT_ZSCORE=${TCELL_EXACT_ZSCORE:-auto}
TCELL_DENSE_SIZE_LIMIT_ENTRIES=${TCELL_DENSE_SIZE_LIMIT_ENTRIES:-100000000}
TCELL_LR_GENE_OUTPUT=${TCELL_LR_GENE_OUTPUT:-tcell_lr_top_genes.csv}
TCELL_COSINE_OUTPUT=${TCELL_COSINE_OUTPUT:-tcell_cosine_similarity.csv}
CELL_TYPE_PROFILE_OUTPUT=${CELL_TYPE_PROFILE_OUTPUT:-tissue_cell_type_profiles.csv}

REFERENCE_DATA=${REFERENCE_DATA:-}
REFERENCE_CELL_TYPE_KEY=${REFERENCE_CELL_TYPE_KEY:-cell_type}
REFERENCE_COUNTS_LAYER=${REFERENCE_COUNTS_LAYER:-}
TCELL_COSINE_CELL_TYPE_PATTERN=${TCELL_COSINE_CELL_TYPE_PATTERN:-(?i)(^|[^A-Za-z])(t cells?|cd4|cd8|treg|t[- ]?regulatory|regulatory t|cytotoxic t)([^A-Za-z]|$)}
TCELL_COSINE_CELL_TYPES=${TCELL_COSINE_CELL_TYPES:-}
TCELL_COSINE_CELL_TYPE_MAP=${TCELL_COSINE_CELL_TYPE_MAP:-}
TCELL_COSINE_MIN_SPATIAL_CELLS=${TCELL_COSINE_MIN_SPATIAL_CELLS:-1}
TCELL_COSINE_MIN_REFERENCE_CELLS=${TCELL_COSINE_MIN_REFERENCE_CELLS:-1}

SKIP_CELL_TYPE_SEPARATION=${SKIP_CELL_TYPE_SEPARATION:-0}
SKIP_TCELL_CONTAMINATION=${SKIP_TCELL_CONTAMINATION:-0}
SKIP_TCELL_NEIGHBORHOOD_METRICS=${SKIP_TCELL_NEIGHBORHOOD_METRICS:-0}
SKIP_TCELL_COSINE_SIMILARITY=${SKIP_TCELL_COSINE_SIMILARITY:-0}
SKIP_CELL_TYPE_PROFILES=${SKIP_CELL_TYPE_PROFILES:-0}
ADDITIONAL_COUNTS_ZERO_HIGH_PROB=${ADDITIONAL_COUNTS_ZERO_HIGH_PROB:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}
source /home/jwinchel/Syndri/scripts/slurm/syndri_reference_cosine_args.sh

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
for input_file in "${ALL_INPUT_FILES[@]}"; do
  if [[ -n "$input_file" ]]; then
    FILTERED_INPUT_FILES+=("$input_file")
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
  echo "Submit with --array=0-$(( ${#ALL_INPUT_FILES[@]} - 1 ))%N, or edit the #SBATCH --array line." >&2
  exit 1
fi

TASK_INPUT_FILE=${ALL_INPUT_FILES[$TASK_ID]}
SAMPLE_NAME=$(basename "$TASK_INPUT_FILE")
SAMPLE_NAME=${SAMPLE_NAME%.h5ad}
SAMPLE_NAME=${SAMPLE_NAME%.h5}
SAMPLE_NAME=${SAMPLE_NAME//[^A-Za-z0-9_.-]/_}

TASK_OUTPUT_DIR="$OUTPUT_DIR"
if [[ "$OUTPUT_SUBDIR_MODE" == "sample" ]]; then
  TASK_OUTPUT_DIR="$OUTPUT_DIR/$SAMPLE_NAME"
elif [[ "$OUTPUT_SUBDIR_MODE" == "array_task" ]]; then
  TASK_OUTPUT_DIR="$OUTPUT_DIR/task_${TASK_ID}"
elif [[ "$OUTPUT_SUBDIR_MODE" != "none" ]]; then
  echo "Invalid OUTPUT_SUBDIR_MODE=$OUTPUT_SUBDIR_MODE. Use sample, array_task, or none." >&2
  exit 1
fi

INPUT_ARGS=(--input-files "$TASK_INPUT_FILE")

OPTIONAL_ARGS=()
if [[ "$SKIP_CELL_TYPE_SEPARATION" == "1" || "$SKIP_CELL_TYPE_SEPARATION" == "true" || "$SKIP_CELL_TYPE_SEPARATION" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--skip-cell-type-separation)
fi
if [[ "$SKIP_TCELL_CONTAMINATION" == "1" || "$SKIP_TCELL_CONTAMINATION" == "true" || "$SKIP_TCELL_CONTAMINATION" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--skip-tcell-contamination)
fi
if [[ "$SKIP_TCELL_NEIGHBORHOOD_METRICS" == "1" || "$SKIP_TCELL_NEIGHBORHOOD_METRICS" == "true" || "$SKIP_TCELL_NEIGHBORHOOD_METRICS" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--skip-tcell-neighborhood-metrics)
fi
if [[ "$SKIP_TCELL_COSINE_SIMILARITY" == "1" || "$SKIP_TCELL_COSINE_SIMILARITY" == "true" || "$SKIP_TCELL_COSINE_SIMILARITY" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--skip-tcell-cosine-similarity)
fi
if [[ "$SKIP_CELL_TYPE_PROFILES" == "1" || "$SKIP_CELL_TYPE_PROFILES" == "true" || "$SKIP_CELL_TYPE_PROFILES" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--skip-cell-type-profiles)
fi
if [[ "$ADDITIONAL_COUNTS_ZERO_HIGH_PROB" == "1" || "$ADDITIONAL_COUNTS_ZERO_HIGH_PROB" == "true" || "$ADDITIONAL_COUNTS_ZERO_HIGH_PROB" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--additional-counts-zero-high-prob)
fi
if [[ "$CELL_TYPE_SPECIFIC_REMOVAL" == "1" || "$CELL_TYPE_SPECIFIC_REMOVAL" == "true" || "$CELL_TYPE_SPECIFIC_REMOVAL" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--cell-type-specific-removal)
fi
if [[ -n "$REMOVAL_GROUP_KEY" ]]; then
  OPTIONAL_ARGS+=(--removal-group-key "$REMOVAL_GROUP_KEY")
fi

echo "Running metrics_eval.py"
echo "  array task: $TASK_ID / $(( ${#ALL_INPUT_FILES[@]} - 1 ))"
echo "  input file: $TASK_INPUT_FILE"
echo "  output dir: $TASK_OUTPUT_DIR"
echo "  sample frac: $SAMPLE_FRAC"
echo "  cleaning mode: $CLEANING_MODE"
echo "  cell-type-specific removal: $CELL_TYPE_SPECIFIC_REMOVAL"
echo "  removal group key: ${REMOVAL_GROUP_KEY:-$LABEL_KEY}"
echo "  probability transform: $PROBABILITY_TRANSFORM"
echo "  additional counts layer: ${ADDITIONAL_COUNTS_LAYER:-<disabled>}"
echo "  tcell cell type key: $TCELL_CELL_TYPE_KEY"
echo "  tcell LR gene output: $TCELL_LR_GENE_OUTPUT"
echo "  tcell cosine output: $TCELL_COSINE_OUTPUT"
echo "  reference data: ${REFERENCE_DATA:-<unset>}"
echo "  skip tcell neighborhood metrics: $SKIP_TCELL_NEIGHBORHOOD_METRICS"
echo "  cell type profile output: $CELL_TYPE_PROFILE_OUTPUT"
echo "  cpus: ${SLURM_CPUS_PER_TASK:-1}"

# EXTRA_ARGS is intentionally word-split so simple additional flags can be supplied via sbatch --export.
# shellcheck disable=SC2086
python /home/jwinchel/Syndri/eval/metrics_eval.py \
  "${INPUT_ARGS[@]}" \
  --output-dir "$TASK_OUTPUT_DIR" \
  --sample-frac "$SAMPLE_FRAC" \
  --label-key "$LABEL_KEY" \
  --exclude-labels "$EXCLUDE_LABELS" \
  --tcell-cell-type-key "$TCELL_CELL_TYPE_KEY" \
  --tcell-sample-key "$TCELL_SAMPLE_KEY" \
  --coord-x "$COORD_X" \
  --coord-y "$COORD_Y" \
  --admix-probs-layer "$ADMIX_PROBS_LAYER" \
  --target-removed-fractions "$TARGET_REMOVED_FRACTIONS" \
  --clean-prob-threshold "$CLEAN_PROB_THRESHOLD" \
  --cleaning-mode "$CLEANING_MODE" \
  --probability-transform "$PROBABILITY_TRANSFORM" \
  --probability-power "$PROBABILITY_POWER" \
  --probability-sigmoid-midpoint "$PROBABILITY_SIGMOID_MIDPOINT" \
  --probability-sigmoid-steepness "$PROBABILITY_SIGMOID_STEEPNESS" \
  --tcell-probability-removal-scale "$TCELL_PROBABILITY_REMOVAL_SCALE" \
  --additional-counts-layer "$ADDITIONAL_COUNTS_LAYER" \
  --additional-counts-case-label "$ADDITIONAL_COUNTS_CASE_LABEL" \
  --leiden-resolution "$LEIDEN_RESOLUTION" \
  --leiden-optimize-resolution "$LEIDEN_OPTIMIZE_RESOLUTION" \
  --leiden-n-neighbors "$LEIDEN_N_NEIGHBORS" \
  --n-pcs "$N_PCS" \
  --library-size-target-sum "$LIBRARY_SIZE_TARGET_SUM" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
  --random-state "$RANDOM_STATE" \
  --tcell-radius-um "$TCELL_RADIUS_UM" \
  --tcell-top-n "$TCELL_TOP_N" \
  --tcell-lr-gene-output "$TCELL_LR_GENE_OUTPUT" \
  --tcell-cosine-output "$TCELL_COSINE_OUTPUT" \
  --reference-data "$REFERENCE_DATA" \
  --reference-cell-type-key "$REFERENCE_CELL_TYPE_KEY" \
  --reference-counts-layer "$REFERENCE_COUNTS_LAYER" \
  --tcell-cosine-cell-type-pattern "$TCELL_COSINE_CELL_TYPE_PATTERN" \
  --tcell-cosine-cell-types "$TCELL_COSINE_CELL_TYPES" \
  --tcell-cosine-cell-type-map "$TCELL_COSINE_CELL_TYPE_MAP" \
  --tcell-cosine-min-spatial-cells "$TCELL_COSINE_MIN_SPATIAL_CELLS" \
  --tcell-cosine-min-reference-cells "$TCELL_COSINE_MIN_REFERENCE_CELLS" \
  --cell-type-profile-output "$CELL_TYPE_PROFILE_OUTPUT" \
  --tcell-min-cells-per-label "$TCELL_MIN_CELLS_PER_LABEL" \
  --tcell-downsample-if-n-cells "$TCELL_DOWNSAMPLE_IF_N_CELLS" \
  --tcell-max-cells-per-label "$TCELL_MAX_CELLS_PER_LABEL" \
  --tcell-exact-zscore "$TCELL_EXACT_ZSCORE" \
  --tcell-dense-size-limit-entries "$TCELL_DENSE_SIZE_LIMIT_ENTRIES" \
  "${OPTIONAL_ARGS[@]}" \
  $EXTRA_ARGS \
  "${SYNDRI_COSINE_ARGS[@]}"

# This script is array-oriented. Each task selects one file by SLURM_ARRAY_TASK_ID
# and writes to OUTPUT_DIR/<sample_name>/ by default.
#
# Examples:
# sbatch /home/jwinchel/slurmfiles/metrics_eval.sbatch
# sbatch --array=0-18%4 --export=ALL,OUTPUT_DIR=/compute/oven-0-13/jwinchel/spatial_adata/metrics_eval_full /home/jwinchel/slurmfiles/metrics_eval.sbatch
# sbatch --array=0-7%2 --export=ALL,INPUT_LIST_FILE=/path/to/files.txt,OUTPUT_DIR=/path/to/out /home/jwinchel/slurmfiles/metrics_eval.sbatch
#
# To get the correct array range for the default glob:
# python - <<'PY'
# from pathlib import Path
# files = sorted(Path('/compute/oven-0-13/jwinchel/spatial_adata/heimdall_inference').glob('GSM*rctd_annotated_val_med_with_admix_probs.h5ad'))
# print(f'--array=0-{len(files)-1}%2')
# PY
