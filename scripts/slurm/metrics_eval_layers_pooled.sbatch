#!/bin/bash
#SBATCH -p cpu,gpu-small,gpu-large
#SBATCH --mem=96Gb
#SBATCH --job-name syndri-xenium-noisy-metrics
#SBATCH --output=/home/jwinchel/Syndri/logs/syndri-xenium-noisy-metrics-%A_%a.txt
#SBATCH --exclude=oven-0-27
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-heimdall}"

SPATIAL_DIR=${SPATIAL_DIR:-/work/magroup/skrieger/Syndri/data/xenium_noise_experiment}
INPUT_GLOB=${INPUT_GLOB:-GSM950*.h5ad}
OUTPUT_DIR=${OUTPUT_DIR:-/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/metrics_eval_layers_pooled}
SAMPLE_REPLICATE_REGEX=${SAMPLE_REPLICATE_REGEX:-(B[0-9]+_B[0-9]+(?:_[0-9]+)?)}
CELL_TYPE_OUTPUT=${CELL_TYPE_OUTPUT:-cell_type_separation_metrics_layers_pooled.csv}
TCELL_OUTPUT=${TCELL_OUTPUT:-tcell_contamination_metrics_layers_pooled.csv}
TCELL_EFFECT_OUTPUT=${TCELL_EFFECT_OUTPUT:-tcell_contamination_effects_layers_pooled.csv}
TCELL_LR_GENE_OUTPUT=${TCELL_LR_GENE_OUTPUT:-tcell_lr_top_genes_layers_pooled.csv}
TCELL_COSINE_OUTPUT=${TCELL_COSINE_OUTPUT:-tcell_cosine_similarity_layers_pooled.csv}
CELL_TYPE_PROFILE_OUTPUT=${CELL_TYPE_PROFILE_OUTPUT:-tissue_cell_type_profiles_layers_pooled.csv}
CONFIG_OUTPUT=${CONFIG_OUTPUT:-metrics_eval_config_layers_pooled.json}
SKIP_EXISTING=${SKIP_EXISTING:-1}
REQUIRE_ALL_OUTPUTS_FOR_SKIP=${REQUIRE_ALL_OUTPUTS_FOR_SKIP:-1}
LAYERED_BASELINE_ONLY=${LAYERED_BASELINE_ONLY:-0}
APPEND_OUTPUT=${APPEND_OUTPUT:-0}
REPLACE_LAYER_COUNT_BASELINES=${REPLACE_LAYER_COUNT_BASELINES:-0}
DEDUPLICATE_ZERO_DROPOUT_INPUTS=${DEDUPLICATE_ZERO_DROPOUT_INPUTS:-1}

# Layer selection. Use INPUT_LAYERS_SEP=: when passing explicit layers through sbatch --export.
INPUT_LAYERS=${INPUT_LAYERS:-}
INPUT_LAYERS_SEP=${INPUT_LAYERS_SEP:-,}
N_LAYERS=${N_LAYERS:-3}
LAYER_START=${LAYER_START:-0}
INPUT_LAYER_PREFIX=${INPUT_LAYER_PREFIX:-trial}
INPUT_LAYER_TEMPLATE=${INPUT_LAYER_TEMPLATE:-${INPUT_LAYER_PREFIX}_%d}

LABEL_KEY=${LABEL_KEY:-harmony_cell_type}
TCELL_CELL_TYPE_KEY=${TCELL_CELL_TYPE_KEY:-$LABEL_KEY}
TCELL_SAMPLE_KEY=${TCELL_SAMPLE_KEY:-batch}
COORD_X=${COORD_X:-x_centroid}
COORD_Y=${COORD_Y:-y_centroid}
EXCLUDE_LABELS=${EXCLUDE_LABELS:-malignant cell}
FILTER_TCELL_INVALID_CELLS=${FILTER_TCELL_INVALID_CELLS:-${TCELL_FILTER_INVALID_CELLS:-0}}
TCELL_EXCLUDE_LABELS=${TCELL_EXCLUDE_LABELS:-$EXCLUDE_LABELS}

REFERENCE_DATA=${REFERENCE_DATA:-/compute/oven-0-13/jwinchel/scRNA-seq/adata/chromium_GSM9509174_breast_integrated.h5ad}
REFERENCE_CELL_TYPE_KEY=${REFERENCE_CELL_TYPE_KEY:-cell_type}
REFERENCE_COUNTS_LAYER=${REFERENCE_COUNTS_LAYER:-}
TCELL_COSINE_CELL_TYPE_PATTERN=${TCELL_COSINE_CELL_TYPE_PATTERN:-(?i)(^|[^A-Za-z])(t cells?|cd4|cd8|treg|t[- ]?regulatory|regulatory t|cytotoxic t)([^A-Za-z]|$)}
TCELL_COSINE_CELL_TYPES=${TCELL_COSINE_CELL_TYPES:-}
TCELL_COSINE_CELL_TYPE_MAP=${TCELL_COSINE_CELL_TYPE_MAP:-}
TCELL_COSINE_MIN_SPATIAL_CELLS=${TCELL_COSINE_MIN_SPATIAL_CELLS:-1}
TCELL_COSINE_MIN_REFERENCE_CELLS=${TCELL_COSINE_MIN_REFERENCE_CELLS:-1}

SYNDRI_PROBS_TEMPLATE=${SYNDRI_PROBS_TEMPLATE:-}
MLP_PROBS_TEMPLATE=${MLP_PROBS_TEMPLATE:-}
SPLIT_COUNTS_TEMPLATE=${SPLIT_COUNTS_TEMPLATE:-}
if [[ -z "$SYNDRI_PROBS_TEMPLATE" ]]; then
  SYNDRI_PROBS_TEMPLATE='admix_probs_{layer}'
fi
if [[ -z "$MLP_PROBS_TEMPLATE" ]]; then
  MLP_PROBS_TEMPLATE='mlp_admix_probs_{layer}'
fi
if [[ -z "$SPLIT_COUNTS_TEMPLATE" ]]; then
  SPLIT_COUNTS_TEMPLATE='split_purified_counts_{layer}'
fi
SYNDRI_CASE_LABEL=${SYNDRI_CASE_LABEL:-syndri}
MLP_CASE_LABEL=${MLP_CASE_LABEL:-mlp}
SPLIT_CASE_LABEL=${SPLIT_CASE_LABEL:-split}

# Fixed configuration from syndri_metric_eval_grid.sh: nonlinear power transform + cell-type-specific removal.
TARGET_REMOVED_FRACTIONS=${TARGET_REMOVED_FRACTIONS:-0.01,0.02,0.05,0.1,0.2}
CLEANING_MODE=${CLEANING_MODE:-probability_scale}
CELL_TYPE_SPECIFIC_REMOVAL=${CELL_TYPE_SPECIFIC_REMOVAL:-1}
REMOVAL_GROUP_KEY=${REMOVAL_GROUP_KEY:-$LABEL_KEY}
PROBABILITY_TRANSFORM=${PROBABILITY_TRANSFORM:-power}
PROBABILITY_POWER=${PROBABILITY_POWER:-2.0}
TCELL_PROBABILITY_REMOVAL_SCALE=${TCELL_PROBABILITY_REMOVAL_SCALE:-1.0}

SAMPLE_FRAC=${SAMPLE_FRAC:-1.0}
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

SKIP_CELL_TYPE_SEPARATION=${SKIP_CELL_TYPE_SEPARATION:-0}
SKIP_TCELL_CONTAMINATION=${SKIP_TCELL_CONTAMINATION:-0}
SKIP_TCELL_NEIGHBORHOOD_METRICS=${SKIP_TCELL_NEIGHBORHOOD_METRICS:-0}
SKIP_TCELL_COSINE_SIMILARITY=${SKIP_TCELL_COSINE_SIMILARITY:-0}
SKIP_CELL_TYPE_PROFILES=${SKIP_CELL_TYPE_PROFILES:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [[ -n "$INPUT_LAYERS" ]]; then
  LAYER_CSV=${INPUT_LAYERS//${INPUT_LAYERS_SEP}/,}
else
  INPUT_LAYER_NAMES=()
  for ((i = 0; i < N_LAYERS; i++)); do
    layer_idx=$((LAYER_START + i))
    printf -v layer_name "$INPUT_LAYER_TEMPLATE" "$layer_idx"
    INPUT_LAYER_NAMES+=("$layer_name")
  done
  LAYER_CSV=$(IFS=,; echo "${INPUT_LAYER_NAMES[*]}")
fi

mapfile -t SAMPLE_REPLICATES < <(
  python - "$SPATIAL_DIR" "$INPUT_GLOB" "$SAMPLE_REPLICATE_REGEX" <<'PY'
import re
import sys
from pathlib import Path

spatial_dir = Path(sys.argv[1])
glob = sys.argv[2]
regex = re.compile(sys.argv[3])
values = []
for path in sorted(spatial_dir.glob(glob)):
    match = regex.search(path.name)
    if match:
        values.append(match.group(0))
for value in sorted(set(values)):
    print(value)
PY
)

if (( ${#SAMPLE_REPLICATES[@]} == 0 )); then
  echo "No sample replicates found." >&2
  echo "  SPATIAL_DIR=$SPATIAL_DIR" >&2
  echo "  INPUT_GLOB=$INPUT_GLOB" >&2
  echo "  SAMPLE_REPLICATE_REGEX=$SAMPLE_REPLICATE_REGEX" >&2
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#SAMPLE_REPLICATES[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=$TASK_ID is out of range for ${#SAMPLE_REPLICATES[@]} sample replicate(s)." >&2
  echo "Submit with --array=0-$(( ${#SAMPLE_REPLICATES[@]} - 1 ))%N." >&2
  exit 1
fi

SAMPLE_REPLICATE=${SAMPLE_REPLICATES[$TASK_ID]}
SAFE_SAMPLE_REPLICATE=${SAMPLE_REPLICATE//[^A-Za-z0-9_.-]/_}
TASK_OUTPUT_DIR="$OUTPUT_DIR/$SAFE_SAMPLE_REPLICATE"
mkdir -p "$TASK_OUTPUT_DIR" /home/jwinchel/Syndri/logs

EXPECTED_OUTPUTS=("$TASK_OUTPUT_DIR/$CONFIG_OUTPUT")
if [[ "$SKIP_CELL_TYPE_SEPARATION" != "1" && "$SKIP_CELL_TYPE_SEPARATION" != "true" && "$SKIP_CELL_TYPE_SEPARATION" != "TRUE" ]]; then
  EXPECTED_OUTPUTS+=("$TASK_OUTPUT_DIR/$CELL_TYPE_OUTPUT")
fi
if [[ "$SKIP_TCELL_CONTAMINATION" != "1" && "$SKIP_TCELL_CONTAMINATION" != "true" && "$SKIP_TCELL_CONTAMINATION" != "TRUE" ]]; then
  EXPECTED_OUTPUTS+=(
    "$TASK_OUTPUT_DIR/$TCELL_OUTPUT"
    "$TASK_OUTPUT_DIR/$TCELL_EFFECT_OUTPUT"
    "$TASK_OUTPUT_DIR/$TCELL_LR_GENE_OUTPUT"
  )
  if [[ "$SKIP_TCELL_COSINE_SIMILARITY" != "1" && "$SKIP_TCELL_COSINE_SIMILARITY" != "true" && "$SKIP_TCELL_COSINE_SIMILARITY" != "TRUE" && -n "$REFERENCE_DATA" ]]; then
    EXPECTED_OUTPUTS+=("$TASK_OUTPUT_DIR/$TCELL_COSINE_OUTPUT")
  fi
  if [[ "$SKIP_CELL_TYPE_PROFILES" != "1" && "$SKIP_CELL_TYPE_PROFILES" != "true" && "$SKIP_CELL_TYPE_PROFILES" != "TRUE" ]]; then
    EXPECTED_OUTPUTS+=("$TASK_OUTPUT_DIR/$CELL_TYPE_PROFILE_OUTPUT")
  fi
fi

if [[ "$SKIP_EXISTING" == "1" || "$SKIP_EXISTING" == "true" || "$SKIP_EXISTING" == "TRUE" ]]; then
  existing_count=0
  for output_file in "${EXPECTED_OUTPUTS[@]}"; do
    if [[ -s "$output_file" ]]; then
      existing_count=$((existing_count + 1))
    fi
  done

  if [[ "$REQUIRE_ALL_OUTPUTS_FOR_SKIP" == "1" || "$REQUIRE_ALL_OUTPUTS_FOR_SKIP" == "true" || "$REQUIRE_ALL_OUTPUTS_FOR_SKIP" == "TRUE" ]]; then
    if (( existing_count == ${#EXPECTED_OUTPUTS[@]} )); then
      echo "Skipping $SAMPLE_REPLICATE: all expected outputs already exist in $TASK_OUTPUT_DIR"
      exit 0
    fi
  elif (( existing_count > 0 )); then
    echo "Skipping $SAMPLE_REPLICATE: at least one expected output already exists in $TASK_OUTPUT_DIR"
    exit 0
  fi
fi

mapfile -t TASK_INPUT_FILES < <(
  python - "$SPATIAL_DIR" "$INPUT_GLOB" "$SAMPLE_REPLICATE_REGEX" "$SAMPLE_REPLICATE" <<'PY'
import re
import sys
from pathlib import Path

spatial_dir = Path(sys.argv[1])
glob = sys.argv[2]
regex = re.compile(sys.argv[3])
target = sys.argv[4]
for path in sorted(spatial_dir.glob(glob)):
    match = regex.search(path.name)
    if match and match.group(0) == target:
        print(path)
PY
)

if (( ${#TASK_INPUT_FILES[@]} == 0 )); then
  echo "No input files found for sample replicate $SAMPLE_REPLICATE." >&2
  exit 1
fi

INPUT_ARGS=(--input-files "${TASK_INPUT_FILES[@]}")

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
if [[ "$CELL_TYPE_SPECIFIC_REMOVAL" == "1" || "$CELL_TYPE_SPECIFIC_REMOVAL" == "true" || "$CELL_TYPE_SPECIFIC_REMOVAL" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--cell-type-specific-removal)
fi
if [[ "$LAYERED_BASELINE_ONLY" == "1" || "$LAYERED_BASELINE_ONLY" == "true" || "$LAYERED_BASELINE_ONLY" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--layered-baseline-only)
fi
if [[ "$APPEND_OUTPUT" == "1" || "$APPEND_OUTPUT" == "true" || "$APPEND_OUTPUT" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--append-output)
fi
if [[ "$REPLACE_LAYER_COUNT_BASELINES" == "1" || "$REPLACE_LAYER_COUNT_BASELINES" == "true" || "$REPLACE_LAYER_COUNT_BASELINES" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--replace-layer-count-baselines)
fi
if [[ "$DEDUPLICATE_ZERO_DROPOUT_INPUTS" == "1" || "$DEDUPLICATE_ZERO_DROPOUT_INPUTS" == "true" || "$DEDUPLICATE_ZERO_DROPOUT_INPUTS" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--deduplicate-zero-dropout-inputs)
fi
if [[ "$FILTER_TCELL_INVALID_CELLS" == "1" || "$FILTER_TCELL_INVALID_CELLS" == "true" || "$FILTER_TCELL_INVALID_CELLS" == "TRUE" ]]; then
  OPTIONAL_ARGS+=(--filter-tcell-invalid-cells)
fi

echo "Running pooled layered metrics"
echo "  task: $TASK_ID / $(( ${#SAMPLE_REPLICATES[@]} - 1 ))"
echo "  sample replicate: $SAMPLE_REPLICATE"
echo "  input files: ${#TASK_INPUT_FILES[@]}"
printf '    %s\n' "${TASK_INPUT_FILES[@]}"
echo "  layers: $LAYER_CSV"
echo "  output dir: $TASK_OUTPUT_DIR"
echo "  label key: $LABEL_KEY"
echo "  cleaning mode: $CLEANING_MODE"
echo "  cell-type-specific removal: $CELL_TYPE_SPECIFIC_REMOVAL"
echo "  removal group key: $REMOVAL_GROUP_KEY"
echo "  probability transform: $PROBABILITY_TRANSFORM"
echo "  probability power: $PROBABILITY_POWER"
echo "  layered baseline only: $LAYERED_BASELINE_ONLY"
echo "  append output: $APPEND_OUTPUT"
echo "  deduplicate zero-dropout inputs: $DEDUPLICATE_ZERO_DROPOUT_INPUTS"
echo "  skip tcell neighborhood metrics: $SKIP_TCELL_NEIGHBORHOOD_METRICS"
echo "  filter tcell invalid cells: $FILTER_TCELL_INVALID_CELLS"
echo "  tcell exclude labels: ${TCELL_EXCLUDE_LABELS:-<none>}"
echo "  reference data: ${REFERENCE_DATA:-<unset>}"
echo "  tcell cosine output: $TCELL_COSINE_OUTPUT"

# EXTRA_ARGS is intentionally word-split so simple additional flags can be supplied via sbatch --export.
# shellcheck disable=SC2086
python /home/jwinchel/Syndri/eval/metrics_eval.py \
  "${INPUT_ARGS[@]}" \
  --output-dir "$TASK_OUTPUT_DIR" \
  --cell-type-output "$CELL_TYPE_OUTPUT" \
  --tcell-output "$TCELL_OUTPUT" \
  --tcell-effect-output "$TCELL_EFFECT_OUTPUT" \
  --tcell-lr-gene-output "$TCELL_LR_GENE_OUTPUT" \
  --tcell-cosine-output "$TCELL_COSINE_OUTPUT" \
  --cell-type-profile-output "$CELL_TYPE_PROFILE_OUTPUT" \
  --config-output "$CONFIG_OUTPUT" \
  --evaluation-layers "$LAYER_CSV" \
  --sample-replicate-regex "$SAMPLE_REPLICATE_REGEX" \
  --label-key "$LABEL_KEY" \
  --tcell-cell-type-key "$TCELL_CELL_TYPE_KEY" \
  --tcell-sample-key "$TCELL_SAMPLE_KEY" \
  --reference-data "$REFERENCE_DATA" \
  --reference-cell-type-key "$REFERENCE_CELL_TYPE_KEY" \
  --reference-counts-layer "$REFERENCE_COUNTS_LAYER" \
  --tcell-cosine-cell-type-pattern "$TCELL_COSINE_CELL_TYPE_PATTERN" \
  --tcell-cosine-cell-types "$TCELL_COSINE_CELL_TYPES" \
  --tcell-cosine-cell-type-map "$TCELL_COSINE_CELL_TYPE_MAP" \
  --tcell-cosine-min-spatial-cells "$TCELL_COSINE_MIN_SPATIAL_CELLS" \
  --tcell-cosine-min-reference-cells "$TCELL_COSINE_MIN_REFERENCE_CELLS" \
  --coord-x "$COORD_X" \
  --coord-y "$COORD_Y" \
  --exclude-labels "$EXCLUDE_LABELS" \
  --tcell-exclude-labels "$TCELL_EXCLUDE_LABELS" \
  --syndri-probs-template "$SYNDRI_PROBS_TEMPLATE" \
  --mlp-probs-template "$MLP_PROBS_TEMPLATE" \
  --split-counts-template "$SPLIT_COUNTS_TEMPLATE" \
  --syndri-case-label "$SYNDRI_CASE_LABEL" \
  --mlp-case-label "$MLP_CASE_LABEL" \
  --split-case-label "$SPLIT_CASE_LABEL" \
  --target-removed-fractions "$TARGET_REMOVED_FRACTIONS" \
  --cleaning-mode "$CLEANING_MODE" \
  --removal-group-key "$REMOVAL_GROUP_KEY" \
  --probability-transform "$PROBABILITY_TRANSFORM" \
  --probability-power "$PROBABILITY_POWER" \
  --tcell-probability-removal-scale "$TCELL_PROBABILITY_REMOVAL_SCALE" \
  --sample-frac "$SAMPLE_FRAC" \
  --leiden-resolution "$LEIDEN_RESOLUTION" \
  --leiden-optimize-resolution "$LEIDEN_OPTIMIZE_RESOLUTION" \
  --leiden-n-neighbors "$LEIDEN_N_NEIGHBORS" \
  --n-pcs "$N_PCS" \
  --library-size-target-sum "$LIBRARY_SIZE_TARGET_SUM" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
  --random-state "$RANDOM_STATE" \
  --tcell-radius-um "$TCELL_RADIUS_UM" \
  --tcell-top-n "$TCELL_TOP_N" \
  --tcell-min-cells-per-label "$TCELL_MIN_CELLS_PER_LABEL" \
  --tcell-downsample-if-n-cells "$TCELL_DOWNSAMPLE_IF_N_CELLS" \
  --tcell-max-cells-per-label "$TCELL_MAX_CELLS_PER_LABEL" \
  --tcell-exact-zscore "$TCELL_EXACT_ZSCORE" \
  --tcell-dense-size-limit-entries "$TCELL_DENSE_SIZE_LIMIT_ENTRIES" \
  "${OPTIONAL_ARGS[@]}" \
  $EXTRA_ARGS

# Current default directory has 19 breast sample-replicates, so:
# sbatch --array=0-18%2 /home/jwinchel/Syndri/scripts/slurm/metrics_eval_layers_pooled.sbatch
#
# Explicit layers through Slurm export:
# sbatch --array=0-18%2 --export=ALL,INPUT_LAYERS=trial_0:trial_1:trial_2,INPUT_LAYERS_SEP=: /home/jwinchel/Syndri/scripts/slurm/metrics_eval_layers_pooled.sbatch
#
# Append the zero-dropout condition from .X to existing outputs:
# sbatch --array=0-18%2 --export=ALL,SKIP_EXISTING=0,INPUT_LAYERS=X,APPEND_OUTPUT=1,REPLACE_LAYER_COUNT_BASELINES=1 /home/jwinchel/Syndri/scripts/slurm/metrics_eval_layers_pooled.sbatch

# 07/28/2026
# sbatch --array=0-18%2 --export=ALL,SKIP_EXISTING=0,INPUT_LAYERS=X:trial_0:trial_1:trial_2,INPUT_LAYERS_SEP=:,TCELL_COSINE_CELL_TYPE_PATTERN='T' /home/jwinchel/Syndri/scripts/slurm/metrics_eval_layers_pooled.sbatch

# sbatch --array=0-18%2   --export=ALL,SKIP_EXISTING=0,SKIP_CELL_TYPE_SEPARATION=1,SKIP_TCELL_NEIGHBORHOOD_METRICS=1,SKIP_CELL_TYPE_PROFILES=1,INPUT_LAYERS=X:trial_0:trial_1:trial_2,INPUT_LAYERS_SEP=:,TCELL_COSINE_CELL_TYPE_PATTERN='T'   /home/jwinchel/Syndri/scripts/slurm/metrics_eval_layers_pooled.sbatch


# sbatch --array=0-18%2 \
#   --export=ALL,SKIP_EXISTING=0,APPEND_OUTPUT=1,SKIP_CELL_TYPE_SEPARATION=1,SKIP_TCELL_NEIGHBORHOOD_METRICS=1,SKIP_CELL_TYPE_PROFILES=1,INPUT_LAYERS=X:trial_0:trial_1:trial_2,INPUT_LAYERS_SEP=:,TCELL_COSINE_CELL_TYPE_PATTERN=T \
#   /home/jwinchel/Syndri/scripts/slurm/metrics_eval_layers_pooled.sbatch


# 08/05/2026: cosmx lab meeting results
# sbatch --array=0-7%2 \
#   --export=ALL,SPATIAL_DIR=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment,INPUT_GLOB='cosmx_nsclc*1pct.h5ad',SAMPLE_REPLICATE_REGEX='Lung[0-9]+(-Rep[0-9]+)?',INPUT_LAYERS=X,LABEL_KEY=cell_type,TCELL_CELL_TYPE_KEY=cell_type,REMOVAL_GROUP_KEY=cell_type,COORD_X=CenterX_global_px,COORD_Y=CenterY_global_px,REFERENCE_DATA=/compute/oven-0-13/jwinchel/scRNA-seq/adata/GSE127465_tumor_integrated.h5ad,REFERENCE_CELL_TYPE_KEY=cell_type,SYNDRI_PROBS_TEMPLATE='admix_probs_{layer}',MLP_PROBS_TEMPLATE='mlp_admix_probs_{layer}',SPLIT_COUNTS_TEMPLATE='split_purified_counts_{layer}',OUTPUT_DIR=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/metrics_eval_layers_pooled_cosmx_X,SKIP_EXISTING=0,EXCLUDE_LABELS='malignant cell,nan',TCELL_COSINE_CELL_TYPE_PATTERN='T' \
#   /home/jwinchel/Syndri/scripts/slurm/metrics_eval_layers_pooled.sbatch


# 08/21/2026: first-pass optimized xenium results
# sbatch --array=0-18%2 \
#   --export=ALL,SKIP_EXISTING=0,APPEND_OUTPUT=0,SKIP_CELL_TYPE_SEPARATION=0,SKIP_TCELL_NEIGHBORHOOD_METRICS=0,SKIP_CELL_TYPE_PROFILES=0,INPUT_LAYERS=X,INPUT_LAYERS_SEP=:,SYNDRI_PROBS_TEMPLATE='admix_probs_matchedcelltypes_optimized_0{layer:.0}',SYNDRI_CASE_LABEL=syndri_matchedcelltypes_optimized_0,TCELL_COSINE_CELL_TYPE_PATTERN=T,INPUT_GLOB='GSM950*_1pct.h5ad',OUTPUT_DIR=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/optimized_matchedcelltypes_eval \
#   /home/jwinchel/Syndri/scripts/slurm/metrics_eval_layers_pooled.sbatch


# 08/25/2026: first-pass optimized xenium results (unexpressed genes, no same-cell type neighbors, no query cell type)
# sbatch --array=0-18%2 \
#   --export=ALL,SKIP_EXISTING=0,APPEND_OUTPUT=0,SKIP_CELL_TYPE_SEPARATION=0,SKIP_TCELL_NEIGHBORHOOD_METRICS=0,SKIP_CELL_TYPE_PROFILES=0,INPUT_LAYERS=X,INPUT_LAYERS_SEP=:,SYNDRI_PROBS_TEMPLATE='admix_probs_optimized_unexpressedgenes_',SYNDRI_CASE_LABEL=syndri_matchedcelltypes_optimized_0,TCELL_COSINE_CELL_TYPE_PATTERN=T,INPUT_GLOB='GSM950*_1pct.h5ad',OUTPUT_DIR=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/optimized_unexpressed_nosametypeneighbors_eval \
#   /home/jwinchel/Syndri/scripts/slurm/metrics_eval_layers_pooled.sbatch


# 08/27/2026: first-pass optimized xenium results (unexpressed genes, no same-cell type neighbors, with query cell type)
# sbatch --array=0-18%2 \
#   --export=ALL,SKIP_EXISTING=0,APPEND_OUTPUT=0,SKIP_CELL_TYPE_SEPARATION=0,SKIP_TCELL_NEIGHBORHOOD_METRICS=0,SKIP_CELL_TYPE_PROFILES=0,INPUT_LAYERS=X,INPUT_LAYERS_SEP=:,SYNDRI_PROBS_TEMPLATE='admix_probs_optimized_unexpressedgenes_withquerycelltype',SYNDRI_CASE_LABEL=admix_probs_optimized_unexpressedgenes_withquerycelltype_X,TCELL_COSINE_CELL_TYPE_PATTERN=T,INPUT_GLOB='GSM950*_1pct.h5ad',OUTPUT_DIR=/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/optimized_unexpressed_nosametypeneighbors_withquerycelltype_eval \
#   /home/jwinchel/Syndri/scripts/slurm/metrics_eval_layers_pooled.sbatch