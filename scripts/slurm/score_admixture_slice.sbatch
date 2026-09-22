#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH --mem=96Gb
#SBATCH --cpus-per-task=8
#SBATCH -t 2-00:00:00
#SBATCH --job-name=score-admixture
#SBATCH --output=/home/jwinchel/Syndri/logs/score-admixture-%A_%a.txt
set -euo pipefail

: "${ADMIXTURE_MANIFEST:?Set ADMIXTURE_MANIFEST to the TSV input manifest}"
task=${SLURM_ARRAY_TASK_ID:-0}
line=$(sed -n "$((task + 1))p" "$ADMIXTURE_MANIFEST")
if [[ -z "$line" ]]; then
  echo "No manifest row for array task $task" >&2
  exit 1
fi

IFS=$'\t' read -r dataset spatial reference output spatial_ct reference_ct coord_x coord_y fov_key <<< "$line"

source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-heimdall}"

label_map=${LABEL_MAP_JSON:-}
[[ -n "$label_map" ]] || label_map='{}'
if [[ "$dataset" == nsclc_cosmx && -z "${LABEL_MAP_JSON:-}" ]]; then
  label_map='{"T CD4 naive":"T CD4+","T CD4 memory":"T CD4+","T CD8 naive":"T CD8+","T CD8 memory":"T CD8+","pDC":"DC","mDC":"DC"}'
fi

args=(
  --backend "${ADMIXTURE_BACKEND:-python}"
  --rscript "${ADMIXTURE_RSCRIPT:-/home/jwinchel/miniconda3/envs/r_env/bin/Rscript}"
  --celladmix-repo "${CELLADMIX_REPO:-/home/jwinchel/cellAdmix}"
  --celladmix-notebooks-repo "${CELLADMIX_NOTEBOOKS_REPO:-/home/jwinchel/cellAdmix-Notebooks}"
  --spatial-h5ad "$spatial"
  --output-dir "$output"
  --spatial-cell-type-key "$spatial_ct"
  --reference-cell-type-key "$reference_ct"
  --coord-x "$coord_x"
  --coord-y "$coord_y"
  --probability-layer "${PROBABILITY_LAYER:-syndri_admix_probs_X}"
  --split-layer "${SPLIT_LAYER-split_purified_counts_X}"
  --reference-layer "${REFERENCE_LAYER:-X}"
  --correlation-threshold "${CORRELATION_THRESHOLD:-0.2}"
  --target-removed-fractions "${TARGET_REMOVED_FRACTIONS:-0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5}"
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
  --label-map-json "$label_map"
)
[[ "$reference" != "AUTO" ]] && args+=(--reference-h5ad "$reference")
[[ -n "$fov_key" ]] && args+=(--fov-key "$fov_key")
[[ "${OVERWRITE:-0}" == 1 ]] && args+=(--overwrite)
[[ "${SKIP_CORRELATION_FILTER:-0}" == 1 ]] && args+=(--skip-correlation-filter)
[[ "${ADJUST_SCORES:-1}" == 0 ]] && args+=(--no-adjust)
[[ "$dataset" == nsclc_cosmx ]] && args+=(--collapse-prefix tumor=tumor)

python /home/jwinchel/Syndri/eval/score_admixture_slice.py "${args[@]}"
