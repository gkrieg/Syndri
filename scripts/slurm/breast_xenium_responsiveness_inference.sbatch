#!/usr/bin/env bash
#SBATCH -p gpu-small,gpu-large
#SBATCH --mem=64Gb
#SBATCH --job-name=breast-response-infer
#SBATCH --output=/home/jwinchel/Syndri/logs/breast-response-infer-%A_%a.txt
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Run inference for one prepared responsiveness AnnData per Slurm array task.
set -euo pipefail

module load cuda-12.1
cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-heimdall}"

inference_script=/home/jwinchel/Syndri/Heimdall/scripts/run_admixture_inference.py
input_dir=${INPUT_DIR:-/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/results/responsiveness/contexts}
input_glob=${INPUT_GLOB:-*.h5ad}
input_list_file=${INPUT_LIST_FILE:-}
input_files=${INPUT_FILES:-}
output_dir=${OUTPUT_DIR:-/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/results/responsiveness/predictions}
checkpoint_dir=${CHECKPOINT_DIR:-/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/chromium_breast_GSM95019174_augmented_matchedcelltypes_optimized_unexpressedgenes_withquerycelltype}
reference_data=${REFERENCE_DATA:-/work/magroup/skrieger/Syndri/data/trainsets/chromium_GSM9509174_breast_matchedcelltypes.h5ad}
gene_list=${GENE_LIST:-}
milestone=${MILESTONE:-50}
batch_size=${BATCH_SIZE:-16}
output_suffix=${OUTPUT_SUFFIX:-responsiveness_predictions}
python_bin=${PYTHON_BIN:-/home/jwinchel/miniconda3/envs/heimdall/bin/python}
extra_overrides=${EXTRA_OVERRIDES:-}
force=${FORCE:-0}
dry_run=${DRY_RUN:-0}

if [[ -n "$input_list_file" ]]; then
  [[ -f "$input_list_file" ]] || { echo "Missing INPUT_LIST_FILE: $input_list_file" >&2; exit 1; }
  mapfile -t files < <(sed '/^[[:space:]]*$/d' "$input_list_file")
elif [[ -n "$input_files" ]]; then
  IFS=',' read -r -a files <<< "$input_files"
else
  shopt -s nullglob
  files=("$input_dir"/$input_glob)
  shopt -u nullglob
fi

(( ${#files[@]} > 0 )) || {
  echo "No prepared context files selected." >&2
  echo "INPUT_DIR=$input_dir" >&2
  echo "INPUT_GLOB=$input_glob" >&2
  exit 1
}
mapfile -t files < <(printf '%s\n' "${files[@]}" | sort -u)

task_id=${SLURM_ARRAY_TASK_ID:-0}
if (( task_id < 0 || task_id >= ${#files[@]} )); then
  echo "Array index $task_id is outside the ${#files[@]} selected context files." >&2
  echo "Submit with --array=0-$(( ${#files[@]} - 1 ))%N." >&2
  exit 1
fi

input_path=${files[$task_id]}
if [[ "$input_path" != /* ]]; then
  input_path="$input_dir/$input_path"
fi
[[ -f "$input_path" ]] || { echo "Input does not exist: $input_path" >&2; exit 1; }
stem=$(basename -- "$input_path" .h5ad)

args=(
  --st-data "$input_path"
  --reference-data "$reference_data"
  --checkpoint-dir "$checkpoint_dir"
  --milestone "$milestone"
  --output-dir "$output_dir"
  --prepared-name "${stem}_inference_prepared.h5ad"
  --output-name "${stem}_${output_suffix}.h5ad"
  --cell-type-col harmony_cell_type
  --coord-cols x_centroid,y_centroid
  --batch-size "$batch_size"
  --use-precomputed-neighbor
  --preserve-split-column
  --extra-overrides "fc.args.query_cell_type_col=harmony_cell_type${extra_overrides:+ $extra_overrides}"
)
[[ -n "$gene_list" ]] && args+=(--gene-list "$gene_list")

echo "Task $task_id/${#files[@]}: $(basename -- "$input_path")"
output_path="$output_dir/${stem}_${output_suffix}.h5ad"
if [[ -f "$output_path" && "$force" != 1 && "$force" != true && "$force" != TRUE ]]; then
  echo "Skipping existing prediction: $output_path"
elif [[ "$dry_run" == 1 || "$dry_run" == true || "$dry_run" == TRUE ]]; then
  printf 'DRY RUN: %q' "$python_bin"
  printf ' %q' "$inference_script" "${args[@]}"
  printf '\n'
else
  "$python_bin" "$inference_script" "${args[@]}"
fi
