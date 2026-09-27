#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH --mem=64Gb
#SBATCH --job-name=breast-contexts
#SBATCH --output=/home/jwinchel/Syndri/logs/breast-contexts-%A_%a.txt
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=8

# Build responsiveness contexts for one selected tissue per Slurm array task.
set -euo pipefail

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-heimdall}"

formatter=/home/jwinchel/Syndri/Heimdall/scripts/prepare_responsiveness_contexts.py
input_dir=${INPUT_DIR:-/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/interim/spatial}
input_glob=${INPUT_GLOB:-*_rctd_annotated.h5ad}
input_list_file=${INPUT_LIST_FILE:-}
input_files=${INPUT_FILES:-}
output_dir=${OUTPUT_DIR:-/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/results/responsiveness/contexts}
if [[ -n "${CELL_TYPE_B64:-}" ]]; then
  cell_type=$(printf '%s' "$CELL_TYPE_B64" | base64 --decode)
else
  cell_type=${CELL_TYPE:-cytotoxic T cell}
fi
output_suffix=${OUTPUT_SUFFIX:-cytotoxic}
max_output_rows=${MAX_OUTPUT_ROWS:-5000000}
n_contexts=${N_CONTEXTS:-200}
n_query_cells=${N_QUERY_CELLS:-300}
n_repeats=${N_REPEATS:-5}
random_seed=${RANDOM_SEED:-0}
include_original_context=${INCLUDE_ORIGINAL_CONTEXT:-1}
python_bin=${PYTHON_BIN:-/home/jwinchel/miniconda3/envs/heimdall/bin/python}
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
  echo "No context inputs selected." >&2
  echo "INPUT_DIR=$input_dir" >&2
  echo "INPUT_GLOB=$input_glob" >&2
  exit 1
}
mapfile -t files < <(printf '%s\n' "${files[@]}" | sort -u)

task_id=${SLURM_ARRAY_TASK_ID:-0}
if (( task_id < 0 || task_id >= ${#files[@]} )); then
  echo "Array index $task_id is outside the ${#files[@]} selected input files." >&2
  echo "Submit with --array=0-$(( ${#files[@]} - 1 ))%N." >&2
  exit 1
fi

input_path=${files[$task_id]}
if [[ "$input_path" != /* ]]; then
  input_path="$input_dir/$input_path"
fi
[[ -f "$input_path" ]] || { echo "Input does not exist: $input_path" >&2; exit 1; }

args=(
  "$input_path"
  --output-dir "$output_dir"
  --cell-type "$cell_type"
  --cell-type-col harmony_cell_type
  --coord-cols x_centroid,y_centroid
  --k-neighbors 5
  --min-source-counts 10
  --suffix "$output_suffix"
  --n-contexts "$n_contexts"
  --n-query-cells "$n_query_cells"
  --n-repeats "$n_repeats"
  --random-seed "$random_seed"
  --max-output-rows "$max_output_rows"
)
if [[ "$include_original_context" == 0 || "$include_original_context" == false || "$include_original_context" == FALSE ]]; then
  args+=(--no-include-original-context)
fi

echo "Task $task_id/${#files[@]}: $(basename -- "$input_path")"
if [[ "$dry_run" == 1 || "$dry_run" == true || "$dry_run" == TRUE ]]; then
  printf 'DRY RUN: %q' "$python_bin"
  printf ' %q' "$formatter" "${args[@]}"
  printf '\n'
else
  "$python_bin" "$formatter" "${args[@]}"
fi
