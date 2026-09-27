#!/usr/bin/env bash
# Prepare precomputed responsiveness contexts for selected breast Xenium tissues.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sbatch_file="$script_dir/slurm/breast_xenium_responsiveness_contexts.sbatch"

input_dir=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/interim/spatial
output_dir=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/results/responsiveness/contexts
python_bin=/home/jwinchel/miniconda3/envs/heimdall/bin/python
cell_type=cytotoxic
max_output_rows=5000000
n_contexts=200
n_query_cells=300
n_repeats=5
random_seed=0
include_original_context=1
dry_run=0
array_concurrency=2
dependency=
sbatch_args=()

usage() {
  cat <<'EOF'
Usage:
  run_breast_xenium_responsiveness_contexts.sh [options] SELECTOR [SELECTOR ...]

Selectors may be:
  - a complete .h5ad path
  - a filename or stem within --input-dir
  - a quoted shell pattern matched against filenames within --input-dir

Options:
  --cell-type NAME       cytotoxic, cd4_memory, treg, or an exact annotation
                         value (default: cytotoxic)
  --input-dir DIR        Tissue-specific input directory
  --output-dir DIR       Prepared-context output directory
  --max-output-rows N    Per-tissue quadratic output safety limit; 0 disables
                         the limit (default: 5000000)
  --n-contexts N         Representative contexts per tissue (default: 200)
  --n-query-cells N      Query cells per tissue (default: 300)
  --n-repeats N          Independent sampling repeats (default: 5)
  --random-seed N        Base random seed (default: 0)
  --no-original-context  Do not add a guaranteed original baseline per query
  --python PATH          Python executable
  --array-concurrency N  Maximum simultaneous tissue jobs (default: 2)
  --dependency SPEC      Slurm dependency, e.g. afterok:12345
  --sbatch-arg VALUE     Additional sbatch argument; may be repeated
  --dry-run              Show resolved tissues and submission without submitting
  -h, --help             Show this help

Examples:
  # One tissue, selected by filename stem
  ./scripts/run_breast_xenium_responsiveness_contexts.sh \
    --cell-type cd4_memory GSM9509174_breast_breast_B1_B1_3_rctd_annotated

  # Several explicitly selected tissues
  ./scripts/run_breast_xenium_responsiveness_contexts.sh \
    --cell-type treg \
    GSM9509172_breast_breast_B1_B1_1_rctd_annotated \
    GSM9509173_breast_breast_B1_B1_2_rctd_annotated

  # Every B1 tissue; quote the pattern so this script resolves it
  ./scripts/run_breast_xenium_responsiveness_contexts.sh \
    --cell-type cytotoxic 'GSM*_breast_breast_B1_*_rctd_annotated.h5ad'

  # Inspect selection without creating files
  ./scripts/run_breast_xenium_responsiveness_contexts.sh \
    --dry-run --cell-type cd4_memory 'GSM950917[2-4]*'
EOF
}

selectors=()
while (( $# > 0 )); do
  case "$1" in
    --cell-type)
      [[ $# -ge 2 ]] || { echo "--cell-type requires a value" >&2; exit 2; }
      cell_type=$2
      shift 2
      ;;
    --input-dir)
      [[ $# -ge 2 ]] || { echo "--input-dir requires a value" >&2; exit 2; }
      input_dir=$2
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      output_dir=$2
      shift 2
      ;;
    --max-output-rows)
      [[ $# -ge 2 ]] || { echo "--max-output-rows requires a value" >&2; exit 2; }
      max_output_rows=$2
      shift 2
      ;;
    --n-contexts)
      [[ $# -ge 2 ]] || { echo "--n-contexts requires a value" >&2; exit 2; }
      n_contexts=$2; shift 2 ;;
    --n-query-cells)
      [[ $# -ge 2 ]] || { echo "--n-query-cells requires a value" >&2; exit 2; }
      n_query_cells=$2; shift 2 ;;
    --n-repeats)
      [[ $# -ge 2 ]] || { echo "--n-repeats requires a value" >&2; exit 2; }
      n_repeats=$2; shift 2 ;;
    --random-seed)
      [[ $# -ge 2 ]] || { echo "--random-seed requires a value" >&2; exit 2; }
      random_seed=$2; shift 2 ;;
    --no-original-context)
      include_original_context=0; shift ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }
      python_bin=$2
      shift 2
      ;;
    --array-concurrency)
      [[ $# -ge 2 ]] || { echo "--array-concurrency requires a value" >&2; exit 2; }
      array_concurrency=$2
      shift 2
      ;;
    --dependency)
      [[ $# -ge 2 ]] || { echo "--dependency requires a value" >&2; exit 2; }
      dependency=$2
      shift 2
      ;;
    --sbatch-arg)
      [[ $# -ge 2 ]] || { echo "--sbatch-arg requires a value" >&2; exit 2; }
      sbatch_args+=("$2")
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      selectors+=("$@")
      break
      ;;
    -* )
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      selectors+=("$1")
      shift
      ;;
  esac
done

(( ${#selectors[@]} > 0 )) || {
  echo "Provide at least one tissue selector; refusing to select every tissue implicitly." >&2
  usage >&2
  exit 2
}
[[ -d "$input_dir" ]] || { echo "Input directory does not exist: $input_dir" >&2; exit 1; }
[[ -f "$sbatch_file" ]] || { echo "Sbatch script does not exist: $sbatch_file" >&2; exit 1; }
[[ -x "$python_bin" ]] || { echo "Python executable does not exist or is not executable: $python_bin" >&2; exit 1; }
[[ "$max_output_rows" =~ ^[0-9]+$ ]] || { echo "--max-output-rows must be a nonnegative integer" >&2; exit 2; }
[[ "$n_contexts" =~ ^[1-9][0-9]*$ ]] || { echo "--n-contexts must be positive" >&2; exit 2; }
[[ "$n_query_cells" =~ ^[1-9][0-9]*$ ]] || { echo "--n-query-cells must be positive" >&2; exit 2; }
[[ "$n_repeats" =~ ^[1-9][0-9]*$ ]] || { echo "--n-repeats must be positive" >&2; exit 2; }
[[ "$random_seed" =~ ^[0-9]+$ ]] || { echo "--random-seed must be nonnegative" >&2; exit 2; }
[[ "$array_concurrency" =~ ^[1-9][0-9]*$ ]] || { echo "--array-concurrency must be positive" >&2; exit 2; }

case "$cell_type" in
  cytotoxic)
    annotation="cytotoxic T cell"
    suffix=cytotoxic
    ;;
  cd4|cd4_memory)
    annotation="CD4-positive, alpha-beta memory T cell"
    suffix=cd4_memory
    ;;
  treg)
    annotation="natural T-regulatory cell"
    suffix=treg
    ;;
  *)
    annotation=$cell_type
    suffix=$(printf '%s' "$cell_type" | tr '[:upper:] ' '[:lower:]_' | tr -cd '[:alnum:]_-')
    [[ -n "$suffix" ]] || suffix=selected_cell_type
    ;;
esac

declare -A selected=()
files=()
add_file() {
  local candidate=$1
  [[ -f "$candidate" ]] || return 0
  candidate=$(realpath -- "$candidate")
  if [[ -z ${selected["$candidate"]+x} ]]; then
    selected["$candidate"]=1
    files+=("$candidate")
  fi
}

shopt -s nullglob
for selector in "${selectors[@]}"; do
  matched_before=${#files[@]}

  if [[ -f "$selector" ]]; then
    add_file "$selector"
  elif [[ -f "$input_dir/$selector" ]]; then
    add_file "$input_dir/$selector"
  elif [[ "$selector" != *.h5ad && -f "$input_dir/$selector.h5ad" ]]; then
    add_file "$input_dir/$selector.h5ad"
  else
    # Deliberately expand only inside input_dir. Keeping the selector quoted at
    # the call site prevents the invoking shell from expanding it elsewhere.
    matches=("$input_dir"/$selector)
    for candidate in "${matches[@]}"; do
      [[ "$candidate" == *.h5ad ]] && add_file "$candidate"
    done
    if (( ${#files[@]} == matched_before )) && [[ "$selector" != *.h5ad ]]; then
      matches=("$input_dir"/$selector.h5ad)
      for candidate in "${matches[@]}"; do
        add_file "$candidate"
      done
    fi
  fi

  if (( ${#files[@]} == matched_before )); then
    echo "Selector matched no new .h5ad files: $selector" >&2
    exit 1
  fi
done
shopt -u nullglob

echo "Selected ${#files[@]} tissue(s)."
echo "Cell type: $annotation"
echo "Output directory: $output_dir"
printf '  %s\n' "${files[@]}"

if (( dry_run )); then
  echo "DRY RUN: would submit array 0-$(( ${#files[@]} - 1 ))%$array_concurrency"
  [[ -n "$dependency" ]] && echo "Dependency: $dependency"
  exit 0
fi

command -v sbatch >/dev/null || { echo "sbatch is not available" >&2; exit 1; }
mkdir -p -- "$output_dir"
manifest=$(mktemp "$output_dir/context_inputs.XXXXXX.txt")
printf '%s\n' "${files[@]}" > "$manifest"
annotation_b64=$(printf '%s' "$annotation" | base64 | tr -d '\n')

command=(
  sbatch
  --array="0-$(( ${#files[@]} - 1 ))%$array_concurrency"
  --export="ALL,INPUT_LIST_FILE=$manifest,INPUT_DIR=$input_dir,OUTPUT_DIR=$output_dir,CELL_TYPE_B64=$annotation_b64,OUTPUT_SUFFIX=$suffix,MAX_OUTPUT_ROWS=$max_output_rows,N_CONTEXTS=$n_contexts,N_QUERY_CELLS=$n_query_cells,N_REPEATS=$n_repeats,RANDOM_SEED=$random_seed,INCLUDE_ORIGINAL_CONTEXT=$include_original_context,PYTHON_BIN=$python_bin"
)
[[ -n "$dependency" ]] && command+=(--dependency="$dependency")
command+=("${sbatch_args[@]}" "$sbatch_file")
"${command[@]}"
echo "Input manifest: $manifest"
