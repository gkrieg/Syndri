#!/usr/bin/env bash
# Run Syndri inference on selected breast Xenium responsiveness-context files.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sbatch_file="$script_dir/slurm/breast_xenium_responsiveness_inference.sbatch"

input_dir=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/results/responsiveness/contexts
output_dir=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/results/responsiveness/predictions
python_bin=/home/jwinchel/miniconda3/envs/heimdall/bin/python
checkpoint_dir=/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/chromium_breast_GSM95019174_augmented_matchedcelltypes_optimized_unexpressedgenes_withquerycelltype
reference_data=/work/magroup/skrieger/Syndri/data/trainsets/chromium_GSM9509174_breast_matchedcelltypes.h5ad
gene_list=
milestone=50
batch_size=16
output_suffix=responsiveness_predictions
force=0
dry_run=0
extra_overrides=()
array_concurrency=2
dependency=
sbatch_args=()

usage() {
  cat <<'EOF'
Usage:
  run_breast_xenium_responsiveness_inference.sh [options] SELECTOR [SELECTOR ...]

Selectors may be:
  - a complete prepared-context .h5ad path
  - a filename or stem within --input-dir
  - a quoted shell pattern matched against filenames within --input-dir

Options:
  --input-dir DIR          Prepared-context input directory
  --output-dir DIR         Prediction output directory
  --checkpoint-dir DIR     Syndri checkpoint directory
  --reference-data FILE    Reference AnnData containing cell_type_dict
  --gene-list FILE         Optional ordered gene-list text file
  --milestone N            Checkpoint milestone (default: 50)
  --batch-size N           Inference batch size (default: 16)
  --output-suffix TEXT     Output filename suffix
  --extra-override VALUE   Additional Hydra override; may be repeated
  --python PATH            Python executable
  --array-concurrency N    Maximum simultaneous inference jobs (default: 2)
  --dependency SPEC        Slurm dependency, e.g. afterok:12345
  --sbatch-arg VALUE       Additional sbatch argument; may be repeated
  --force                  Recompute prediction files that already exist
  --dry-run                Show resolved inputs and commands without running
  -h, --help               Show this help

Examples:
  # One prepared tissue
  ./scripts/run_breast_xenium_responsiveness_inference.sh \
    GSM9509174_breast_breast_B1_B1_3_rctd_annotated_cd4_memory

  # Several inputs
  ./scripts/run_breast_xenium_responsiveness_inference.sh \
    '*B1_B1_1*cd4_memory.h5ad' '*B1_B1_2*cd4_memory.h5ad'

  # Every prepared cytotoxic context file
  ./scripts/run_breast_xenium_responsiveness_inference.sh '*_cytotoxic.h5ad'

  # Inspect commands only
  ./scripts/run_breast_xenium_responsiveness_inference.sh \
    --dry-run 'GSM950917[2-4]*_cd4_memory.h5ad'
EOF
}

selectors=()
while (( $# > 0 )); do
  case "$1" in
    --input-dir)
      [[ $# -ge 2 ]] || { echo "--input-dir requires a value" >&2; exit 2; }
      input_dir=$2; shift 2 ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      output_dir=$2; shift 2 ;;
    --checkpoint-dir)
      [[ $# -ge 2 ]] || { echo "--checkpoint-dir requires a value" >&2; exit 2; }
      checkpoint_dir=$2; shift 2 ;;
    --reference-data)
      [[ $# -ge 2 ]] || { echo "--reference-data requires a value" >&2; exit 2; }
      reference_data=$2; shift 2 ;;
    --gene-list)
      [[ $# -ge 2 ]] || { echo "--gene-list requires a value" >&2; exit 2; }
      gene_list=$2; shift 2 ;;
    --milestone)
      [[ $# -ge 2 ]] || { echo "--milestone requires a value" >&2; exit 2; }
      milestone=$2; shift 2 ;;
    --batch-size)
      [[ $# -ge 2 ]] || { echo "--batch-size requires a value" >&2; exit 2; }
      batch_size=$2; shift 2 ;;
    --output-suffix)
      [[ $# -ge 2 ]] || { echo "--output-suffix requires a value" >&2; exit 2; }
      output_suffix=$2; shift 2 ;;
    --extra-override)
      [[ $# -ge 2 ]] || { echo "--extra-override requires a value" >&2; exit 2; }
      extra_overrides+=("$2"); shift 2 ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }
      python_bin=$2; shift 2 ;;
    --array-concurrency)
      [[ $# -ge 2 ]] || { echo "--array-concurrency requires a value" >&2; exit 2; }
      array_concurrency=$2; shift 2 ;;
    --dependency)
      [[ $# -ge 2 ]] || { echo "--dependency requires a value" >&2; exit 2; }
      dependency=$2; shift 2 ;;
    --sbatch-arg)
      [[ $# -ge 2 ]] || { echo "--sbatch-arg requires a value" >&2; exit 2; }
      sbatch_args+=("$2"); shift 2 ;;
    --force)
      force=1; shift ;;
    --dry-run)
      dry_run=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; selectors+=("$@"); break ;;
    -* )
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      selectors+=("$1"); shift ;;
  esac
done

(( ${#selectors[@]} > 0 )) || {
  echo "Provide at least one context-file selector; refusing to select every file implicitly." >&2
  usage >&2
  exit 2
}
[[ -d "$input_dir" ]] || { echo "Input directory does not exist: $input_dir" >&2; exit 1; }
[[ -f "$sbatch_file" ]] || { echo "Sbatch script does not exist: $sbatch_file" >&2; exit 1; }
[[ -x "$python_bin" ]] || { echo "Python executable does not exist or is not executable: $python_bin" >&2; exit 1; }
[[ -d "$checkpoint_dir" ]] || { echo "Checkpoint directory does not exist: $checkpoint_dir" >&2; exit 1; }
[[ -f "$checkpoint_dir/config.txt" ]] || { echo "Missing checkpoint config: $checkpoint_dir/config.txt" >&2; exit 1; }
[[ -f "$checkpoint_dir/model-$milestone.pt" ]] || { echo "Missing checkpoint: $checkpoint_dir/model-$milestone.pt" >&2; exit 1; }
[[ -f "$reference_data" ]] || { echo "Reference AnnData does not exist: $reference_data" >&2; exit 1; }
[[ -z "$gene_list" || -f "$gene_list" ]] || { echo "Gene list does not exist: $gene_list" >&2; exit 1; }
[[ "$milestone" =~ ^[0-9]+$ ]] || { echo "--milestone must be a nonnegative integer" >&2; exit 2; }
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo "--batch-size must be a positive integer" >&2; exit 2; }
[[ "$array_concurrency" =~ ^[1-9][0-9]*$ ]] || { echo "--array-concurrency must be positive" >&2; exit 2; }

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

echo "Selected ${#files[@]} prepared context file(s)."
echo "Output directory: $output_dir"
printf '  %s\n' "${files[@]}"

override_string=
for override in "${extra_overrides[@]}"; do
  override_string+="${override_string:+ }$override"
done

if (( dry_run )); then
  echo "DRY RUN: would submit array 0-$(( ${#files[@]} - 1 ))%$array_concurrency"
  [[ -n "$dependency" ]] && echo "Dependency: $dependency"
  exit 0
fi

command -v sbatch >/dev/null || { echo "sbatch is not available" >&2; exit 1; }
mkdir -p -- "$output_dir"
manifest=$(mktemp "$output_dir/inference_inputs.XXXXXX.txt")
printf '%s\n' "${files[@]}" > "$manifest"

export_value="ALL,INPUT_LIST_FILE=$manifest,INPUT_DIR=$input_dir,OUTPUT_DIR=$output_dir,CHECKPOINT_DIR=$checkpoint_dir,REFERENCE_DATA=$reference_data,MILESTONE=$milestone,BATCH_SIZE=$batch_size,OUTPUT_SUFFIX=$output_suffix,PYTHON_BIN=$python_bin,FORCE=$force"
[[ -n "$gene_list" ]] && export_value+=",GENE_LIST=$gene_list"
[[ -n "$override_string" ]] && export_value+=",EXTRA_OVERRIDES=$override_string"
command=(
  sbatch
  --array="0-$(( ${#files[@]} - 1 ))%$array_concurrency"
  --export="$export_value"
)
[[ -n "$dependency" ]] && command+=(--dependency="$dependency")
command+=("${sbatch_args[@]}" "$sbatch_file")
"${command[@]}"
echo "Input manifest: $manifest"
