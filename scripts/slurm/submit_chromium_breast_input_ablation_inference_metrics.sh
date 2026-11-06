#!/usr/bin/env bash
set -euo pipefail

repo=/home/jwinchel/Syndri
model_root="$repo/Heimdall/admixture_prediction_results/chromium_breast_GSM95019174_augmented_matchedcelltypes_optimized_unexpressedgenes_withquerycelltype_input_ablations"
data_root=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium
source_dir="$data_root/results/split/merged_h5ad"
output_root="$data_root/results/input_ablations"
inference_reference=/work/magroup/skrieger/Syndri/data/trainsets/chromium_GSM9509174_breast_matchedcelltypes.h5ad
metrics_reference="$data_root/interim/single-cell/breast_cancer_chromium_genesubset.h5ad"
inference_script="$repo/scripts/slurm/admixture_inference_layers.sbatch"
non_tcell_script="$repo/scripts/slurm/metrics_eval_final.sbatch"
tcell_script="$repo/scripts/slurm/metrics_eval_tcell_final.sbatch"
metadata_python=/home/jwinchel/miniconda3/envs/rctd/bin/python

concurrent_inference_tasks=${CONCURRENT_INFERENCE_TASKS:-3}
concurrent_metric_tasks=${CONCURRENT_METRIC_TASKS:-2}
rerun_inference=${RERUN_INFERENCE:-0}

if [[ -n "${ABLATIONS:-}" ]]; then
  IFS=',' read -r -a ablations <<< "$ABLATIONS"
else
  ablations=(query_only neighbors_only no_cell_type_tokens)
fi
declare -A milestones=(
  [query_only]=53
  [neighbors_only]=53
  [no_cell_type_tokens]=53
)

for ablation in "${ablations[@]}"; do
  if [[ ! -v "milestones[$ablation]" ]]; then
    echo "Unknown ablation $ablation; choose query_only, neighbors_only, or no_cell_type_tokens." >&2
    exit 1
  fi
done

for value_name in concurrent_inference_tasks concurrent_metric_tasks; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name^^} must be a positive integer: $value" >&2
    exit 1
  fi
done
if [[ ! "$rerun_inference" =~ ^(0|1)$ ]]; then
  echo "RERUN_INFERENCE must be 0 or 1: $rerun_inference" >&2
  exit 1
fi

[[ -f "$inference_reference" ]] || { echo "Missing inference reference: $inference_reference" >&2; exit 1; }
[[ -f "$metrics_reference" ]] || { echo "Missing metrics reference: $metrics_reference" >&2; exit 1; }
mkdir -p "$repo/logs" "$output_root"

shopt -s nullglob
source_files=("$source_dir"/*rctd_annotated.h5ad)
shopt -u nullglob
if (( ${#source_files[@]} != 19 )); then
  echo "Expected 19 Breast Xenium inputs in $source_dir; found ${#source_files[@]}." >&2
  exit 1
fi

# Write the training-reference genes once, in the exact order expected by the
# checkpoint tokenizer.
gene_list="$output_root/training_reference_genes.txt"
"$metadata_python" - "$inference_reference" "$gene_list" <<'PY'
import sys
from pathlib import Path
import h5py

with h5py.File(sys.argv[1], "r") as handle:
    var = handle["var"]
    index = var[var.attrs["_index"]]
    if isinstance(index, h5py.Group) and "values" in index:
        if index["mask"][()].any():
            raise ValueError("Reference gene index contains missing values.")
        genes = index["values"].asstr()[()].tolist()
    elif isinstance(index, h5py.Group):
        codes = index["codes"][()]
        if (codes < 0).any():
            raise ValueError("Reference gene index contains missing values.")
        genes = index["categories"].asstr()[()][codes].tolist()
    else:
        genes = index.asstr()[()].tolist()

if not genes or len(genes) != len(set(genes)):
    raise ValueError("Reference genes must be nonempty and unique.")
Path(sys.argv[2]).write_text("\n".join(genes) + "\n", encoding="utf-8")
print(f"Wrote {len(genes)} reference genes to {sys.argv[2]}")
PY

submit_metrics() {
  local ablation=$1
  local dependency=$2
  local final_dir="$output_root/$ablation/final"
  local metrics_root="$output_root/$ablation/metrics"
  local manifest="$output_root/$ablation/metric_inputs.txt"
  local dependency_args=()

  printf '%s\n' "$final_dir"/*.h5ad > "$manifest"
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:$dependency")
  fi

  local common_exports
  common_exports="ALL,INPUT_LIST_FILE=$manifest,SAMPLE_FRAC=1.0"

  local non_tcell_job
  non_tcell_job=$(sbatch --parsable \
    --job-name="abl-nontcell-$ablation" \
    --array="0-18%${concurrent_metric_tasks}" \
    "${dependency_args[@]}" \
    --export="$common_exports,OUTPUT_ROOT=$metrics_root/non_tcell,LABEL_KEY=harmony_cell_type,EXCLUDE_LABELS=nan,SYNDRI_PROFILE_CELL_TYPE_COL=cell_type" \
    "$non_tcell_script")

  local tcell_job
  tcell_job=$(sbatch --parsable \
    --job-name="abl-tcell-$ablation" \
    --array="0-18%${concurrent_metric_tasks}" \
    "${dependency_args[@]}" \
    --export="$common_exports,OUTPUT_ROOT=$metrics_root/tcell,LABEL_KEY=harmony_cell_type,REFERENCE_DATA=$metrics_reference,REFERENCE_LABEL_KEY=cell_type,SYNDRI_PROFILE_CELL_TYPE_COL=cell_type,COORD_X=x_centroid,COORD_Y=y_centroid,COORD_UNITS=micrometers,NEIGHBORHOOD_RADIUS=15,PHYSICAL_RADIUS_UM=15" \
    "$tcell_script")

  echo "  non-T-cell metrics job: $non_tcell_job"
  echo "  T-cell metrics job: $tcell_job"
}

for ablation in "${ablations[@]}"; do
  checkpoint_dir="$model_root/$ablation"
  milestone=${milestones[$ablation]}
  ablation_root="$output_root/$ablation"
  final_dir="$ablation_root/final"
  inference_manifest="$ablation_root/inference_inputs.txt"
  pending_manifest="$ablation_root/pending_inference_inputs.txt"

  [[ -f "$checkpoint_dir/config.txt" ]] || { echo "Missing $checkpoint_dir/config.txt" >&2; exit 1; }
  [[ -f "$checkpoint_dir/model-$milestone.pt" ]] || {
    echo "Missing $checkpoint_dir/model-$milestone.pt" >&2
    exit 1
  }

  mkdir -p "$final_dir" "$ablation_root/metrics/non_tcell" "$ablation_root/metrics/tcell"
  : > "$inference_manifest"
  : > "$pending_manifest"

  for source in "${source_files[@]}"; do
    target="$final_dir/$(basename "$source")"
    if [[ ! -f "$target" ]]; then
      cp --reflink=auto -- "$source" "$target"
    fi
    printf '%s\n' "$target" >> "$inference_manifest"

    has_prediction=$(
      "$metadata_python" - "$target" <<'PY'
import sys
import h5py
with h5py.File(sys.argv[1], "r") as handle:
    if "split_purified_counts_X" not in handle["layers"]:
        raise SystemExit(f"Missing split_purified_counts_X in {sys.argv[1]}")
    print(int("syndri_admix_probs_X" in handle["layers"]))
PY
    )
    if [[ "$rerun_inference" == 1 || "$has_prediction" == 0 ]]; then
      printf '%s\n' "$target" >> "$pending_manifest"
    fi
  done

  pending_count=$(wc -l < "$pending_manifest")
  inference_job=""
  if (( pending_count > 0 )); then
    inference_job=$(sbatch --parsable \
      --job-name="abl-infer-$ablation" \
      --array="0-$((pending_count - 1))%${concurrent_inference_tasks}" \
      --export="ALL,ST_DATA_LIST_FILE=$pending_manifest,REFERENCE_DATA=$inference_reference,CHECKPOINT_DIR=$checkpoint_dir,CONFIG_SOURCE=checkpoint,MILESTONE=$milestone,BATCH_SIZE=16,SPLIT_VALUE=val,SPLIT_COL=split,GENE_LIST=$gene_list,CELL_TYPE_COL=harmony_cell_type,PRESERVE_SPLIT_COLUMN=0,COORD_X=x_centroid,COORD_Y=y_centroid,APPLY_ADMIXTURE=0,FAST_INFERENCE=1,INPUT_LAYERS=X,PRED_LAYERS=syndri_admix_probs_X,OVERWRITE_LAYERS=$rerun_inference,EXTRA_OVERRIDES=fc.args.query_cell_type_col=harmony_cell_type" \
      "$inference_script")
    echo "$ablation: inference job $inference_job for $pending_count slices (milestone $milestone)"
  else
    echo "$ablation: all 19 inference layers already exist; submitting metrics directly."
  fi

  submit_metrics "$ablation" "$inference_job"
done

echo "Submitted all Breast Xenium input-ablation inference and metric jobs."
echo "Outputs: $output_root/{query_only,neighbors_only,no_cell_type_tokens}"
