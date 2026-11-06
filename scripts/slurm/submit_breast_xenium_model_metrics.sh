#!/usr/bin/env bash
set -euo pipefail

# Prepare each inference output with its matching SPLIT baseline layer, then
# submit the standard non-T-cell/cell-separation and T-cell/profile metric jobs.

repo=/home/jwinchel/Syndri
data_root=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium
results_root="$data_root/results"
metrics_reference="$data_root/interim/single-cell/breast_cancer_chromium_genesubset.h5ad"
metadata_python=/home/jwinchel/miniconda3/envs/rctd/bin/python
heimdall_python=/home/jwinchel/miniconda3/envs/heimdall/bin/python
concurrency=${CONCURRENT_METRIC_TASKS:-2}

models=(breast_xenium_k10_separators chromium_breast_xenium_full_retrain)

[[ -x "$metadata_python" ]] || { echo "Missing metadata Python: $metadata_python" >&2; exit 1; }
[[ -x "$heimdall_python" ]] || { echo "Missing Heimdall Python: $heimdall_python" >&2; exit 1; }
[[ -f "$metrics_reference" ]] || { echo "Missing metrics reference: $metrics_reference" >&2; exit 1; }
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo "CONCURRENT_METRIC_TASKS must be a positive integer" >&2; exit 1; }

mkdir -p "$repo/logs"

for model in "${models[@]}"; do
  output_dir="$results_root/$model"
  manifest="$output_dir/metric_inputs.txt"
  shopt -s nullglob
  files=("$output_dir"/GSM*rctd_annotated.h5ad)
  shopt -u nullglob
  (( ${#files[@]} == 19 )) || {
    echo "Expected 19 inferred outputs in $output_dir; found ${#files[@]}" >&2
    exit 1
  }

  "$metadata_python" - "${files[@]}" <<'PY'
import sys
import h5py

for name in sys.argv[1:]:
    with h5py.File(name, "r") as handle:
        if "syndri_admix_probs_X" not in handle.get("layers", {}):
            raise ValueError(f"Missing inference layer syndri_admix_probs_X in {name}")
print(f"Validated Syndri probability layer in all {len(sys.argv) - 1} outputs")
PY

  printf '%s\n' "${files[@]}" > "$manifest"
  non_tcell_root="$output_dir/metrics/non_tcell"
  tcell_root="$output_dir/metrics/tcell"
  mkdir -p "$non_tcell_root" "$tcell_root"
  common="ALL,INPUT_LIST_FILE=$manifest,SAMPLE_FRAC=1.0,LABEL_KEY=harmony_cell_type"

  non_tcell_job=$(sbatch --parsable \
    --job-name="syndri-metrics-nontcell-${model}" \
    --array="0-18%${concurrency}" \
    --export="$common,OUTPUT_ROOT=$non_tcell_root,EXCLUDE_LABELS=nan,SYNDRI_PROFILE_CELL_TYPE_COL=cell_type,OMIT_ADDITIONAL_COUNTS=1" \
    "$repo/scripts/slurm/metrics_eval_final.sbatch")

  tcell_job=$(sbatch --parsable \
    --job-name="syndri-metrics-tcell-${model}" \
    --array="0-18%${concurrency}" \
    --export="$common,OUTPUT_ROOT=$tcell_root,REFERENCE_DATA=$metrics_reference,REFERENCE_LABEL_KEY=cell_type,SYNDRI_PROFILE_CELL_TYPE_COL=cell_type,COORD_X=x_centroid,COORD_Y=y_centroid,COORD_UNITS=micrometers,NEIGHBORHOOD_RADIUS=15,PHYSICAL_RADIUS_UM=15,OMIT_ADDITIONAL_COUNTS=1" \
    "$repo/scripts/slurm/metrics_eval_tcell_final.sbatch")

  echo "$model: non-T-cell/cell-separation metrics job $non_tcell_job"
  echo "$model: T-cell/profile metrics job $tcell_job"
done
