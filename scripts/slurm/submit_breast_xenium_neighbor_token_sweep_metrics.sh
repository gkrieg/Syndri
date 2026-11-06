#!/usr/bin/env bash
set -euo pipefail

repo=/home/jwinchel/Syndri
data_root=/work/magroup/skrieger/Syndri/data/datasets/breast_xenium
results_root="$data_root/results/breast_xenium_10neighbor_token_sweep"
metrics_reference="$data_root/interim/single-cell/breast_cancer_chromium_genesubset.h5ad"
inference_jobs=(103881 103882 103883 103884 103885 103886)
models=(
  breast_xenium_k10_visible1
  breast_xenium_k10_visible2
  breast_xenium_k10_visible4
  breast_xenium_k10_visible6
  breast_xenium_k10_visible8
  breast_xenium_k10_visible10
)
concurrency=${CONCURRENT_METRIC_TASKS:-2}

[[ -f "$metrics_reference" ]] || { echo "Missing metrics reference: $metrics_reference" >&2; exit 1; }
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || {
  echo "CONCURRENT_METRIC_TASKS must be a positive integer" >&2
  exit 1
}
mkdir -p "$repo/logs"

for i in "${!models[@]}"; do
  model=${models[$i]}
  inference_job=${inference_jobs[$i]}
  output_dir="$results_root/$model"
  manifest="$output_dir/metric_inputs.txt"
  shopt -s nullglob
  files=("$output_dir"/GSM*rctd_annotated.h5ad)
  shopt -u nullglob
  (( ${#files[@]} == 19 )) || {
    echo "Expected 19 staged inference files in $output_dir; found ${#files[@]}" >&2
    exit 1
  }
  printf '%s\n' "${files[@]}" > "$manifest"

  non_tcell_root="$output_dir/metrics/non_tcell"
  tcell_root="$output_dir/metrics/tcell"
  mkdir -p "$non_tcell_root" "$tcell_root"
  dependency=(--dependency="afterok:$inference_job")
  common="ALL,INPUT_LIST_FILE=$manifest,SAMPLE_FRAC=1.0,LABEL_KEY=harmony_cell_type"

  non_tcell_job=$(sbatch --parsable \
    --job-name="sweep-nontcell-${model##*visible}" \
    --cpus-per-task=4 \
    --array="0-18%${concurrency}" \
    "${dependency[@]}" \
    --export="$common,OUTPUT_ROOT=$non_tcell_root,EXCLUDE_LABELS=nan,SYNDRI_PROFILE_CELL_TYPE_COL=cell_type,OMIT_ADDITIONAL_COUNTS=1" \
    "$repo/scripts/slurm/metrics_eval_final.sbatch")

  tcell_job=$(sbatch --parsable \
    --job-name="sweep-tcell-${model##*visible}" \
    --cpus-per-task=4 \
    --array="0-18%${concurrency}" \
    "${dependency[@]}" \
    --export="$common,OUTPUT_ROOT=$tcell_root,REFERENCE_DATA=$metrics_reference,REFERENCE_LABEL_KEY=cell_type,SYNDRI_PROFILE_CELL_TYPE_COL=cell_type,COORD_X=x_centroid,COORD_Y=y_centroid,COORD_UNITS=micrometers,NEIGHBORHOOD_RADIUS=15,PHYSICAL_RADIUS_UM=15,OMIT_ADDITIONAL_COUNTS=1" \
    "$repo/scripts/slurm/metrics_eval_tcell_final.sbatch")

  echo "$model (after inference $inference_job): non-T-cell/cell-separation metrics $non_tcell_job; T-cell/profile metrics $tcell_job"
done
