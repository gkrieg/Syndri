#!/usr/bin/env bash
set -euo pipefail

if (( $# > 0 )); then
  datasets=("$@")
else
  datasets=(agingbrain_merfish brain_merfish nsclc_cosmx breast_xenium)
fi
base=/work/magroup/skrieger/Syndri/data/datasets
backend=${ADMIXTURE_BACKEND:-python}
case "$backend" in
  python) result_subdir=admixture_scores ;;
  celladmix) result_subdir=admixture_scores_celladmix ;;
  *) echo "Unknown ADMIXTURE_BACKEND: $backend" >&2; exit 1 ;;
esac
export ADMIXTURE_BACKEND="$backend"
manifest=/home/jwinchel/Syndri/logs/score-admixture-inputs.$(date +%Y%m%d-%H%M%S).tsv
mkdir -p /home/jwinchel/Syndri/logs
: > "$manifest"

add_dataset() {
  local dataset=$1 reference=$2 spatial_ct=$3 reference_ct=$4 coord_x=$5 coord_y=$6 fov_key=${7:-}
  local file output
  while IFS= read -r file; do
    output="$base/$dataset/results/metrics/$result_subdir/$(basename "$file" .h5ad)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$dataset" "$file" "$reference" "$output" \
      "$spatial_ct" "$reference_ct" "$coord_x" "$coord_y" "$fov_key" >> "$manifest"
  done < <(find "$base/$dataset/results/final" -maxdepth 1 -type f -name '*.h5ad' | sort)
}

for dataset in "${datasets[@]}"; do
  case "$dataset" in
    agingbrain_merfish)
      add_dataset "$dataset" \
        AUTO \
        cell_type cell_type center_x center_y
      ;;
    brain_merfish)
      add_dataset "$dataset" \
        AUTO \
        class class x y
      ;;
    nsclc_cosmx)
      add_dataset "$dataset" \
        AUTO \
        cell_type_harmonized cell_type_harmonized CenterX_global_px CenterY_global_px fov
      ;;
    breast_xenium)
      add_dataset "$dataset" \
        /work/magroup/skrieger/Syndri/data/trainsets/chromium_GSM9509174_breast_matchedcelltypes.h5ad \
        harmony_cell_type cell_type x_centroid y_centroid
      ;;
  esac
done

mapfile -t rows < "$manifest"
if (( ${#rows[@]} == 0 )); then
  echo "No final H5AD inputs found." >&2
  exit 1
fi

echo "Submitting ${#rows[@]} slices using manifest $manifest"
sbatch \
  --job-name="score-admixture-$backend" \
  --array="0-$((${#rows[@]} - 1))%${MAX_CONCURRENT:-2}" \
  --export="ALL,ADMIXTURE_MANIFEST=$manifest" \
  /home/jwinchel/Syndri/scripts/slurm/score_admixture_slice.sbatch
