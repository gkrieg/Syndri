#!/usr/bin/env bash
set -euo pipefail

if (( $# > 0 )); then
  datasets=("$@")
else
  datasets=(agingbrain_merfish brain_merfish nsclc_cosmx breast_xenium)
fi
base=/work/magroup/skrieger/Syndri/data/datasets
script=/home/jwinchel/Syndri/scripts/slurm/backfill_count_summaries.sbatch
manifest=/home/jwinchel/Syndri/logs/count-summaries-inputs.$(date +%Y%m%d-%H%M%S).txt
mkdir -p /home/jwinchel/Syndri/logs

: > "$manifest"
for dataset in "${datasets[@]}"; do
  case "$dataset" in
    nsclc_cosmx)
      while IFS= read -r file; do
        printf '%s\t%s\t%s\n' "$file" cell_type_harmonized cell_type_harmonized >> "$manifest"
      done < <(
        find "$base/$dataset/results/metrics/non_tcell" \
          -type f -path '*/st_integrated_slice-*/non_tcell_metrics.csv' -print | sort
      )
      ;;
    agingbrain_merfish|brain_merfish|breast_xenium)
      while IFS= read -r file; do
        printf '%s\t\t\n' "$file" >> "$manifest"
      done < <(
        find "$base/$dataset/results/metrics/non_tcell" \
          -type f -name non_tcell_metrics.csv -print | sort
      )
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      exit 1
      ;;
  esac
done
sort -o "$manifest" "$manifest"

mapfile -t files < "$manifest"
if (( ${#files[@]} == 0 )); then
  echo "No non_tcell_metrics.csv files found." >&2
  exit 1
fi

echo "Submitting ${#files[@]} metrics files from $manifest"
sbatch \
  --array="0-$((${#files[@]} - 1))%4" \
  --export="ALL,METRICS_LIST_FILE=$manifest" \
  "$script"
