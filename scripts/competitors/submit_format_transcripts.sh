#!/usr/bin/env bash
set -euo pipefail
base=/work/magroup/skrieger/Syndri/data/datasets
manifest=$(mktemp /home/jwinchel/Syndri/logs/competitor-format-inputs.XXXXXXXX.tsv)
for dataset in nsclc_cosmx breast_xenium; do
  for final in "$base/$dataset/results/final/"*.h5ad; do
    [[ -f "$final" ]] || continue
    out="$base/$dataset/raw/competitors/transcripts/$(basename "${final%.h5ad}").parquet"
    [[ -e "$out" ]] && continue
    printf '%s\t%s\n' "$dataset" "$final" >> "$manifest"
  done
done
rows=$(wc -l < "$manifest")
if (( rows == 0 )); then echo 'All formatted transcripts already exist'; exit 0; fi
echo "Formatting $rows final AnnData samples; manifest: $manifest"
sbatch --array="0-$((rows-1))%2" --export="ALL,COMPETITOR_FORMAT_MANIFEST=$manifest" \
  /home/jwinchel/Syndri/scripts/competitors/format_transcripts.sbatch
