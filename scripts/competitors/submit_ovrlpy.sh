#!/usr/bin/env bash
set -euo pipefail
base=/work/magroup/skrieger/Syndri/data/datasets
manifest=$(mktemp /home/jwinchel/Syndri/logs/competitor-ovrlpy-inputs.XXXXXXXX.txt)
for dataset in nsclc_cosmx breast_xenium; do
  for final in "$base/$dataset/results/final/"*.h5ad; do
    [[ -f "$final" ]] || continue
    out="$base/$dataset/results/competitors/ovrlpy/$(basename "$final")"
    [[ -e "$out" ]] && continue
    printf '%s\n' "$final" >> "$manifest"
  done
done
rows=$(wc -l < "$manifest")
if (( rows == 0 )); then echo 'All ovrlpy outputs already exist'; exit 0; fi
echo "Computing ovrlpy for $rows samples; manifest: $manifest"
sbatch --array="0-$((rows-1))%2" --export="ALL,COMPETITOR_OVRLPY_MANIFEST=$manifest" \
  /home/jwinchel/Syndri/scripts/competitors/run_ovrlpy.sbatch
