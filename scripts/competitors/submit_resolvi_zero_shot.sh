#!/usr/bin/env bash
set -euo pipefail
base=/work/magroup/skrieger/Syndri/data/datasets
manifest=$(mktemp /home/jwinchel/Syndri/logs/competitor-resolvi-zero-shot-inputs.XXXXXXXX.txt)
for dataset in agingbrain_merfish brain_merfish nsclc_cosmx breast_xenium; do
  for final in "$base/$dataset/results/final/"*.h5ad; do
    [[ -f "$final" ]] || continue
    output="$base/$dataset/results/competitors/resolvi_zero_shot/$(basename "$final")"
    [[ -e "$output" ]] && continue
    printf '%s\n' "$final" >> "$manifest"
  done
done
rows=$(wc -l < "$manifest")
if (( rows == 0 )); then echo 'All zero-shot ResolVI outputs already exist'; exit 0; fi
echo "Queued inputs: $rows; manifest: $manifest"
sbatch --array="0-$((rows - 1))%2" \
  --export="ALL,RESOLVI_ZERO_SHOT_MANIFEST=$manifest" \
  /home/jwinchel/Syndri/scripts/competitors/run_resolvi_zero_shot.sbatch
