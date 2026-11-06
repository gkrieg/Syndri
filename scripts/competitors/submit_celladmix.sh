#!/usr/bin/env bash
set -euo pipefail
base=/work/magroup/skrieger/Syndri/data/datasets
manifest=$(mktemp /home/jwinchel/Syndri/logs/competitor-celladmix-inputs.XXXXXXXX.txt)
for dataset in nsclc_cosmx breast_xenium; do
  for final in "$base/$dataset/results/final/"*.h5ad; do
    [[ -f "$final" ]] || continue
    name=$(basename "$final" .h5ad)
    out="$base/$dataset/results/competitors/celladmix/factorization/$name"
    [[ -e "$out/assignments.csv" || -e "$out/factor_summary.csv" || -e "$out/factor_loadings.csv" ]] && continue
    printf '%s\n' "$final" >> "$manifest"
  done
done
rows=$(wc -l < "$manifest")
if (( rows == 0 )); then echo 'All cellAdmix factorizations already exist'; exit 0; fi
echo "Factorizing $rows samples with cellAdmix; manifest: $manifest"
sbatch --array="0-$((rows-1))%1" --export="ALL,COMPETITOR_CELLADMIX_MANIFEST=$manifest" \
  /home/jwinchel/Syndri/scripts/competitors/run_celladmix.sbatch
