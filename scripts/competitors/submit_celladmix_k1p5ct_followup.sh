#!/usr/bin/env bash
set -euo pipefail
repo=/home/jwinchel/Syndri
manifest=$repo/logs/celladmix-factorization-k1p5ct-20261005-123021.tsv
batch=$repo/scripts/competitors/celladmix_k1p5ct_followup.sbatch
test -s "$manifest"
test -f "$batch"
n=$(wc -l < "$manifest")
test "$n" -eq 27
range="0-$((n-1))%1"
annotation=$(sbatch --parsable --job-name=celladmix-annotation-k1p5ct \
  --array="$range" --cpus-per-task=4 \
  --export="ALL,CELLADMIX_FOLLOWUP_MANIFEST=$manifest,CELLADMIX_STAGE=annotation" "$batch")
echo "annotation=$annotation"
correction=$(sbatch --parsable --job-name=celladmix-correction-k1p5ct \
  --array="$range" --dependency="afterok:$annotation" \
  --export="ALL,CELLADMIX_FOLLOWUP_MANIFEST=$manifest,CELLADMIX_STAGE=correction" "$batch")
echo "correction=$correction (afterok:$annotation)"
metrics=$(sbatch --parsable --job-name=celladmix-metrics-k1p5ct \
  --array="$range" --dependency="afterok:$correction" \
  --export="ALL,CELLADMIX_FOLLOWUP_MANIFEST=$manifest,CELLADMIX_STAGE=metrics" "$batch")
echo "metrics=$metrics (afterok:$correction)"
