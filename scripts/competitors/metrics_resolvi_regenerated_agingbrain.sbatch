#!/usr/bin/env bash
#SBATCH --job-name=metrics-agingbrain-resolvi-regenerated
#SBATCH --partition=cpu,gpu-small,gpu-large
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/metrics-agingbrain-resolvi-regenerated-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/metrics-agingbrain-resolvi-regenerated-%A_%a.err
set -euo pipefail

: "${RESOLVI_METRICS_MANIFEST:?Set RESOLVI_METRICS_MANIFEST}"
: "${RESOLVI_METRICS_OUTPUT_ROOT:?Set RESOLVI_METRICS_OUTPUT_ROOT}"

input=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RESOLVI_METRICS_MANIFEST")
[[ -f "$input" ]] || { echo "Missing ResolVI input: $input" >&2; exit 1; }
sample=$(basename "$input" .h5ad)
output="$RESOLVI_METRICS_OUTPUT_ROOT/$sample"
if [[ -e "$output/non_tcell_metrics.csv" || -e "$output/non_tcell_metrics_config.json" ]]; then
  echo "Refusing to overwrite metrics in $output" >&2
  exit 1
fi
mkdir -p "$output"

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall

python eval/metrics_eval.py \
  --input-files "$input" \
  --output-dir "$output" \
  --cell-type-output non_tcell_metrics.csv \
  --config-output non_tcell_metrics_config.json \
  --label-key cell_type \
  --exclude-labels nan \
  --additional-counts-layer resolvi_corrected_counts \
  --additional-counts-case-label ResolVI \
  --only-additional-counts \
  --sample-frac 1.0 \
  --sample-random-state 0 \
  --n-pcs 50 \
  --library-size-target-sum 10000 \
  --leiden-resolution 1 \
  --leiden-optimize-resolution false \
  --leiden-n-neighbors 15 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-4}" \
  --random-state 0 \
  --skip-tcell-contamination \
  --skip-tcell-neighborhood-metrics \
  --skip-tcell-cosine-similarity \
  --skip-cell-type-profiles

python - "$output/non_tcell_metrics.csv" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
table = pd.read_csv(path)
required = ["silhouette_score", "calinski_harabasz", "davies_bouldin",
            "leiden_nmi_vs_label", "leiden_adjusted_rand_vs_label",
            "median_genes_per_cell", "median_transcripts_per_cell"]
if (len(table) != 1 or table.iloc[0]["case"] != "ResolVI"
        or table["error"].fillna("").astype(str).str.len().gt(0).any()
        or table[required].isna().any().any()):
    raise SystemExit(f"Incomplete ResolVI metrics: {path}")
print(f"Completed ResolVI metrics: {path}")
PY
