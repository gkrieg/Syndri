#!/usr/bin/env bash
#SBATCH -p cpu,gpu-small,gpu-large
#SBATCH --mem=96Gb
#SBATCH --job-name=syndri-final-metrics
#SBATCH --output=/home/jwinchel/Syndri/logs/syndri-final-metrics-%A_%a.txt
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=8
set -euo pipefail

cd /home/jwinchel/Syndri
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-heimdall}"

: "${INPUT_LIST_FILE:?Set INPUT_LIST_FILE}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${LABEL_KEY:?Set LABEL_KEY}"
mapfile -t files < "$INPUT_LIST_FILE"
task=${SLURM_ARRAY_TASK_ID:-0}
if (( task < 0 || task >= ${#files[@]} )); then
  echo "Array index $task is outside the input manifest." >&2
  exit 1
fi
input=${files[$task]}
sample=$(basename "$input" .h5ad)
output="$OUTPUT_ROOT/$sample"
fractions=${TARGET_REMOVED_FRACTIONS:-0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5}
mkdir -p "$output"

# Each call evaluates raw X, the removal sweep, and SPLIT on the same cells.
# Empty exclude-labels includes every annotated cell type, including malignant
# cells. Skipping the T-cell metric family does not exclude T cells themselves.
python /home/jwinchel/Syndri/eval/metrics_eval.py \
  --input-files "$input" \
  --output-dir "$output" \
  --cell-type-output non_tcell_metrics.csv \
  --config-output non_tcell_metrics_config.json \
  --label-key "$LABEL_KEY" \
  --exclude-labels "${EXCLUDE_LABELS:-}" \
  --admix-probs-layer syndri_admix_probs_X \
  --additional-counts-layer split_purified_counts_X \
  --additional-counts-case-label SPLIT \
  --target-removed-fractions "$fractions" \
  --cleaning-mode probability_scale \
  --cell-type-specific-removal \
  --removal-group-key "$LABEL_KEY" \
  --probability-transform power \
  --probability-power 2 \
  --sample-frac "${SAMPLE_FRAC:-1.0}" \
  --sample-random-state 0 \
  --n-pcs 50 \
  --library-size-target-sum 10000 \
  --leiden-resolution 1 \
  --leiden-optimize-resolution false \
  --leiden-n-neighbors 15 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
  --random-state 0 \
  --skip-tcell-contamination \
  --skip-tcell-neighborhood-metrics \
  --skip-tcell-cosine-similarity \
  --skip-cell-type-profiles

# The evaluator records individual failures in CSV instead of raising them.
# Surface these as failed Slurm tasks rather than reporting false success.
python - "$output/non_tcell_metrics.csv" "$fractions" <<'PY'
import sys
import pandas as pd

path, fractions = sys.argv[1:]
table = pd.read_csv(path)
expected = 2 + len(fractions.split(","))
metrics = ["silhouette_score", "calinski_harabasz", "davies_bouldin",
           "leiden_nmi_vs_label", "leiden_adjusted_rand_vs_label"]
errors = table["error"].fillna("").astype(str).str.len() > 0
if len(table) != expected or errors.any() or table[metrics].isna().any().any():
    print(table[["case", "error"]].to_string(index=False))
    raise SystemExit(f"Incomplete metrics in {path}; expected {expected} successful cases.")
print(f"Completed {len(table)} cases with all five metrics: {path}")
PY
