#!/usr/bin/env bash
# Recompute the standard non-T-cell metric sweep in an isolated QC comparison.
#SBATCH --partition=cpu,gpu-small,gpu-large
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/syndri-qc-metrics-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/syndri-qc-metrics-%A_%a.err
set -euo pipefail

: "${INPUT_LIST_FILE:?}"
: "${OUTPUT_ROOT:?}"
: "${LABEL_KEY:?}"
: "${SEPARATION_MIN_COUNTS:?}"
: "${SEPARATION_MIN_GENES:?}"
mapfile -t inputs < "$INPUT_LIST_FILE"
input=${inputs[${SLURM_ARRAY_TASK_ID:?}]}
test -s "$input"
name=$(basename "$input" .h5ad)
output=$OUTPUT_ROOT/$name
if [[ -e "$output" ]]; then
  echo "Refusing to overwrite existing result directory: $output" >&2
  exit 1
fi
mkdir -p "$output"

/home/jwinchel/miniconda3/envs/heimdall/bin/python \
  /home/jwinchel/Syndri/eval/metrics_eval.py \
  --input-files "$input" --output-dir "$output" \
  --cell-type-output non_tcell_metrics.csv \
  --config-output non_tcell_metrics_config.json \
  --label-key "$LABEL_KEY" --exclude-labels nan \
  --admix-probs-layer syndri_admix_probs_X \
  --additional-counts-layer split_purified_counts_X \
  --additional-counts-case-label SPLIT \
  --target-removed-fractions 0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5 \
  --cleaning-mode probability_scale --cell-type-specific-removal \
  --removal-group-key "$LABEL_KEY" --probability-transform power \
  --probability-power 2 --clean-prob-threshold 0.5 \
  --separation-min-counts "$SEPARATION_MIN_COUNTS" \
  --separation-min-genes "$SEPARATION_MIN_GENES" \
  --sample-frac 1.0 --sample-random-state 0 \
  --n-pcs 50 --library-size-target-sum 10000 \
  --leiden-resolution 1 --leiden-optimize-resolution false \
  --leiden-n-neighbors 15 --n-jobs "${SLURM_CPUS_PER_TASK:-4}" \
  --random-state 0 --skip-tcell-contamination \
  --skip-tcell-neighborhood-metrics --skip-tcell-cosine-similarity \
  --skip-cell-type-profiles

/home/jwinchel/miniconda3/envs/heimdall/bin/python - "$output/non_tcell_metrics.csv" "$SEPARATION_MIN_COUNTS" "$SEPARATION_MIN_GENES" <<'PY'
import pandas as pd
import sys

path, min_counts, min_genes = sys.argv[1:]
table = pd.read_csv(path)
cases = {'admixed', 'SPLIT', *(
    f'cleaned_admixed_removed_frac_{fraction}'
    for fraction in ('0.01', '0.02', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5')
)}
required = ['silhouette_score', 'calinski_harabasz', 'davies_bouldin',
            'leiden_nmi_vs_label', 'leiden_adjusted_rand_vs_label',
            'median_genes_per_cell', 'median_transcripts_per_cell',
            'n_cells_before_qc', 'n_cells_dropped_qc']
if (len(table) != 10 or set(table['case']) != cases
        or table['error'].fillna('').astype(str).str.len().any()
        or table[required].isna().any().any()):
    raise SystemExit(f'Incomplete or failed metrics: {path}')
if float(min_counts) == 0 and int(min_genes) == 0 and table['n_cells_dropped_qc'].sum() != 0:
    raise SystemExit(f'Unfiltered run unexpectedly dropped cells: {path}')
print(f'Validated {len(table)} cases: {path}')
PY
