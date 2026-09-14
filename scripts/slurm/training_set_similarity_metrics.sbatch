#!/bin/bash
#SBATCH -p cpu,gpu-small,gpu-large
#SBATCH --mem=48Gb
#SBATCH --job-name syndri-training-set-similarity
#SBATCH --output=/home/jwinchel/Syndri/logs/training-set-similarity-%A_%a.txt
#SBATCH -t 1-23:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail

cd /home/jwinchel/Syndri
mkdir -p /home/jwinchel/Syndri/logs

source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-heimdall}"

MANIFEST=${MANIFEST:-}
OUTPUT_DIR=${OUTPUT_DIR:-/compute/oven-0-13/jwinchel/spatial_adata/training_set_similarity_metrics}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-training_set_similarity}
MANIFEST_DELIMITER=${MANIFEST_DELIMITER:-auto}
EXTRA_ARGS=${EXTRA_ARGS:-}

if [[ -z "$MANIFEST" ]]; then
  echo "MANIFEST is required. Provide a CSV/TSV with spatial_path and single_cell_path columns." >&2
  exit 1
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "Running training_set_similarity_metrics.py"
echo "  manifest: $MANIFEST"
echo "  manifest index: $TASK_ID"
echo "  output dir: $OUTPUT_DIR"
echo "  output prefix: $OUTPUT_PREFIX"
echo "  cpus: ${SLURM_CPUS_PER_TASK:-1}"

# EXTRA_ARGS is intentionally word-split so simple additional flags can be supplied via sbatch --export.
# shellcheck disable=SC2086
python /home/jwinchel/Syndri/eval/training_set_similarity_metrics.py \
  --manifest "$MANIFEST" \
  --manifest-index "$TASK_ID" \
  --manifest-delimiter "$MANIFEST_DELIMITER" \
  --output-dir "$OUTPUT_DIR/task_${TASK_ID}" \
  --output-prefix "$OUTPUT_PREFIX" \
  $EXTRA_ARGS
