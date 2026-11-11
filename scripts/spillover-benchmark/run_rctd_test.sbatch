#!/bin/bash
#SBATCH --job-name=spillover-rctd-test
#SBATCH --partition=cpu,gpu-large
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/spillover-rctd-test-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/spillover-rctd-test-%A_%a.err
set -euo pipefail

input=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SPILLOVER_TEST_MANIFEST")
[[ -f "$input" ]] || { echo "Missing test slice: $input" >&2; exit 1; }
mkdir -p "$SPILLOVER_RCTD_OUTPUT_DIR"
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate rctd
exec python /home/jwinchel/Syndri/code/rctd_annotation.py \
  --reference "$SPILLOVER_SC_REFERENCE" \
  --spatial "$input" \
  --layer-pattern '^X$' \
  --cell-type-col cell_type \
  --output-dir "$SPILLOVER_RCTD_OUTPUT_DIR"
