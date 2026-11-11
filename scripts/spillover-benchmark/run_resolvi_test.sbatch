#!/bin/bash
#SBATCH --job-name=spillover-resolvi-test
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/spillover-resolvi-test-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/spillover-resolvi-test-%A_%a.err
set -euo pipefail

input=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SPILLOVER_TEST_MANIFEST")
[[ -f "$input" ]] || { echo "Missing test slice: $input" >&2; exit 1; }
mkdir -p "$SPILLOVER_RESOLVI_OUTPUT_ROOT"
exec /home/jwinchel/.venv/resolvi/bin/python \
  /home/jwinchel/Syndri/scripts/competitors/run_resolvi.py "$input" \
  --dataset nsclc_cosmx --epochs "${RESOLVI_EPOCHS:-100}" \
  --samples "${RESOLVI_SAMPLES:-3}" --output-root "$SPILLOVER_RESOLVI_OUTPUT_ROOT"
