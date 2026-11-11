#!/usr/bin/env bash
#SBATCH --job-name=competitor-resolvi
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/competitor-resolvi-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/competitor-resolvi-%A_%a.err
set -euo pipefail
final=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMPETITOR_RESOLVI_MANIFEST")
[[ -f "$final" ]] || { echo 'Missing final AnnData' >&2; exit 1; }
args=(--epochs "${RESOLVI_EPOCHS:-100}" --samples "${RESOLVI_SAMPLES:-3}")
if [[ -n "${RESOLVI_OUTPUT_ROOT:-}" ]]; then args+=(--output-root "$RESOLVI_OUTPUT_ROOT"); fi
exec /home/jwinchel/.venv/resolvi/bin/python \
  /home/jwinchel/Syndri/scripts/competitors/run_resolvi.py "$final" "${args[@]}"
