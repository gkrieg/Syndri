#!/usr/bin/env bash
#SBATCH --job-name=competitor-resolvi-zero-shot
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/competitor-resolvi-zero-shot-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/competitor-resolvi-zero-shot-%A_%a.err
set -euo pipefail
final=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RESOLVI_ZERO_SHOT_MANIFEST")
[[ -f "$final" ]] || { echo "Missing final AnnData: $final" >&2; exit 1; }
exec /home/jwinchel/.venv/resolvi/bin/python \
  /home/jwinchel/Syndri/scripts/competitors/run_resolvi_zero_shot.py "$final" \
  --epochs "${RESOLVI_EPOCHS:-100}" --samples "${RESOLVI_SAMPLES:-3}"
