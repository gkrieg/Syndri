#!/usr/bin/env bash
#SBATCH --job-name=competitor-ovrlpy
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/competitor-ovrlpy-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/competitor-ovrlpy-%A_%a.err
set -euo pipefail
final=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMPETITOR_OVRLPY_MANIFEST")
[[ -f "$final" ]] || { echo 'Missing final AnnData' >&2; exit 1; }
exec /home/jwinchel/miniconda3/envs/ovrlpy/bin/python \
  /home/jwinchel/Syndri/scripts/competitors/run_ovrlpy.py "$final" --workers 8
