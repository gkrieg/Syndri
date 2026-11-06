#!/usr/bin/env bash
#SBATCH --job-name=competitor-format
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/competitor-format-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/competitor-format-%A_%a.err
set -euo pipefail
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMPETITOR_FORMAT_MANIFEST")
IFS=$'\t' read -r dataset final <<< "$row"
[[ -n "$dataset" && -f "$final" ]] || { echo "Invalid manifest row" >&2; exit 1; }
exec /home/jwinchel/miniconda3/envs/rctd/bin/python \
  /home/jwinchel/Syndri/scripts/competitors/format_transcripts.py "$dataset" "$final"
