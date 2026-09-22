#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH --mem=48Gb
#SBATCH --job-name=count-summaries
#SBATCH --output=/home/jwinchel/Syndri/logs/count-summaries-%A_%a.txt
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=2
set -euo pipefail

: "${METRICS_LIST_FILE:?Set METRICS_LIST_FILE to a newline-delimited CSV manifest}"
mapfile -t files < "$METRICS_LIST_FILE"
task=${SLURM_ARRAY_TASK_ID:-0}
if (( task < 0 || task >= ${#files[@]} )); then
  echo "Array index $task is outside the metrics manifest." >&2
  exit 1
fi

source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-heimdall}"

IFS=$'\t' read -r metrics_csv label_key removal_group_key <<< "${files[$task]}"
args=(--metrics-csv "$metrics_csv")
[[ -n "${label_key:-}" ]] && args+=(--label-key "$label_key")
[[ -n "${removal_group_key:-}" ]] && args+=(--removal-group-key "$removal_group_key")

/home/jwinchel/miniconda3/envs/heimdall/bin/python \
  /home/jwinchel/Syndri/eval/backfill_count_summaries.py "${args[@]}"
