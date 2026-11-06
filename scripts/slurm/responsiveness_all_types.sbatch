#!/usr/bin/env bash
#SBATCH --partition=cpu,gpu-small,gpu-large
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/response-alltypes-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/response-alltypes-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
if [[ "$PANEL_MODE" == infer ]]; then module load cuda-12.1; fi
cd /home/jwinchel/Syndri
/home/jwinchel/miniconda3/envs/heimdall/bin/python -u "$PANEL_RUN/all_types_panel.py" "$PANEL_MODE" "$PANEL_RUN" --index "${SLURM_ARRAY_TASK_ID:-0}"
