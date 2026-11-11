#!/usr/bin/env bash
#SBATCH --partition=cpu,gpu-small,gpu-large
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/response-selected-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/response-selected-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLBACKEND=Agg
python=/home/jwinchel/miniconda3/envs/heimdall/bin/python
cd /home/jwinchel/Syndri
case "$PANEL_MODE" in
 infer) module load cuda-12.1; "$python" -u "$PANEL_RUN/all_types_panel.py" infer "$PANEL_RUN" --index "$SLURM_ARRAY_TASK_ID" ;;
 prepare) "$python" -u "$PANEL_RUN/all_types_panel.py" prepare "$PANEL_RUN" --index "$SLURM_ARRAY_TASK_ID" ;;
 variation) "$python" -u "$PANEL_RUN/context_variability.py" "$PANEL_RUN" "$SLURM_ARRAY_TASK_ID" ;;
 finalize) "$python" -u "$PANEL_RUN/finalize_selected_panel.py" "$PANEL_RUN" ;;
esac
