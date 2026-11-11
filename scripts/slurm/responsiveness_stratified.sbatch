#!/usr/bin/env bash
#SBATCH --partition=cpu,gpu-small,gpu-large
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/response-stratified-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/response-stratified-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLBACKEND=Agg
python=/home/jwinchel/miniconda3/envs/heimdall/bin/python
cd "$PANEL_RUN"
case "$PANEL_MODE" in
 prepare) "$python" -u "$PANEL_RUN/stratified_panel.py" prepare "$PANEL_RUN" --index "$SLURM_ARRAY_TASK_ID" ;;
 infer) module load cuda-12.1; "$python" -u "$PANEL_RUN/stratified_panel.py" infer "$PANEL_RUN" --index "$SLURM_ARRAY_TASK_ID" ;;
 finalize) "$python" -u "$PANEL_RUN/finalize_stratified_panel.py" "$PANEL_RUN" ;;
esac
