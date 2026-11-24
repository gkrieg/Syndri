#!/usr/bin/env bash
#SBATCH --job-name=fig3-transcript-localization
#SBATCH --partition=cpu
#SBATCH --nodelist=oven-1-24
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/fig3-transcript-localization-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/fig3-transcript-localization-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
cd /home/jwinchel/Syndri
exec /home/jwinchel/miniconda3/envs/heimdall/bin/python -u \
  scripts/spillover-benchmark/render_pool_fraction_notebook.py \
  --notebook figures/Figure3_SyndriSIM_TranscriptLocalization.ipynb
