#!/usr/bin/env bash
#SBATCH --job-name=cosmx-fibroblast-pooled
#SBATCH --partition=cpu,gpu-small,gpu-large
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/cosmx-fibroblast-pooled-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/cosmx-fibroblast-pooled-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PATH=/home/jwinchel/miniconda3/envs/rctd/bin:$PATH
exec /home/jwinchel/miniconda3/envs/rctd/bin/python -u /home/jwinchel/Syndri/scripts/competitors/execute_cosmx_pooled.py
