#!/usr/bin/env bash
#SBATCH --partition=cpu,gpu-small,gpu-large
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/all-type-cosine-final-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/all-type-cosine-final-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLBACKEND=Agg
/home/jwinchel/miniconda3/envs/heimdall/bin/python -u "$COSINE_RUN/finalize.py" "$COSINE_RUN"
