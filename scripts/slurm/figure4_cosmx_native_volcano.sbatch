#!/usr/bin/env bash
#SBATCH --job-name=cosmx-native-volcano
#SBATCH --partition=cpu,gpu-small,gpu-large
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/cosmx-native-volcano-%j.out
#SBATCH --error=/home/jwinchel/Syndri/logs/cosmx-native-volcano-%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export PATH=/home/jwinchel/miniconda3/envs/rctd/bin:$PATH
exec /home/jwinchel/miniconda3/envs/rctd/bin/python -u /home/jwinchel/Syndri/scripts/competitors/execute_cosmx_native_volcano.py
