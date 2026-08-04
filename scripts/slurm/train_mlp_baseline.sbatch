#!/bin/bash
#SBATCH -p gpu-large
#SBATCH --mem=32Gb
#SBATCH --job-name mlp-baseline-train
#SBATCH --output /home/jwinchel/Syndri/logs/mlp-baseline-train-%J.txt
#SBATCH -t 0-47:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH
module load cuda-12.1
# activate conda environnment
# cd to correct directory
cd ~/Syndri
# python run command.py`
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall
python admixture_mlp_baseline/train.py --experiment $EXPERIMENT --save-checkpoint

# sbatch --export=EXPERIMENT=admixture_prediction Syndri/scripts/slurm/train_mlp_baseline.sbatch
