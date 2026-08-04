#!/bin/bash
#SBATCH -p gpu-large
#SBATCH --mem=64Gb
#SBATCH --job-name syndri-train
#SBATCH --output /home/jwinchel/Syndri/logs/syndri-train-%J.txt
#SBATCH -t 0-47:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH
module load cuda-12.1
# activate conda environnment
# cd to correct directory
cd ~/Syndri/Heimdall
# python run command.py`
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall
python train.py +experiments=$EXPERIMENT

# sbatch --export=EXPERIMENT=admixture_prediction Syndri/scripts/slurm/train_syndri.sbatch