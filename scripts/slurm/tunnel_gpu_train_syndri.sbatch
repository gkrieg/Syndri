#!/bin/bash
#SBATCH -p gpu-large
#SBATCH --nodes=1
#SBATCH -w oven-0-13
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64Gb
#SBATCH --job-name vscode
#SBATCH --output vlog-%J.txt
#SBATCH -t 0-47:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
module load cuda-12.1
# activate conda environnment
# cd to correct directory
cd ~
# python run command.py`
echo "********************************************************************"
echo "Starting code-server in Slurm"
echo "Environment information:"
echo "Date:" $(date)
echo "Allocated node:" $(hostname)
echo "Node IP:" $(ip a | grep 131.159)
echo "Path:" $(pwd)
echo "********************************************************************"
./code tunnel
