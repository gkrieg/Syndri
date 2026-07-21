#!/bin/bash
#SBATCH -p gpu-large
#SBATCH --mem=64Gb
#SBATCH --job-name pw-sweep
#SBATCH --output pw-sweep-%A_%a.txt
#SBATCH -t 0-47:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

module load cuda-12.1

cd ~/Heimdall
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall

POS_WEIGHTS=(1.0 4.0 9.7)
POS_WEIGHT=${POS_WEIGHTS[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT=${EXPERIMENT:-admixture_prediction}
POS_WEIGHT_TAG=${POS_WEIGHT//./p}

python train.py \
  +experiments=$EXPERIMENT \
  loss.args.pos_weight=$POS_WEIGHT \
  run_name=admixture_posweight_${POS_WEIGHT_TAG} \
  work_dir=admixture_prediction_results/posweight_${POS_WEIGHT_TAG}
