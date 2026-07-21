#!/bin/bash
#SBATCH -p gpu-large
#SBATCH --mem=32Gb
#SBATCH --job-name poswt_sweep
#SBATCH --output /home/jwinchel/Syndri/logs/pw-sweep-%A_%a.txt
#SBATCH -t 0-47:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

module load cuda-12.1

cd ~/Syndri/Heimdall
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall

POS_WEIGHTS=(3.0 5.0 8.0)
POS_WEIGHT=${POS_WEIGHTS[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT=${EXPERIMENT:-admixture_prediction}
FG=${FG:-identity}
MODEL=${MODEL:-transformer_syndri_medium}
POS_WEIGHT_TAG=${POS_WEIGHT//./p}
FG_TAG=${FG//./_}
MODEL_TAG=${MODEL//./_}

python train.py \
  +experiments=$EXPERIMENT \
  fg=$FG \
  model=$MODEL \
  loss.args.pos_weight=$POS_WEIGHT \
  run_name=admixture_${FG_TAG}_${MODEL_TAG}_posweight_${POS_WEIGHT_TAG}_nodropout \
  work_dir=admixture_prediction_results/${FG_TAG}_${MODEL_TAG}_posweight_${POS_WEIGHT_TAG}_nodropout
