#!/bin/bash
#SBATCH --job-name=spillover-syndri-test
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/spillover-syndri-test-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/spillover-syndri-test-%A_%a.err
set -euo pipefail

repo=/home/jwinchel/Syndri
input=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SPILLOVER_TEST_MANIFEST")
[[ -f "$input" ]] || { echo "Missing test slice: $input" >&2; exit 1; }
mkdir -p "$SPILLOVER_SYNDRI_OUTPUT_DIR"
name=$(basename "$input" .h5ad)
module load cuda-12.1
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall
python "$repo/Heimdall/scripts/run_admixture_inference.py" \
  --st-data "$input" \
  --reference-data "$SPILLOVER_MODEL_REFERENCE" \
  --output-dir "$SPILLOVER_SYNDRI_OUTPUT_DIR" \
  --prepared-name "${name}_prepared.h5ad" \
  --output-name "${name}_syndri_predictions.h5ad" \
  --checkpoint-dir /home/jwinchel/Syndri/Heimdall/admixture_prediction_results/nsclc_cosmx_train_val_k9_tau0_0p95 \
  --milestone 29 \
  --config-source checkpoint \
  --cell-type-col cell_type \
  --coord-cols CenterX_global_px,CenterY_global_px \
  --tissue-col spatial_slice \
  --prob-layer admix_probs \
  --split-value test \
  --batch-size "${SYNDRI_BATCH_SIZE:-256}"

python "$repo/scripts/spillover-benchmark/clean_syndri_predictions.py" \
  "$SPILLOVER_SYNDRI_OUTPUT_DIR/${name}_syndri_predictions.h5ad" \
  --probability-layer admix_probs \
  --counts-layer admixed_counts \
  --output-layer syndri_corrected_counts \
  --threshold "${SYNDRI_PROBABILITY_THRESHOLD:-0.5}"
