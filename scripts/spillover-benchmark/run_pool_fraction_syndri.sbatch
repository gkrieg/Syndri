#!/usr/bin/env bash
#SBATCH --job-name=poolfraction-syndri
#SBATCH --partition=gpu-large
#SBATCH --nodelist=oven-1-3
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jwinchel/Syndri/logs/poolfraction-syndri-%A_%a.out
#SBATCH --error=/home/jwinchel/Syndri/logs/poolfraction-syndri-%A_%a.err
set -euo pipefail
module load cuda-12.1
repo=/home/jwinchel/Syndri
python_bin=/home/jwinchel/miniconda3/envs/heimdall/bin/python
checkpoint_dir="$repo/outputs/SyndriSIM_pool_fraction/syndri_model"
input=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SPILLOVER_TEST_MANIFEST")
[[ -f "$input" ]] || { echo "Missing test slice: $input" >&2; exit 1; }
[[ -f "$checkpoint_dir/model-29.pt" && -f "$checkpoint_dir/training_complete.json" ]] || {
  echo "Matched milestone-29 retraining has not completed: $checkpoint_dir" >&2; exit 1;
}
mkdir -p "$SPILLOVER_SYNDRI_OUTPUT_DIR"
name=$(basename "$input" .h5ad)
[[ ! -f "$SPILLOVER_SYNDRI_OUTPUT_DIR/${name}_syndri_predictions.h5ad" ]] || {
  echo "Prediction output already exists; refusing overwrite" >&2; exit 1;
}
"$python_bin" "$repo/Heimdall/scripts/run_admixture_inference.py" \
  --st-data "$input" --reference-data "$SPILLOVER_MODEL_REFERENCE" \
  --output-dir "$SPILLOVER_SYNDRI_OUTPUT_DIR" --prepared-name "${name}_prepared.h5ad" \
  --output-name "${name}_syndri_predictions.h5ad" --checkpoint-dir "$checkpoint_dir" \
  --milestone 29 --config-source checkpoint --cell-type-col cell_type \
  --coord-cols CenterX_global_px,CenterY_global_px --tissue-col spatial_slice \
  --prob-layer admix_probs --split-value test --batch-size 256
"$python_bin" "$repo/scripts/spillover-benchmark/clean_syndri_predictions.py" \
  "$SPILLOVER_SYNDRI_OUTPUT_DIR/${name}_syndri_predictions.h5ad" \
  --probability-layer admix_probs --counts-layer admixed_counts \
  --output-layer syndri_corrected_counts --threshold 0.5
