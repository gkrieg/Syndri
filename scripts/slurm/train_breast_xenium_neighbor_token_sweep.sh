#!/usr/bin/env bash
#SBATCH --partition=gpu-large
#SBATCH --mem=64G
#SBATCH --job-name=syndri-breast-neighbor-tokens
#SBATCH --output=/home/jwinchel/Syndri/logs/syndri-breast-neighbor-tokens-%A_%a.txt
#SBATCH --time=0-47:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%2

set -euo pipefail

repo=/home/jwinchel/Syndri
output_root="$repo/Heimdall/admixture_prediction_results/breast_xenium_10neighbor_token_sweep"
visible_counts=(10 8 6 4 2 1)
gene_context_length=506

# Run this file with bash to submit the six-job array. A Slurm task runs the
# training command below for its assigned visible-neighbor count.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  mkdir -p "$repo/logs"
  sbatch "$0"
  exit
fi

task_id=$SLURM_ARRAY_TASK_ID
if (( task_id < 0 || task_id >= ${#visible_counts[@]} )); then
  echo "Unexpected SLURM_ARRAY_TASK_ID: $task_id" >&2
  exit 1
fi

visible_neighbors=${visible_counts[$task_id]}
sampled_neighbors=10
query_tokens=1
max_input_length=$((gene_context_length + visible_neighbors + query_tokens))
run_name="breast_xenium_k10_visible${visible_neighbors}"
work_dir="$output_root/$run_name"

if [[ -e "$work_dir/milestone.txt" ]]; then
  echo "Existing checkpoint in $work_dir; refusing to resume a run intended to start from scratch." >&2
  exit 1
fi

module load cuda-12.1
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall
cd "$repo/Heimdall"

echo "Sampling admixture from $sampled_neighbors nearest neighbors."
echo "Exposing $visible_neighbors nearest-neighbor cell types plus the query cell type."
echo "Gene positions: $gene_context_length; total input positions: $max_input_length"
echo "Output: $work_dir"

python train.py \
  +experiments=admixture_prediction \
  dataset=chromium_breast_GSM95019174_augmented_optimized \
  model=transformer_syndri_medium \
  tasks=admixture_correction \
  scheduler=cosine \
  trainer=default \
  optimizer=AdamW \
  fc=admixture_test \
  fg=identity \
  fe=identity \
  loss=MaskedBCEWithLogitsLoss \
  seed=54 \
  dataset.preprocess_args.data_path=/work/magroup/skrieger/Syndri/data/trainsets/chromium_GSM9509174_breast_matchedcelltypes.h5ad \
  dataset.preprocess_args.fix_admixture=false \
  "dataset.preprocess_args.admixture_args.k_neighbors=$sampled_neighbors" \
  dataset.preprocess_args.admixture_args.only_sample_unexpressed_genes=true \
  "tasks.args.context_length=$gene_context_length" \
  tasks.args.batchsize=32 \
  tasks.args.epochs=1000 \
  tasks.args.splits.col=split \
  trainer.precision=32-true \
  trainer.per_device_batch_size=64 \
  trainer.accumulate_grad_batches=5 \
  trainer.grad_norm_clip=1.0 \
  optimizer.args.lr=0.002 \
  optimizer.args.weight_decay=0.1 \
  loss.args.pos_weight=4.0 \
  fc.args.query_cell_type_token_mode=metadata \
  fc.args.query_cell_type_col=cell_type \
  fc.args.include_query_cell_type_token=true \
  fc.args.include_neighbor_cell_type_tokens=true \
  "fc.args.num_neighbor_cell_type_tokens=$visible_neighbors" \
  "fc.args.max_input_length=$max_input_length" \
  fe.args.drop_zeros=true \
  "run_name=$run_name" \
  "work_dir=$work_dir"
