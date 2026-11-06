#!/usr/bin/env bash
#SBATCH --partition=gpu-large
#SBATCH --mem=64G
#SBATCH --job-name=syndri-breast-separators
#SBATCH --output=/home/jwinchel/Syndri/logs/syndri-breast-separators-%j.txt
#SBATCH --time=0-47:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
set -euo pipefail

# Breast Xenium experiment: matched Chromium training reference used by the
# preceding neighbor-token sweep. All sampled neighbor types remain visible.
sampled_neighbors=${K_NEIGHBORS:-10}
if [[ ! "$sampled_neighbors" =~ ^[1-9][0-9]*$ ]]; then
  echo "K_NEIGHBORS must be a positive integer." >&2
  exit 1
fi
gene_context_length=506
max_input_length=$((gene_context_length + sampled_neighbors + 1 + 3))
repo=/home/jwinchel/Syndri
run_name="breast_xenium_k${sampled_neighbors}_separators"
work_dir="$repo/Heimdall/admixture_prediction_results/$run_name"
if [[ -e "$work_dir/milestone.txt" ]]; then
  echo "Existing checkpoint in $work_dir; choose a new work_dir to train from scratch." >&2
  exit 1
fi

module load cuda-12.1
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall
cd "$repo/Heimdall"

python train.py \
  +experiments=admixture_prediction \
  dataset=chromium_breast_GSM95019174_augmented_optimized \
  model=transformer_syndri_medium \
  tasks=admixture_correction \
  scheduler=cosine trainer=default optimizer=AdamW \
  fc=admixture_test fg=identity fe=identity loss=MaskedBCEWithLogitsLoss \
  seed=54 \
  dataset.preprocess_args.data_path=/work/magroup/skrieger/Syndri/data/trainsets/chromium_GSM9509174_breast_matchedcelltypes.h5ad \
  dataset.preprocess_args.fix_admixture=false \
  "dataset.preprocess_args.admixture_args.k_neighbors=$sampled_neighbors" \
  dataset.preprocess_args.admixture_args.only_sample_unexpressed_genes=true \
  "tasks.args.context_length=$gene_context_length" \
  tasks.args.batchsize=32 tasks.args.epochs=1000 tasks.args.splits.col=split \
  trainer.precision=32-true trainer.per_device_batch_size=64 \
  trainer.accumulate_grad_batches=5 trainer.grad_norm_clip=1.0 \
  optimizer.args.lr=0.002 optimizer.args.weight_decay=0.1 \
  loss.args.pos_weight=4.0 \
  fc.args.query_cell_type_token_mode=metadata \
  fc.args.query_cell_type_col=cell_type \
  fc.args.include_query_cell_type_token=true \
  fc.args.include_neighbor_cell_type_tokens=true \
  "fc.args.num_neighbor_cell_type_tokens=$sampled_neighbors" \
  fc.args.include_separator_tokens=true \
  "fc.args.max_input_length=$max_input_length" \
  fe.args.drop_zeros=true \
  "run_name=$run_name" "work_dir=$work_dir"

# Submit: sbatch scripts/slurm/train_breast_xenium_separators.sbatch
# For the older five-neighbor baseline: sbatch --export=ALL,K_NEIGHBORS=5 ...
# Inference: use --config-source checkpoint; separators are restored from
# config.txt. Use --extra-overrides fc.args.query_cell_type_col=harmony_cell_type
# when predicting on the Breast Xenium spatial slices.
