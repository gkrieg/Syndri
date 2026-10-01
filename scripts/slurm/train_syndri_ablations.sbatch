#!/usr/bin/env bash
#SBATCH -p gpu-large
#SBATCH --mem=64G
#SBATCH --job-name=syndri-ablation
#SBATCH --output=/home/jwinchel/Syndri/logs/syndri-ablation-%A_%a.txt
#SBATCH --time=0-47:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

set -euo pipefail

experiment=${EXPERIMENT:-admixture_prediction}
output_root=${OUTPUT_ROOT:-/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/ablations}
gene_context_length=${GENE_CONTEXT_LENGTH:-1532}
neighbor_token_count=${NEIGHBOR_TOKEN_COUNT:-3}

if [[ ! "${gene_context_length}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GENE_CONTEXT_LENGTH must be a positive integer: ${gene_context_length}" >&2
  exit 1
fi
if [[ ! "${neighbor_token_count}" =~ ^[0-9]+$ ]]; then
  echo "NEIGHBOR_TOKEN_COUNT must be a nonnegative integer: ${neighbor_token_count}" >&2
  exit 1
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    ablation_name=query_only
    include_query=true
    include_neighbors=false
    ;;
  1)
    ablation_name=neighbors_only
    include_query=false
    include_neighbors=true
    ;;
  2)
    ablation_name=no_cell_type_tokens
    include_query=false
    include_neighbors=false
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

num_prefix_tokens=0
if [[ "${include_query}" == true ]]; then
  ((num_prefix_tokens += 1))
fi
if [[ "${include_neighbors}" == true ]]; then
  ((num_prefix_tokens += neighbor_token_count))
fi
max_input_length=$((gene_context_length + num_prefix_tokens))

module load cuda-12.1
source /home/jwinchel/miniconda3/etc/profile.d/conda.sh
conda activate heimdall

cd /home/jwinchel/Syndri/Heimdall

echo "Training Syndri ablation: ${ablation_name}"
echo "  include query cell-type token: ${include_query}"
echo "  include neighbor cell-type tokens: ${include_neighbors}"
echo "  gene context length: ${gene_context_length}"
echo "  total input length: ${max_input_length}"
echo "  output: ${output_root}/${ablation_name}"

python train.py \
  "+experiments=${experiment}" \
  "fc.args.include_query_cell_type_token=${include_query}" \
  "fc.args.include_neighbor_cell_type_tokens=${include_neighbors}" \
  "fc.args.max_input_length=${max_input_length}" \
  "run_name=${ablation_name}" \
  "work_dir=${output_root}/${ablation_name}" \
  "$@"

# Submit all three conditions with:
# sbatch /home/jwinchel/Syndri/scripts/slurm/train_syndri_ablations.sbatch
