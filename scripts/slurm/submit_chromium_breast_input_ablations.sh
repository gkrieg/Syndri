#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/jwinchel/Syndri
batch_script=${repo_root}/scripts/slurm/train_syndri_ablations.sbatch
output_root=${repo_root}/Heimdall/admixture_prediction_results/chromium_breast_GSM95019174_augmented_matchedcelltypes_optimized_unexpressedgenes_withquerycelltype_input_ablations

# The reference run used max_input_length=512 with five neighbor tokens and
# one query token, leaving 506 positions for genes. The array job adjusts the
# total length per ablation so all three jobs retain that same 506-gene budget.
gene_context_length=506
neighbor_token_count=5

mkdir -p "${repo_root}/logs" "${output_root}"

# Pin the training configuration recorded in:
# Heimdall/admixture_prediction_results/
# chromium_breast_GSM95019174_augmented_matchedcelltypes_optimized_
# unexpressedgenes_withquerycelltype/config.txt
hydra_overrides=(
  "dataset=chromium_breast_GSM95019174_augmented_optimized"
  "model=transformer_syndri_medium"
  "tasks=admixture_correction"
  "scheduler=cosine"
  "trainer=default"
  "optimizer=AdamW"
  "fc=admixture_test"
  "fg=identity"
  "fe=identity"
  "loss=MaskedBCEWithLogitsLoss"
  "seed=54"
  "dataset.preprocess_args.fix_admixture=false"
  "dataset.preprocess_args.admixture_args.k_neighbors=5"
  "dataset.preprocess_args.admixture_args.only_sample_unexpressed_genes=true"
  "tasks.args.context_length=506"
  "tasks.args.batchsize=32"
  "tasks.args.epochs=1000"
  "tasks.args.splits.col=split"
  "trainer.precision=32-true"
  "trainer.per_device_batch_size=64"
  "trainer.accumulate_grad_batches=5"
  "trainer.grad_norm_clip=1.0"
  "optimizer.args.lr=0.002"
  "optimizer.args.weight_decay=0.1"
  "loss.args.pos_weight=4.0"
  "fc.args.query_cell_type_token_mode=metadata"
  "fc.args.query_cell_type_col=cell_type"
  "fe.args.drop_zeros=true"
)

sbatch \
  --job-name=brca-input-ablation \
  --export="ALL,EXPERIMENT=admixture_prediction,OUTPUT_ROOT=${output_root},GENE_CONTEXT_LENGTH=${gene_context_length},NEIGHBOR_TOKEN_COUNT=${neighbor_token_count}" \
  "${batch_script}" \
  "${hydra_overrides[@]}"
