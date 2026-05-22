# Admixture MLP Baseline

This is a small baseline outside the Heimdall package that reuses Heimdall's
Hydra configs, `CellRepresentation`, dataloaders, and masked BCE loss.

The model input is:

- dense admixed gene expression reconstructed from the Heimdall admixture batch
- one-hot query cell type
- flattened one-hot neighboring-cell-type tokens

The model outputs one logit per gene. For the existing Heimdall admixture labels,
the script gathers the logits back to each batch's sampled `gene_order` and uses
the configured `MaskedBCEWithLogitsLoss`.

Run from the repository root:

```bash
cd /home/jwinchel/Syndri
python admixture_mlp_baseline/train.py --experiment admixture_prediction --no-wandb trainer.fastdev=true
```

Example with W&B enabled and a learning-rate override:

```bash
cd /home/jwinchel/Syndri
python admixture_mlp_baseline/train.py --experiment admixture_prediction optimizer.args.lr=1e-3
```

With W&B enabled, the trainer logs one payload per epoch with metric names like
`train/loss`, `valid/AUPRC`, and `test/AUROC`, using `epoch` as the chart step.
Non-finite metrics are printed and skipped because W&B will not plot them.

Useful script arguments:

- `--hidden-dims 1024,512`
- `--dropout 0.1`
- `--num-workers 4`
- `--no-wandb`
- `--save-checkpoint`
