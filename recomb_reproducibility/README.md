# Overview of Syndri

Syndri identifies transcript admixture in spatial transcriptomics data using gene expression and cellular context. Its transformer predicts a probability of admixture for each cell–gene entry, using the query cell's expression profile, cell type, and neighboring cell types.

The workflow consists of two main parts:

1. **Reference-based training:** map scRNA-seq reference expression to spatial coordinates, add synthetic admixture from neighboring cells, and train the model using the known added-expression labels.
2. **Spatial prediction and correction:** apply the trained model to spatial counts and use the predicted probabilities to identify and remove likely admixture.

The model and training code use the [Heimdall framework](../Heimdall/README.md). This directory documents the code and analyses used to reproduce the Syndri experiments.

**Paper:** SYNDRI: Context-aware removal of transcript spillover in
imaging-based spatial transcriptomics, Winchell, J., Ma, J. Krieger, S. 

---

# Running Syndri

## Input Data Format

Syndri uses AnnData objects saved as `.h5ad` files. Unless a command specifies otherwise, run the examples below from the **Syndri repository root**, one directory above this README. Replace all `FILL IN` values before running the commands.

### 1. Gene Expression Data

- Store unnormalized counts in `.X`, with dimensions `[number of cells, number of genes]`. Sparse matrices are recommended.
- An alternative count layer can be selected for inference with `--expression-layer`.
- Gene identifiers in `.var_names` must use the species and identifier convention expected by the trained model and its gene metadata. Use unique cell and gene identifiers.

### 2. Spatial Coordinates

- Store the two coordinate columns in `.obs`, for example `.obs['x']` and `.obs['y']`.
- Select those columns with `--coord-cols x,y` during inference or `dataset.preprocess_args.admixture_args.coord_cols` during training.
- For a file containing multiple slices, provide a slice identifier through `--tissue-col` or the training configuration's `tissue_col` so neighbors are constructed within slices.

If coordinates are stored in `.obsm['spatial']`, copy them to the selected observation columns before saving the input:

```python
adata.obs['x'] = adata.obsm['spatial'][:, 0]
adata.obs['y'] = adata.obsm['spatial'][:, 1]
```

### 3. Cell-Type Annotations and Reference Data

- Store cell-type labels in `.obs`, and use the corresponding column name in the training and inference settings.
- Harmonize labels between the scRNA-seq reference and spatial data before training and prediction.
- Preserve the training label-to-token mapping in `.uns['cell_type_dict']`. Inference can copy this mapping from the reference AnnData supplied with `--reference-data`; use the same mapping as the checkpoint.
- Cells with missing cell-type labels are dropped by the inference preparation script.

Dataset-specific preprocessing and harmonization notebooks are in [data/](../data/).

### 4. Training Partitions

Prepared training data require a partition column, usually `.obs['split']`, containing `train` and `val` labels. The experiment configuration selects the partition column and values. Keep any independent benchmark test data separate from model fitting and parameter selection.

---

## Installation

### Step 1: Obtain the Repository

```bash
git clone "FILL IN: public Syndri repository URL" Syndri
cd Syndri
```

### Step 2: Create a Conda Environment

The repository includes an environment export with Python 3.10, PyTorch, and analysis dependencies:

```bash
conda env create --name syndri --file environment.yml
conda activate syndri
```

**FILL IN:** confirm the final tested environment, operating system, CUDA/driver versions, GPU memory requirements, and expected runtimes. The existing environment file contains Linux-specific package builds and a local installation prefix.

### Step 3: Install Heimdall

```bash
python -m pip install -e ./Heimdall
```

The model package configuration and dependency list are in [Heimdall/pyproject.toml](../Heimdall/pyproject.toml) and [Heimdall/requirements.txt](../Heimdall/requirements.txt). Comparator methods have additional dependencies; see [scripts/competitors/README.md](../scripts/competitors/README.md).

---

## Data and Trained Models

**FILL IN:** public download links, accession numbers, preprocessing versions, and licenses for the spatial data and matched scRNA-seq references.

**FILL IN:** download links for the prepared training/validation/test data, trained checkpoints, and the gene metadata required by Heimdall.

For inference, each checkpoint directory should include its `config.txt` and the selected `model-<milestone>.pt` file. Record the matching reference AnnData and cell-type dictionary alongside each model.

Configure local metadata paths in [Heimdall/config/global_vars.yaml](../Heimdall/config/global_vars.yaml), including `data_path` and `ensembl_dir`. Check any explicit metadata paths in the selected experiment or checkpoint configuration as well.

---

## Running the Code

### Step 1: Prepare Training Data

Use the dataset-specific notebooks in [data/](../data/) to harmonize annotations and construct pseudo-spatial reference data. The mapping utilities are in [code/mapping.py](../code/mapping.py), and synthetic admixture is generated by [Heimdall/Heimdall/admixture_sampling.py](../Heimdall/Heimdall/admixture_sampling.py).

For the NSCLC CosMx simulation benchmark, start with [CalibrationExperiment.ipynb](../data/nsclc_cosmx/CalibrationExperiment.ipynb). It prepares `train_val.h5ad` and `test.h5ad` and includes the test-admixture workflow. The associated [benchmark README](../scripts/spillover-benchmark/README.md) describes parameter calibration and method submission.

The configured CosMx benchmark training experiment uses **K = 9** and **tau sampled uniformly from [0, 0.95]**. These are dataset-specific settings; consult each model's saved configuration when reproducing another experiment.

### Step 2: Train a Model

Review [admixture_prediction.yaml](../Heimdall/config/experiments/admixture_prediction.yaml) for the training input, species, coordinates, slice identifiers, cell-type column, admixture settings, and output directory. Then run:

```bash
cd Heimdall
WANDB_MODE=disabled python train.py +experiments=admixture_prediction
cd ..
```

Training saves its configuration and model checkpoints in the configured `work_dir`. Use a separate output directory for each experiment.

For a Slurm cluster, [train_syndri.sbatch](../scripts/slurm/train_syndri.sbatch) provides a submission template:

```bash
sbatch --export=ALL,EXPERIMENT=admixture_prediction scripts/slurm/train_syndri.sbatch
```

Update its account/partition settings, paths, and environment activation for your cluster before submission.

### Step 3: Predict Admixture in Spatial Data

The inference entry point loads the saved checkpoint configuration and writes an AnnData object containing a probability layer:

```bash
python Heimdall/scripts/run_admixture_inference.py \
  --st-data "FILL IN: spatial slice .h5ad path" \
  --reference-data "FILL IN: matching reference .h5ad path" \
  --checkpoint-dir "FILL IN: checkpoint directory" \
  --milestone FILL_IN_MILESTONE_INTEGER \
  --config-source checkpoint \
  --cell-type-col cell_type \
  --coord-cols x,y \
  --no-apply-admixture \
  --prob-layer admix_probs \
  --prepared-name slice_prepared.h5ad \
  --output-name slice_with_admix_probs.h5ad \
  --output-dir results/syndri_inference
```

Adapt the coordinate and annotation column names to the input data and checkpoint. The `--no-apply-admixture` option predicts on the supplied counts. By default, the script assigns all prepared cells to the inference partition; use `--preserve-split-column` with `--split-col` and `--split-value` to select an existing partition instead.

Checkpoint configurations can contain paths from the original training machine. Use `--extra-overrides` to replace those metadata paths when needed. Run `python Heimdall/scripts/run_admixture_inference.py --help` for all options.

The output layer `.layers['admix_probs']` contains cell–gene admixture probabilities. Count correction is a separate step: a hard correction sets an entry's counts to zero when its probability meets the chosen threshold. The general inference command above writes probabilities; it does not automatically create corrected counts.

**FILL IN:** final model/milestone manifest and correction-selection settings used for each released dataset and figure. The analysis code supports several selection rules; see [reference-based selection](../eval/SYNDRI_REFERENCE_THRESHOLDS.md) and the settings in each figure notebook.

---

## Reproducing the Figures and Evaluations

Set the data, checkpoint, result, and cache paths in each notebook before running it. Existing notebooks and Slurm scripts include paths specific to the original computing environment.

| Analysis | Notebook or documentation |
| --- | --- |
| Spillover localization | [Figure1_SpilloverLocalization.ipynb](../figures/Figure1_SpilloverLocalization.ipynb) |
| Cell-type UMAPs | [Brain MERFISH](../figures/Figure2_UMAPs_BrainMERFISH.ipynb), [NSCLC CosMx](../figures/Figure2_UMAPs_CosMx.ipynb), [breast Xenium](../figures/Figure2_UMAPs_Xenium.ipynb) |
| Synthetic admixture benchmark, MMD, PR curves, and removal metrics | [Figure3_SyndriSIM.ipynb](../figures/Figure3_SyndriSIM.ipynb) |
| CosMx fibroblast analysis | [Figure4_CosMx_FibroblastCleaning.ipynb](../figures/Figure4_CosMx_FibroblastCleaning.ipynb) |
| Brain MERFISH analysis | [Figure4_BrainMERFISH.ipynb](../figures/Figure4_BrainMERFISH.ipynb) |
| RCTD annotation comparisons | [SuppFig1_RCTD.ipynb](../figures/SuppFig1_RCTD.ipynb) |
| Cross-dataset performance | [AllDatasetMetricsSimplified.ipynb](../eval/AllDatasetMetricsSimplified.ipynb) |
| Variance preservation | [VariancePreservation.ipynb](../context_analysis/VariancePreservation.ipynb) |
| Comparator execution | [Comparator README](../scripts/competitors/README.md) |

**FILL IN:** final figure-to-panel manifest, archived result/cache downloads, and the exact execution order for reproducing the submitted manuscript.

---

## Citation

FILL IN: Syndri BibTeX entry and the citation for the underlying Heimdall framework.

## License

Syndri is distributed under the [MIT License](../LICENSE).

## Contact

FILL IN: maintainer name, email address, and issue-tracker URL.
