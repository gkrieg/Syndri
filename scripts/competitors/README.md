# Spatial competitor adapters

The source AnnData files are under `/work/magroup/skrieger/Syndri/data/datasets/<dataset>/results/final`.
These adapters never modify those files. They write a new AnnData with the same cells and genes under each dataset's `results/competitors/<method>/` directory. Outputs are created exclusively and an existing output causes a failure.

`format_transcripts.py` converts the raw CosMx `*_tx_file.csv` and Xenium `*_transcripts.parquet` into per-sample `raw/competitors/transcripts/*.parquet`. The normalized columns are `mol_id,cell_id,gene,x,y,z` (coordinates in micrometers). CosMx cell IDs are resolved by `(fov, cell_ID)` against the final AnnData. Xenium cell IDs are joined through final `obs["cell_id"]`. Unassigned molecules and controls are omitted. The format command checks that every retained cell ID and gene occurs in the final AnnData.

Run one sample at a time (or use the `submit_*.sh` scripts for the supported sample lists):

```
/home/jwinchel/miniconda3/envs/rctd/bin/python scripts/competitors/format_transcripts.py DATASET FINAL_H5AD
/home/jwinchel/.venv/resolvi/bin/python scripts/competitors/run_resolvi.py FINAL_H5AD
/home/jwinchel/miniconda3/envs/ovrlpy/bin/python scripts/competitors/run_ovrlpy.py FINAL_H5AD
```

ResolVI runs on all four final datasets using `/home/jwinchel/.venv/resolvi/bin/python`; `install_resolvi.sh` reproduces the isolated scvi-tools 1.3.3 environment. Its output layer is `resolvi_corrected_counts`, posterior median `px_rate`, and it retains the original count scale through the model posterior. It uses the final AnnData cell type labels and coordinates. The model is fitted independently by sample. Cells without a cell type or spatial coordinate retain raw counts in this layer and are marked by `obs["resolvi_scored"]`.

`submit_resolvi_zero_shot.sh` generates an additional leave-one-slice-out ResolVI result for every final slice. Each model is trained on all other slices from the same dataset, with spatial neighbors computed within each source slice. The held-out slice is scored with the trained shared parameters and fresh, neutral per-cell parameters; no query training takes place. Outputs go to `results/competitors/resolvi_zero_shot/` and retain the same `resolvi_corrected_counts` layer and `resolvi_scored` flag as the per-slice results. The output metadata records the held-out slice, training slices, training cell count, and fit mode. Existing outputs are never replaced.

ovrlpy's native output is a cell integrity score, not corrected expression. Its output is an `obs["ovrlpy_vsi"]` score and a `layers["ovrlpy_vsi"]` matrix that repeats the cell score over genes solely for layer-based comparison. **Do not treat this layer as corrected counts.**

The cloned cellAdmix correction workflow fits molecule factors and calls for identifying admixture factors by cell type. Its notebook makes this choice by inspecting factors. The local clone is installed in `/home/jwinchel/.local/lib/R/competitors` with R 4.5. `CRF` needed a compatibility patch to replace removed `Calloc` and `Free` macros. `install_celladmix.sh` reproduces the isolated dependency installation.

The cellAdmix stage and export commands are:

```
/home/jwinchel/miniconda3/envs/rctd/bin/python scripts/competitors/stage_celladmix.py FINAL_H5AD /tmp/transcripts.csv
/home/jwinchel/miniconda3/envs/r_env/bin/Rscript scripts/competitors/run_celladmix.R /tmp/transcripts.csv /tmp/assignments.csv /tmp/factor_summary.csv /tmp/factor_loadings.csv
/home/jwinchel/miniconda3/envs/rctd/bin/python scripts/competitors/export_celladmix.py FINAL_H5AD /tmp/assignments.csv factor_map.csv
```

`factor_map.csv` has `cell_type,factor` columns. The factor map is selected after inspecting the factor summary; no automatic factor deletion is performed. The exported layer is `celladmix_corrected_counts`.

The AnnData output adapters accept `--output-root` to stage in another directory. The default output directories are in each dataset's `results/competitors` tree. Preexisting files are never replaced.
