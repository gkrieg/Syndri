# Syndri data transfer manifest (scp)

Scope: files outside the Syndri Git checkout or ignored by its `.gitignore`, needed to rerun the current CosMx spillover benchmark and the plots in `context_analysis/ResponsivenessQueryNativeMarkers.ipynb`. The repository code and tracked marker CSV are not listed. `*.h5ad` and `context_analysis/results/*` are Git-ignored; data under `/work/magroup/...` is outside the checkout.

## CosMx Syndri-SIM / spillover benchmark

### Rerun the existing held-out test benchmark

Copy these prepared artifacts (7 H5ADs, about 4.14 GB total):

- `/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/spillover-benchmark/train_val.h5ad` — Syndri model reference.
- `/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/spillover-benchmark/test.h5ad` — calibration/parameter-search held-out set.
- `/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/spillover-benchmark/test_admixture/train_val_sc_reference_unique.h5ad` — competitor and diagnostic SC reference.
- All four files under `/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/spillover-benchmark/test_admixture/slices/`: `Lung12.h5ad`, `Lung13.h5ad`, `Lung9_Rep1.h5ad`, `Lung9_Rep2.h5ad`. These are the saved synthetic-admixture test inputs consumed by the method jobs.

A benchmark parameter-search rerun additionally needs the eight spatial slices under `/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/results/final/` named `st_integrated_slice-*.h5ad` (about 3.21 GB total). The test diagnostic uses only the four held-out slices: Lung12, Lung13, Lung9_Rep1, Lung9_Rep2.

The diagnostic also takes `/home/jwinchel/Syndri/recomb_reproducibility/assets/markers/cosmx_deseq2.csv`; this is tracked in Git, so it does not need copying separately. The README's example `data/spatial/` location is stale for this checkout; the present integrated spatial H5ADs are in `results/final/`.

### Optional: regenerate the prepared calibration and test inputs

The files above already exist, so these larger upstream inputs are unnecessary unless you intend to rerun the preparation notebook itself: `/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/raw/single-cell/sc_concat.h5ad`, `/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/raw/spatial/st_concat.h5ad`, plus the eight `results/final/st_integrated_slice-*.h5ad` listed above. For the transcript-coordinate simulation pilot, also copy the original Lung12 CosMx flat-file directory `/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/raw/Lung12/Lung12-Flat_files_and_images/` if present on the source machine; it contains `CellLabels/`, FOV positions, and `tx_file.csv`.

## Context-analysis notebook plots

For the notebook as written (including its plot cells that rebuild summaries), copy the following external/ignored inputs. They are processed intermediate data, not the original breast Xenium data.

- The current result bundle: `/home/jwinchel/Syndri/context_analysis/results/query_native_marker_responsiveness_20261020_115220/`. It contains the plot-ready CSV/CSV.GZ tables, JSON design/marker metadata, and analysis helper scripts. It is ignored by Git. You can omit its existing `.png`/`.pdf` files if you only need to redraw plots.
- From `/home/jwinchel/Syndri/context_analysis/results/all_type_responsiveness_stratified300_20261019_162748/contexts/`, all eight `*.h5ad` context panels.
- From `/home/jwinchel/Syndri/context_analysis/results/all_type_responsiveness_stratified300_20261019_162748/predictions/`, the eight `*_responsiveness_predictions.h5ad` files (not the `*_prepared.h5ad` files). Context panels plus predictions total about 5.45 GB.
- The eight source spatial inputs named in the current run's `design.json`, under `/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/interim/spatial/` (about 1.15 GB total). They are read by the notebook's spatial-colocalization cell.
- `/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/interim/single-cell/breast_cancer_chromium_genesubset.h5ad` (about 7 MB), used for the vascular reference-expression plots.

If you plan to adjust only the figures that use existing summary tables and will skip cells that recompute from AnnData, you can omit the eight context panels, eight prediction files, and eight source spatial files. Keep the current result bundle and the single-cell reference for the standalone vascular plausibility plot (cell 34 checks that reference and saved predictions exist even when using its cached summary). The notebook's absolute paths point to `/home/jwinchel/Syndri` and `/work/magroup/...`; on the Mac, change those path constants or recreate the same directory structure.

## Copy commands

Replace `jwinchel@oven.compbio.cs.cmu.edu` with the SSH host alias for this machine and `~/Desktop/Syndri-data/` with the destination. Copy the benchmark prepared inputs and optional parameter-search spatial slices:

```bash
mkdir -p ~/Desktop/Syndri-data
scp -r jwinchel@oven.compbio.cs.cmu.edu:/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/spillover-benchmark/train_val.h5ad jwinchel@oven.compbio.cs.cmu.edu:/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/spillover-benchmark/test.h5ad jwinchel@oven.compbio.cs.cmu.edu:/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/spillover-benchmark/test_admixture/train_val_sc_reference_unique.h5ad jwinchel@oven.compbio.cs.cmu.edu:/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/spillover-benchmark/test_admixture/slices jwinchel@oven.compbio.cs.cmu.edu:/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/results/final/st_integrated_slice-*.h5ad ~/Desktop/Syndri-data/
```

For notebook plot inputs (about 6.6 GB plus the result bundle):

```bash
scp -r jwinchel@oven.compbio.cs.cmu.edu:/home/jwinchel/Syndri/context_analysis/results/query_native_marker_responsiveness_20261020_115220 jwinchel@oven.compbio.cs.cmu.edu:/home/jwinchel/Syndri/context_analysis/results/all_type_responsiveness_stratified300_20261019_162748/contexts jwinchel@oven.compbio.cs.cmu.edu:/home/jwinchel/Syndri/context_analysis/results/all_type_responsiveness_stratified300_20261019_162748/predictions jwinchel@oven.compbio.cs.cmu.edu:/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/interim/spatial jwinchel@oven.compbio.cs.cmu.edu:/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/interim/single-cell/breast_cancer_chromium_genesubset.h5ad ~/Desktop/Syndri-data/
```

The first command includes all eight benchmark spatial slices for the parameter search; remove that wildcard if only rerunning the saved test-set methods/diagnostic. SCP may prompt for each path if the shell on the remote host does not expand globs; copying the `results/final/` directory is the simple fallback, but transfers other files too.
