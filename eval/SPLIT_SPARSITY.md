Run `scripts/slurm/split_sparsity.sbatch` on merged h5ad outputs from
`run_split_h5ad_layers.sbatch`. This measures existing SPLIT results; it does not
rerun SPLIT. Each input must contain the original layer and its corresponding
`split_purified_counts_<sanitized-layer>` layer.

To run all four prepared datasets using their `results/final` files:

```bash
cd /home/jwinchel/Syndri
DRY_RUN=1 bash scripts/slurm/submit_split_sparsity_all_datasets.sh
bash scripts/slurm/submit_split_sparsity_all_datasets.sh
```

The launcher submits one array per dataset (two concurrent tasks per array by
default; override `MAX_CONCURRENT`). It freezes input lists in `logs/`, uses
the appropriate reference label column, and writes CSVs under
`/work/magroup/skrieger/Syndri/data/datasets/<dataset>/results/split/sparsity_comparison/`.
Set `SPARSITY_OUTPUT_ROOT` to override this with `<root>/<dataset>/`.
Pass dataset names as arguments for a subset.
It uses `results/final` to avoid counting both old and new CosMx slice files in
the SPLIT output directory. The two other dataset directories, `nsclc_xenium`
and `stamp`, currently have no final/SPLIT h5ad outputs and are not included.

Reference paths come from each file's RCTD metadata, preserving aging-brain
donor references. Breast Xenium metadata lacks a reference path, so the
launcher uses the `breast_cancer_chromium_genesubset.h5ad` reference configured
in `submit_split_all_datasets.sh`; override with `BREAST_REFERENCE_FILE`.
Profiles use the annotation helper, retaining reference cells with **10–2,000
shared-panel UMIs inclusive**. Cells below 10 or above 2,000 are excluded before
profile construction. This is separate from RCTD's `UMI_min_sigma=100`, which
controls spatial cells used for sigma estimation. Set `CELL_MIN`, `N_MAX_CELLS`,
`REFERENCE_MIN_UMI`, and `REFERENCE_MAX_UMI` to override profile settings.

```bash
sbatch --array=0-7%2 \
  --export=ALL,SPATIAL_DIR=/path/to/merged_h5ad,INPUT_LAYERS=X,REFERENCE_FILE=/path/to/reference.h5ad \
  /home/jwinchel/Syndri/scripts/slurm/split_sparsity.sbatch
```

Set the array range to match the input count. `INPUT_GLOB`, `INPUT_LIST_FILE`
(one path per line), and `INPUT_FILES` work like the RCTD submission script.
Without an array, only the first input is processed. Use `INPUT_LAYERS=X:trial_0`
for multiple original layers. If `REFERENCE_FILE` is omitted, each layer's RCTD
metadata supplies it. Set `CELL_TYPE_COL` to the reference annotation column.
`CELL_MIN=25` and `N_MAX_CELLS=10000` match annotation defaults.
`REFERENCE_MIN_UMI=10` and `REFERENCE_MAX_UMI=2000` match the annotation helper.
The minimum of 25 retained cells per type still applies.

Rerun the launcher to regenerate statistics with these bounds. Cache keys
include both settings and helper implementation, so older profiles are not reused or deleted. If multiple
versions exist, use the new CSV's `profile_cache` path as an explicit
`SYNDRI_REFERENCE_PROFILES` file for adaptive metric jobs to avoid ambiguity.

Profiles call `rctd_annotation.build_reference` directly, using reference `X`
restricted to the spatial/reference gene intersection **before** filtering and
normalization. These are the scRNA-seq `Reference.profiles` (means of per-cell
UMI-normalized counts), before spatial platform correction or DE gene selection.
The shared ordered gene panel is also used for both spatial measurements.

Reusable profiles are saved as cell-type-by-gene h5ad files under
`data/cache/rctd_profiles` (`CACHE_DIR` override). The key includes the resolved
reference path, file size/mtime, ordered gene panel, reference settings, and
implementation source hash. Concurrent jobs share a locked, atomically written
cache. File size/mtime is used instead of a full reference content checksum.
The h5ad contains provenance in `uns['cache_metadata_json']`.

When running the single-dataset sbatch script directly, one CSV per input is
saved under `eval/split_sparsity` (`OUTPUT_DIR` override); the all-dataset
launcher sets the dataset-specific directory described above. Each CSV has
with one row per cell type, input layer, and stage (`scrna_profile`,
`spatial_before`, `spatial_after`). Repeating an input replaces its CSV.

- `zero_fraction`: exact-zero entries divided by all cell-by-gene entries.
  For a reference profile this is the fraction of zero genes in that vector;
  `n_cells=1` denotes one profile, not the number of reference cells.
- `mean_profile_zero_fraction`: fraction of genes with zero mean expression
  within the spatial cell type, useful for comparing aggregate spatial profiles
  to scRNA-seq profiles.
- Counts, denominators, excluded-cell count, label source, and cache path are
  included. Types without spatial cells have undefined (blank) fractions.

Spatial groups default to the highest original `rctd_weights_<layer>` weight,
matching SPLIT preparation. Zero-weight, negative-weight, or nonfinite rows are
excluded. Alternatively, set `SPATIAL_CELL_TYPE_COL` to an existing obs column;
missing/empty labels are excluded. The same cells and labels are used before
and after SPLIT. Spatial types filtered out of the reference are still reported,
but have no reference-profile row.

Statistics describe the **merged layers**: original/zero values inserted by
SPLIT merge's `FILL_MISSING` policy are included. They do not imply every included
cell was successfully purified. No input h5ad is modified.
