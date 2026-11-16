# Spillover admixture parameter search

Run `run_calibration_experiment.sbatch` first to create
`spillover-benchmark/train_val.h5ad`. Then submit the parameter search:

```bash
sbatch /home/jwinchel/Syndri/scripts/spillover-benchmark/search_admixture_parameters.sbatch
```

The search uses only rows marked `split == 'train'` and the spatial slices
recorded in their `spatial_slice` column. For each slice, it matches each
pseudo-spatial cell to the original spatial row by the original cell ID stored
in the pseudo-spatial observation name. The sbatch script samples the same up
to 1,000 query cells per cell type for every configuration and seed;
lower-population cell types use all available cells.

The current grid searches `K = 3, 4, 5, 6, 7, 8, 9`, `tau_min = 0`, and
`tau_max = 0.65, 0.75, 0.85, 0.95`. Each configuration uses seeds 0, 1,
and 2. Tau is the target fraction of the final mixed counts contributed by
admixture. Values at or above 1 cannot be reached with finite query counts, so
the sampler uses all available eligible neighbor counts for those targets.
Neighbor selection uses Heimdall's
`build_neighbor_dict` defaults except that K is varied; admixture counts are
sampled by `sample_admixture_from_neighbor_dict` from the Heimdall package.

MMD is computed separately for each slice and cell type on shared genes, after
library-size normalization to 10,000 counts and `log1p`. The RBF kernel uses a
median-distance bandwidth fixed from the unmixed mapped cells and their paired
spatial cells. The summary ranks configurations by the equally weighted mean
MMD-squared across equally weighted slice/cell-type strata within each seed,
then averages those seed scores. The summary also reports score variation
across seeds. Per-cell achieved admixture burdens are recorded because
whole-gene addition can exceed or fall short of the sampled target, and
available neighbor counts can limit it.

Each run writes `parameter_search_replicates.csv` and
`parameter_search_summary.csv`, plus per-cell achieved burdens in
`cell_admixture_burdens.csv.gz`, in a timestamped directory under
`spillover-benchmark/parameter-search/`. The query sample cap and seeds can be
changed with the corresponding command-line options in the sbatch script.

## Test-set synthetic-admixture corrections

Run the final cells in `data/nsclc_cosmx/CalibrationExperiment.ipynb` to create
four per-slice test inputs and `test_admixture/train_val_sc_reference_unique.h5ad`.
The slice files keep original counts in `original_counts`, injected counts in
`synthetic_admixture_counts`, and admixed input counts in both `X` and
`admixed_counts`. Admixture uses K=9, tau uniform on [0, 0.95], the trained
model's source-count and gene-eligibility settings, and an independent spatial
neighbor graph per slice.

After those files exist, submit the method jobs with:

```bash
bash /home/jwinchel/Syndri/scripts/spillover-benchmark/submit_test_benchmark.sh
```

This submits RCTD first, then SPLIT after RCTD succeeds; ResolVI and Syndri run
per slice in parallel. Outputs are under `test_admixture/{rctd,split,resolvi,syndri}`.
ResolVI fits each slice separately. Syndri uses milestone 29 and saves its
probability layer plus `syndri_corrected_counts`, which zeros genes predicted
as admixture at probability >=0.5. Change `SYNDRI_BATCH_SIZE` or
`SYNDRI_PROBABILITY_THRESHOLD` in `run_syndri_test.sbatch` to tune inference or
that hard threshold before submission.

## Held-out test transcript coordinates

`simulate_test_transcripts.py` adds transcript locations to the existing test
slice matrices. It reads native and injected counts without changing the
admixture sampler. Supply the original spatial AnnData and the corresponding
CosMx flat-file directory with `CellLabels/`, FOV positions, and `tx_file.csv`.
For a single-FOV, 100-cell pilot, run from the repository root:

```bash
python scripts/spillover-benchmark/simulate_test_transcripts.py \
  --admixed-slice data/test_admixture/slices/Lung12.h5ad \
  --spatial-adata data/spatial/st_integrated_slice-Lung12.h5ad \
  --raw-dir data/raw/Lung12/Lung12-Flat_files_and_images \
  --output-dir outputs/SyndriSIM_transcripts_pilot \
  --fov 1 --max-cells 100 --seed 20261113 --decay-um 2
```

Replace the three input paths with the available local files. The complete
saved slice supplies donor counts and the original neighbor pool, including
donors in other FOVs; only the selected recipients get transcript rows.

The decay length defaults to 2 micrometers. Legacy coordinates remain in
pixels (0.18 micrometers per pixel); z values follow the original slice's
empirical discrete-plane distribution. Native positions are uniform inside
recipient masks; donated positions are biased toward the donor region.
Gene-specific source proportions determine donor allocations. Because the
current sampler adds whole pooled genes, each contributing donor supplies
exactly its original count for that gene.

Each output directory contains:

- `<slice>_tx_file.csv`: original CosMx columns followed by ground truth,
  including source/assigned cell IDs, FOVs, and admixture status. Unsimulated
  instrument fields such as `CellComp` are empty.
- `donor_allocations.parquet`: counts and gene-specific proportions for each
  source–recipient–gene allocation.
- `simulated_counts.h5ad`: selected recipient counts and clean/added layers.
- `source_and_assigned_cells.csv`: segmentation keys and centroids.
- `simulation.json`: random seed, units, transforms, and z distribution.
- `validation.json`: results of all seven required integrity checks.

Geometry and the original slice's z profile are cached under the output root.
Existing transcript CSVs are never overwritten. The implementation is shared
with the standalone package in
`recomb_reproducibility/syndri_repro/transcript_locations.py`.

## Test-set expression diagnostics

`diagnose_test_admixture.py` measures all cells in the four saved test slices,
their matched real CosMx cells, the clean mapped cells, and the unique scRNA-seq
reference. It reads existing counts without generating admixture or coordinates.

```bash
python scripts/spillover-benchmark/diagnose_test_admixture.py \
  --test-dir data/test_admixture/slices \
  --spatial-dir data/spatial \
  --reference data/test_admixture/train_val_sc_reference_unique.h5ad \
  --markers data/markers/candidate_cell_type_markers_deseq2.csv \
  --output-dir outputs/SyndriSIM_test_diagnostics
```

The spatial directory should contain `st_integrated_slice-<slice>.h5ad` files.
Replace the input paths with their local locations. The default marker panel
uses three positive, significant DESeq2 markers per type, ranked by adjusted p
value and descending fold change. Change `--markers-per-cell-type` or
`--marker-padj` to adjust it. A non-native marker is a detected marker of
another type that is not also selected as a marker of the query's own type.

For all significant markers unique to their cell type, use the full
`all_cell_type_vs_rest_deseq2.csv` table with
`--markers-per-cell-type all --unique-markers-only`. Uniqueness means positive
fold change and adjusted p <= `--marker-padj` for exactly one cell type across
the supplied table. This is evaluated before ranking or limiting the number of
markers. `marker_counts_by_cell_type.csv` records the resulting panel sizes.

The four primary measurements are detected genes, detected non-native marker
genes, achieved added/(native+added) fraction, and genes available in the native
neighbor pool but unexpressed in the clean query. The latter two depend on known
synthetic ground truth: real-data and reference values are missing rather than
invented. Eligibility is measured before adding admixture using the original
neighbor builder and the saved slice-specific settings.

Results include cached per-cell Parquet data, pooled and cell-type summaries,
cell-type-balanced means, the selected marker panel, validation records, and
PNG/PDF/SVG plots. The script's output README describes each column and plot.
Repeat the command to reuse a matching cache, add `--recompute` to rebuild it,
or add `--replot-only` to redraw from the saved per-cell measurements.
