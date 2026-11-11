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
