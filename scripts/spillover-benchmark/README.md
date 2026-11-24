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

## Alternative simulation with donor-type weights

`prepare_weighted_test_admixture.py` calibrates donor-type weights from the three
Lung5 **training** slices, runs a stratified Lung12 pilot, and optionally creates
all four held-out test slices. The implementation is in
`donor_weighted_admixture.py`; the original Heimdall sampler and existing
benchmark inputs are preserved.

```bash
sbatch scripts/spillover-benchmark/prepare_weighted_test_admixture.sbatch
```

The job requests the CPU partition, 8 CPUs, and 64 GB RAM. Set
`SYNDRI_DATA_ROOT`, `SYNDRI_WEIGHTED_OUTPUT`, or `SYNDRI_PYTHON` before submission
to select other data, output, or Python locations. The default output root is
`outputs/SyndriSIM_donor_weighted_exposed_20261120/`. Existing slice outputs are not
overwritten. The underlying Python command accepts explicit paths:

```bash
python scripts/spillover-benchmark/prepare_weighted_test_admixture.py \
  --training data/spillover-benchmark/train_val.h5ad \
  --test-dir data/test_admixture/slices \
  --reference data/test_admixture/train_val_sc_reference_unique.h5ad \
  --spatial-dir data/spatial \
  --markers data/markers/all_cell_type_vs_rest_deseq2.csv \
  --output-dir outputs/donor_weighted \
  --calibration-slices Lung5_Rep1 Lung5_Rep2 Lung5_Rep3 \
  --pilot-slice Lung12 --pilot-cells-per-type 100 \
  --min-calibration-cells 50 --min-calibration-slices 2 \
  --shrinkage .25 --seed 20261120 \
  --run-all
```

Omit `--run-all` to stop after the pilot. With it, a failed pilot prevents full
test generation. The pilot uses up to 100 recipients per cell type, with the
complete clean slice available as their source pool. A unit-weight control uses
the same recipients and target-tau streams. Corresponding pilot and full-slice
counts must be identical because each query has its own seeded random streams.

### Calibration and sampling definitions

- Markers are **all** positive DESeq2 hits with adjusted p <= 0.05 that are
  significant for exactly one cell type in the full supplied table.
- For each calibration slice, recipient type, and donor type, marker detection
  is the mean detection probability over the donor's marker set. Use the
  **same selected donor graph from the clean mapped cells** to define exposure
  for native availability and real CosMx detection. Real cells are matched by
  `matched_st_id`, in the same row order, with identical cell-type labels.
  Real detection includes only recipients with at least one selected donor of
  that type. Excess is `max(0, CosMx_detection - reference_detection)`. The
  reference baseline uses the unique original train/validation scRNA-seq cells
  of the recipient type; this reference has no spatial exposure definition.
- A stratum requires at least **50 donor-exposed recipients** in both the
  spatial and clean mapped data, and 50 reference cells. Exclude matching donor
  and recipient types. Average recipient types equally within each slice,
  then average slices equally. A donor weight requires qualifying strata from
  at least **two calibration slices**.
- For each exposed query, availability `E` is the eligible marker count from
  selected donors of that type divided by **both the number of markers and
  the number of those donors**. Eligible genes are absent from the clean query.
  Exclude queries with no selected donor of that type; retain exposed queries
  with zero eligible marker counts as zeros. Average these values over queries,
  recipient types, and slices using the same eligible strata as the excess.
- Raw donor weights are `excess / (E + 1e-8)`. Normalize supported weights to
  mean 1 and shrink them as `0.75 * normalized_weight + 0.25`. Donor types with
  insufficient support or no positive eligible availability receive exactly
  **1**. Supported weights and the complete set both retain mean 1. Support
  thresholds, shrinkage, and epsilon are configurable.
- Retain the original neighbor selection (K=9, source total counts >400,
  different source/query types) and tau uniform on [0, 0.95]. Each donor/gene
  contribution becomes `floor(weight * count) + Bernoulli(fractional_part)`.
  This preserves the weighted expectation and preserves integer counts exactly
  at weight 1. Pool these integer contributions, exclude natively expressed
  genes, and use the original whole-gene sampler and minimum added count of 10.
- Weights apply to **all** donor genes, not only marker genes. They represent
  effective contributions, so weighted counts may exceed a donor's original
  measured count; they are not transcript donation probabilities.

### Outputs and integrity checks

`calibration/` contains donor weights, the marker panel, all pairwise detection
and availability strata, per-slice estimates, and input/settings records.
`donor_weights.csv` reports total and used donor-exposed neighborhood counts,
qualifying recipient strata and slices, support status, and any neutral-weight
reason. `donor_exposure_support_by_slice.csv` gives support per slice and donor
type. The full strata also record the number of selected donors and their mean
per exposed recipient. Counts of supporting neighborhoods are query counts;
a query can support several donor types.
`pilot/` contains the initial slice, unit-weight control, diagnostics, and
validation report. `slices/` contains the four full test inputs and their truth
sidecars; `diagnostics/` compares the reference, clean, weighted simulated, and
matched real counts using the same marker panel.

Every slice stores native counts in `original_counts`, additions in
`synthetic_admixture_counts`, and their sum in both `X` and `admixed_counts`.
Observation metadata records sampled tau, achieved burden, eligible gene/count
capacity, and additions. Donor–recipient–gene Parquet sidecars preserve source
and assigned cell IDs, scRNA-seq IDs, original matched spatial IDs (including
slice/FOV/cell), native donor counts, donor weights, and integer donated counts.
Allocation files are streamed in batches.

Validation reconstructs additions from the saved donor allocations, checks
rounding bounds and identifiers, confirms that additions occur only at native
zeros, and reads the written AnnData back to check all three count matrices.
Full Lung12 generation must reproduce every pilot cell's added counts exactly.
The original transcript-coordinate simulator's unweighted-pool reconstruction
must not be used on these weighted inputs; their saved allocation sidecars
are the authoritative donor assignments. Pass `--donor-allocations` to
`simulate_test_transcripts.py` to use those integer transfers directly.

### Transcript coordinates for whole-gene lambda=0.25 inputs

`run_pool_fraction_transcripts.py` reads the frozen matrices and donor-allocation
sidecars in `outputs/SyndriSIM_pool_fraction/slices/`. It adds locations without
resampling expression, donors, or donor weights. Native transcripts are uniform
within recipient-mask pixels. Donated transcripts use the same recipient mask,
with pixel weights proportional to `exp(-distance_to_donor_region / 2 um)`.
Uniform subpixel offsets follow pixel selection. Distances use the documented
0.18 um pixel scale, while exported coordinates retain the CosMx pixel units,
calibrated image orientation, and FOV offsets. Both populations use the original
slice-wide discrete z-plane frequencies. Instrument measurements remain empty.

Submit a 100-cell Lung12/FOV1 pilot, then all four slices after it passes:

```bash
pilot_job=$(sbatch --parsable --array=0 --time=02:00:00 \
  --export=ALL,POOL_TRANSCRIPT_STAGE=pilot \
  scripts/spillover-benchmark/simulate_pool_fraction_transcripts.sbatch)
sbatch --array=0-3%2 --dependency=afterok:"$pilot_job" \
  --export=ALL,POOL_TRANSCRIPT_STAGE=full \
  scripts/spillover-benchmark/simulate_pool_fraction_transcripts.sbatch
```

Each task requests 16 CPUs and 64 GiB on the CPU node `oven-1-24`; at most two
full slices run concurrently. Sampling uses up to 16 worker processes with
one BLAS thread each; per-cell random streams are independent of worker count.
The default coordinate seed is 20261121 and the
decay length is fixed at 2 um. The runner also checks the pilot's seven
validation flags before processing a full slice. The array order is Lung12,
Lung13, Lung9_Rep1, Lung9_Rep2.

Outputs are in `outputs/SyndriSIM_pool_fraction/transcript_coordinates/`:
the pilot is under `Lung12/fov_001_n100/` and full slices under
`<slice>/all_fovs/`. Each contains a CosMx-style transcript CSV, donor-allocation
Parquet, clean/added/combined count AnnData, cell/FOV metadata, provenance, and
the seven reconstruction/geometry validation checks. Cache files are shared
between the pilot and full Lung12 run. Existing transcript CSVs are not
overwritten. These outputs can contain hundreds of millions of transcript rows
across the four full slices.

## Donor-pool fraction calibration

`search_pool_fraction_admixture.py` evaluates an alternative target for the
same weighted donor pool. For each query, let `P` be the eligible weighted
pool total **after integer rounding and exclusion of native-positive genes**.
Draw `lambda ~ Uniform(0, lambda_max)` and set
`target_added_counts = round(lambda * P)`. Randomly permute eligible genes
and add each selected gene's entire pooled count until reaching the target.
Whole-gene addition can exceed the target, but cannot exceed the available
pool. Targets below 10 counts produce no addition. The recorded final burden
is still `added / (native + added)`; `sampled_lambda` and
`achieved_pool_fraction = added / P` are recorded separately.

```bash
sbatch scripts/spillover-benchmark/search_pool_fraction_admixture.sbatch
```

The job uses 8 CPUs and 64 GB RAM. Its default output directory is
`outputs/SyndriSIM_pool_fraction/`; override `SYNDRI_POOL_OUTPUT` for a new run.
`SYNDRI_DATA_ROOT`, `SYNDRI_DONOR_WEIGHT_ROOT`, and `SYNDRI_PYTHON` select inputs,
the frozen donor-weight calibration, and the Python executable. Existing
output directories are rejected to preserve completed runs.

### Selection rule

- Compare `lambda_max` values **0.1, 0.25, 0.5, 0.75, 1.0**, with simulation
  seeds **0, 1, 2**. Include **no admixture** as a separate reference row;
  select lambda from the requested positive grid and report whether it improves
  over that reference.
- Use **Lung5_Rep1, Lung5_Rep2, and Lung5_Rep3 training cells only**. A supported
  query type has at least 50 mapped/real matched pairs in the slice and 50
  unique reference cells. Use a fixed sample of up to **500 queries per type
  and slice**, the full clean slice as the donor pool, and exactly matched real
  recipients. The query sample is identical across candidates and seeds.
- Minimize **cell-type-conditional MMD squared**, averaging types equally
  within each slice, slices equally within each seed, then seeds equally.
  Diagnostics do not contribute to the selection score.
- Fix preprocessing to **library-size normalization to 10,000 followed by
  log1p**, using all 956 shared genes. Use a biased RBF MMD-squared estimator.
  Determine one bandwidth per slice/type from the clean and matched real
  sample using the median positive squared pairwise distance. Reuse that
  bandwidth, target cells, and kernel definition for every candidate and seed.
- Freeze donor weights from the prior exposure-conditioned calibration, with
  unchanged K=9, source-count threshold, native-zero restriction, and rounding.
  No test slice is opened until lambda selection has been written to disk.

### Separate diagnostics and generated data

`search/lambda_search_summary.csv` contains the selection scores, seed SDs,
and baseline. `search/selected_parameters.json` records the selected positive
lambda, its comparison with no admixture, and the averaging/preprocessing rules.
`search/fixed_mmd_kernels.csv`, `evaluation_support.csv`, and
`fixed_query_cells.parquet` record the fixed evaluation design.

`search/separate_diagnostic_summary.csv` reports detected-gene and non-native
marker counts, signed mean differences, and one-dimensional Wasserstein
distances against real data. Cross-type gene-pair co-detection is evaluated on
**all unique significant DESeq2 markers**, over all selected queries, using
only pairs whose markers belong to different cell types. Report the mean
absolute error in joint detection, the corresponding covariance error after
subtracting the product of marginal detection rates, and marginal detection
error. These diagnostics are also saved by slice/type/seed as appropriate;
the search plots show them separately from MMD squared. The co-detection panel
here contains all calibration markers; the notebook's three-marker-per-type
heatmap is a smaller, separately ranked panel.

After selection, the script creates the 100-cells-per-type Lung12 pilot.
With `--run-all` (enabled in the sbatch script), successful pilot checks gate
generation of all four held-out test slices. Saved native, added, combined,
and donor-allocation truth follow the same format and integrity checks as the
weighted sampler above. Additional assertions check `round(lambda * P)`,
minimum target behavior, and that additions stay within the eligible pool.
Test data are used only for generation and post-selection diagnostics.

To generate data at an explicit pool-fraction setting without a search, the
existing driver also accepts `--target-mode pool_fraction --lambda-max VALUE`.
Its default `final_fraction` mode retains the previous tau calculation.

### Figure 3 predictions for the pool-fraction inputs

`submit_pool_fraction_benchmark.py` submits the original Figure 3 method
recipes against the four validated files in
`outputs/SyndriSIM_pool_fraction/slices/`. It records submitted job IDs,
dependencies, explicit environment settings, references, and parameters in
`outputs/SyndriSIM_pool_fraction/manifests/method_jobs.json`. Repeated
submissions of the same stage are rejected to avoid duplicate work.

```bash
python scripts/spillover-benchmark/submit_pool_fraction_benchmark.py --stage competitors
python scripts/spillover-benchmark/submit_pool_fraction_benchmark.py --stage syndri
python scripts/spillover-benchmark/submit_pool_fraction_benchmark.py --stage figures
```

- **RCTD:** the original doublet-mode annotation runner, unique original
  train/validation scRNA-seq reference, shared-panel reference UMI limits
  10–2,000, minimum 25 cells per reference type, and original remaining defaults.
- **SPLIT:** the original RCTD-to-SPLIT pipeline, input/reference layer `X`,
  reference label `cell_type`, 50,000-cell chunks, no residual-contamination
  removal, and original handling of missing/zero-weight cells. Each array task
  waits for its matching RCTD task (`aftercorr`). Neither program uses repeated
  mapped copies as its reference.
- **ResolVI:** a fresh per-slice semisupervised fit using scvi-tools 1.3.3,
  seed 0, 100 epochs, and the median of three posterior samples. The original
  coordinate conversion, labels, model defaults, and correction export are used.
- **SYNDRI:** fresh training from the original checkpoint's saved configuration,
  with the same `train_val.h5ad`, split, architecture, seed 54, batch settings,
  optimizer, loss, and scheduler. A process-local epoch-store subclass replaces
  tau sampling with the frozen globally weighted whole-gene lambda=0.25 sampler.
  Training admixture refreshes every epoch; validation admixture stays fixed,
  using the original epoch-store seed rules. Native counts stay in the original
  input; added counts and their binary labels come from the cached epoch matrices.

SYNDRI stops after training and saving milestone **29** (30 epochs), matching
the checkpoint used by the original Figure 3. Its nominal **1,000-epoch
learning-rate horizon is retained**: setting the configuration to 30 epochs
would change all preceding learning rates. The new checkpoint directory is
`outputs/SyndriSIM_pool_fraction/syndri_model/`. Core Heimdall code and original
checkpoints are not changed. The frozen configuration and training recipe are
written to `manifests/syndri_training_config.yaml` and
`manifests/syndri_training_recipe.json`.

Four SYNDRI inference tasks wait for training and use the new milestone 29,
the original train/validation reference for vocabulary consistency, batch size
256, and hard zeroing at probability 0.5. All method outputs use the exact paths
expected by `figures/Figure3_SyndriSIM_PoolFraction.ipynb`. The figures stage
waits for all prediction arrays, then refreshes the notebook and metric caches.
Its removal-AP panel scores SYNDRI's hard-zeroed counts at the configured
threshold, while its continuous PR/ROC curves use `admix_probs`, matching the
original panel definitions.

RCTD and SPLIT wrappers clear inherited virtualenv/conda PATH state before
calling the original runners so Python 3.6 from an unrelated virtualenv cannot
override their Python 3.10 environment. SYNDRI uses an explicit interpreter.

Resource policy for this run: GPU tasks are restricted to the H100 node
`oven-1-3`, with one GPU and eight CPUs per task. SYNDRI requests 64 GiB host
RAM and ResolVI 96 GiB, within the recommended 32-CPU/275-GB share per H100.
RCTD and SPLIT use the CPU-only partition (`oven-1-24`), requesting eight CPUs
and 64/32 GiB respectively, with at most two tasks per array at once. Their
combined maximum is 32 CPUs and 192 GiB, leaving capacity on the CPU node.
Figure rendering uses eight CPUs and 64 GiB after predictions finish.

## Independent binomial transcript-donation experiment

`search_binomial_admixture.py` evaluates a separate alternative using
`binomial_transcript_admixture.py`. Existing whole-gene samplers, datasets,
notebooks, mappings, and train/validation/test assignments remain available.

For query `i`, use its existing selected neighbors and frozen calibrated
donor-type weights. Divide each selected donor's weight by the arithmetic mean
weight over **all selected neighbors**, including neighbors without eligible
gene counts. Draw `lambda_i ~ Uniform(0, lambda_max)` and use
`p_ij = min(1, lambda_i * normalized_weight_ij)`. For each eligible donor/gene
entry draw `D_ijg ~ Binomial(original_donor_count_jg, p_ij)` in a vectorized
operation. Only genes with zero native query expression are eligible in this
experiment. Donated counts are added without depleting source counts.

There is no weighted-count rounding, gene permutation, target-count budget,
or 10-count cutoff in this sampler. Applying such a cutoff would violate
independent transcript donation. Each donor can contribute at most its original
count for a given gene/recipient pair. Probabilities can saturate at one; the
per-cell diagnostics record how many selected donors do so. A neighborhood with
fewer than nine eligible donors uses the neighbors returned by the unchanged
graph-selection procedure.

```bash
sbatch scripts/spillover-benchmark/search_binomial_admixture.sbatch
```

The job requests 8 CPUs and 64 GB RAM and writes to the separate directory
`outputs/SyndriSIM_binomial_transcripts/`. Set `SYNDRI_BINOMIAL_OUTPUT` for
another run. Other overrides are `SYNDRI_DATA_ROOT`,
`SYNDRI_DONOR_WEIGHT_ROOT`, `SYNDRI_WHOLE_GENE_ROOT`, and `SYNDRI_PYTHON`.
An existing output directory is rejected.

The default search uses lambda values **0.0001, 0.0005, 0.001, 0.0025, 0.005,
0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0**. Values below 0.1 were added
after an initial coarse-grid calibration indicated excessive gene detection
already at 0.1. It uses the same three seeds, three training/calibration
slices, 500 fixed queries per supported cell type,
normalization, RBF MMD² kernels, and equal hierarchical averaging described
above. A no-admixture baseline is included. Lambda is selected only on
calibration data. Before reading any test inputs, the driver checks that
queries, kernels, support, baselines, and frozen donor weights match the
completed whole-gene run.

After selection, the job validates a 100-cells-per-type Lung12 pilot, then
generates all four test slices if the pilot passes (`--run-all`). Saved AnnData
uses the same native/added/combined layers. Donor allocation sidecars retain
the original schema and additionally store `normalized_donor_weight`,
`neighborhood_weight_mean`, `sampled_lambda`, and `donation_probability`.
The saved positive donor–recipient–gene allocations, rather than inferred
donors, should be used for subsequent transcript-location generation.
Per-cell statistics distinguish expected additions, actual additions, sampled
lambda, available **unweighted original** donor counts, and final admixture
burden. Lambda is not the final contamination fraction.

Integrity assertions reconstruct every added count from donor allocations,
check native-zero eligibility, original donor count bounds, graph membership,
local normalization, probabilities, identifiers, written count layers, and
pilot/full reproducibility. Native counts are compared against the existing
whole-gene test set before post-selection diagnostics.

`comparison/` contains calibration and held-out comparison CSVs and plots for
MMD², detected-gene and non-native-marker Wasserstein distances, and cross-type
co-detection mean absolute error. The latter uses all unique significant
DESeq2 markers. Test evaluation uses the same fixed queries and clean/real
bandwidths for both simulators; it does not select lambda. The existing
whole-gene implementation uses global weights, while the requested binomial
version normalizes them locally. Therefore this comparison measures both
changes together and does not isolate transcript sampling alone.

## Transcript-based comparators on whole-gene lambda=0.25 test slices

`run_pool_fraction_transcript_methods.py` runs `prepare`, `celladmix`, and
`ovrlpy` stages for Lung12, Lung13, Lung9_Rep1, and Lung9_Rep2 (array indices
0–3). The associated sbatch file reads `POOL_TRANSCRIPT_METHOD_STAGE`.
Preparation must succeed for a slice before either method runs.

Preparation streams the existing simulated transcript CSVs, exports one
cellAdmix CSV per slice and Parquet files per FOV, and exactly reconstructs
the input count matrix. Model inputs contain only molecule ID, assigned cell,
cell type, gene, and x/y/z coordinates. Molecule IDs are zero-based row
numbers in the original simulated CSV. Source-cell IDs and admixture truth
are excluded. X/y are converted using the simulation's 0.18 um pixel scale;
z planes are multiplied by 0.8 um, matching the existing CosMx method adapters.
Original transcript files and expression matrices are preserved.

cellAdmix uses the existing factorization and annotation scripts:
`ceil(1.5 * n_cell_types)` factors (21 for 14 types), approximately 1,000
cell-type-balanced training cells in total, `h=20`, `nmol.dsamp=2000`,
CRF `num.nn=10`, and seed 0. The existing bridge test and distant-cell
false-positive check select type-specific factors to remove. No factor is
selected using synthetic admixture truth. Outputs include molecule assignments,
factor loadings/summaries, bridge results, a factor map, and
`celladmix/<slice>.h5ad` with `celladmix_corrected_counts`.

ovrlpy 1.2.0 fits each FOV separately with the existing `Ovrlp` defaults,
`n_components=min(30,n_genes-1)`, seed 0, and `analyse(fit_umap=False)`.
A distance-preserving z shift accommodates negative CosMx planes. The output
`ovrlpy/<slice>.h5ad` contains native cell-level `obs['ovrlpy_vsi']` scores
and count layers `ovrlpy_corrected_counts_vsi_lt_0_5` and
`ovrlpy_corrected_counts_vsi_lt_0_7`. These count layers remove molecules
whose pixel VSI falls below the stated threshold; nonfinite VSI retains the
molecule. Per-molecule VSI values are saved as Parquet under
`ovrlpy/transcript_scores/<slice>/`. Cell scores are not broadcast as counts.

All paths above are relative to `outputs/SyndriSIM_pool_fraction/`. Jobs use
16 CPUs per task on `oven-1-24`, no GPUs: preparation uses 32 GiB,
cellAdmix 192 GiB, and ovrlpy 128 GiB. The repaired cellAdmix array permits
two simultaneous slices (32 CPUs and 384 GiB total); preparation and ovrlpy
have already completed. Per-stage Slurm commands,
dependencies, parameters, and input paths are recorded in
`manifests/transcript_method_jobs.json`.

### Bounded cellAdmix CRF execution

The pool-fraction runner uses `run_celladmix_batched.R`. It fits the same
seed-0, slice-wide NMF and checkpoints it before CRF processing. The CRF
uses at most 128 complete cells and 250,000 transcripts per batch by default
(`CELLADMIX_BATCH_CELLS`, `CELLADMIX_BATCH_TRANSCRIPTS`). A single cell larger
than the transcript limit stays intact to preserve its graph. Each batch runs
in a fresh R process with the requested 16 workers, releasing its memory on
exit. The parent retains the input table and a single integer assignment
vector, without splitting the entire slice or retaining all CRF results.

Full-slice missing-gene loadings are frozen before batching, so normalization
is identical to the original full-slice CRF. Neighbors, factor potentials,
transition settings, iteration limits, annotation, and original molecule order
are preserved. Each complete batch is checkpointed under
`celladmix/factorization/<slice>/crf_batches/`; rerunning reuses the NMF and
completed batches when the input signature and settings match. Logs report
`CRF batch i/N START` and `DONE` for progress tracking. The existing original
competitor runner remains available.

### Bounded factor annotation

`celladmix_annotation_bounded.py` and `.R` replace full-transcript graph
materialization in the pool-fraction runner. `CELLADMIX_ANNOTATION_BACKEND=original`
retains the untouched original annotation path; the default is `bounded`.
The new `run_pool_fraction_annotation_bounded.sbatch` reuses saved assignments
and requests 8 CPUs / 96 GiB, typically with array `0-3%2`.

The bounded path streams paired input/assignment CSVs, verifies molecule,
cell, gene, and type identities, and stages only coordinates, factors, and
cell summaries on disk. A persistent exact global 3D cKDTree is queried in
100,000-transcript chunks. It retains the first qualifying molecule and
lexically first adjacent cell for each recipient-cell/neighbor-type pair,
then applies the original duplicate-removal rules. All FOVs remain in the
same neighbor index. No transcripts, cells, or FOVs are downsampled.

Bridge tests run serially in original type-pair order with seed 0, sampling
200 cells, using the original null-generation, motion, Wilcoxon, factor
extraction, and false-positive functions/settings. Only the geometry for
post-pruning cells is loaded. Pairwise RANN searches are cached across
factors because they are identical and consume no random numbers. The
false-positive check consumes the exact cell-factor count summaries and a
cell-level graph, rather than the complete transcript dataframe.

Caches are under `celladmix/annotation/<slice>/bounded_cache/`; bridge-test
checkpoints (including RNG state) are under `bridge_pairs/`. Progress is
reported in logs and `bounded_cache/progress.json`. Input signatures and
implementation hashes guard reuse. Staging retains complete cells and
original within-cell molecule order. Exact-distance ties may be ordered
differently between FNN and SciPy; validation measures actual neighbor sets.
The installed cellAdmix package is never modified.

Validation commands:

```bash
python scripts/spillover-benchmark/validate_celladmix_bounded.py \
  --kind fixture --output outputs/SyndriSIM_pool_fraction/annotation_validation/new_fixture
python scripts/spillover-benchmark/validate_celladmix_bounded.py \
  --kind actual --output outputs/SyndriSIM_pool_fraction/annotation_validation/new_actual
```

These compare against the original one-worker implementation: global
neighbor sets, cell-pair selection, count/centroid summaries, bridge
statistics, candidate factor annotations, false-positive filtering, and
final factor maps. A separate case exercises a nonempty retained map and
rejection of a native factor. Passing subset checks establishes equivalence
on those datasets, not a guarantee of bitwise agreement on every full slice.
