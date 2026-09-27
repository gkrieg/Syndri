Cell-type reference-cosine gene-prefix selection
================================================

Enable the additional `syndri_reference_cosine` metric case with:

```bash
SYNDRI_REFERENCE_COSINE=1 SYNDRI_ONE_SE_MULTIPLIER=1 \
  bash /home/jwinchel/Syndri/scripts/slurm/submit_syndri_sliding_metrics_all_datasets.sh
```

Despite the launcher's historical name, this case no longer uses a rolling
window or a fixed probability threshold. `SYNDRI_ONE_SE_MULTIPLIER` defaults to
1. Set it to 0 to choose the earliest literal maximum of the mean curve.

Reference profiles
------------------

Run the sparsity comparison first to populate
`/home/jwinchel/Syndri/data/cache/rctd_profiles/`. Profiles are reused directly
from those cell-type-by-gene h5ad files, preserving RCTD reference construction.
The cache match uses the original reference path, its size and modification
time, the ordered shared panel, and the profile filtering settings.

Use `SYNDRI_REFERENCE_PROFILES=/path/to/profile.h5ad` to choose an explicit
profile file. `SYNDRI_CELL_TYPE_KEY` chooses the spatial annotation matched to
the reference profile names and defaults to the metric label key. Cells with a
missing reference type are preserved and marked in diagnostics.

Selection rule
--------------

For each cell type `c`:

1. For every gene `g`, compute its Syndri score as the mean probability among
   cells of type `c` in which `g` is expressed. Cells where `g` is unexpressed
   do not contribute to this mean.
2. Rank the genes once by decreasing score. No fixed probability threshold is
   applied.
3. For every prefix size `k = 0, ..., G`, remove the first `k` ranked genes from
   every cell of type `c`. Compute each cell's change from its original cosine
   similarity to the type's reference profile, then average those changes.
4. Let `k_max` be the earliest maximum of that mean curve. Select the smallest
   `k` whose mean improvement is at least
   `mean(k_max) - multiplier * SE(k_max)`.

The SE is the exact nonparametric-bootstrap standard error of the sample mean
under the empirical cell distribution. Computing it from the first two moments
is algebraically exact and avoids noisy, expensive Monte Carlo resampling.
Every cell of a type receives the same selected gene prefix. A gene absent from
a particular cell has no effect on that cell and is not counted as removed.

Cosines use linear expression on the shared spatial/reference panel without
renormalizing the remaining genes. Genes outside the reference panel remain
unchanged. The implementation updates only the removed genes' dot-product and
squared-norm contributions; it does not reconstruct a vector for every `k`.

Outputs
-------

Aggregate metric CSVs identify the case with:

- `threshold_selection = cell_type_smallest_near_optimal_k`
- `syndri_probability_ordering = cell_type_expressed_mean`
- `syndri_one_se_multiplier`

`admix_prob_threshold` remains empty because this method has no probability
threshold.

Per-cell diagnostics are saved under `syndri_thresholds/*.csv`. Useful columns
include `cell_type`, `selected_k`, `k_max`, `cosine_before`, `cosine_after`,
`n_removed_genes`, `removed_count`, and `status`.

The complete cell-type curves are saved under `syndri_cosine_curves/*.csv`.
They include `k`, the gene added at that rank, its cell-type score, mean cosine
improvement, bootstrap SE, the one-SE target, and flags for `k_max` and the
selected `k`.

Notebook use
------------

```python
import sys
sys.path.insert(0, "/home/jwinchel/Syndri/eval")
from syndri_thresholds import clean_cells_by_reference

cleaned, cell_diagnostics, curves = clean_cells_by_reference(
    X,
    probability_matrix,
    cell_type_labels,
    profiles,
    one_se_multiplier=1.0,
    return_curves=True,
)
```

`X` and the probability matrix may be dense or sparse. `profiles` is a pandas
DataFrame whose rows are cell types and whose columns match the input gene
order.
