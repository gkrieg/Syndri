Optional per-cell reference-cosine threshold selection
====================================================

Enable the additional `syndri_reference_cosine` case in the metric job scripts:

```bash
SYNDRI_REFERENCE_COSINE=1 SYNDRI_COSINE_WINDOW=5 \
  bash /home/jwinchel/Syndri/scripts/slurm/submit_metrics_all_datasets.sh

SYNDRI_REFERENCE_COSINE=1 SYNDRI_COSINE_WINDOW=5 \
  bash /home/jwinchel/Syndri/scripts/slurm/submit_tcell_metrics_datasets.sh
```

Each launcher retains its existing dataset selection. The option is also
supported by `metrics_eval.sbatch`, `metrics_eval_layers_pooled.sbatch`,
`metrics_eval_final.sbatch`, and `metrics_eval_tcell_final.sbatch`. Defaults
remain unchanged when the option is disabled. Existing removal-fraction sweeps,
raw counts, and SPLIT cases still run. In layered mode only Syndri gets this
additional case; MLP retains its existing sweep.

Run the sparsity comparison first to populate
`/home/jwinchel/Syndri/data/cache/rctd_profiles/`. Profiles are reused directly
from those cell-type-by-gene h5ad files, preserving RCTD reference construction.
No RCTD dependency is needed in the metric environment. By default the cache is
matched against the input's original RCTD reference path, current reference
size/mtime, and shared ordered gene panel. If metadata has no reference path,
`REFERENCE_DATA` is the fallback (set automatically for breast Xenium by the
non-T-cell launcher, and already set by the T-cell launcher).

Use `SYNDRI_REFERENCE_PROFILES=/path/to/profile.h5ad` to select an explicit
profile file, or set it to another cache directory. Missing/ambiguous cache
matches fail with instructions instead of choosing a reference arbitrarily.
When specifying a file explicitly, ensure its source reference, gene panel,
and filtering settings are appropriate for the dataset.

`SYNDRI_CELL_TYPE_KEY` selects the spatial obs labels matched to reference
profile names; it defaults to the metric label key. The non-T-cell dataset
launcher uses `cell_type_harmonized` for this additional CosMx case. Each cell
uses its assigned type's reference, not the best-matching type. Missing types
and zero references leave cells unchanged and are recorded in diagnostics;
if no labels match, the run fails with an actionable error.

Equivalent Python CLI options:

```bash
python eval/metrics_eval.py \
  --input-files /path/to/spatial.h5ad --output-dir /path/to/metrics \
  --label-key cell_type --admix-probs-layer syndri_admix_probs_X \
  --additional-counts-layer split_purified_counts_X \
  --syndri-reference-cosine --syndri-cosine-window 5 \
  --syndri-reference-profiles /path/to/profile.h5ad
```

Outputs use the existing metric output directory. Per-cell selection diagnostics
are in `syndri_thresholds/<sample>.<source-and-layer-hash>.csv`, including cell ID,
input layer, chosen probability threshold, cosine before/after, rolling cosine,
number of removed genes, removed counts on the shared panel, status, label key,
window, and profile path. Repeated runs for the same source/layer replace that
diagnostic file. Input h5ad files are never modified.

Selection semantics
-------------------

For each cell:

1. Order expressed genes by descending Syndri probability. Genes with equal
   probabilities form an indivisible removal step, since a threshold must
   remove all genes with `probability >= threshold`.
2. Include the unchanged cell (`threshold=inf`) and every distinct threshold.
   An all-zero result has undefined cosine and cannot win.
3. Compute a **trailing** rolling mean of cosine over these candidate states,
   including the unchanged state, with `min_periods=1`. The window is measured
   in distinct thresholds, not transcripts or individual genes within a tie.
4. Scan the whole curve and choose its highest rolling mean. Floating-point
   ties prefer fewer removals. `window=1` gives the exact unsmoothed maximum.

The selected endpoint maximizes the smoothed objective; its individual cosine
need not be the maximum or exceed the original cosine. Both values are saved.
The reference and spatial cell are compared on the same shared genes using
**linear expression**, without log transformation. Genes outside the reference
panel are retained unchanged and do not enter the optimization. This objective
differs from the existing T-cell **log pseudobulk** cosine metric, which is still
computed as before. Similarity to the reference used for selection is an
optimization result, not an independent validation measure.

Efficiency and notebook use
----------------------------

With dot product `D = sum(x*r)` and squared cell norm `S = sum(x*x)`, removing a
set of genes changes only `D -= sum(x_removed*r_removed)` and
`S -= sum(x_removed*x_removed)`. The reference norm stays fixed. The implementation
uses equivalent reverse cumulative sums to avoid cancellation when nearly all
genes are removed, then cumulative sums for the rolling mean. It never rebuilds
an expression vector for each threshold. Cost is `O(m log m)` for sorting plus
`O(m)` arithmetic per cell with `m` expressed genes; batch reference norms are
computed once per cell type. Sparse counts/probabilities remain sparse.

```python
import sys
sys.path.insert(0, "/home/jwinchel/Syndri/eval")
from syndri_thresholds import select_cell_threshold, clean_cells_by_reference

# All three vectors share the same gene order. counts can also be a CSR row.
result = select_cell_threshold(counts, probabilities, reference_profile,
                               window=5, return_trace=True)
threshold = result.threshold
cleaned = counts.copy()  # dense vector example
cleaned[probabilities >= threshold] = 0

import pandas as pd
curve = pd.DataFrame(result.trace)
curve.plot(x="n_removed_genes", y=["cosine", "rolling_cosine"])

# Batch: profiles is a DataFrame with cell-type rows and aligned gene columns.
cleaned_csr, diagnostics = clean_cells_by_reference(
    X, probability_matrix, cell_type_labels, profiles, window=5)
```

The standalone selector only needs NumPy/SciPy. The batch wrapper also uses
pandas. AnnData and the metric evaluator are confined to the separate adapter.
