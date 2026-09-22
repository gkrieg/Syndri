# Authors' R admixture backend

Run all datasets:

```bash
ADMIXTURE_BACKEND=celladmix bash /home/jwinchel/Syndri/scripts/slurm/submit_score_admixture_all_datasets.sh
```

Append dataset names to restrict the run. `OVERWRITE=1` permits replacing existing
scores. Outputs go to `results/metrics/admixture_scores_celladmix/<slice>/`, with
the same per-cell CSV columns used by existing plotting code. Change the plotting
loader's directory from `admixture_scores` to `admixture_scores_celladmix`.

The adapter sources `cellAdmix/R/admix_scores.R` and
`cellAdmix-Notebooks/R/benchmarking.R` without modifying the clones. Adjacency,
correlation preservation, and scoring execute the actual R functions. Scoring
uses `estimate_contamination_scores`, the matrix implementation underlying the
notebook's `get_dbl_scores`. This preserves fractional SPLIT counts without
expanding them into transcript rows. A fixture test compares the matrix results
against `get_dbl_scores` itself (`test_celladmix_notebook_parity.R`).

Intentional adaptations: adjacency uses cell centroids and is constructed within
each FOV where available; AnnData input preparation and Syndri count removal use
the existing Python helpers. Reference labels and paths follow the existing
dataset configuration, including `.uns` inference and the explicit breast
reference. Therefore using this backend alone does not reproduce the paper's
input data, annotations, or molecule-coordinate adjacency.

The default is count-weighted, background-adjusted scoring with the notebook
parameters (`signal.thres=.4`, `p.c=.25`, `min.expr.frac=.05`,
`max.prob.high=.1`). Correlation filtering uses raw and reference counts normalized
by their original full-panel cell totals, with threshold .2, fixed across all
conditions. Zero-count cells remain as missing scores. Cells without retained
admixture genes keep the authors' zero result and are marked
`score_status=no_admixture_genes`; these zeros are not evidence of clean cells.
Review `gene_correlation_preservation.csv` and `n_admixture_genes` before plotting.
`SKIP_CORRELATION_FILTER=1` explicitly disables this filter for a sensitivity run.

Each result includes clone commit IDs in `admixture_score_config.json`, R package
versions in `R_session_info.txt`, and the fixed gene-by-type contamination
probabilities in `admixture_gene_probabilities.rds` (rather than the Python
backend's pickle). The scoring CSV is written only after every condition succeeds.
The cloned adjacency implementation requires at least two workers and three
represented cell types; the adapter reports unsupported inputs as errors.
