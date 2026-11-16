# SYNDRI-SIM test diagnostics

All saved test cells are used; no new admixture is generated. The real CosMx
cells are matched to simulated recipients by `matched_st_id`, with identical
cell-type composition and gene panels. The unique original train/validation
scRNA-seq reference is included once, without repeating its rows across slices.
Clean mapped test scRNA-seq cells provide the paired before-admixture baseline.

- **Detected genes:** number of shared-panel genes with counts > 0 per cell.
- **Non-native marker genes:** detected genes in the union of selected markers
  for other cell types, excluding every marker selected for the query's own type.
  Markers are positive DESeq2 hits with adjusted p <= the configured threshold,
  ranked by adjusted p and then descending log2 fold change. They are a proxy
  for cross-type expression, not proof of contamination in real data.
- **Achieved admixture fraction:** added counts / (native + added counts), only
  for simulated ST. It is not added / native counts. Unknown fractions for real
  data and the reference are missing, not set to zero.
- **Eligible genes per query:** genes present in the original selected native
  neighbor pool and absent from the clean query, before adding admixture.
  Neighbor selection reuses the original builder and saved settings. Every added
  whole-gene count is verified against its original neighbor pool. The same
  eligibility values accompany the clean and admixed versions of each query;
  they do not count the residual zeros after admixture. No comparable native
  spatial neighbor pool is known for real data or the unique reference, so those
  values are missing.

`per_cell_diagnostics.parquet` caches every measurement. Additional columns
record added genes, eligible transcript counts, remaining eligible genes, and
the fraction of eligible genes selected. `diagnostic_summary.csv` gives pooled,
slice, and slice/cell-type summaries; `cell_type_balanced_means.csv` averages
within-type means equally to reduce cell-composition confounding.
`selected_markers.csv`, `shared_genes.csv`, `input_signature.json`, and
`validation.json` record definitions, inputs, and checks.

Boxes show medians, interquartile ranges, and 5th–95th percentile whiskers, with
all cells contributing. `diagnostics_by_cell_type` shows within-type means.
PNG, PDF, and SVG figures use a maximum width of 7.5 inches.

Rerunning the same command reuses cached measurements when the input metadata,
settings, and script match. Use `--recompute` to rebuild them or `--replot-only`
to render the existing measurements without reading input count matrices.
