# Analysis definitions and reproducibility

This document describes the cohorts, calculations, defaults, and interpretation limits behind the SYNDRI results. Use the [README](README.md) for installation, data acquisition, parameter-file examples, checkpoints, commands, output locations, and cache management.

## Inputs and cohorts

**Raw** denotes the uncorrected input counts. Analyses compare Raw and SYNDRI using processed counts, harmonized cell-type labels, spatial coordinates, and matching scRNA-seq references.

| Experiment | Included spatial cohort |
| --- | --- |
| Whole mouse brain MERFISH (BM) | Three sections; measured, non-imputed counts |
| NSCLC CosMx (NC) | Eight sections |
| Breast cancer Xenium (BX) | Nineteen sections |
| CosMx simulation (SIM) | Four test sections: `Lung12`, `Lung13`, `Lung9_Rep1`, `Lung9_Rep2` |

The [pretrained-model table](README.md#pretrained-models) identifies the checkpoints used for these experiments. Reference analyses for BX use SoupX counts, stored as `.X` in the included reference matrix.

### Pseudo-spatial mapping and partitions

Mapping uses 30 Harmony PCs, up to 50 candidates of the same cell type, reuse penalty 9, a reuse cap of 5,000, and seed 0. Included training matrices preserve the inputs used for the pretrained checkpoints.

For SIM, the scRNA-seq reference is partitioned by patient before mapping: p5 is validation, p1 is test, and the remaining patients are training. Treg cells are excluded. Each partition is integrated independently with its associated spatial sections. Remapped copies retain their source identities so that copies of the same reference cell do not cross training and validation.

## Count correction and cell selection

### Correction rule

The default figure settings rank expressed cell–gene entries within each cell type and select a shared probability threshold. The target is removal of 50% of that cell type's counts, with a minimum admixture probability of 0.5. Selected gene counts are set to zero.

The alternative per-cell rule selects a threshold independently within each cell. The `correction.unit`, `target_fraction`, and `min_probability` settings determine these choices. The two removal units can select different genes. Whole-gene removal and tied probabilities can cause the achieved count-removal fraction to exceed the target.

### QC cohorts

UMAPs and fibroblast differential-expression analyses use the intersection of Raw and SYNDRI QC-passing cells: at least 10 counts and 5 detected genes in each condition. Cross-dataset metric defaults use the unfiltered Raw/SYNDRI cohort. Results calculated with different cell-selection rules can therefore refer to different populations.

## Evaluation metrics

### Cell-type separation

Counts are normalized to 10,000 per cell and transformed with `log1p`. The analysis uses up to 50 PCA dimensions, a 15-neighbor graph, and Leiden clustering at resolution 1. ARI and NMI compare cluster assignments with the dataset's cell-type labels. Silhouette uses cosine distance in PCA space.

The evaluation random state is 0; independently plotted UMAPs use random state 42.

### Reference similarity

Expression is summed within each cell type to form a pseudobulk profile. Spatial and scRNA-seq profiles are normalized to 10,000 counts, transformed with `log1p`, and compared using cosine similarity on shared genes. Section-level reference similarity is the mean over available cell types.

### Variance preservation

Within each cell type, the analysis computes gene-wise sample variance of normalized, log-transformed expression. Spatial and reference variances are compared using Spearman correlation. A comparison requires at least 50 cells in each dataset; genes must have at least 5% detection in the reference, and genes with zero reference variance are excluded. Section-level scores average finite cell-type correlations.

## Simulation benchmark

Synthetic admixture uses K=9 neighbors and tau uniformly sampled from [0, 0.95]. The default test cohort is listed above. Its membership is defined by `simulation.test_slices` in the analysis configuration.

### MMD before and after admixture

The analysis compares distributions of normalized, log-transformed single-cell expression vectors within each section/cell-type stratum. It reports the biased, nonnegative RBF **MMD²** estimate. A bandwidth based on median squared distance is shared between the before- and after-admixture comparisons.

Defaults use up to 500 query cells per cell type, three random seeds, and equal cell-type weighting when aggregating losses. Each MMD boxplot point is a section/cell-type value averaged across seeds. The calculation compares individual-cell distributions rather than pseudobulk profiles.

### Marker co-detection

The reference, simulated, and real spatial matrices use the top three available shared genes per type from the included DESeq2 marker table. Every gene pair is evaluated across all cells in its dataset. The simulated matrix uses the clean mapped scRNA-seq counts for Lung5_Rep1 before synthetic admixture is added.

### Cell-type adjacency

A physical K-nearest-neighbor graph with K=6 defines directed cell-type adjacency. Enrichment is `log2((observed + 1e-8) / (expected + 1e-8))`, where `observed` is the directed pair frequency and `expected` is the product of the two cell-type abundances. Rows represent query types and columns represent neighbor types in the full matrices.

In the triangular comparison, each pair with `i < j` uses simulated `A[i,j]` above the diagonal and real `A[i,j]` mirrored below it. The complementary plot uses `A[j,i]`. The diagonal is the mean of the real and simulated values. The mirrored lower triangle compares the same directed pair; its displayed row/column position does not define a new directed statistic. Both complete matrices are also available for inspecting all directions.

### Precision–recall and removal summaries

SYNDRI scores are `admix_probs`. Evaluation includes observed cell–gene entries; an entry is positive when its synthetic added count is greater than zero. The positive-prevalence baseline is the fraction of evaluated entries that are positive. AUPRC is calculated as average precision using all evaluated entries.

The binarized curve and best-F1 result use a threshold selected on pooled test labels. They are descriptive oracle summaries, rather than performance estimates for a threshold selected independently on validation data. Curve thinning affects display only, not AUPRC or F1 calculations.

## Localization and fibroblast analyses

### Transcript localization

The localization analysis uses measured transcript-distance, cell-sampling, and composition tables. Per-cell means are recomputed using transcripts inside their assigned cell polygon. Boundary distances are divided by the equivalent circular radius, `sqrt(area/pi)`.

### Fibroblast differential expression

The analysis pools the three Lung5 replicates and compares tumor-interior with stromal fibroblasts on shared measured genes. Counts are normalized to 10,000 per cell and transformed with `log1p`. Scanpy's tie-corrected Wilcoxon test is used with Benjamini–Hochberg adjustment.

Volcano plots show log2 fold changes and adjusted P values. Z plots compare signed Wilcoxon statistics. Pooled cells are the unit of this descriptive analysis; the results are not donor-level biological-replicate tests.

## Context perturbation analysis

### Inputs and selection

The experiment uses the BX model on B1_1, B1_2, B1_3, B5, B12, B10, B17, and B9. Sampling uses seed 42 plus the section index. Eligible queries have at least 10 counts, at least 5 detected genes, finite coordinates and labels, and exactly five eligible neighbors of a different exact cell type. Neighbor sources have more than 10 counts.

Up to 300 queries per exact type/section are balanced across original malignant-neighbor bins (0, 1–2, 3–5) and within-type/section marker-burden quantiles. Burden ties remain together. Query IDs, strata, and donor IDs are included for replay.

### Intervention

For each exact query type, up to five distinct donor cells are selected at each malignant-token count from 0 through 5. Undersupplied counts are not compensated. Ordered neighbor-type sequences are retained, including identical sequences from different donor cells.

Each query's expression and cell-type token remain fixed under its original, duplicate unchanged-control, and donor contexts. Inference uses these precomputed neighbor sequences. Only the neighbor-type tokens change.

### Marker definitions

The 32 breast-cancer panel genes and the authors' ST 4 native-marker mapping are defined in `configs/context.json` and `assets/context/`. Eligible genes have positive original query counts, fixed across interventions. Scoring requires at least one expressed panel gene and one expressed native gene. Pericytes have no native annotation and contribute no native-marker scores.

Marker AUROC measures how often a panel gene's admixture probability exceeds a native gene's probability, giving half credit for ties. Responses are measured relative to the same query's original context. No count-correction threshold is used for these probability comparisons.

### Panel definitions

- **Panel a: intervention schematic.** The editable illustration shows a fixed query cell placed in different cellular contexts.
- **Panel b: burden-stratified response.** Donor responses at the same change in malignant count are averaged within each query. Query medians are calculated per exact type/section/burden/original-context stratum, requiring five queries. Supported exact-type medians are averaged equally within each family. Stratum/x combinations require support in at least three sections. Original-context strata are averaged equally within each section; curves show section medians and interquartile ranges for low, medium, and high query marker burden.
- **Panel c: per-gene response heatmap.** A gene's strict-win fraction is the fraction of expressed native genes with a strictly smaller admixture probability; ties count as zero. Within each section/family/burden/original-context stratum, the change in malignant count is correlated with the change in strict-win fraction. This requires ten distinct queries, three delta levels, and a nonconstant response. The displayed value is the median Spearman rho across supported section–stratum units, requiring at least three sections. Genes are clustered using average-linkage Euclidean distance on the four family responses. Missing values are imputed with a gene's mean only for clustering; gray cells remain unsupported in the display. Annotations count sections whose median stratum rho agrees with the overall median sign, divided by supported sections. They are not significance counts.
- **Panel d: vascular plausibility.** For each endothelial query and expressed panel gene, strict-win fraction is regressed against change in malignant count using representative donor contexts with at least three distinct levels. The displayed beta is the median of the per-query slopes. The analysis also records 2,000 within-section query bootstraps with seed 741. Reference detection prevalence is smoothed as `(n_expressing + 0.5)/(n_cells + 1)`; the x-axis is its log2 endothelial/malignant ratio. Reference expression uses SoupX counts. The displayed line is an ordinary least-squares trend across genes.
- **Panel e: controlled versus original-context slopes.** Each query's controlled AUROC-change slope is fitted against change in malignant-neighbor count, requiring three levels. The original-context AUROC slope is fitted against original malignant-neighbor count within each exact type/section/burden, requiring ten queries, three count levels, and five queries in each retained original-context bin. Bins receive equal weight, as do queries within each bin. The controlled coefficient is the weighted mean of the same queries' intervention slopes. Paired-valid exact-type coefficients are averaged equally within family/section/burden. Points show individual sections; summaries show coordinate-wise section medians and interquartile ranges. The dashed line marks equality.

## Reproducibility limits and provenance

### Included results and regenerated analyses

The included matrices, annotations, checkpoints, and probabilities preserve the available inputs for reproducing existing outputs. Fresh Harmony integration, mapping, and stochastic training can produce different numerical results. Recomputed probabilities may also vary slightly across floating-point hardware and software environments.

Correction settings, cohort selection, and marker definitions affect the results independently of model weights. Existing corrected-count layers reflect the settings used when they were generated. See [cache management](README.md#caching) when changing inputs or analysis settings.

### Comparison scope

The repository evaluates Raw and SYNDRI directly. Comparator implementations and correction results are excluded. These outputs do not reconstruct rankings that depend on additional methods' performance.

### Data representation and records

Inputs retain count matrices, required annotations, simulation truth, and SYNDRI probabilities. Stored probabilities are restricted to expressed entries because inference predicts observed gene tokens and evaluation uses nonzero input entries. Counts and identifier order are preserved.

The following records identify the assets, transformations, and runtime used:

- [assets/manifest.json](assets/manifest.json): asset sizes and checksums.
- [provenance/asset_sources.json](provenance/asset_sources.json): input identifiers and transformations.
- [provenance/context_sources.json](provenance/context_sources.json): context-analysis source identifiers and hashes.
- [provenance/runtime_sources.json](provenance/runtime_sources.json): model-runtime source and included-file hashes.
