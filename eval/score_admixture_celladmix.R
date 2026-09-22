#!/usr/bin/env Rscript
# Source the authors' clones unchanged. This adapter only handles files,
# centroid/FOV input, and matrices (including fractional SPLIT counts).
suppressPackageStartupMessages({library(Matrix); library(dplyr); library(magrittr)})
argv <- commandArgs(trailingOnly=TRUE)
mode <- argv[[1]]
work <- argv[[2]]
cfg <- jsonlite::fromJSON(file.path(work, "config.json"))
source(file.path(cfg$celladmix_repo, "R", "admix_scores.R"))
source(file.path(cfg$celladmix_notebooks_repo, "R", "benchmarking.R"))
read_table <- function(name) read.csv(file.path(work, name), colClasses="character", check.names=FALSE)
genes <- read_table("genes.csv")$gene
obs <- read_table("spatial_obs.csv")
ref.obs <- read_table("reference_obs.csv")
read_counts <- function(name, cells) {
  m <- as(readMM(file.path(work, name)), "CsparseMatrix")
  dimnames(m) <- list(genes, cells)
  m
}
normalize <- function(m, totals) {
  scales <- ifelse(totals > 0, 10000 / totals, 0)
  m <- m %*% Diagonal(x=scales)
  m@x <- log1p(m@x)
  m
}
if (mode == "prepare") {
  stopifnot(requireNamespace("sccore"), requireNamespace("geometry"))
  set.seed(cfg$random_seed)
  obs$x <- as.numeric(obs$x); obs$y <- as.numeric(obs$y)
  groups <- split(obs[is.finite(obs$x) & is.finite(obs$y) & !is.na(obs$fov), ], ~fov)
  edges <- lapply(groups, function(g) {
    if (nrow(g) < 4) return(NULL)
    estimate_cell_adjacency(g, edge.max.mad=cfg$edge_max_mad, n.cores=cfg$n_jobs)
  }) %>% bind_rows()
  if (!nrow(edges)) stop("No centroid adjacency edges remain")
  K <- estimate_cell_type_adjacency(edges)
  missing.types <- setdiff(unique(obs$cell_type), colnames(K))
  if (length(missing.types)) stop("No adjacency for cell types: ", paste(missing.types, collapse=", "))
  if (ncol(K) < 3) stop("This clone requires at least three adjacent cell types")
  write.csv(K, file.path(cfg$output_dir, "cell_type_adjacency.csv"))
  ref <- read_counts("reference.mtx", ref.obs$cell)
  raw <- read_counts("raw.mtx", obs$cell)
  cors <- setNames(rep(NA_real_, length(genes)), genes)
  expressed <- rowSums(ref) > 0 & rowSums(raw) > 0
  excluded <- rep(FALSE, length(genes))
  if (!cfg$skip_correlation_filter) {
    if (sum(expressed) < 3) stop("Too few expressed genes for correlation filtering")
    cr <- estimate_correlation_preservation(
      normalize(raw, as.numeric(obs$library_size)),
      normalize(ref, as.numeric(ref.obs$library_size)))
    cors[names(cr)] <- cr
    excluded <- !is.finite(cors) | cors < cfg$correlation_threshold
  }
  write.csv(data.frame(gene=genes, correlation=cors, excluded=excluded,
    reason=if (cfg$skip_correlation_filter) "filter_disabled" else
      ifelse(!expressed, "not_expressed_in_both", ifelse(!is.finite(cors), "undefined_correlation",
      ifelse(excluded, "below_threshold", "retained")))),
    file.path(cfg$output_dir, "gene_correlation_preservation.csv"), row.names=FALSE)
  message("Shared genes: ", length(genes), "; excluded: ", sum(excluded))
  if (all(excluded)) warning("All shared genes were excluded; resulting zero scores cannot measure admixture. See gene_correlation_preservation.csv.")
  saveRDS(list(reference=ref, K=K, exclude=genes[excluded]), file.path(work, "state.rds"), compress=FALSE)
  writeLines(capture.output(sessionInfo()), file.path(cfg$output_dir, "R_session_info.txt"))
} else if (mode == "score") {
  state <- readRDS(file.path(work, "state.rds"))
  cm <- read_counts("condition.mtx", obs$cell)
  # Same score parameters and core function used by Notebooks::get_dbl_scores.
  # Its molecule-table reconstruction is bypassed to retain fractional counts.
  result <- estimate_contamination_scores(
    cm.rna=state$reference, cm.spatial=cm,
    annot.rna=setNames(ref.obs$cell_type, ref.obs$cell),
    annot.spatial=setNames(obs$cell_type, obs$cell), cell.type.adj.mat=state$K,
    p.c=cfg$contamination_prior, signal.thres=cfg$signal_threshold,
    min.expr.frac=cfg$min_expr_frac_contam, max.prob.high=cfg$max_prob_high,
    exclude.genes=state$exclude, adjust=cfg$adjust)
  totals <- colSums(cm)
  scores <- result$doublet.scores[obs$cell]
  scores[totals == 0] <- NA_real_
  if (any(!is.finite(scores[totals > 0]))) stop("Nonfinite scores for nonempty cells")
  n.genes <- colSums(result$cont.probs.per.type > 0)
  no.genes <- n.genes[obs$cell_type] == 0
  if (any(no.genes)) warning("No retained admixture genes for: ", paste(names(n.genes)[n.genes == 0], collapse=", "))
  write.csv(data.frame(cell=obs$cell, cell_type=obs$cell_type, scores=scores,
    dataset="spatial", total_counts=totals, n_admixture_genes=n.genes[obs$cell_type],
    score_status=ifelse(totals == 0, "zero_counts", ifelse(no.genes, "no_admixture_genes", "ok"))),
    file.path(work, "scores.csv"), row.names=FALSE)
  saveRDS(result$cont.probs.per.type, file.path(cfg$output_dir, "admixture_gene_probabilities.rds"))
} else stop("Unknown mode: ", mode)
