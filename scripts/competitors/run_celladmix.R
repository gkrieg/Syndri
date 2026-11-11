#!/usr/bin/env Rscript
# Fit the cloned cellAdmix factorization and retain molecule factor assignments.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 4) stop('Usage: run_celladmix.R transcripts.csv assignments.csv summary.csv loadings.csv')
if (any(file.exists(args[2:4]))) stop('Refusing to overwrite an existing result')
.libPaths(c('/home/jwinchel/.local/lib/R/competitors', .libPaths()))
suppressPackageStartupMessages(library(cellAdmix))
suppressPackageStartupMessages(library(data.table))
df <- fread(args[1])
required <- c('mol_id', 'cell_id', 'cell_type', 'gene', 'x', 'y', 'z')
if (!all(required %in% names(df))) stop('Missing normalized transcript columns')
if (anyNA(df[, ..required])) stop('Missing transcript fields')
setnames(df, c('cell_id', 'cell_type'), c('cell', 'celltype'))
if (anyDuplicated(df$mol_id)) stop('Duplicate molecule IDs')
meta <- unique(df[, .(cell, celltype)])
if (anyDuplicated(meta$cell)) stop('Conflicting cell-type annotations')
sample.cells <- as.integer(Sys.getenv('CELLADMIX_SAMPLE_CELLS', '1000'))
nmol.dsamp <- as.integer(Sys.getenv('CELLADMIX_NMOL_DSAMP', '2000'))
cores <- as.integer(Sys.getenv('CELLADMIX_CORES', '10'))
if (anyNA(c(sample.cells, nmol.dsamp, cores)) || any(c(sample.cells, nmol.dsamp, cores) < 1))
  stop('Invalid cellAdmix sampling or core setting')
set.seed(0)
fit <- samp_ct_equal(df, as.data.frame(meta), num.cells.samp=sample.cells)
nmf <- run_knn_nmf(as.data.frame(fit), k=10, h=20, nmol.dsamp=nmol.dsamp, n.cores=cores)
eligible <- df[, .N, by=cell][N > 2, cell]
ok <- df$cell %in% eligible
if (!any(ok)) stop('No cells have enough transcripts for the cellAdmix CRF')
factors <- run_crf_all(as.data.frame(df[ok]), nmf, num.nn=10, n.cores=cores)
factor.id <- rep(NA_integer_, nrow(df))
factor.id[ok] <- as.integer(factors[, 1])
assignments <- data.table(mol_id=df$mol_id, cell_id=df$cell,
                          gene=df$gene, cell_type=df$celltype,
                          factor=factor.id)
summary <- assignments[!is.na(factor), .N, by=.(cell_type, factor)]
summary[, fraction := N / sum(N), by=cell_type]
loadings <- as.data.table(t(nmf@fit@H), keep.rownames='gene')
if (any(file.exists(args[2:4]))) stop('Result appeared during computation')
tmp.assign <- tempfile(pattern='.celladmix-', tmpdir=dirname(args[2]), fileext='.csv')
tmp.summary <- tempfile(pattern='.celladmix-', tmpdir=dirname(args[3]), fileext='.csv')
tmp.loadings <- tempfile(pattern='.celladmix-', tmpdir=dirname(args[4]), fileext='.csv')
tryCatch({
  fwrite(assignments, tmp.assign)
  fwrite(summary, tmp.summary)
  fwrite(loadings, tmp.loadings)
  if (!file.link(tmp.assign, args[2])) stop('Could not publish assignments exclusively')
  if (!file.link(tmp.summary, args[3])) stop('Could not publish summary exclusively')
  if (!file.link(tmp.loadings, args[4])) stop('Could not publish loadings exclusively')
}, finally=unlink(c(tmp.assign, tmp.summary, tmp.loadings)))
