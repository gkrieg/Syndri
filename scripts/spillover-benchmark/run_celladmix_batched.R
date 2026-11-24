#!/usr/bin/env Rscript
# Preserve the slice-wide NMF and per-cell CRF, bounding only execution/storage.
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c('/home/jwinchel/.local/lib/R/competitors', .libPaths()))
suppressPackageStartupMessages(library(cellAdmix))
suppressPackageStartupMessages(library(data.table))
atomic_rds <- function(value, path) {
  tmp <- paste0(path, '.tmp-', Sys.getpid())
  saveRDS(value, tmp, compress=FALSE)
  if (!file.rename(tmp, path)) stop('Cannot publish checkpoint: ', path)
}
if (length(args) == 4L && args[1] == '--batch') {
  b <- readRDS(args[2]); nmf <- readRDS(args[3])
  # Full-slice missing-gene augmentation was frozen before batching. Thus the
  # original run_crf_all normalizes exactly the same loadings for every batch.
  result <- run_crf_all(as.data.frame(b$df), nmf, num.nn=10,
                        n.cores=b$cores)
  factor <- as.integer(result[, 1])
  if (length(factor) != nrow(b$df) || anyNA(factor))
    stop('Incomplete CRF batch; no result will be checkpointed')
  atomic_rds(list(mol_id=b$df$mol_id, factor=factor), args[4])
  quit(save='no')
}
if (length(args) != 4L)
  stop('Usage: run_celladmix_batched.R transcripts.csv assignments.csv summary.csv loadings.csv')
if (any(file.exists(args[2:4]))) stop('Refusing to overwrite existing final outputs')
cores <- as.integer(Sys.getenv('CELLADMIX_CORES', '16'))
batch.cells <- as.integer(Sys.getenv('CELLADMIX_BATCH_CELLS', '128'))
batch.mols <- as.integer(Sys.getenv('CELLADMIX_BATCH_TRANSCRIPTS', '250000'))
k <- as.integer(Sys.getenv('CELLADMIX_K', '21'))
sample.cells <- as.integer(Sys.getenv('CELLADMIX_SAMPLE_CELLS', '1000'))
nmol.dsamp <- as.integer(Sys.getenv('CELLADMIX_NMOL_DSAMP', '2000'))
stopifnot(all(!is.na(c(cores,batch.cells,batch.mols,k,sample.cells,nmol.dsamp))),
          all(c(cores,batch.cells,batch.mols,k,sample.cells,nmol.dsamp) > 0))
checkpoint <- file.path(dirname(args[2]), 'crf_batches')
dir.create(checkpoint, recursive=TRUE, showWarnings=FALSE)
info <- file.info(args[1])
config <- list(schema=1L, input=normalizePath(args[1]), size=info$size,
               mtime=as.numeric(info$mtime), k=k, sample.cells=sample.cells,
               nmol.dsamp=nmol.dsamp, seed=0L, h=20L, num.nn=10L,
               batch.cells=batch.cells, batch.mols=batch.mols)
config.path <- file.path(checkpoint, 'config.rds')
if (file.exists(config.path)) {
  if (!identical(config, readRDS(config.path))) stop('Checkpoint parameters/input changed')
} else atomic_rds(config, config.path)
cat('Reading slice transcripts; only CRF batches are split by cell.\n')
flush.console()
df <- fread(args[1])
required <- c('mol_id','cell_id','cell_type','gene','x','y','z')
stopifnot(all(required %in% names(df)), !anyNA(df[, ..required]),
          !anyDuplicated(df$mol_id))
setnames(df, c('cell_id','cell_type'), c('cell','celltype'))
meta <- unique(df[, .(cell,celltype)])
stopifnot(!anyDuplicated(meta$cell))
nmf.path <- file.path(checkpoint, 'nmf.rds')
if (!file.exists(nmf.path)) {
  set.seed(0)
  fit <- samp_ct_equal(df, as.data.frame(meta), num.cells.samp=sample.cells)
  nmf <- cellAdmix::run_knn_nmf(as.data.frame(fit), k=k, h=20,
                               nmol.dsamp=nmol.dsamp, n.cores=cores)
  atomic_rds(nmf, nmf.path)
  rm(fit); gc()
  cat('NMF checkpoint saved.\n'); flush.console()
} else {
  nmf <- readRDS(nmf.path)
  cat('Reusing NMF checkpoint.\n'); flush.console()
}
# Match original full-slice run_crf_all: excluded cells <=2 molecules never
# contribute to its missing-gene list. Freeze that list before any batching.
sizes <- df[, .N, by=cell]
eligible <- sizes[N > 2L]
if (!nrow(eligible)) stop('No CRF-eligible cells')
setindexv(df, 'cell')
# Get genes without copying the full eligible transcript table.
excluded.genes <- if (all(sizes$N > 2L)) unique(df$gene) else
  df[cell %in% eligible$cell, unique(gene)]
stopifnot(!isTRUE(nmf$gene_sub))
missing <- excluded.genes[!excluded.genes %in% colnames(nmf@fit@H)]
if (length(missing)) {
  addition <- matrix(min(nmf@fit@H), nrow=nrow(nmf@fit@H), ncol=length(missing),
                     dimnames=list(rownames(nmf@fit@H), missing))
  nmf@fit@H <- cbind(nmf@fit@H, addition)
}
frozen.path <- file.path(checkpoint, 'nmf_full_gene_universe.rds')
atomic_rds(nmf, frozen.path)
rm(nmf); gc()
setorder(eligible, cell)
# A single large cell stays intact: its transcript graph cannot be split
# without changing CRF inference. Ordinary batches obey both limits.
batches <- list(); start <- 1L; count <- 0; cells <- 0L
for (i in seq_len(nrow(eligible))) {
  if (cells > 0L && (cells >= batch.cells || count + eligible$N[i] > batch.mols)) {
    batches[[length(batches)+1L]] <- eligible$cell[start:(i-1L)]
    start <- i; count <- 0; cells <- 0L
  }
  count <- count + eligible$N[i]; cells <- cells + 1L
}
batches[[length(batches)+1L]] <- eligible$cell[start:nrow(eligible)]
cat(sprintf('CRF plan: %d cells, %d batches; limits %d cells / %d transcripts.\n',
            nrow(eligible), length(batches), batch.cells, batch.mols))
flush.console()
factor.id <- rep(NA_integer_, nrow(df))
script <- sub('^--file=', '', commandArgs()[grepl('^--file=', commandArgs())][1])
rscript <- file.path(R.home('bin'), 'Rscript')
for (b in seq_along(batches)) {
  indices <- sort(df[.(batches[[b]]), on='cell', which=TRUE])
  result.path <- file.path(checkpoint, sprintf('batch_%06d.rds', b))
  if (!file.exists(result.path)) {
    input.path <- file.path(checkpoint, 'current_batch.rds')
    atomic_rds(list(df=df[indices], cores=cores), input.path)
    cat(sprintf('CRF batch %d/%d START cells=%d transcripts=%d\n',
                b, length(batches), length(batches[[b]]), length(indices)))
    flush.console()
    status <- system2(rscript, c(shQuote(script), '--batch', shQuote(input.path),
                                 shQuote(frozen.path), shQuote(result.path)))
    if (status != 0L || !file.exists(result.path)) stop('CRF batch failed: ', b)
    unlink(input.path)
  }
  result <- readRDS(result.path)
  if (!identical(result$mol_id, df$mol_id[indices]) ||
      length(result$factor) != length(indices) || anyNA(result$factor) ||
      any(result$factor < 1L | result$factor > k)) stop('Invalid checkpoint batch ', b)
  factor.id[indices] <- result$factor
  cat(sprintf('CRF batch %d/%d DONE cells=%d transcripts=%d\n',
              b, length(batches), length(batches[[b]]), length(indices)))
  flush.console()
  rm(result, indices); gc()
}
stopifnot(sum(!is.na(factor.id)) == sum(eligible$N))
assignments <- df[, .(mol_id, cell_id=cell, gene, cell_type=celltype)]
assignments[, factor := factor.id]
summary <- assignments[!is.na(factor), .N, by=.(cell_type,factor)]
summary[, fraction := N/sum(N), by=cell_type]
# Export the original fitted loadings, without the inference-only missing genes.
nmf <- readRDS(nmf.path)
loadings <- as.data.table(t(nmf@fit@H), keep.rownames='gene')
for (i in 2:4) {
  if (file.exists(args[i])) stop('Output appeared during processing')
  value <- list(assignments,summary,loadings)[[i-1L]]
  tmp <- paste0(args[i], '.tmp-', Sys.getpid())
  fwrite(value, tmp)
  if (!file.rename(tmp,args[i])) stop('Cannot publish ', args[i])
}
cat('Complete: molecule assignments retain original input order.\n')
