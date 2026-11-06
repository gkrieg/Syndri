#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 5) stop('Usage: annotate_celladmix_k1p5ct.R transcripts.csv assignments.csv k output_dir cores')
transcripts <- args[1]; assignments <- args[2]; k <- as.integer(args[3]); out <- args[4]; cores <- as.integer(args[5])
if (is.na(k) || k < 1 || is.na(cores) || cores < 1) stop('Invalid factor count or cores')
dir.create(out, recursive=TRUE, showWarnings=FALSE)
paths <- file.path(out, c('bridge_raw.rds', 'bridge_res.rds', 'factor_map.csv'))
if (any(file.exists(paths))) stop('Annotation output already exists; refusing to overwrite')
.libPaths(c('/home/jwinchel/.local/lib/R/competitors', .libPaths()))
suppressPackageStartupMessages(library(cellAdmix))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(Matrix))
df <- fread(transcripts)
ass <- fread(assignments)
stopifnot(!anyDuplicated(df$mol_id), !anyDuplicated(ass$mol_id), nrow(df) == nrow(ass))
setnames(df, c('cell_id','cell_type'), c('cell','celltype'))
setkey(df, mol_id); setkey(ass, mol_id)
if (!identical(df$mol_id, ass$mol_id) || any(df$cell != ass$cell_id) ||
    any(df$celltype != ass$cell_type) || any(df$gene != ass$gene))
  stop('Assignments do not match the staged transcripts')
df[, factor := ass$factor]
df <- df[!is.na(factor)]
if (!nrow(df)) stop('No assigned molecules')
if (any(df$factor < 1 | df$factor > k)) stop('Factor outside expected range')
setorder(df, cell, mol_id)
meta <- df[, .(celltype=celltype[1], x=mean(x), y=mean(y), z=mean(z)), by=cell]
if (anyNA(meta) || anyDuplicated(meta$cell)) stop('Invalid cell metadata')
meta <- as.data.frame(meta)
rownames(meta) <- meta$cell
df <- as.data.frame(df)
rownames(df) <- df$mol_id
crf <- matrix(df$factor, ncol=1)
set.seed(0)
knn <- cellAdmix:::get_mol_knn(df, knn_k=10)
bridge_raw <- cellAdmix::run_bridge_test(df, crf, meta, knn[[1]], knn[[2]],
                                        ncells.samp=200, knn.k=20, n.cores=cores)
bridge_res <- cellAdmix::extract_bridge_res(bridge_raw, unique(meta$celltype), k,
                                             nmf.type='joint', p.thresh=.1, adj.pvals=FALSE)
# The authors' check_fp performs the distant-cell false-positive check.
helper <- new.env(parent=asNamespace('cellAdmix'))
sys.source('/home/jwinchel/cellAdmix/R/vignette_helpers.R', envir=helper)
kept <- helper$check_fp(df, meta, crf, bridge_res, do_clean=TRUE,
                         knn_k=min(100L, nrow(meta)-1L), median_thresh=.1)
if (length(kept)==1 && is.na(kept)) kept <- character()
candidate <- bridge_res[[1]]
mapping <- data.frame(cell_type=character(), factor=integer())
for (factor_name in names(candidate)) {
  factor_id <- as.integer(sub('^f_', '', factor_name))
  for (cell_type in candidate[[factor_name]]) {
    if (paste0(factor_id, '_', cell_type) %in% kept)
      mapping <- rbind(mapping, data.frame(cell_type=cell_type, factor=factor_id))
  }
}
mapping <- unique(mapping)
save_new <- function(obj, path) {
  tmp <- tempfile(pattern='.celladmix-', tmpdir=dirname(path))
  on.exit(unlink(tmp))
  saveRDS(obj, tmp)
  if (!file.link(tmp, path)) stop('Could not publish ', path)
}
save_new(bridge_raw, paths[1])
save_new(bridge_res, paths[2])
tmp <- tempfile(pattern='.celladmix-', tmpdir=out, fileext='.csv')
on.exit(unlink(tmp), add=TRUE)
fwrite(mapping, tmp)
if (!file.link(tmp, paths[3])) stop('Could not publish factor map')
cat('factor mappings:', nrow(mapping), '\n')
