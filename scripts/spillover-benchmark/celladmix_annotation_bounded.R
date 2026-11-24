#!/usr/bin/env Rscript
# Same bridge and false-positive statistics; disk-backed selected cell geometry.
.libPaths(c('/home/jwinchel/.local/lib/R/competitors', .libPaths()))
suppressPackageStartupMessages(library(cellAdmix))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(Matrix))

atomic_rds <- function(value, path) {
  tmp <- paste0(path, '.tmp')
  saveRDS(value, tmp)
  if (!file.rename(tmp, path)) stop('Cannot publish ', path)
}

load_cells <- function(work, meta, ids) {
  selected <- which(meta$cell %in% ids) # original lexical cell / mol_id order
  xyz <- file(file.path(work, 'xyz.dat'), 'rb')
  fac <- file(file.path(work, 'factors.dat'), 'rb')
  on.exit({close(xyz); close(fac)})
  parts <- lapply(selected, function(i) {
    seek(xyz, where=as.double(meta$start[i])*24, origin='start')
    coordinates <- matrix(readBin(xyz, 'double', n=3L*meta$n[i], size=8,
                                   endian='little'), ncol=3, byrow=TRUE)
    seek(fac, where=as.double(meta$start[i])*4, origin='start')
    factors <- readBin(fac, 'integer', n=meta$n[i], size=4, endian='little')
    stopifnot(nrow(coordinates)==meta$n[i],length(factors)==meta$n[i])
    data.frame(cell=meta$cell[i], x=coordinates[,1], y=coordinates[,2],
               z=coordinates[,3], factor=factors)
  })
  do.call(rbind,parts)
}

# Geometry is unchanged across factor-specific scores. Compute its original
# RANN neighbor search once per ordered pair / coordinate frame, instead of
# once per factor. No random draws occur in this function.
score_environment <- function() {
  env <- new.env(parent=asNamespace('cellAdmix'))
  frame <- NULL; by_cell <- NULL; links <- new.env(hash=TRUE,parent=emptyenv())
  env$compute_bridge_scores <- function(df_crf_diff_fracs, factor_test, df_crf,
                                        cell_pair_df, knn_k=10, abs_diff=TRUE) {
    if (!identical(frame,df_crf)) {
      frame <<- df_crf
      by_cell <<- split(df_crf[,c('x','y','z','factor')],df_crf$cell,drop=TRUE)
      links <<- new.env(hash=TRUE,parent=emptyenv())
    }
    result <- vapply(seq_len(nrow(cell_pair_df)),function(i) {
      a <- as.character(cell_pair_df[i,1]); b <- as.character(cell_pair_df[i,2])
      key <- paste(a,b,knn_k,sep='\034')
      if (!exists(key,envir=links,inherits=FALSE)) {
        c1 <- by_cell[[a]]; c2 <- by_cell[[b]]
        points <- rbind(c1[,c('x','y','z')],c2[,c('x','y','z')])
        near <- RANN::nn2(data=points,query=c1[,c('x','y','z')],
                         k=min(knn_k,nrow(points)-1L))$nn.idx
        crossing <- rowSums(near > nrow(c1)) > 0
        f <- as.integer(as.character(c1$factor))
        n_factors <- ncol(df_crf_diff_fracs)
        numerator <- tabulate(f[crossing],nbins=n_factors)
        denominator <- tabulate(f,nbins=n_factors)
        assign(key,numerator/denominator,envir=links)
      }
      difference <- df_crf_diff_fracs[b,factor_test]-df_crf_diff_fracs[a,factor_test]
      if (abs_diff) difference <- abs(difference)
      unname(difference*get(key,envir=links)[factor_test])
    },numeric(1))
    names(result) <- cell_pair_df[,1]
    result
  }
  # Keep the author's complete null-generation procedure, including all draws,
  # geometry shifts and factor-specific sampling, substituting only the cache.
  null <- get('compute_bridge_scores_null',asNamespace('cellAdmix'))
  environment(null) <- env
  env$compute_bridge_scores_null <- null
  env
}

one_pair <- function(work,meta,counts,pairs,index,ncells=200,fast=TRUE) {
  pair <- pairs[pairs$pair_index==index,c('self_cells','adj_cells'),drop=FALSE]
  if (!nrow(pair)) return(NA)
  all_cells <- unique(unlist(pair,use.names=FALSE))
  # The original code constructs this table before pruning. Counts are exactly
  # the same integers; materializing transcripts before pruning is unnecessary.
  ct <- counts[rownames(counts) %in% all_cells,,drop=FALSE]
  fractions <- ct/Matrix::rowSums(ct)
  pair <- cbind.data.frame(pair,as.data.frame.matrix(ct[pair$self_cells,,drop=FALSE]))
  rownames(pair) <- pair$self_cells
  pair <- cellAdmix:::prune_cells_test(pair,ncells_samp=ncells)
  if (nrow(pair)<5) return(NA)
  # Null candidates and real-score geometry only use the pruned pairs' cells.
  df <- load_cells(work,meta,unique(c(pair$self_cells,pair$adj_cells)))
  env <- if (fast) score_environment() else asNamespace('cellAdmix')
  nulls <- lapply(1:3,function(iteration) {
    env$compute_bridge_scores_null(fractions,pair,df,knn_k=20,ncells_samp=ncells)
  })
  combined <- lapply(seq_along(nulls[[1]]),function(f) {
    names_all <- lapply(nulls,function(x) names(x[[f]]))
    shared <- names_all[[1]]
    for (iteration in 2:3) shared <- intersect(shared,names_all[[iteration]])
    values <- lapply(nulls,function(x) x[[f]][shared])
    Matrix::rowMeans(do.call(cbind,values))
  })
  lapply(seq_len(ncol(fractions)),function(f) {
    baseline <- combined[[f]]
    if (is.na(baseline[1])) return(NA)
    selected <- pair[names(baseline),,drop=FALSE]
    observed <- env$compute_bridge_scores(fractions,f,df,selected,knn_k=20,abs_diff=TRUE)
    c(wilcox.test(observed,baseline,paired=TRUE,alternative='greater')[["p.value"]],
      quantile(observed,.75))
  })
}

false_positive <- function(meta,counts,bridge) {
  graph <- cellAdmix::knn_adjacency_matrix(meta[,c('x','y','z')],
                                        k=min(100L,nrow(meta)-1L))
  rownames(graph) <- colnames(graph) <- meta$cell
  types <- factor(meta$celltype)
  summary <- graph %*% sparseMatrix(i=seq_len(nrow(meta)),j=as.integer(types),
                                    x=rep(1,nrow(meta)))
  colnames(summary) <- levels(types)
  fractions <- counts/Matrix::rowSums(counts)
  cellAdmix::check_f_rm(setNames(meta$celltype,meta$cell),bridge[[1]],bridge[[2]],
                       fractions,summary,do.clean=TRUE,median.thresh=.1)
}

factor_map <- function(bridge,kept) {
  if (length(kept)==1L && is.na(kept)) kept <- character()
  mapping <- data.frame(cell_type=character(),factor=integer())
  for (name in names(bridge[[1]])) {
    factor <- as.integer(sub('^f_','',name))
    for (type in bridge[[1]][[name]]) {
      if (paste0(factor,'_',type) %in% kept)
        mapping <- rbind(mapping,data.frame(cell_type=type,factor=factor))
    }
  }
  unique(mapping)
}

main <- function(args) {
  if (length(args)!=3L) stop('Usage: bounded.R cache_dir output_dir factors')
  work <- args[1]; out <- args[2]; k <- as.integer(args[3])
  meta <- read.csv(file.path(work,'cells.csv'),stringsAsFactors=FALSE,
                   colClasses=c(cell='character',celltype='character'))
  counts <- as.matrix(read.csv(file.path(work,'factor_counts.csv'),check.names=FALSE))
  rownames(counts) <- meta$cell
  plan <- read.csv(file.path(work,'pair_plan.csv'),stringsAsFactors=FALSE)
  pairs <- read.csv(file.path(work,'pairs.csv'),stringsAsFactors=FALSE,
                    colClasses=c(self_cells='character',adj_cells='character'))
  checkpoint <- file.path(out,'bridge_pairs')
  dir.create(checkpoint,recursive=TRUE,showWarnings=FALSE)
  raw <- vector('list',nrow(plan)); names(raw) <- plan$name
  set.seed(0)
  for (i in seq_len(nrow(plan))) {
    path <- file.path(checkpoint,sprintf('pair_%03d.rds',i))
    if (file.exists(path)) {
      saved <- readRDS(path)
      if (!identical(saved$name,plan$name[i])) stop('Pair checkpoint order changed')
      raw[[i]] <- saved$result
      assign('.Random.seed',saved$rng,envir=.GlobalEnv)
    } else {
      cat(sprintf('BRIDGE %d/%d START %s candidates=%d\n',i,nrow(plan),plan$name[i],plan$n_pairs[i]))
      flush.console()
      raw[[i]] <- one_pair(work,meta,counts,pairs,i)
      atomic_rds(list(name=plan$name[i],result=raw[[i]],rng=.Random.seed),path)
    }
    jsonlite::write_json(list(stage='bridge',done=i,total=nrow(plan),pair=plan$name[i]),
                         file.path(work,'progress.json'),auto_unbox=TRUE)
    cat(sprintf('BRIDGE %d/%d DONE %s\n',i,nrow(plan),plan$name[i]));flush.console()
    gc()
  }
  atomic_rds(raw,file.path(out,'bridge_raw.rds'))
  cat('FACTOR EXTRACTION START\n');flush.console()
  bridge <- cellAdmix::extract_bridge_res(raw,unique(meta$celltype),k,
                         nmf.type='joint',p.thresh=.1,adj.pvals=FALSE)
  atomic_rds(bridge,file.path(out,'bridge_res.rds'))
  cat('FALSE-POSITIVE CHECK START (cell-level graph)\n');flush.console()
  jsonlite::write_json(list(stage='false_positive',done=nrow(plan),total=nrow(plan)),
                       file.path(work,'progress.json'),auto_unbox=TRUE)
  kept <- false_positive(meta,counts,bridge)
  mapping <- factor_map(bridge,kept)
  tmp <- file.path(out,'factor_map.csv.tmp')
  fwrite(mapping,tmp)
  if (!file.rename(tmp,file.path(out,'factor_map.csv'))) stop('Cannot publish factor map')
  jsonlite::write_json(list(stage='complete',n_factor_mappings=nrow(mapping),
                            pair_tests=nrow(plan),ncells_samp=200,knn_k=20,
                            p_threshold=.1,adjust_pvalues=FALSE,
                            fp_knn_k=min(100,nrow(meta)-1),fp_median_threshold=.1,
                            global_transcript_neighbors=TRUE,transcript_downsampling=FALSE),
                       file.path(out,'annotation_complete.json'),auto_unbox=TRUE,pretty=TRUE)
  cat('COMPLETE factor mappings:',nrow(mapping),'\n')
}

if (sys.nframe()==0L) main(commandArgs(trailingOnly=TRUE))
