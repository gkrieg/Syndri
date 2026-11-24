#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)
source(file.path(args[1],'celladmix_annotation_bounded.R'))
work <- args[2]; k <- as.integer(args[3]); report <- args[4]
meta <- read.csv(file.path(work,'cells.csv'),stringsAsFactors=FALSE,
                 colClasses=c(cell='character',celltype='character'))
counts <- as.matrix(read.csv(file.path(work,'factor_counts.csv'),check.names=FALSE))
rownames(counts) <- meta$cell
df <- load_cells(work,meta,meta$cell)
df$celltype <- meta$celltype[match(df$cell,meta$cell)]
df$mol_id <- seq_len(nrow(df))
rownames(df) <- as.character(df$mol_id)
original_counts <- table(df$cell,factor(df$factor))
stopifnot(identical(unname(original_counts),unname(structure(counts,class='table'))))
original_meta <- as.data.table(df)[,.(celltype=celltype[1],x=mean(x),y=mean(y),z=mean(z)),by=cell]
stopifnot(isTRUE(all.equal(meta[,c('x','y','z')],as.data.frame(original_meta)[,c('x','y','z')],tolerance=1e-12)))
rownames(meta) <- meta$cell
cat('REFERENCE GLOBAL KNN\n');flush.console()
knn <- cellAdmix:::get_mol_knn(df,knn_k=10)
neighbor <- FNN::get.knn(as.matrix(df[,c('x','y','z')]),k=10)$nn.index
con <- file(file.path(work,'reference_neighbors.dat'),'wb')
writeBin(as.integer(t(neighbor))-1L,con,size=4,endian='little');close(con)
plan <- read.csv(file.path(work,'pair_plan.csv'),stringsAsFactors=FALSE)
pairs <- read.csv(file.path(work,'pairs.csv'),stringsAsFactors=FALSE,
                  colClasses=c(self_cells='character',adj_cells='character'))
for (i in seq_len(nrow(plan))) {
  reference <- cellAdmix:::get_adj_pairs(df,meta,knn[[1]],knn[[2]],plan$ct1[i],plan$ct2[i])
  bounded <- pairs[pairs$pair_index==i,c('self_cells','adj_cells'),drop=FALSE]
  if (is.null(reference)) stopifnot(nrow(bounded)==0L) else {
    stopifnot(identical(as.character(reference$self_cells),bounded$self_cells),
              identical(as.character(reference$adj_cells),bounded$adj_cells))
  }
}
cat('CELL PAIRS EXACT; REFERENCE BRIDGE TESTS\n');flush.console()
set.seed(0)
original <- cellAdmix::run_bridge_test(df,matrix(df$factor,ncol=1),meta,
                         knn[[1]],knn[[2]],ncells.samp=200,knn.k=20,n.cores=1)
set.seed(0)
bounded <- lapply(seq_len(nrow(plan)),function(i) one_pair(work,meta,counts,pairs,i,fast=TRUE))
names(bounded) <- plan$name
comparison <- all.equal(original,bounded,tolerance=1e-12,check.attributes=FALSE)
stopifnot(isTRUE(comparison))
bridge0 <- cellAdmix::extract_bridge_res(original,unique(meta$celltype),k,nmf.type='joint',p.thresh=.1,adj.pvals=FALSE)
bridge1 <- cellAdmix::extract_bridge_res(bounded,unique(meta$celltype),k,nmf.type='joint',p.thresh=.1,adj.pvals=FALSE)
stopifnot(isTRUE(all.equal(bridge0,bridge1,tolerance=1e-12)))
helper <- new.env(parent=asNamespace('cellAdmix'))
sys.source('/home/jwinchel/cellAdmix/R/vignette_helpers.R',envir=helper)
keep0 <- helper$check_fp(df,meta,matrix(df$factor,ncol=1),bridge0,do_clean=TRUE,
                         knn_k=min(100L,nrow(meta)-1L),median_thresh=.1)
keep1 <- false_positive(meta,counts,bridge1)
stopifnot(identical(keep0,keep1),identical(factor_map(bridge0,keep0),factor_map(bridge1,keep1)))
numeric0 <- unlist(original); numeric1 <- unlist(bounded)
valid <- is.finite(numeric0)&is.finite(numeric1)
result <- list(status='passed',n_cells=nrow(meta),n_transcripts=nrow(df),
                cell_counts_exact=TRUE,cell_centroids_tolerance=1e-12,
                cell_pairs_exact=TRUE,bridge_statistics_equal=TRUE,
                max_bridge_difference=if(any(valid)) max(abs(numeric0[valid]-numeric1[valid])) else 0,
                n_finite_bridge_statistics=sum(valid),
                bridge_factor_annotations_equal=TRUE,false_positive_filter_equal=TRUE,
                final_factor_map_equal=TRUE,n_factor_mappings=nrow(factor_map(bridge1,keep1)),
                original_implementation='installed cellAdmix, 1 worker, seed 0')
jsonlite::write_json(result,report,pretty=TRUE,auto_unbox=TRUE,na='null')
print(result)

# Exercise a NONEMPTY factor map and a rejected native factor independently
# of whether this small test cohort has enough power for a significant bridge.
forced_meta <- data.frame(cell=sprintf('test_%04d',1:1200),
                         celltype=rep(c('A','B','C'),each=400),
                         x=rep((0:399)%%20,3)+rep(c(0,100,200),each=400),
                         y=rep((0:399)%/%20,3),z=0)
rownames(forced_meta) <- forced_meta$cell
forced_meta$x[1:100] <- forced_meta$x[1:100]+100
forced_meta$x[101:200] <- forced_meta$x[101:200]+200
forced_counts <- rbind(matrix(rep(c(78,2,20),400),ncol=3,byrow=TRUE),
                       matrix(rep(c(3,95,2),400),ncol=3,byrow=TRUE),
                       matrix(rep(c(3,2,95),400),ncol=3,byrow=TRUE))
forced_counts[1:100,] <- matrix(rep(c(60,20,20),100),ncol=3,byrow=TRUE)
rownames(forced_counts) <- forced_meta$cell;colnames(forced_counts) <- as.character(1:3)
forced_df <- data.frame(cell=rep(forced_meta$cell,each=100),
                        factor=unlist(lapply(1:1200,function(i) rep(1:3,forced_counts[i,]))))
forced_bridge <- list(list(f_2='A',f_3='A'),list(f_2='B',f_3='C'))
forced0 <- helper$check_fp(forced_df,forced_meta,matrix(forced_df$factor,ncol=1),
                           forced_bridge,do_clean=TRUE,knn_k=100,median_thresh=.1)
forced1 <- false_positive(forced_meta,forced_counts,forced_bridge)
stopifnot(identical(forced0,forced1),'2_A' %in% forced1,!('3_A' %in% forced1))
result$nonempty_false_positive_map_exact <- TRUE
result$nonempty_test_kept <- forced1
jsonlite::write_json(result,report,pretty=TRUE,auto_unbox=TRUE,na='null')
