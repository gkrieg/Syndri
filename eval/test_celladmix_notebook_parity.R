# Usage: Rscript test_celladmix_notebook_parity.R /path/to/fixture
# Fixture contains spatial/reference.mtx, *_obs.csv, genes.csv, and output/.
# Verifies that the matrix bridge reproduces the actual transcript-table
# notebook wrapper for integer counts, including its background adjustment.
suppressPackageStartupMessages({library(Matrix); library(Seurat); library(dplyr); library(magrittr)})
source("/home/jwinchel/cellAdmix/R/admix_scores.R")
source("/home/jwinchel/cellAdmix/R/utils.R")
source("/home/jwinchel/cellAdmix-Notebooks/R/benchmarking.R")
p <- commandArgs(trailingOnly=TRUE)[[1]]
genes <- read.csv(file.path(p, "genes.csv"))$gene
read_input <- function(name) {
  obs <- read.csv(file.path(p, paste0(name, "_obs.csv")), row.names=1)
  cm <- as(readMM(file.path(p, paste0(name, ".mtx"))), "CsparseMatrix")
  dimnames(cm) <- list(genes, rownames(obs))
  list(cm=cm, obs=obs)
}
r <- read_input("reference"); s <- read_input("spatial")
so <- CreateSeuratObject(r$cm, meta.data=r$obs)
ij <- summary(s$cm)
stopifnot(all(ij$x == as.integer(ij$x)))
tx <- data.frame(gene=genes[ij$i], cell=colnames(s$cm)[ij$j],
                 celltype=s$obs$cell_type[ij$j])
tx <- tx[rep(seq_len(nrow(tx)), ij$x), ]
K <- as.matrix(read.csv(file.path(p, "output/cell_type_adjacency.csv"), row.names=1))
ex <- read.csv(file.path(p, "output/gene_correlation_preservation.csv"))
expected <- get_dbl_scores(list(.4, ex$gene[ex$excluded], .25, .05, NULL, .1), tx, so, K)
actual <- read.csv(file.path(p, "output/admixture_scores.csv")) %>% filter(setting == "raw")
stopifnot(isTRUE(all.equal(expected$scores, actual$scores[match(expected$cell, actual$cell)], tolerance=1e-12)))
cat("Matrix bridge agrees with cloned notebook get_dbl_scores for", nrow(expected), "nonempty cells\n")
