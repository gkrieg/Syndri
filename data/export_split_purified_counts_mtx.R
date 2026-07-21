library(Matrix)

args <- commandArgs(trailingOnly = TRUE)

output_root <- if (length(args) >= 1 && nzchar(args[[1]])) {
  args[[1]]
} else {
  "/home/jwinchel/Syndri/data/brca_split_output_by_sample"
}

overwrite <- if (length(args) >= 2 && nzchar(args[[2]])) {
  tolower(args[[2]]) %in% c("true", "t", "1", "yes", "y")
} else {
  tolower(Sys.getenv("SPLIT_EXPORT_OVERWRITE", "false")) %in% c("true", "t", "1", "yes", "y")
}

input_name <- "split_purified_counts.rds"
mtx_name <- "split_purified_counts.mtx"
genes_name <- "split_purified_counts_genes.txt"
cells_name <- "split_purified_counts_cells.txt"

find_count_files <- function(root) {
  list.files(
    root,
    pattern = paste0("^", input_name, "$"),
    recursive = TRUE,
    full.names = TRUE
  )
}

export_one <- function(counts_path, overwrite) {
  sample_dir <- dirname(counts_path)
  sample_id <- basename(sample_dir)

  mtx_path <- file.path(sample_dir, mtx_name)
  genes_path <- file.path(sample_dir, genes_name)
  cells_path <- file.path(sample_dir, cells_name)
  output_paths <- c(mtx_path, genes_path, cells_path)

  if (!overwrite && all(file.exists(output_paths))) {
    message("[", sample_id, "] outputs already exist; skipping")
    return(data.frame(
      sample_id = sample_id,
      status = "skipped",
      n_genes = NA_integer_,
      n_cells = NA_integer_,
      counts_path = counts_path,
      error = NA_character_,
      stringsAsFactors = FALSE
    ))
  }

  message("[", sample_id, "] reading ", counts_path)
  counts <- readRDS(counts_path)

  if (is.null(rownames(counts))) {
    stop("purified counts matrix has no gene row names")
  }
  if (is.null(colnames(counts))) {
    stop("purified counts matrix has no cell column names")
  }

  if (!inherits(counts, "sparseMatrix")) {
    counts <- as(counts, "sparseMatrix")
  }

  message("[", sample_id, "] writing Matrix Market files")
  writeMM(counts, mtx_path)
  writeLines(rownames(counts), genes_path)
  writeLines(colnames(counts), cells_path)

  data.frame(
    sample_id = sample_id,
    status = "completed",
    n_genes = nrow(counts),
    n_cells = ncol(counts),
    counts_path = counts_path,
    error = NA_character_,
    stringsAsFactors = FALSE
  )
}

if (!dir.exists(output_root)) {
  stop("Output root does not exist: ", output_root)
}

counts_paths <- sort(find_count_files(output_root))
if (length(counts_paths) == 0) {
  stop("No ", input_name, " files found under: ", output_root)
}

message("Output root: ", output_root)
message("Files found: ", length(counts_paths))
message("Overwrite: ", overwrite)

results <- lapply(counts_paths, function(counts_path) {
  tryCatch(
    export_one(counts_path, overwrite),
    error = function(err) {
      sample_id <- basename(dirname(counts_path))
      message("[", sample_id, "] failed: ", conditionMessage(err))
      data.frame(
        sample_id = sample_id,
        status = "failed",
        n_genes = NA_integer_,
        n_cells = NA_integer_,
        counts_path = counts_path,
        error = conditionMessage(err),
        stringsAsFactors = FALSE
      )
    }
  )
})

manifest <- do.call(rbind, results)
manifest_path <- file.path(output_root, "split_purified_counts_mtx_manifest.csv")
write.csv(manifest, manifest_path, row.names = FALSE)

message("Wrote manifest: ", manifest_path)

if (any(manifest$status == "failed")) {
  failed <- manifest$sample_id[manifest$status == "failed"]
  stop("Export failed for ", length(failed), " sample(s): ", paste(failed, collapse = ", "))
}
