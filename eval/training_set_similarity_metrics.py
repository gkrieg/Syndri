from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ks_2samp, pearsonr, spearmanr, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
HEIMDALL_ROOT = REPO_ROOT / "Heimdall"
if str(HEIMDALL_ROOT) not in sys.path:
    sys.path.append(str(HEIMDALL_ROOT))


NULL_STRINGS = {"", "none", "null", "nan", "na"}


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str | None) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def parse_int_csv(value: str | None) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_tau_range(value: str | None) -> tuple[float, float]:
    values = parse_float_csv(str(value).replace("[", "").replace("]", "") if value is not None else None)
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"Expected two comma-separated tau values, got {value!r}.")
    if values[0] < 0 or values[0] >= 1 or values[1] > 1 or values[0] > values[1]:
        raise argparse.ArgumentTypeError("tau range must satisfy 0 <= low <= high <= 1 and low < 1.")
    return float(values[0]), float(values[1])


def optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.lower() in NULL_STRINGS:
        return None
    return normalized


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare spatial transcriptomics counts with simulated pseudo-ST counts "
            "from single-cell AnnData objects."
        )
    )
    parser.add_argument("--spatial-path", default="", help="Spatial transcriptomics h5ad.")
    parser.add_argument("--single-cell-path", default="", help="Single-cell h5ad with raw and admixed counts.")
    parser.add_argument("--output-dir", default="training_set_similarity_metrics")
    parser.add_argument("--output-prefix", default="training_set_similarity")
    parser.add_argument("--config-id", default="")

    parser.add_argument(
        "--manifest",
        default="",
        help=(
            "Optional CSV/TSV manifest. Each array task reads one row selected by "
            "--manifest-index. Columns may include any CLI option name without leading dashes."
        ),
    )
    parser.add_argument("--manifest-index", type=int, default=None)
    parser.add_argument("--manifest-delimiter", default="auto", choices=["auto", "comma", "tab"])

    parser.add_argument("--spatial-layer", default="", help="Spatial count layer. Empty or X uses .X.")
    parser.add_argument("--single-cell-raw-layer", default="", help="Raw single-cell layer. Empty or X uses .X.")
    parser.add_argument(
        "--single-cell-admix-layer",
        default="auto",
        help=(
            "Layer containing additive admixed transcript counts. 'auto' tries "
            "admixed_counts, admixed, and admixture_counts."
        ),
    )
    parser.add_argument(
        "--single-cell-sim-layer",
        default="",
        help="Optional precomputed simulated pseudo-ST layer. If set, raw + admixed construction is skipped.",
    )
    parser.add_argument(
        "--admixture-scales",
        default="1.0",
        help="Comma-separated scales for raw + scale * admixed_counts.",
    )
    parser.add_argument(
        "--sample-admixture-from-neighbors",
        type=str_to_bool,
        default=False,
        help="Generate admixed counts from Heimdall neighbor sampling instead of reading --single-cell-admix-layer.",
    )
    parser.add_argument("--neighbor-dict-key", default="neighbor_dict")
    parser.add_argument("--neighbor-id-col", default="")
    parser.add_argument(
        "--build-neighbor-dict",
        type=str_to_bool,
        default=True,
        help="If neighbor_dict_key is absent, build neighbors from single-cell coordinates.",
    )
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--coord-x", default="sampled_st_x")
    parser.add_argument("--coord-y", default="sampled_st_y")
    parser.add_argument("--tissue-key", default="harmony_spatial_tissue")
    parser.add_argument("--neighbor-cell-type-key", default="")
    parser.add_argument("--min-source-transcript-count", type=int, default=5)
    parser.add_argument("--use-same-cell-type-neighbors", type=str_to_bool, default=False)
    parser.add_argument("--min-sampled-transcript-threshold", type=int, default=0)
    parser.add_argument("--only-sample-unexpressed-genes", type=str_to_bool, default=True)
    parser.add_argument("--mask-out-random-genes", type=str_to_bool, default=False)
    parser.add_argument("--mask-out-frac", type=float, default=0.0)
    parser.add_argument("--tau-range", default="0.4,0.7")
    parser.add_argument("--admixture-repeats", type=int, default=1)

    parser.add_argument("--spatial-cell-type-key", default="cell_type")
    parser.add_argument("--single-cell-cell-type-key", default="cell_type")
    parser.add_argument(
        "--group-mode",
        choices=["all", "cell_type", "both"],
        default="both",
        help="Whether to compute gene/cell metrics globally, by matched cell type, or both.",
    )
    parser.add_argument("--min-cells-per-group", type=int, default=20)
    parser.add_argument("--max-cells-per-domain", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=0)

    parser.add_argument("--top-gene-counts", default="50,100,500")
    parser.add_argument("--skip-positive-medians", action="store_true")
    parser.add_argument(
        "--classifier",
        type=str_to_bool,
        default=True,
        help="Run logistic-regression domain classifier. Lower AUC means simulated pseudo-ST is harder to distinguish.",
    )
    parser.add_argument("--classifier-by-group", action="store_true")
    parser.add_argument("--classifier-max-cells-per-domain", type=int, default=5000, help="Use 0 for all cells.")
    parser.add_argument("--classifier-max-features", type=int, default=2000)
    parser.add_argument("--classifier-cv", type=int, default=5)
    parser.add_argument("--classifier-normalize", choices=["log1p_cpm", "none"], default="log1p_cpm")
    parser.add_argument("--library-size-target-sum", type=float, default=1e4)
    parser.add_argument("--classifier-max-iter", type=int, default=1000)

    parser.add_argument("--write-per-gene", action="store_true")
    return parser


def read_manifest_row(path: Path, index: int, delimiter: str) -> dict[str, str]:
    text = path.read_text()
    if delimiter == "tab":
        dialect_delimiter = "\t"
    elif delimiter == "comma":
        dialect_delimiter = ","
    else:
        sample = "\n".join(line for line in text.splitlines()[:10] if line.strip())
        dialect_delimiter = "\t" if "\t" in sample and sample.count("\t") >= sample.count(",") else ","

    rows = list(csv.DictReader(text.splitlines(), delimiter=dialect_delimiter))
    if not rows:
        raise ValueError(f"Manifest {path} has no data rows.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Manifest index {index} is out of range for {len(rows)} rows.")
    row = {str(k).strip().replace("-", "_"): str(v).strip() for k, v in rows[index].items() if k is not None}
    aliases = {
        "spatial_h5ad": "spatial_path",
        "spatial_adata": "spatial_path",
        "single_cell_h5ad": "single_cell_path",
        "single_cell_adata": "single_cell_path",
        "sc_path": "single_cell_path",
        "sc_h5ad": "single_cell_path",
        "raw_layer": "single_cell_raw_layer",
        "admix_layer": "single_cell_admix_layer",
        "admixed_layer": "single_cell_admix_layer",
        "sim_layer": "single_cell_sim_layer",
        "simulated_layer": "single_cell_sim_layer",
        "admixture_scale": "admixture_scales",
        "k": "k_neighbors",
        "min_source_counts": "min_source_transcript_count",
        "min_source_count": "min_source_transcript_count",
        "min_sampled_threshold": "min_sampled_transcript_threshold",
        "min_sampled_transcript_count": "min_sampled_transcript_threshold",
        "same_cell_type_neighbors": "use_same_cell_type_neighbors",
        "only_unexpressed_genes": "only_sample_unexpressed_genes",
        "tau": "tau_range",
        "cell_type_key": "spatial_cell_type_key",
        "sc_cell_type_key": "single_cell_cell_type_key",
    }
    for source, dest in aliases.items():
        if source in row and dest not in row:
            row[dest] = row[source]
    return row


def apply_manifest_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if not args.manifest:
        return args
    if args.manifest_index is None:
        raise ValueError("--manifest-index is required when --manifest is set.")

    row = read_manifest_row(Path(args.manifest).expanduser(), args.manifest_index, args.manifest_delimiter)
    actions = {action.dest: action for action in parser._actions}
    for key, value in row.items():
        if key not in actions or value == "":
            continue
        action = actions[key]
        if isinstance(action, argparse._StoreTrueAction):
            setattr(args, key, str_to_bool(value))
        elif action.type is not None:
            setattr(args, key, action.type(value))
        else:
            setattr(args, key, value)
    return args


def matrix_from_adata(adata: ad.AnnData, layer: str | None) -> sparse.csr_matrix:
    layer = optional_str(layer)
    if layer is None or layer == "X":
        X = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not found. Available layers: {sorted(adata.layers.keys())}")
        X = adata.layers[layer]
    if sparse.issparse(X):
        return X.tocsr().astype(np.float64)
    return sparse.csr_matrix(np.asarray(X, dtype=np.float64))


def find_admix_layer(adata: ad.AnnData, requested: str | None) -> str | None:
    requested = optional_str(requested)
    if requested is None:
        return None
    if requested != "auto":
        if requested not in adata.layers:
            raise KeyError(f"Layer {requested!r} not found. Available layers: {sorted(adata.layers.keys())}")
        return requested
    for candidate in ["admixed_counts", "admixed", "admixture_counts"]:
        if candidate in adata.layers:
            return candidate
    raise KeyError(
        "Could not auto-detect an admixed-count layer. Pass --single-cell-admix-layer. "
        f"Available layers: {sorted(adata.layers.keys())}"
    )


def generated_admixture_config(args: argparse.Namespace) -> dict[str, Any]:
    tau_low, tau_high = parse_tau_range(args.tau_range)
    return {
        "sample_admixture_from_neighbors": bool(args.sample_admixture_from_neighbors),
        "neighbor_dict_key": args.neighbor_dict_key,
        "neighbor_id_col": optional_str(args.neighbor_id_col) or "",
        "build_neighbor_dict": bool(args.build_neighbor_dict),
        "k_neighbors": int(args.k_neighbors),
        "coord_x": args.coord_x,
        "coord_y": args.coord_y,
        "tissue_key": optional_str(args.tissue_key) or "",
        "neighbor_cell_type_key": optional_str(args.neighbor_cell_type_key) or optional_str(args.single_cell_cell_type_key) or "",
        "min_source_transcript_count": int(args.min_source_transcript_count),
        "use_same_cell_type_neighbors": bool(args.use_same_cell_type_neighbors),
        "min_sampled_transcript_threshold": int(args.min_sampled_transcript_threshold),
        "only_sample_unexpressed_genes": bool(args.only_sample_unexpressed_genes),
        "mask_out_random_genes": bool(args.mask_out_random_genes),
        "mask_out_frac": float(args.mask_out_frac),
        "tau_low": tau_low,
        "tau_high": tau_high,
        "admixture_repeats": int(args.admixture_repeats),
    }


def sample_admixed_counts_from_neighbors(
    single_cell: ad.AnnData,
    raw_counts: sparse.csr_matrix,
    shared_gene_idx: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    from Heimdall.admixture_sampling import build_neighbor_dict, sample_admixture_from_neighbor_dict

    sampler_adata = single_cell
    sampler_adata.X = raw_counts
    neighbor_key = args.neighbor_dict_key
    cell_type_key = optional_str(args.neighbor_cell_type_key) or optional_str(args.single_cell_cell_type_key)
    id_col = optional_str(args.neighbor_id_col)

    if neighbor_key not in sampler_adata.uns:
        if not args.build_neighbor_dict:
            raise KeyError(
                f"Neighbor dictionary {neighbor_key!r} not found in single-cell .uns and --build-neighbor-dict is false."
            )
        if cell_type_key is None:
            raise ValueError("--neighbor-cell-type-key or --single-cell-cell-type-key is required to build neighbors.")
        sampler_adata = build_neighbor_dict(
            sampler_adata,
            k_neighbors=args.k_neighbors,
            coord_cols=(args.coord_x, args.coord_y),
            tissue_col=optional_str(args.tissue_key),
            cell_type_col=cell_type_key,
            min_source_counts=args.min_source_transcript_count,
            use_same_cell_type_neighbors=args.use_same_cell_type_neighbors,
            id_col=id_col,
            uns_key=neighbor_key,
        )

    admixed, _ = sample_admixture_from_neighbor_dict(
        sampler_adata,
        tau_range=parse_tau_range(args.tau_range),
        n_repeats=args.admixture_repeats,
        min_sampled_transcript_threshold=args.min_sampled_transcript_threshold,
        uns_key=neighbor_key,
        id_col=id_col,
        only_sample_unexpressed_genes=args.only_sample_unexpressed_genes,
        mask_out_random_genes=args.mask_out_random_genes,
        mask_out_frac=args.mask_out_frac,
        rng=rng,
        return_sparse=args.admixture_repeats == 1,
    )
    if args.admixture_repeats != 1:
        raise ValueError("Only --admixture-repeats 1 is currently supported by the similarity evaluator.")
    if sparse.issparse(admixed):
        return admixed[:, shared_gene_idx].tocsr().astype(np.float64)
    return sparse.csr_matrix(np.asarray(admixed, dtype=np.float64))[:, shared_gene_idx].tocsr()


def row_sum(X: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(X.sum(axis=1)).ravel()


def row_max(X: sparse.csr_matrix) -> np.ndarray:
    if X.shape[0] == 0:
        return np.array([], dtype=float)
    values = X.max(axis=1)
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values).ravel()


def cell_metric_vectors(X: sparse.csr_matrix) -> dict[str, np.ndarray]:
    return {
        "detected_genes_per_cell": X.getnnz(axis=1).astype(float),
        "total_counts_per_cell": row_sum(X).astype(float),
        "max_count_per_cell": row_max(X).astype(float),
    }


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def describe(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = finite(values)
    if values.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p25": np.nan,
            f"{prefix}_p75": np.nan,
            f"{prefix}_p95": np.nan,
        }
    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_p25": float(np.quantile(values, 0.25)),
        f"{prefix}_p75": float(np.quantile(values, 0.75)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
    }


def compare_distributions(spatial: np.ndarray, simulated: np.ndarray) -> dict[str, float]:
    spatial = finite(spatial)
    simulated = finite(simulated)
    if spatial.size == 0 or simulated.size == 0:
        return {
            "wasserstein": np.nan,
            "normalized_wasserstein": np.nan,
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "abs_log2_median_ratio": np.nan,
            "abs_log2_mean_ratio": np.nan,
        }
    wd = float(wasserstein_distance(spatial, simulated))
    ks = ks_2samp(spatial, simulated)
    pooled_median = float(np.median(np.concatenate([spatial, simulated])))
    eps = 1e-12
    return {
        "wasserstein": wd,
        "normalized_wasserstein": wd / max(abs(pooled_median), eps),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "abs_log2_median_ratio": float(abs(math.log2((np.median(simulated) + eps) / (np.median(spatial) + eps)))),
        "abs_log2_mean_ratio": float(abs(math.log2((np.mean(simulated) + eps) / (np.mean(spatial) + eps)))),
    }


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan, int(x.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            result = pearsonr(x, y)
        elif method == "spearman":
            result = spearmanr(x, y)
        else:
            raise ValueError(method)
    return float(result.statistic), float(result.pvalue), int(x.size)


def detection_fraction(X: sparse.csr_matrix) -> np.ndarray:
    if X.shape[0] == 0:
        return np.full(X.shape[1], np.nan)
    csc = X.tocsc()
    return np.diff(csc.indptr).astype(float) / float(X.shape[0])


def mean_expression(X: sparse.csr_matrix) -> np.ndarray:
    if X.shape[0] == 0:
        return np.full(X.shape[1], np.nan)
    return np.asarray(X.mean(axis=0)).ravel()


def positive_mean_expression(X: sparse.csr_matrix) -> np.ndarray:
    csc = X.tocsc()
    nnz = np.diff(csc.indptr).astype(float)
    sums = np.asarray(csc.sum(axis=0)).ravel()
    out = np.full(X.shape[1], np.nan)
    mask = nnz > 0
    out[mask] = sums[mask] / nnz[mask]
    return out


def positive_median_expression(X: sparse.csr_matrix) -> np.ndarray:
    csc = X.tocsc()
    medians = np.full(X.shape[1], np.nan)
    for gene_idx in range(X.shape[1]):
        start, stop = csc.indptr[gene_idx], csc.indptr[gene_idx + 1]
        if stop > start:
            medians[gene_idx] = float(np.median(csc.data[start:stop]))
    return medians


def correlation_rows(
    config: dict[str, Any],
    group: str,
    metric_name: str,
    spatial_values: np.ndarray,
    simulated_values: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ["pearson", "spearman"]:
        statistic, pvalue, n_genes = safe_corr(spatial_values, simulated_values, method)
        rows.append(
            {
                **config,
                "group": group,
                "metric": metric_name,
                "correlation_method": method,
                "correlation": statistic,
                "pvalue": pvalue,
                "n_genes": n_genes,
            }
        )
    return rows


def rank_agreement_rows(
    config: dict[str, Any],
    group: str,
    genes: pd.Index,
    spatial_mean: np.ndarray,
    simulated_mean: np.ndarray,
    top_gene_counts: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid = np.isfinite(spatial_mean) & np.isfinite(simulated_mean)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return rows

    spatial_order = valid_idx[np.argsort(-spatial_mean[valid_idx], kind="mergesort")]
    simulated_order = valid_idx[np.argsort(-simulated_mean[valid_idx], kind="mergesort")]
    simulated_ranks = np.empty(len(genes), dtype=float)
    simulated_ranks.fill(np.nan)
    simulated_ranks[simulated_order] = np.arange(1, simulated_order.size + 1, dtype=float)

    for top_k in top_gene_counts:
        k = min(int(top_k), spatial_order.size, simulated_order.size)
        if k <= 1:
            continue
        spatial_top = spatial_order[:k]
        simulated_top = set(simulated_order[:k].tolist())
        overlap = len(set(spatial_top.tolist()) & simulated_top)
        rank_corr, rank_pvalue, n_ranked = safe_corr(
            np.arange(1, k + 1, dtype=float),
            simulated_ranks[spatial_top],
            "spearman",
        )
        rows.append(
            {
                **config,
                "group": group,
                "top_k": k,
                "overlap_count": overlap,
                "overlap_fraction": overlap / float(k),
                "spearman_on_spatial_top_genes": rank_corr,
                "spearman_pvalue": rank_pvalue,
                "n_ranked_genes": n_ranked,
            }
        )
    return rows


def normalize_log1p_cpm(X: sparse.csr_matrix, target_sum: float) -> sparse.csr_matrix:
    X = X.tocsr(copy=True).astype(np.float64)
    sums = row_sum(X)
    scale = np.zeros_like(sums, dtype=float)
    positive = sums > 0
    scale[positive] = target_sum / sums[positive]
    X = sparse.diags(scale).dot(X).tocsr()
    X.data = np.log1p(X.data)
    return X


def sample_rows(indices: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if max_n <= 0 or indices.size <= max_n:
        return indices
    return np.sort(rng.choice(indices, size=max_n, replace=False))


def top_variance_features(X: sparse.csr_matrix, max_features: int) -> np.ndarray:
    if max_features <= 0 or X.shape[1] <= max_features:
        return np.arange(X.shape[1])
    mean = np.asarray(X.mean(axis=0)).ravel()
    second = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    variance = second - mean**2
    top = np.argpartition(variance, -max_features)[-max_features:]
    return np.sort(top)


def domain_classifier_row(
    config: dict[str, Any],
    scope: str,
    X_spatial: sparse.csr_matrix,
    X_simulated: sparse.csr_matrix,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n_available = min(X_spatial.shape[0], X_simulated.shape[0])
    n = n_available if args.classifier_max_cells_per_domain <= 0 else min(args.classifier_max_cells_per_domain, n_available)
    if n < 4:
        return {
            **config,
            "classifier_scope": scope,
            "auc_mean": np.nan,
            "auc_sd": np.nan,
            "accuracy_mean": np.nan,
            "accuracy_sd": np.nan,
            "n_spatial_cells": int(X_spatial.shape[0]),
            "n_simulated_cells": int(X_simulated.shape[0]),
            "n_features": int(X_spatial.shape[1]),
            "status": "skipped_too_few_cells",
        }

    spatial_idx = sample_rows(np.arange(X_spatial.shape[0]), n, rng)
    simulated_idx = sample_rows(np.arange(X_simulated.shape[0]), n, rng)
    X = sparse.vstack([X_spatial[spatial_idx], X_simulated[simulated_idx]], format="csr")
    y = np.array([0] * len(spatial_idx) + [1] * len(simulated_idx), dtype=int)

    if args.classifier_normalize == "log1p_cpm":
        X = normalize_log1p_cpm(X, args.library_size_target_sum)

    feature_idx = top_variance_features(X, args.classifier_max_features)
    X = X[:, feature_idx].tocsr()
    min_class = int(np.bincount(y).min())
    n_splits = min(args.classifier_cv, min_class)
    if n_splits < 2:
        status = "skipped_too_few_cells_per_class"
        auc = acc = np.array([np.nan])
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                class_weight="balanced",
                max_iter=args.classifier_max_iter,
                solver="liblinear",
                random_state=args.random_state,
            ),
        )
        scoring = {"auc": "roc_auc", "accuracy": "accuracy"}
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=1)
        auc = scores["test_auc"]
        acc = scores["test_accuracy"]
        status = "ok"

    return {
        **config,
        "classifier_scope": scope,
        "auc_mean": float(np.nanmean(auc)),
        "auc_sd": float(np.nanstd(auc)),
        "accuracy_mean": float(np.nanmean(acc)),
        "accuracy_sd": float(np.nanstd(acc)),
        "n_spatial_cells": int(X_spatial.shape[0]),
        "n_simulated_cells": int(X_simulated.shape[0]),
        "n_sampled_per_domain": int(n),
        "n_features": int(X.shape[1]),
        "n_cv_splits": int(n_splits),
        "status": status,
    }


def labels_for(adata: ad.AnnData, key: str | None) -> np.ndarray:
    key = optional_str(key)
    if key is None:
        return np.full(adata.n_obs, "all", dtype=object)
    if key not in adata.obs:
        raise KeyError(f"obs column {key!r} not found. Available columns: {list(adata.obs.columns)}")
    labels = adata.obs[key].astype("string").fillna("nan").astype(str).to_numpy()
    labels[np.isin(np.char.lower(labels.astype(str)), list(NULL_STRINGS))] = "nan"
    return labels


def group_index(labels: np.ndarray, group: str) -> np.ndarray:
    if group == "all":
        return np.arange(labels.size)
    return np.flatnonzero(labels == group)


def selected_groups(
    spatial_labels: np.ndarray,
    sc_labels: np.ndarray,
    group_mode: str,
    min_cells: int,
) -> list[str]:
    groups: list[str] = []
    if group_mode in {"all", "both"}:
        groups.append("all")
    if group_mode in {"cell_type", "both"}:
        spatial_counts = pd.Series(spatial_labels).value_counts()
        sc_counts = pd.Series(sc_labels).value_counts()
        common = sorted((set(spatial_counts.index) & set(sc_counts.index)) - {"nan", "all"})
        groups.extend(
            group
            for group in common
            if spatial_counts[group] >= min_cells and sc_counts[group] >= min_cells
        )
    return groups


def clean_scale_for_id(scale: float) -> str:
    return f"{scale:g}".replace("-", "m").replace(".", "p")


def make_config(args: argparse.Namespace, scale: float, admix_layer: str | None) -> dict[str, Any]:
    base = optional_str(args.config_id)
    if base is None:
        spatial_name = Path(args.spatial_path).stem
        sc_name = Path(args.single_cell_path).stem
        base = f"{spatial_name}__{sc_name}"
    config_id = base if len(parse_float_csv(args.admixture_scales)) == 1 else f"{base}__scale_{clean_scale_for_id(scale)}"
    config = {
        "config_id": config_id,
        "spatial_path": str(Path(args.spatial_path).expanduser()),
        "single_cell_path": str(Path(args.single_cell_path).expanduser()),
        "spatial_layer": optional_str(args.spatial_layer) or "X",
        "single_cell_raw_layer": optional_str(args.single_cell_raw_layer) or "X",
        "single_cell_admix_layer": admix_layer or "",
        "single_cell_sim_layer": optional_str(args.single_cell_sim_layer) or "",
        "admixture_scale": float(scale),
    }
    config.update(generated_admixture_config(args))
    return config


def add_cell_metric_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    group: str,
    X_spatial: sparse.csr_matrix,
    X_simulated: sparse.csr_matrix,
) -> None:
    spatial_metrics = cell_metric_vectors(X_spatial)
    simulated_metrics = cell_metric_vectors(X_simulated)
    for metric_name, spatial_values in spatial_metrics.items():
        simulated_values = simulated_metrics[metric_name]
        rows.append(
            {
                **config,
                "group": group,
                "metric": metric_name,
                **describe(spatial_values, "spatial"),
                **describe(simulated_values, "simulated"),
                **compare_distributions(spatial_values, simulated_values),
            }
        )


def add_gene_metric_rows(
    gene_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    per_gene_rows: list[dict[str, Any]],
    config: dict[str, Any],
    group: str,
    genes: pd.Index,
    X_spatial: sparse.csr_matrix,
    X_simulated: sparse.csr_matrix,
    args: argparse.Namespace,
) -> None:
    spatial_detect = detection_fraction(X_spatial)
    simulated_detect = detection_fraction(X_simulated)
    spatial_mean = mean_expression(X_spatial)
    simulated_mean = mean_expression(X_simulated)
    spatial_positive_mean = positive_mean_expression(X_spatial)
    simulated_positive_mean = positive_mean_expression(X_simulated)

    metric_pairs = [
        ("detection_fraction", spatial_detect, simulated_detect),
        ("mean_expression", spatial_mean, simulated_mean),
        ("positive_mean_expression", spatial_positive_mean, simulated_positive_mean),
    ]
    if not args.skip_positive_medians:
        spatial_positive_median = positive_median_expression(X_spatial)
        simulated_positive_median = positive_median_expression(X_simulated)
        metric_pairs.append(("positive_median_expression", spatial_positive_median, simulated_positive_median))

    for metric_name, spatial_values, simulated_values in metric_pairs:
        gene_rows.extend(correlation_rows(config, group, metric_name, spatial_values, simulated_values))

    rank_rows.extend(
        rank_agreement_rows(
            config,
            group,
            genes,
            spatial_mean,
            simulated_mean,
            parse_int_csv(args.top_gene_counts),
        )
    )

    if args.write_per_gene:
        for gene, sp_det, sim_det, sp_mean, sim_mean, sp_pos_mean, sim_pos_mean in zip(
            genes,
            spatial_detect,
            simulated_detect,
            spatial_mean,
            simulated_mean,
            spatial_positive_mean,
            simulated_positive_mean,
        ):
            per_gene_rows.append(
                {
                    **config,
                    "group": group,
                    "gene": gene,
                    "spatial_detection_fraction": sp_det,
                    "simulated_detection_fraction": sim_det,
                    "spatial_mean_expression": sp_mean,
                    "simulated_mean_expression": sim_mean,
                    "spatial_positive_mean_expression": sp_pos_mean,
                    "simulated_positive_mean_expression": sim_pos_mean,
                }
            )


def summarize(
    cell_rows: list[dict[str, Any]],
    gene_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    classifier_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    config_ids = sorted({row["config_id"] for row in cell_rows + gene_rows + rank_rows + classifier_rows})
    for config_id in config_ids:
        record: dict[str, Any] = {"config_id": config_id}
        for metric in ["detected_genes_per_cell", "total_counts_per_cell", "max_count_per_cell"]:
            values = [
                row["normalized_wasserstein"]
                for row in cell_rows
                if row["config_id"] == config_id and row["metric"] == metric and np.isfinite(row["normalized_wasserstein"])
            ]
            record[f"{metric}_median_normalized_wasserstein"] = float(np.median(values)) if values else np.nan

        for metric in ["detection_fraction", "positive_mean_expression", "positive_median_expression"]:
            values = [
                row["correlation"]
                for row in gene_rows
                if row["config_id"] == config_id
                and row["metric"] == metric
                and row["correlation_method"] == "pearson"
                and np.isfinite(row["correlation"])
            ]
            record[f"{metric}_median_pearson"] = float(np.median(values)) if values else np.nan

        for top_k in sorted({row["top_k"] for row in rank_rows if row["config_id"] == config_id}):
            values = [
                row["overlap_fraction"]
                for row in rank_rows
                if row["config_id"] == config_id and row["top_k"] == top_k and np.isfinite(row["overlap_fraction"])
            ]
            record[f"top_{top_k}_median_overlap_fraction"] = float(np.median(values)) if values else np.nan

        overall_classifier = [
            row
            for row in classifier_rows
            if row["config_id"] == config_id and row["classifier_scope"] == "all" and row["status"] == "ok"
        ]
        if overall_classifier:
            record["classifier_auc"] = overall_classifier[0]["auc_mean"]
            record["classifier_accuracy"] = overall_classifier[0]["accuracy_mean"]
            record["classifier_auc_distance_from_0p5"] = abs(overall_classifier[0]["auc_mean"] - 0.5)
        records.append(record)
    return pd.DataFrame(records)


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def main() -> None:
    parser = build_parser()
    args = apply_manifest_args(parser.parse_args(), parser)
    rng = np.random.default_rng(args.random_state)

    if not args.spatial_path or not args.single_cell_path:
        raise ValueError("--spatial-path and --single-cell-path are required.")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial = ad.read_h5ad(Path(args.spatial_path).expanduser())
    single_cell = ad.read_h5ad(Path(args.single_cell_path).expanduser())
    shared_genes = spatial.var_names.intersection(single_cell.var_names)
    if len(shared_genes) == 0:
        raise ValueError("Spatial and single-cell AnnData objects have no shared genes.")

    spatial_gene_idx = spatial.var_names.get_indexer(shared_genes)
    sc_gene_idx = single_cell.var_names.get_indexer(shared_genes)
    X_spatial = matrix_from_adata(spatial, args.spatial_layer)[:, spatial_gene_idx].tocsr()
    X_raw_full = matrix_from_adata(single_cell, args.single_cell_raw_layer)
    X_raw = X_raw_full[:, sc_gene_idx].tocsr()

    sim_layer = optional_str(args.single_cell_sim_layer)
    admix_layer = None
    if sim_layer is None:
        if args.sample_admixture_from_neighbors:
            if args.admixture_repeats != 1:
                raise ValueError("Only --admixture-repeats 1 is currently supported by the similarity evaluator.")
            admix_layer = "generated_from_neighbors"
            X_admix = sample_admixed_counts_from_neighbors(single_cell, X_raw_full, sc_gene_idx, args, rng)
        else:
            admix_layer = find_admix_layer(single_cell, args.single_cell_admix_layer)
            X_admix = matrix_from_adata(single_cell, admix_layer)[:, sc_gene_idx].tocsr()
    else:
        X_sim_layer = matrix_from_adata(single_cell, sim_layer)[:, sc_gene_idx].tocsr()
        X_admix = None

    spatial_labels = labels_for(spatial, args.spatial_cell_type_key)
    sc_labels = labels_for(single_cell, args.single_cell_cell_type_key)
    groups = selected_groups(spatial_labels, sc_labels, args.group_mode, args.min_cells_per_group)
    if not groups:
        raise ValueError("No groups selected. Check group keys and --min-cells-per-group.")

    cell_rows: list[dict[str, Any]] = []
    gene_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []
    per_gene_rows: list[dict[str, Any]] = []

    for scale in parse_float_csv(args.admixture_scales):
        config = make_config(args, scale, admix_layer)
        if sim_layer is None:
            X_simulated_full = (X_raw + (X_admix * scale)).tocsr()
        else:
            X_simulated_full = X_sim_layer

        for group in groups:
            sp_idx = group_index(spatial_labels, group)
            sc_idx = group_index(sc_labels, group)
            if sp_idx.size < args.min_cells_per_group or sc_idx.size < args.min_cells_per_group:
                continue
            sp_idx = sample_rows(sp_idx, args.max_cells_per_domain, rng)
            sc_idx = sample_rows(sc_idx, args.max_cells_per_domain, rng)
            X_sp_group = X_spatial[sp_idx].tocsr()
            X_sim_group = X_simulated_full[sc_idx].tocsr()

            add_cell_metric_rows(cell_rows, config, group, X_sp_group, X_sim_group)
            add_gene_metric_rows(
                gene_rows,
                rank_rows,
                per_gene_rows,
                config,
                group,
                shared_genes,
                X_sp_group,
                X_sim_group,
                args,
            )

            if args.classifier and (group == "all" or args.classifier_by_group):
                classifier_rows.append(domain_classifier_row(config, group, X_sp_group, X_sim_group, args, rng))

    prefix = args.output_prefix
    write_table(cell_rows, output_dir / f"{prefix}_cell_metrics.csv")
    write_table(gene_rows, output_dir / f"{prefix}_gene_correlations.csv")
    write_table(rank_rows, output_dir / f"{prefix}_rank_agreement.csv")
    write_table(classifier_rows, output_dir / f"{prefix}_classifier.csv")
    if args.write_per_gene:
        write_table(per_gene_rows, output_dir / f"{prefix}_per_gene.csv")

    summary = summarize(cell_rows, gene_rows, rank_rows, classifier_rows)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    config_payload = {
        "args": vars(args),
        "n_shared_genes": int(len(shared_genes)),
        "groups": groups,
        "outputs": {
            "summary": f"{prefix}_summary.csv",
            "cell_metrics": f"{prefix}_cell_metrics.csv",
            "gene_correlations": f"{prefix}_gene_correlations.csv",
            "rank_agreement": f"{prefix}_rank_agreement.csv",
            "classifier": f"{prefix}_classifier.csv",
        },
    }
    (output_dir / f"{prefix}_config.json").write_text(json.dumps(config_payload, indent=2))
    print(f"Wrote metrics to {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
