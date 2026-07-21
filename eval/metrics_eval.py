from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scib_metrics
from scipy import sparse
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from neighbor_prediction import add_log_normalized_layer, run_specificity_contamination_assessment


DEFAULT_SPATIAL_DIR = "/compute/oven-0-13/jwinchel/spatial_adata/heimdall_inference"
DEFAULT_INPUT_GLOB = "GSM950*rctd_annotated_val_med_with_admix_probs.h5ad"
DEFAULT_OUTPUT_DIR = "/compute/oven-0-13/jwinchel/spatial_adata/metrics_eval"


def parse_float_list(value: str) -> list[float]:
    if value is None or value == "":
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    if value is None or value == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


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
            "Compute cell-type separation metrics and T-cell contamination metrics "
            "for full spatial tissue AnnData files."
        )
    )

    parser.add_argument("--spatial-dir", default=DEFAULT_SPATIAL_DIR)
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--input-files", nargs="*", default=None)
    parser.add_argument("--input-list-file", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cell-type-output", default="cell_type_separation_metrics_worstadj.csv")
    parser.add_argument("--tcell-output", default="tcell_contamination_metrics_worstadj.csv")
    parser.add_argument("--tcell-effect-output", default="tcell_contamination_effects_worstadj.csv")
    parser.add_argument("--tcell-lr-gene-output", default="tcell_lr_top_genes_worstadj.csv")
    parser.add_argument("--cell-type-profile-output", default="tissue_cell_type_profiles_worstadj.csv")
    parser.add_argument("--config-output", default="metrics_eval_config_worstadj.json")
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--sample-random-state", type=int, default=0)

    parser.add_argument("--label-key", default="harmony_cell_type")
    parser.add_argument("--exclude-labels", default="malignant cell")
    parser.add_argument("--source-file-key", default="source_file")
    parser.add_argument("--tissue-name-regex", default=None)
    parser.add_argument("--admix-probs-layer", default="admix_probs")
    parser.add_argument("--target-removed-fractions", default="0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--clean-prob-threshold", type=float, default=0.5)
    parser.add_argument("--cleaning-mode", choices=["zero", "probability_scale"], default="zero")
    parser.add_argument("--cell-type-specific-removal", action="store_true")
    parser.add_argument(
        "--removal-group-key",
        default=None,
        help="obs column used for cell-type-specific transcript removal. Defaults to --label-key.",
    )
    parser.add_argument(
        "--probability-transform",
        choices=["identity", "power", "sigmoid"],
        default="power",
    )
    parser.add_argument("--probability-power", type=float, default=2.0)
    parser.add_argument("--probability-sigmoid-midpoint", type=float, default=0.5)
    parser.add_argument("--probability-sigmoid-steepness", type=float, default=10.0)
    parser.add_argument("--tcell-probability-removal-scale", type=float, default=1.0)

    parser.add_argument("--additional-counts-layer", default="split_purified_counts")
    parser.add_argument("--additional-counts-case-label", default="SPLIT")
    parser.add_argument("--additional-counts-zero-high-prob", action="store_true")

    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--leiden-optimize-resolution", type=str_to_bool, default=False)
    parser.add_argument("--leiden-n-neighbors", type=int, default=15)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--library-size-target-sum", type=float, default=1e4)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)

    parser.add_argument("--skip-cell-type-separation", action="store_true")
    parser.add_argument("--skip-tcell-contamination", action="store_true")
    parser.add_argument("--skip-cell-type-profiles", action="store_true")

    parser.add_argument("--tcell-cell-type-key", default="rctd_first_type")
    parser.add_argument("--tcell-sample-key", default="batch")
    parser.add_argument("--coord-key", default="spatial")
    parser.add_argument("--coord-x", default="x_centroid")
    parser.add_argument("--coord-y", default="y_centroid")
    parser.add_argument("--tcell-radius-um", type=float, default=15.0)
    parser.add_argument("--tcell-top-n", type=int, default=20)
    parser.add_argument("--tcell-min-cells-per-label", type=int, default=100)
    parser.add_argument("--tcell-downsample-if-n-cells", type=int, default=50_000)
    parser.add_argument("--tcell-max-cells-per-label", type=int, default=25_000)
    parser.add_argument("--tcell-exact-zscore", default="auto")
    parser.add_argument("--tcell-dense-size-limit-entries", type=float, default=1e8)
    parser.add_argument("--cleaned-condition-label", default="cleaned_expression")
    parser.add_argument("--purified-condition-label", default="split")
    parser.add_argument("--uncorrected-condition-label", default="uncorrected")

    return parser


def discover_input_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    if args.input_files:
        files.extend(Path(f).expanduser() for f in args.input_files)
    if args.input_list_file:
        list_path = Path(args.input_list_file).expanduser()
        files.extend(
            Path(line.strip()).expanduser()
            for line in list_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not files:
        spatial_dir = Path(args.spatial_dir).expanduser()
        files.extend(sorted(spatial_dir.glob(args.input_glob)))

    files = [f.resolve() for f in files]
    if not files:
        raise FileNotFoundError(
            f"No input files found. spatial_dir={args.spatial_dir!r}, input_glob={args.input_glob!r}"
        )
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Input AnnData file does not exist: {f}")
    return files


def tissue_name_from_path(path: Path, regex: str | None = None) -> str:
    if regex:
        match = re.search(regex, path.name)
        if not match:
            raise ValueError(f"tissue-name-regex {regex!r} did not match {path.name!r}.")
        return match.group(1) if match.groups() else match.group(0)

    name = path.name
    for suffix in (
        "_rctd_annotated_val_med_with_admix_probs.h5ad",
        "_med_with_admix_probs.h5ad",
        ".h5ad",
        ".h5",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def sanitize_numeric_matrix(X):
    if sparse.issparse(X):
        X = X.copy()
        X.data = np.nan_to_num(X.data)
        return X
    return np.nan_to_num(X)


def sample_adata(adata: ad.AnnData, args: argparse.Namespace) -> ad.AnnData:
    if args.sample_frac >= 1.0:
        return adata
    if not 0.0 < args.sample_frac < 1.0:
        raise ValueError("--sample-frac must be in (0, 1] if provided.")

    if args.label_key in adata.obs:
        sampled_index = (
            adata.obs.groupby(args.label_key, observed=True)
            .sample(frac=args.sample_frac, random_state=args.sample_random_state)
            .index
        )
        return adata[sampled_index].copy()

    rng = np.random.default_rng(args.sample_random_state)
    n_keep = max(1, int(round(adata.n_obs * args.sample_frac)))
    keep = rng.choice(adata.n_obs, size=n_keep, replace=False)
    return adata[keep].copy()


def counts_from_layer(adata: ad.AnnData, layer_key: str | None):
    if layer_key is None or layer_key == "":
        return adata.X
    if layer_key not in adata.layers:
        raise KeyError(f"Layer {layer_key!r} not found in adata.layers.")
    return adata.layers[layer_key]


def jsonable_float(value):
    if value is None:
        return None
    value = float(value)
    if np.isnan(value):
        return None
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return value


def json_dumps_float_dict(values: dict) -> str:
    if not values:
        return ""
    return json.dumps({str(k): jsonable_float(v) for k, v in values.items()}, sort_keys=True)


def expressed_counts_and_probabilities(probs, X) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(X):
        xcoo = X.tocoo()
        counts = xcoo.data.astype(float)
        pvals = np.asarray(probs[xcoo.row, xcoo.col]).ravel().astype(float)
    else:
        X = np.asarray(X)
        probs = probs.toarray() if sparse.issparse(probs) else np.asarray(probs)
        nz = X > 0
        counts = X[nz].astype(float)
        pvals = probs[nz].astype(float)

    good = np.isfinite(pvals) & np.isfinite(counts) & (counts > 0)
    return counts[good], pvals[good]


def transformed_probability_weights(pvals, args: argparse.Namespace) -> np.ndarray:
    pvals = np.nan_to_num(np.asarray(pvals, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    pvals = np.clip(pvals, 0.0, 1.0)

    if args.probability_transform == "identity":
        weights = pvals
    elif args.probability_transform == "power":
        weights = np.power(pvals, args.probability_power)
    elif args.probability_transform == "sigmoid":
        steepness = float(args.probability_sigmoid_steepness)
        midpoint = float(args.probability_sigmoid_midpoint)
        raw = 1.0 / (1.0 + np.exp(-steepness * (pvals - midpoint)))
        lo = 1.0 / (1.0 + np.exp(-steepness * (0.0 - midpoint)))
        hi = 1.0 / (1.0 + np.exp(-steepness * (1.0 - midpoint)))
        weights = (raw - lo) / (hi - lo) if hi > lo else raw
    else:
        raise ValueError(f"Unsupported probability transform: {args.probability_transform!r}")

    return np.clip(weights, 0.0, 1.0)


def zeroed_counts(X, admix_probs, thresh: float):
    if sparse.issparse(X):
        cleaned = X.copy()
        row_idx, col_idx = cleaned.nonzero()
        probs_for_nonzero_counts = admix_probs[row_idx, col_idx]
        cleaned.data[np.asarray(probs_for_nonzero_counts).ravel() >= thresh] = 0
        cleaned.eliminate_zeros()
        return cleaned

    if sparse.issparse(admix_probs):
        admix_probs = admix_probs.toarray()

    return np.where(admix_probs >= thresh, 0, X)


def probability_scaled_counts(
    X,
    admix_probs,
    args: argparse.Namespace,
    removal_scale: float,
):
    if removal_scale < 0:
        raise ValueError("removal_scale must be non-negative.")

    if sparse.issparse(X):
        cleaned = X.astype(float, copy=True)
        row_idx, col_idx = cleaned.nonzero()
        probs_for_nonzero_counts = np.asarray(admix_probs[row_idx, col_idx]).ravel().astype(float)
        weights = transformed_probability_weights(probs_for_nonzero_counts, args)
        if np.isinf(removal_scale):
            removal_fraction = (weights > 0).astype(float)
        else:
            removal_fraction = np.clip(removal_scale * weights, 0.0, 1.0)
        cleaned.data = cleaned.data * (1.0 - removal_fraction)
        cleaned.eliminate_zeros()
        return cleaned

    if sparse.issparse(admix_probs):
        admix_probs = admix_probs.toarray()

    weights = transformed_probability_weights(admix_probs, args)
    if np.isinf(removal_scale):
        removal_fraction = (weights > 0).astype(float)
    else:
        removal_fraction = np.clip(removal_scale * weights, 0.0, 1.0)
    return np.asarray(X) * (1.0 - removal_fraction)


def cleaned_counts(
    X,
    admix_probs,
    threshold: float | None = None,
    args: argparse.Namespace | None = None,
    removal_scale: float | None = None,
):
    if args is None or args.cleaning_mode == "zero":
        if threshold is None:
            raise ValueError("threshold is required for zero cleaning mode.")
        return zeroed_counts(X, admix_probs, threshold)

    if args.cleaning_mode == "probability_scale":
        scale = args.tcell_probability_removal_scale if removal_scale is None else removal_scale
        return probability_scaled_counts(X, admix_probs, args, scale)

    raise ValueError(f"Unsupported cleaning mode: {args.cleaning_mode!r}")


def threshold_for_removed_fraction(probs, X, target_removed_frac: float) -> tuple[float, float]:
    r = min(max(float(target_removed_frac), 0.0), 1.0)
    if r == 0.0:
        return np.inf, 0.0
    if r == 1.0:
        return -np.inf, 1.0

    counts, pvals = expressed_counts_and_probabilities(probs, X)
    if counts.size == 0:
        raise ValueError("No expressed transcripts found (X > 0).")

    total = float(counts.sum())
    target = r * total
    order = np.argsort(pvals)[::-1]
    p_sorted = pvals[order]
    c_sorted = counts[order]
    cum = np.cumsum(c_sorted)
    idx = min(int(np.searchsorted(cum, target, side="left")), len(p_sorted) - 1)
    thr = float(p_sorted[idx])
    achieved = float(c_sorted[p_sorted >= thr].sum()) / total
    return thr, achieved


def removal_scale_for_removed_fraction(
    probs,
    X,
    target_removed_frac: float,
    args: argparse.Namespace,
) -> tuple[float, float]:
    r = min(max(float(target_removed_frac), 0.0), 1.0)
    if r == 0.0:
        return 0.0, 0.0

    counts, pvals = expressed_counts_and_probabilities(probs, X)
    if counts.size == 0:
        raise ValueError("No expressed transcripts found (X > 0).")

    weights = transformed_probability_weights(pvals, args)
    total = float(counts.sum())
    target = r * total
    positive = weights > 0
    if not np.any(positive):
        raise ValueError("Probability transform produced zero weights for all expressed transcripts.")

    max_removed = float(counts[positive].sum())
    if target >= max_removed:
        return np.inf, max_removed / total

    def removed_for(scale: float) -> float:
        return float(np.sum(counts * np.minimum(1.0, scale * weights)))

    lo = 0.0
    hi = 1.0
    while removed_for(hi) < target:
        hi *= 2.0

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if removed_for(mid) < target:
            lo = mid
        else:
            hi = mid

    achieved = removed_for(hi) / total
    return hi, achieved


def cleaned_counts_for_removed_fraction(
    X,
    admix_probs,
    target_removed_fraction: float,
    args: argparse.Namespace,
):
    if args.cleaning_mode == "zero":
        threshold, achieved_removed_fraction = threshold_for_removed_fraction(
            admix_probs,
            X,
            target_removed_fraction,
        )
        return (
            cleaned_counts(X, admix_probs, threshold=threshold, args=args),
            threshold,
            achieved_removed_fraction,
            np.nan,
        )

    if args.cleaning_mode == "probability_scale":
        removal_scale, achieved_removed_fraction = removal_scale_for_removed_fraction(
            admix_probs,
            X,
            target_removed_fraction,
            args,
        )
        return (
            cleaned_counts(X, admix_probs, args=args, removal_scale=removal_scale),
            np.nan,
            achieved_removed_fraction,
            removal_scale,
        )

    raise ValueError(f"Unsupported cleaning mode: {args.cleaning_mode!r}")


def cleaned_counts_for_removed_fraction_by_group(
    X,
    admix_probs,
    target_removed_fraction: float,
    labels,
    args: argparse.Namespace,
):
    labels = pd.Series(labels).reset_index(drop=True)
    if len(labels) != X.shape[0]:
        raise ValueError(
            f"Removal labels length ({len(labels)}) does not match matrix rows ({X.shape[0]})."
        )

    valid = ~labels.isna().to_numpy()
    if not np.any(valid):
        raise ValueError("No non-null labels available for cell-type-specific removal.")
    label_values = labels.astype(str).to_numpy()

    group_thresholds = {}
    group_scales = {}
    group_achieved = {}
    group_totals = {}

    is_sparse = sparse.issparse(X)
    if is_sparse:
        out_rows = []
        out_cols = []
        out_data = []
    else:
        out = np.asarray(X).copy()
        if args.cleaning_mode == "probability_scale":
            out = out.astype(float, copy=False)

    invalid_idx = np.flatnonzero(~valid)
    if is_sparse and invalid_idx.size:
        coo = X[invalid_idx].tocoo()
        out_rows.append(invalid_idx[coo.row])
        out_cols.append(coo.col)
        out_data.append(coo.data)

    for label in pd.unique(label_values[valid]):
        row_idx = np.flatnonzero(valid & (label_values == label))
        X_group = X[row_idx]
        probs_group = admix_probs[row_idx]
        counts, _ = expressed_counts_and_probabilities(probs_group, X_group)
        group_total = float(counts.sum())
        if group_total == 0.0:
            group_thresholds[label] = np.nan
            group_scales[label] = np.nan
            group_achieved[label] = np.nan
            group_totals[label] = 0.0
            X_cleaned_group = X_group.copy()
        else:
            X_cleaned_group, threshold, achieved, removal_scale = cleaned_counts_for_removed_fraction(
                X_group,
                probs_group,
                target_removed_fraction,
                args,
            )
            group_thresholds[label] = threshold
            group_scales[label] = removal_scale
            group_achieved[label] = achieved
            group_totals[label] = group_total

        if is_sparse:
            coo = X_cleaned_group.tocoo()
            out_rows.append(row_idx[coo.row])
            out_cols.append(coo.col)
            out_data.append(coo.data)
        else:
            out[row_idx] = X_cleaned_group

    total = sum(group_totals.values())
    achieved_removed_fraction = (
        sum(
            group_achieved[group] * group_totals[group]
            for group in group_totals
            if np.isfinite(group_achieved[group])
        )
        / total
        if total > 0
        else np.nan
    )

    if is_sparse:
        if out_data:
            out = sparse.coo_matrix(
                (
                    np.concatenate(out_data),
                    (np.concatenate(out_rows), np.concatenate(out_cols)),
                ),
                shape=X.shape,
            ).tocsr()
        else:
            out = sparse.csr_matrix(X.shape, dtype=X.dtype)
        out.eliminate_zeros()

    threshold = np.nan
    removal_scale = np.nan
    return (
        out,
        threshold,
        achieved_removed_fraction,
        removal_scale,
        json_dumps_float_dict(group_thresholds),
        json_dumps_float_dict(group_scales),
        json_dumps_float_dict(group_achieved),
    )


def cleaned_counts_for_removed_fraction_maybe_grouped(
    X,
    admix_probs,
    target_removed_fraction: float,
    args: argparse.Namespace,
    labels=None,
):
    if args.cell_type_specific_removal:
        if labels is None:
            raise ValueError("labels are required for cell-type-specific removal.")
        return cleaned_counts_for_removed_fraction_by_group(
            X,
            admix_probs,
            target_removed_fraction,
            labels,
            args,
        )

    X_cleaned, threshold, achieved, removal_scale = cleaned_counts_for_removed_fraction(
        X,
        admix_probs,
        target_removed_fraction,
        args,
    )
    return X_cleaned, threshold, achieved, removal_scale, "", "", ""


def prepare_matrix_and_labels(X, labels):
    labels = pd.Series(labels).astype("category")
    valid = ~labels.isna().to_numpy()
    X = X[valid]
    labels = labels[valid].cat.remove_unused_categories()
    return X, labels.to_numpy()


def preprocess_counts_for_metrics(X, args: argparse.Namespace):
    metric_adata = ad.AnnData(X=X.copy())
    sc.pp.normalize_total(metric_adata, target_sum=args.library_size_target_sum)
    sc.pp.log1p(metric_adata)
    n_comps = min(args.n_pcs, metric_adata.n_obs - 1, metric_adata.n_vars - 1)
    if n_comps < 1:
        raise ValueError("PCA metrics require at least two observations and two variables.")
    sc.pp.pca(metric_adata, n_comps=n_comps, random_state=args.random_state)
    return metric_adata.obsm["X_pca"]


def neighbors_for_scib_leiden_metrics(embedding, args: argparse.Namespace):
    return pynndescent(
        embedding,
        n_neighbors=min(args.leiden_n_neighbors, embedding.shape[0]),
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )


def empty_metric_row(
    tissue: str,
    source_file: str,
    case_name: str,
    target_removed_fraction=np.nan,
    threshold=np.nan,
    achieved_removed_fraction=np.nan,
    cleaning_mode="",
    probability_transform="",
    probability_removal_scale=np.nan,
    cell_type_specific_removal=False,
    group_admix_prob_thresholds="",
    group_probability_removal_scales="",
    group_achieved_removed_fractions="",
) -> dict:
    return {
        "tissue": tissue,
        "source_file": source_file,
        "case": case_name,
        "cleaning_mode": cleaning_mode,
        "cell_type_specific_removal": cell_type_specific_removal,
        "probability_transform": probability_transform,
        "probability_removal_scale": probability_removal_scale,
        "group_admix_prob_thresholds": group_admix_prob_thresholds,
        "group_probability_removal_scales": group_probability_removal_scales,
        "group_achieved_removed_fractions": group_achieved_removed_fractions,
        "target_removed_fraction": target_removed_fraction,
        "admix_prob_threshold": threshold,
        "achieved_removed_fraction": achieved_removed_fraction,
        "n_cells": np.nan,
        "n_genes": np.nan,
        "n_labels": np.nan,
        "silhouette_score": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
        "leiden_nmi_vs_label": np.nan,
        "leiden_adjusted_rand_vs_label": np.nan,
        "error": "",
    }


def score_case(
    tissue: str,
    source_file: str,
    case_name: str,
    X,
    labels,
    args: argparse.Namespace,
    target_removed_fraction=np.nan,
    threshold=np.nan,
    achieved_removed_fraction=np.nan,
    probability_removal_scale=np.nan,
    cleaning_mode="",
    probability_transform="",
    cell_type_specific_removal=False,
    group_admix_prob_thresholds="",
    group_probability_removal_scales="",
    group_achieved_removed_fractions="",
) -> dict:
    row = empty_metric_row(
        tissue,
        source_file,
        case_name,
        target_removed_fraction=target_removed_fraction,
        threshold=threshold,
        achieved_removed_fraction=achieved_removed_fraction,
        cleaning_mode=cleaning_mode,
        probability_transform=probability_transform,
        probability_removal_scale=probability_removal_scale,
        cell_type_specific_removal=cell_type_specific_removal,
        group_admix_prob_thresholds=group_admix_prob_thresholds,
        group_probability_removal_scales=group_probability_removal_scales,
        group_achieved_removed_fractions=group_achieved_removed_fractions,
    )

    try:
        X_raw, labels_prepared = prepare_matrix_and_labels(X, labels)
        unique_labels = pd.unique(labels_prepared)
        row["n_cells"] = int(X_raw.shape[0])
        row["n_genes"] = int(X_raw.shape[1])
        row["n_labels"] = int(len(unique_labels))
        if len(unique_labels) < 2:
            raise ValueError("At least two labels are required for separation metrics.")

        X_preprocessed = preprocess_counts_for_metrics(X_raw, args)
        neighbors = neighbors_for_scib_leiden_metrics(X_preprocessed, args)
        leiden_metrics = scib_metrics.nmi_ari_cluster_labels_leiden(
            neighbors,
            labels_prepared,
            optimize_resolution=args.leiden_optimize_resolution,
            resolution=args.leiden_resolution,
            n_jobs=args.n_jobs,
            seed=args.random_state,
        )
        row.update(
            {
                "silhouette_score": silhouette_score(
                    X_preprocessed, labels_prepared, metric="cosine"
                ),
                "calinski_harabasz": calinski_harabasz_score(X_preprocessed, labels_prepared),
                "davies_bouldin": davies_bouldin_score(X_preprocessed, labels_prepared),
                "leiden_nmi_vs_label": leiden_metrics["nmi"],
                "leiden_adjusted_rand_vs_label": leiden_metrics["ari"],
            }
        )
    except Exception as exc:  # Keep the batch moving and record the failed case.
        row["error"] = repr(exc)

    return row


def append_missing_layer_row(
    rows: list[dict],
    tissue: str,
    source_file: str,
    case_name: str,
    layer_key: str,
) -> None:
    row = empty_metric_row(tissue, source_file, case_name)
    row["error"] = f"Missing layer {layer_key!r}."
    rows.append(row)


def run_cell_type_separation(
    adata: ad.AnnData,
    tissue: str,
    source_file: str,
    args: argparse.Namespace,
) -> list[dict]:
    if args.label_key not in adata.obs:
        row = empty_metric_row(tissue, source_file, "all")
        row["error"] = f"Missing label key {args.label_key!r}."
        return [row]
    if args.admix_probs_layer not in adata.layers:
        row = empty_metric_row(tissue, source_file, "all")
        row["error"] = f"Missing admixture probability layer {args.admix_probs_layer!r}."
        return [row]

    labels = adata.obs[args.label_key]
    exclude_labels = set(parse_str_list(args.exclude_labels))
    if exclude_labels:
        keep = ~labels.astype(str).isin(exclude_labels)
        adata = adata[keep].copy()
        labels = adata.obs[args.label_key]

    rows = [
        score_case(tissue, source_file, "admixed", adata.X, labels, args),
    ]

    removal_labels = None
    if args.cell_type_specific_removal:
        removal_group_key = args.removal_group_key or args.label_key
        if removal_group_key not in adata.obs:
            row = empty_metric_row(tissue, source_file, "cell_type_specific_removal")
            row["error"] = f"Missing removal group key {removal_group_key!r}."
            rows.append(row)
            return rows
        removal_labels = adata.obs[removal_group_key]

    admix_probs = adata.layers[args.admix_probs_layer]
    for target_removed_fraction in parse_float_list(args.target_removed_fractions):
        try:
            (
                X_cleaned,
                thresh,
                achieved_removed_fraction,
                removal_scale,
                group_thresholds,
                group_scales,
                group_achieved,
            ) = (
                cleaned_counts_for_removed_fraction_maybe_grouped(
                    adata.X,
                    admix_probs,
                    target_removed_fraction,
                    args,
                    labels=removal_labels,
                )
            )
            rows.append(
                score_case(
                    tissue,
                    source_file,
                    f"cleaned_admixed_removed_frac_{target_removed_fraction:g}",
                    X_cleaned,
                    labels,
                    args,
                    target_removed_fraction=target_removed_fraction,
                    threshold=thresh,
                    achieved_removed_fraction=achieved_removed_fraction,
                    probability_removal_scale=removal_scale,
                    cleaning_mode=args.cleaning_mode,
                    cell_type_specific_removal=args.cell_type_specific_removal,
                    probability_transform=(
                        args.probability_transform
                        if args.cleaning_mode == "probability_scale"
                        else ""
                    ),
                    group_admix_prob_thresholds=group_thresholds,
                    group_probability_removal_scales=group_scales,
                    group_achieved_removed_fractions=group_achieved,
                )
            )
        except Exception as exc:
            row = empty_metric_row(
                tissue,
                source_file,
                f"cleaned_admixed_removed_frac_{target_removed_fraction:g}",
                target_removed_fraction=target_removed_fraction,
            )
            row["error"] = repr(exc)
            rows.append(row)

    if args.additional_counts_layer:
        additional_case_label = args.additional_counts_case_label or args.additional_counts_layer
        if args.additional_counts_layer not in adata.layers:
            append_missing_layer_row(
                rows,
                tissue,
                source_file,
                additional_case_label,
                args.additional_counts_layer,
            )
            return rows

        additional_counts = counts_from_layer(adata, args.additional_counts_layer)
        rows.append(score_case(tissue, source_file, additional_case_label, additional_counts, labels, args))

        if args.additional_counts_zero_high_prob:
            for target_removed_fraction in parse_float_list(args.target_removed_fractions):
                try:
                    (
                        X_cleaned,
                        thresh,
                        achieved_removed_fraction,
                        removal_scale,
                        group_thresholds,
                        group_scales,
                        group_achieved,
                    ) = (
                        cleaned_counts_for_removed_fraction_maybe_grouped(
                            additional_counts,
                            admix_probs,
                            target_removed_fraction,
                            args,
                            labels=removal_labels,
                        )
                    )
                    rows.append(
                        score_case(
                            tissue,
                            source_file,
                            f"{additional_case_label}_removed_frac_{target_removed_fraction:g}",
                            X_cleaned,
                            labels,
                            args,
                            target_removed_fraction=target_removed_fraction,
                            threshold=thresh,
                            achieved_removed_fraction=achieved_removed_fraction,
                            probability_removal_scale=removal_scale,
                            cleaning_mode=args.cleaning_mode,
                            cell_type_specific_removal=args.cell_type_specific_removal,
                            probability_transform=(
                                args.probability_transform
                                if args.cleaning_mode == "probability_scale"
                                else ""
                            ),
                            group_admix_prob_thresholds=group_thresholds,
                            group_probability_removal_scales=group_scales,
                            group_achieved_removed_fractions=group_achieved,
                        )
                    )
                except Exception as exc:
                    row = empty_metric_row(
                        tissue,
                        source_file,
                        f"{additional_case_label}_removed_frac_{target_removed_fraction:g}",
                        target_removed_fraction=target_removed_fraction,
                    )
                    row["error"] = repr(exc)
                    rows.append(row)

    return rows


def ensure_tcell_inputs(adata: ad.AnnData, tissue: str, args: argparse.Namespace) -> None:
    if args.tcell_sample_key not in adata.obs:
        adata.obs[args.tcell_sample_key] = tissue

    if args.coord_key not in adata.obsm:
        missing = [col for col in (args.coord_x, args.coord_y) if col not in adata.obs]
        if missing:
            raise KeyError(f"Missing coordinate columns required for T-cell metrics: {missing}")
        adata.obsm[args.coord_key] = adata.obs[[args.coord_x, args.coord_y]].to_numpy(dtype=float)


def jsonize_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].map(lambda x: isinstance(x, (list, tuple, set))).any():
            df[col] = df[col].map(
                lambda x: json.dumps(list(x)) if isinstance(x, (list, tuple, set)) else x
            )
    return df


def gene_symbol_lookup(adata: ad.AnnData, genes: list[str]) -> dict[str, str]:
    if "name" in adata.var.columns:
        values = adata.var["name"].astype(str)
    elif "gene_symbol" in adata.var.columns:
        values = adata.var["gene_symbol"].astype(str)
    elif "symbol" in adata.var.columns:
        values = adata.var["symbol"].astype(str)
    else:
        return {gene: gene for gene in genes}

    lookup = values.to_dict()
    return {gene: lookup.get(gene, gene) for gene in genes}


def compute_cell_type_profiles(
    adata: ad.AnnData,
    layer: str,
    genes: list[str],
    cell_type_key: str,
    tissue: str,
    source_file: str,
    condition: str,
) -> pd.DataFrame:
    genes = [gene for gene in dict.fromkeys(genes) if gene in adata.var_names]
    if not genes:
        return pd.DataFrame()
    if cell_type_key not in adata.obs:
        return pd.DataFrame(
            [
                {
                    "tissue": tissue,
                    "source_file": source_file,
                    "condition": condition,
                    "error": f"Missing cell type key {cell_type_key!r}.",
                }
            ]
        )

    gene_symbols = gene_symbol_lookup(adata, genes)
    gene_indices = adata.var_names.get_indexer(genes)
    X = adata.layers[layer][:, gene_indices] if layer is not None else adata.X[:, gene_indices]
    cell_types = adata.obs[cell_type_key].astype(str).to_numpy()

    rows = []
    for cell_type in pd.unique(cell_types):
        row_idx = np.flatnonzero(cell_types == cell_type)
        if row_idx.size == 0:
            continue
        X_group = X[row_idx]
        if sparse.issparse(X_group):
            mean_expression = np.asarray(X_group.mean(axis=0)).ravel()
            pct_expressed = np.asarray((X_group > 0).mean(axis=0)).ravel() * 100.0
        else:
            X_group = np.asarray(X_group)
            mean_expression = X_group.mean(axis=0)
            pct_expressed = (X_group > 0).mean(axis=0) * 100.0

        for gene, mean_value, pct_value in zip(genes, mean_expression, pct_expressed):
            rows.append(
                {
                    "tissue": tissue,
                    "source_file": source_file,
                    "condition": condition,
                    "cell_type": cell_type,
                    "gene": gene,
                    "gene_symbol": gene_symbols.get(gene, gene),
                    "mean_expression": float(mean_value),
                    "pct_expressed": float(pct_value),
                    "n_cells": int(row_idx.size),
                    "layer": layer,
                    "error": "",
                }
            )

    return pd.DataFrame(rows)


def safe_layer_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def run_tcell_contamination(
    adata: ad.AnnData,
    tissue: str,
    source_file: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.tcell_cell_type_key not in adata.obs:
        return pd.DataFrame(
            [
                {
                    "tissue": tissue,
                    "source_file": source_file,
                    "condition": "all",
                    "error": f"Missing T-cell label key {args.tcell_cell_type_key!r}.",
                }
            ]
        ), pd.DataFrame(), pd.DataFrame()
    if args.admix_probs_layer not in adata.layers:
        return pd.DataFrame(
            [
                {
                    "tissue": tissue,
                    "source_file": source_file,
                    "condition": "all",
                    "error": f"Missing admixture probability layer {args.admix_probs_layer!r}.",
                }
            ]
        ), pd.DataFrame(), pd.DataFrame()

    try:
        ensure_tcell_inputs(adata, tissue, args)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "tissue": tissue,
                    "source_file": source_file,
                    "condition": "all",
                    "error": repr(exc),
                }
            ]
        ), pd.DataFrame(), pd.DataFrame()

    lognorm_uncorrected = "_metrics_lognorm_uncorrected"
    lognorm_purified = "_metrics_lognorm_purified"

    add_log_normalized_layer(
        adata,
        counts_layer=None,
        out_layer=lognorm_uncorrected,
        target_sum=args.library_size_target_sum,
    )

    specificity_results: list[pd.DataFrame] = []
    coefficient_results: list[pd.DataFrame] = []
    profile_results: list[pd.DataFrame] = []
    specificity_layers: list[dict] = [
        {
            "condition": args.uncorrected_condition_label,
            "layer": lognorm_uncorrected,
            "target_removed_fraction": np.nan,
            "admix_prob_threshold": np.nan,
            "achieved_removed_fraction": np.nan,
            "probability_removal_scale": np.nan,
            "group_admix_prob_thresholds": "",
            "group_probability_removal_scales": "",
            "group_achieved_removed_fractions": "",
        }
    ]

    removal_labels = None
    if args.cell_type_specific_removal:
        removal_group_key = args.removal_group_key or args.label_key
        if removal_group_key not in adata.obs:
            return pd.DataFrame(
                [
                    {
                        "tissue": tissue,
                        "source_file": source_file,
                        "condition": "cell_type_specific_removal",
                        "error": f"Missing removal group key {removal_group_key!r}.",
                    }
                ]
            ), pd.DataFrame(), pd.DataFrame()
        removal_labels = adata.obs[removal_group_key]

    admix_probs = adata.layers[args.admix_probs_layer]
    for target_removed_fraction in parse_float_list(args.target_removed_fractions):
        condition = f"cleaned_admixed_removed_frac_{target_removed_fraction:g}"
        try:
            (
                X_cleaned,
                thresh,
                achieved_removed_fraction,
                removal_scale,
                group_thresholds,
                group_scales,
                group_achieved,
            ) = cleaned_counts_for_removed_fraction_maybe_grouped(
                adata.X,
                admix_probs,
                target_removed_fraction,
                args,
                labels=removal_labels,
            )
            raw_layer = f"_metrics_raw_{safe_layer_suffix(condition)}"
            lognorm_layer = f"_metrics_lognorm_{safe_layer_suffix(condition)}"
            adata.layers[raw_layer] = X_cleaned
            add_log_normalized_layer(
                adata,
                counts_layer=raw_layer,
                out_layer=lognorm_layer,
                target_sum=args.library_size_target_sum,
            )
            specificity_layers.append(
                {
                    "condition": condition,
                    "layer": lognorm_layer,
                    "target_removed_fraction": target_removed_fraction,
                    "admix_prob_threshold": thresh,
                    "achieved_removed_fraction": achieved_removed_fraction,
                    "probability_removal_scale": removal_scale,
                    "group_admix_prob_thresholds": group_thresholds,
                    "group_probability_removal_scales": group_scales,
                    "group_achieved_removed_fractions": group_achieved,
                }
            )
        except Exception as exc:
            specificity_results.append(
                pd.DataFrame(
                    [
                        {
                            "tissue": tissue,
                            "source_file": source_file,
                            "condition": condition,
                            "target_removed_fraction": target_removed_fraction,
                            "error": repr(exc),
                        }
                    ]
                )
            )

    if args.additional_counts_layer:
        if args.additional_counts_layer in adata.layers:
            add_log_normalized_layer(
                adata,
                counts_layer=args.additional_counts_layer,
                out_layer=lognorm_purified,
                target_sum=args.library_size_target_sum,
            )
            specificity_layers.append(
                {
                    "condition": args.purified_condition_label,
                    "layer": lognorm_purified,
                    "target_removed_fraction": np.nan,
                    "admix_prob_threshold": np.nan,
                    "achieved_removed_fraction": np.nan,
                    "probability_removal_scale": np.nan,
                    "group_admix_prob_thresholds": "",
                    "group_probability_removal_scales": "",
                    "group_achieved_removed_fractions": "",
                }
            )
            if args.additional_counts_zero_high_prob:
                additional_counts = counts_from_layer(adata, args.additional_counts_layer)
                additional_case_label = args.additional_counts_case_label or args.additional_counts_layer
                for target_removed_fraction in parse_float_list(args.target_removed_fractions):
                    condition = f"{additional_case_label}_removed_frac_{target_removed_fraction:g}"
                    try:
                        (
                            X_cleaned,
                            thresh,
                            achieved_removed_fraction,
                            removal_scale,
                            group_thresholds,
                            group_scales,
                            group_achieved,
                        ) = cleaned_counts_for_removed_fraction_maybe_grouped(
                            additional_counts,
                            admix_probs,
                            target_removed_fraction,
                            args,
                            labels=removal_labels,
                        )
                        raw_layer = f"_metrics_raw_{safe_layer_suffix(condition)}"
                        lognorm_layer = f"_metrics_lognorm_{safe_layer_suffix(condition)}"
                        adata.layers[raw_layer] = X_cleaned
                        add_log_normalized_layer(
                            adata,
                            counts_layer=raw_layer,
                            out_layer=lognorm_layer,
                            target_sum=args.library_size_target_sum,
                        )
                        specificity_layers.append(
                            {
                                "condition": condition,
                                "layer": lognorm_layer,
                                "target_removed_fraction": target_removed_fraction,
                                "admix_prob_threshold": thresh,
                                "achieved_removed_fraction": achieved_removed_fraction,
                                "probability_removal_scale": removal_scale,
                                "group_admix_prob_thresholds": group_thresholds,
                                "group_probability_removal_scales": group_scales,
                                "group_achieved_removed_fractions": group_achieved,
                            }
                        )
                    except Exception as exc:
                        specificity_results.append(
                            pd.DataFrame(
                                [
                                    {
                                        "tissue": tissue,
                                        "source_file": source_file,
                                        "condition": condition,
                                        "target_removed_fraction": target_removed_fraction,
                                        "error": repr(exc),
                                    }
                                ]
                            )
                        )
        else:
            missing = pd.DataFrame(
                [
                    {
                        "tissue": tissue,
                        "source_file": source_file,
                        "condition": args.purified_condition_label,
                        "error": f"Missing layer {args.additional_counts_layer!r}.",
                    }
                ]
            )
            specificity_results.append(missing)

    for layer_info in specificity_layers:
        condition = layer_info["condition"]
        layer = layer_info["layer"]
        try:
            res, coefficients, _ = run_specificity_contamination_assessment(
                adata,
                sample_key=args.tcell_sample_key,
                cell_type_key=args.tcell_cell_type_key,
                coord_key=args.coord_key,
                layer=layer,
                radius_um=args.tcell_radius_um,
                top_n=args.tcell_top_n,
                min_cells_per_label=args.tcell_min_cells_per_label,
                downsample_if_n_cells=args.tcell_downsample_if_n_cells,
                max_cells_per_label=args.tcell_max_cells_per_label,
                exact_zscore=args.tcell_exact_zscore,
                dense_size_limit_entries=args.tcell_dense_size_limit_entries,
                random_state=args.random_state,
            )
            res["condition"] = condition
            res["tissue"] = tissue
            res["source_file"] = source_file
            res["target_removed_fraction"] = layer_info["target_removed_fraction"]
            res["admix_prob_threshold"] = layer_info["admix_prob_threshold"]
            res["achieved_removed_fraction"] = layer_info["achieved_removed_fraction"]
            res["cleaning_mode"] = args.cleaning_mode
            res["probability_transform"] = (
                args.probability_transform if args.cleaning_mode == "probability_scale" else ""
            )
            res["probability_removal_scale"] = layer_info["probability_removal_scale"]
            res["cell_type_specific_removal"] = args.cell_type_specific_removal
            res["group_admix_prob_thresholds"] = layer_info["group_admix_prob_thresholds"]
            res["group_probability_removal_scales"] = layer_info["group_probability_removal_scales"]
            res["group_achieved_removed_fractions"] = layer_info["group_achieved_removed_fractions"]
            res["error"] = ""
            specificity_results.append(res)

            if not coefficients.empty:
                coefficients = coefficients.copy()
                coefficients["condition"] = condition
                coefficients["tissue"] = tissue
                coefficients["source_file"] = source_file
                coefficients["target_removed_fraction"] = layer_info["target_removed_fraction"]
                coefficients["admix_prob_threshold"] = layer_info["admix_prob_threshold"]
                coefficients["achieved_removed_fraction"] = layer_info["achieved_removed_fraction"]
                coefficients["cleaning_mode"] = args.cleaning_mode
                coefficients["probability_transform"] = (
                    args.probability_transform if args.cleaning_mode == "probability_scale" else ""
                )
                coefficients["probability_removal_scale"] = layer_info["probability_removal_scale"]
                coefficients["cell_type_specific_removal"] = args.cell_type_specific_removal
                coefficients["group_admix_prob_thresholds"] = layer_info["group_admix_prob_thresholds"]
                coefficients["group_probability_removal_scales"] = layer_info[
                    "group_probability_removal_scales"
                ]
                coefficients["group_achieved_removed_fractions"] = layer_info[
                    "group_achieved_removed_fractions"
                ]
                coefficients["gene_symbol"] = coefficients["gene"].map(
                    gene_symbol_lookup(adata, coefficients["gene"].astype(str).unique().tolist())
                )
                coefficients["error"] = ""
                coefficient_results.append(coefficients)

                if not args.skip_cell_type_profiles:
                    profile_genes = coefficients["gene"].astype(str).unique().tolist()
                    profile = compute_cell_type_profiles(
                        adata=adata,
                        layer=layer,
                        genes=profile_genes,
                        cell_type_key=args.tcell_cell_type_key,
                        tissue=tissue,
                        source_file=source_file,
                        condition=condition,
                    )
                    if not profile.empty:
                        profile["target_removed_fraction"] = layer_info["target_removed_fraction"]
                        profile["admix_prob_threshold"] = layer_info["admix_prob_threshold"]
                        profile["achieved_removed_fraction"] = layer_info[
                            "achieved_removed_fraction"
                        ]
                        profile["cleaning_mode"] = args.cleaning_mode
                        profile["probability_transform"] = (
                            args.probability_transform
                            if args.cleaning_mode == "probability_scale"
                            else ""
                        )
                        profile["probability_removal_scale"] = layer_info[
                            "probability_removal_scale"
                        ]
                        profile["cell_type_specific_removal"] = args.cell_type_specific_removal
                        profile["group_admix_prob_thresholds"] = layer_info[
                            "group_admix_prob_thresholds"
                        ]
                        profile["group_probability_removal_scales"] = layer_info[
                            "group_probability_removal_scales"
                        ]
                        profile["group_achieved_removed_fractions"] = layer_info[
                            "group_achieved_removed_fractions"
                        ]
                    profile_results.append(profile)
        except Exception as exc:
            specificity_results.append(
                pd.DataFrame(
                    [
                        {
                            "tissue": tissue,
                            "source_file": source_file,
                            "condition": condition,
                            "target_removed_fraction": layer_info["target_removed_fraction"],
                            "admix_prob_threshold": layer_info["admix_prob_threshold"],
                            "achieved_removed_fraction": layer_info["achieved_removed_fraction"],
                            "cleaning_mode": args.cleaning_mode,
                            "probability_transform": (
                                args.probability_transform
                                if args.cleaning_mode == "probability_scale"
                                else ""
                            ),
                            "probability_removal_scale": layer_info["probability_removal_scale"],
                            "cell_type_specific_removal": args.cell_type_specific_removal,
                            "group_admix_prob_thresholds": layer_info["group_admix_prob_thresholds"],
                            "group_probability_removal_scales": layer_info[
                                "group_probability_removal_scales"
                            ],
                            "group_achieved_removed_fractions": layer_info[
                                "group_achieved_removed_fractions"
                            ],
                            "error": repr(exc),
                        }
                    ]
                )
            )

    if not specificity_results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    specificity_df = jsonize_list_columns(pd.concat(specificity_results, ignore_index=True))
    coefficient_df = (
        pd.concat(coefficient_results, ignore_index=True)
        if coefficient_results
        else pd.DataFrame()
    )
    profile_df = (
        pd.concat(profile_results, ignore_index=True)
        if profile_results
        else pd.DataFrame()
    )
    return specificity_df, coefficient_df, profile_df


def summarize_tcell_effects(tcell_metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = {
        "tissue",
        "source_file",
        "condition",
        "target_cell_type_i",
        "contaminating_cell_type_j",
        "marker_overlap_n",
    }
    if tcell_metrics.empty or not required.issubset(tcell_metrics.columns):
        return pd.DataFrame()

    metric_rows = tcell_metrics[tcell_metrics["error"].fillna("") == ""].copy()
    if metric_rows.empty:
        return pd.DataFrame()

    effect = (
        metric_rows.groupby(
            ["tissue", "source_file", "condition", "target_cell_type_i", "contaminating_cell_type_j"],
            dropna=False,
        )["marker_overlap_n"]
        .mean()
        .reset_index()
        .pivot(
            index=["tissue", "source_file", "target_cell_type_i", "contaminating_cell_type_j"],
            columns="condition",
            values="marker_overlap_n",
        )
        .reset_index()
    )

    uncorrected = args.uncorrected_condition_label
    index_cols = {"tissue", "source_file", "target_cell_type_i", "contaminating_cell_type_j"}
    condition_cols = [col for col in effect.columns if col not in index_cols]
    if uncorrected in condition_cols:
        safe_uncorrected = safe_layer_suffix(str(uncorrected)).lower()
        for condition in condition_cols:
            if condition == uncorrected:
                continue
            safe_condition = safe_layer_suffix(str(condition)).lower()
            effect[f"delta_{safe_condition}_minus_{safe_uncorrected}"] = (
                effect[condition] - effect[uncorrected]
            )

    return effect


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} rows: {path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    files = discover_input_files(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / args.config_output
    config_path.write_text(
        json.dumps({**vars(args), "input_files": [str(f) for f in files]}, indent=2) + "\n"
    )
    print(f"Wrote config: {config_path}")
    print(f"Found {len(files)} input file(s). sample_frac={args.sample_frac}")

    cell_type_rows: list[dict] = []
    tcell_frames: list[pd.DataFrame] = []
    tcell_coefficient_frames: list[pd.DataFrame] = []
    cell_type_profile_frames: list[pd.DataFrame] = []

    for path in tqdm(files, desc="Spatial files"):
        tissue = tissue_name_from_path(path, args.tissue_name_regex)
        print(f"\nReading {path}")
        adata = ad.read_h5ad(path)
        adata.obs[args.source_file_key] = str(path)
        adata.obs["_metrics_tissue"] = tissue
        adata.X = sanitize_numeric_matrix(adata.X)
        adata = sample_adata(adata, args)

        if not args.skip_cell_type_separation:
            print(f"Computing cell-type separation metrics: {tissue}")
            cell_type_rows.extend(run_cell_type_separation(adata, tissue, str(path), args))

        if not args.skip_tcell_contamination:
            print(f"Computing T-cell contamination metrics: {tissue}")
            tcell_df, coefficient_df, profile_df = run_tcell_contamination(
                adata.copy(),
                tissue,
                str(path),
                args,
            )
            tcell_frames.append(tcell_df)
            tcell_coefficient_frames.append(coefficient_df)
            cell_type_profile_frames.append(profile_df)

    if not args.skip_cell_type_separation:
        cell_type_df = pd.DataFrame(cell_type_rows)
        write_csv(cell_type_df, output_dir / args.cell_type_output)

    if not args.skip_tcell_contamination:
        tcell_df = (
            pd.concat(tcell_frames, ignore_index=True)
            if tcell_frames
            else pd.DataFrame()
        )
        write_csv(tcell_df, output_dir / args.tcell_output)
        write_csv(summarize_tcell_effects(tcell_df, args), output_dir / args.tcell_effect_output)
        tcell_coefficient_df = (
            pd.concat(tcell_coefficient_frames, ignore_index=True)
            if tcell_coefficient_frames
            else pd.DataFrame()
        )
        write_csv(tcell_coefficient_df, output_dir / args.tcell_lr_gene_output)
        cell_type_profile_df = (
            pd.concat(cell_type_profile_frames, ignore_index=True)
            if cell_type_profile_frames
            else pd.DataFrame()
        )
        write_csv(cell_type_profile_df, output_dir / args.cell_type_profile_output)


if __name__ == "__main__":
    main()
