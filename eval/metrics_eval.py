from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Sequence

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
    parser.add_argument("--cell-type-output", default="cell_type_separation_metrics.csv")
    parser.add_argument("--tcell-output", default="tcell_contamination_metrics.csv")
    parser.add_argument("--tcell-effect-output", default="tcell_contamination_effects.csv")
    parser.add_argument("--tcell-lr-gene-output", default="tcell_lr_top_genes.csv")
    parser.add_argument("--tcell-cosine-output", default="tcell_cosine_similarity.csv")
    parser.add_argument("--cell-type-profile-output", default="tissue_cell_type_profilesj.csv")
    parser.add_argument("--config-output", default="metrics_eval_config.json")
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--sample-random-state", type=int, default=0)
    parser.add_argument("--separation-min-counts", type=float, default=0.0,
                        help="Minimum count total per cell for cell-type separation metrics; 0 disables this filter.")
    parser.add_argument("--separation-min-genes", type=int, default=0,
                        help="Minimum number of expressed genes per cell for cell-type separation metrics; 0 disables this filter.")

    parser.add_argument("--label-key", default="harmony_cell_type")
    parser.add_argument("--exclude-labels", default="malignant cell")
    parser.add_argument("--source-file-key", default="source_file")
    parser.add_argument("--tissue-name-regex", default=None)
    parser.add_argument("--admix-probs-layer", default="admix_probs")
    parser.add_argument("--target-removed-fractions", default="0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--clean-prob-threshold", type=float, default=0.5,
                        help="Minimum admixture probability eligible for target-fraction removal.")
    parser.add_argument("--syndri-reference-cosine", action="store_true",
                        help="Add a per-cell Syndri zeroing case selected by rolling reference cosine.")
    parser.add_argument("--syndri-cosine-window", type=int, default=5,
                        help="Trailing window of distinct expressed-gene probability thresholds; 1 disables smoothing.")
    parser.add_argument("--syndri-reference-profiles",
                        default=str(SCRIPT_DIR.parent / "data/cache/rctd_profiles"),
                        help="Cell-type-by-gene profile h5ad, or directory of split_sparsity profile caches.")
    parser.add_argument("--syndri-cell-type-key", default="",
                        help="obs column matching reference profile names; defaults to --label-key.")
    parser.add_argument("--syndri-profile-min-umi", type=int, default=10)
    parser.add_argument("--syndri-profile-max-umi", type=int, default=2000)
    parser.add_argument("--syndri-profile-cell-min", type=int, default=25)
    parser.add_argument("--syndri-profile-max-cells", type=int, default=10000)
    parser.add_argument("--syndri-profile-cell-type-col", default="cell_type")
    parser.add_argument(
        "--syndri-one-se-multiplier",
        type=float,
        default=1.0,
        help="Choose the smallest cell-type gene prefix within this many SEs of the best mean cosine improvement.",
    )
    parser.add_argument(
        "--only-syndri-reference-cosine",
        action="store_true",
        help="Evaluate only the adaptive per-cell Syndri reference-cosine case.",
    )
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
    parser.add_argument(
        "--only-additional-counts",
        action="store_true",
        help=("Evaluate only --additional-counts-layer. Use with --append-output "
              "to replace that case while preserving existing rows."),
    )

    parser.add_argument(
        "--evaluation-layers",
        default="",
        help=(
            "Comma-separated count layers to evaluate in layered mode, e.g. trial_0,trial_1,trial_2. "
            "For each layer, evaluates Syndri-cleaned, MLP-cleaned, and SPLIT-purified counts."
        ),
    )
    parser.add_argument(
        "--layered-baseline-only",
        action="store_true",
        help="In layered mode, compute only the uncorrected baseline rows for each input layer.",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append new rows to existing output CSVs and de-duplicate by metric identity columns.",
    )
    parser.add_argument(
        "--replace-layer-count-baselines",
        action="store_true",
        help=(
            "When appending, remove older method-specific *_uncorrected rows so the shared "
            "uncorrected rows are the only uncorrected baselines."
        ),
    )
    parser.add_argument(
        "--deduplicate-zero-dropout-inputs",
        action="store_true",
        help=(
            "In layered mode, evaluate the X/0pct condition once per sample-replicate "
            "instead of once for every dropout file in a pooled task."
        ),
    )
    parser.add_argument("--syndri-probs-template", default="admix_probs_{layer}")
    parser.add_argument("--mlp-probs-template", default="mlp_admix_probs_{layer}")
    parser.add_argument("--split-counts-template", default="split_purified_counts_{layer}")
    parser.add_argument("--syndri-case-label", default="syndri")
    parser.add_argument("--mlp-case-label", default="mlp")
    parser.add_argument("--split-case-label", default="split")
    parser.add_argument(
        "--sample-replicate-regex",
        default=r"(B\d+_B\d+(?:_\d+)?)",
        help="Regex used to extract original sample-replicate names from input filenames.",
    )

    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--leiden-optimize-resolution", type=str_to_bool, default=False)
    parser.add_argument("--leiden-n-neighbors", type=int, default=15)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--library-size-target-sum", type=float, default=1e4)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)

    parser.add_argument("--skip-cell-type-separation", action="store_true")
    parser.add_argument("--skip-tcell-contamination", action="store_true")
    parser.add_argument(
        "--skip-tcell-neighborhood-metrics",
        action="store_true",
        help="Skip the existing T-cell neighborhood/LR assessment while still allowing cosine similarity.",
    )
    parser.add_argument("--skip-tcell-cosine-similarity", action="store_true")
    parser.add_argument("--skip-cell-type-profiles", action="store_true")

    parser.add_argument(
        "--reference-data",
        default="",
        help="Single-cell reference AnnData used for T-cell pseudo-bulk cosine similarity.",
    )
    parser.add_argument(
        "--reference-cell-type-key",
        default="",
        help="Cell type column in the single-cell reference. Defaults to --tcell-cell-type-key.",
    )
    parser.add_argument(
        "--reference-counts-layer",
        default="",
        help="Counts layer in the single-cell reference for pseudo-bulk profiles. Defaults to .X.",
    )
    parser.add_argument(
        "--tcell-cosine-cell-type-pattern",
        default=r"(?i)(^|\b)(t cells?|cd4|cd8|treg|t[- ]?regulatory|regulatory t|cytotoxic t)(\b|$)",
        help="Regex selecting spatial cell types to include in the T-cell cosine metric. Empty means all.",
    )
    parser.add_argument(
        "--tcell-cosine-cell-types",
        default="",
        help="Comma-separated exact spatial cell types to include. Overrides --tcell-cosine-cell-type-pattern.",
    )
    parser.add_argument(
        "--tcell-cosine-cell-type-map",
        default="",
        help="Optional comma-separated spatial=reference cell type name mapping.",
    )
    parser.add_argument("--tcell-cosine-min-spatial-cells", type=int, default=1)
    parser.add_argument("--tcell-cosine-min-reference-cells", type=int, default=1)

    parser.add_argument("--tcell-cell-type-key", default="rctd_first_type")
    parser.add_argument("--tcell-sample-key", default="batch")
    parser.add_argument(
        "--filter-tcell-invalid-cells",
        action="store_true",
        help=(
            "Before T-cell contamination metrics, drop cells with missing/string-'nan' "
            "T-cell labels, excluded T-cell labels, missing removal-group labels, or "
            "non-finite coordinates."
        ),
    )
    parser.add_argument(
        "--tcell-exclude-labels",
        default="",
        help=(
            "Comma-separated T-cell labels to exclude when --filter-tcell-invalid-cells "
            "is enabled. Defaults to --exclude-labels when unset."
        ),
    )
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


def sample_replicate_from_path(path: Path, regex: str | None = None) -> str:
    if regex:
        match = re.search(regex, path.name)
        if match:
            return match.groupdict().get("sample_replicate") or match.group(0)
    return tissue_name_from_path(path)


def dropout_from_path(path: Path) -> tuple[str, float]:
    match = re.search(r"dropout_([0-9]+(?:\.[0-9]+)?)pct", path.name)
    if not match:
        return "", np.nan
    value = float(match.group(1))
    return f"{match.group(1)}pct", value / 100.0


def format_layer_template(template: str, layer: str) -> str:
    if "{layer}" in template:
        return template.format(layer=layer)
    return f"{template}{layer}"


def resolve_layer_key(
    adata: ad.AnnData,
    key: str,
    *,
    fallback_keys: Sequence[str] = (),
) -> str | None:
    for candidate in [key, *fallback_keys]:
        if candidate == "X" or candidate in adata.layers:
            return candidate
    return None


def layer_dropout_metadata(
    input_layer: str,
    dropout_label: str,
    dropout_fraction: float,
) -> tuple[str, float]:
    if input_layer == "X":
        return "0pct", 0.0
    return dropout_label, dropout_fraction


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
    if layer_key is None or layer_key == "" or layer_key == "X":
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
    minimum = float(getattr(args, "clean_prob_threshold", 0.5))
    if not np.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError("clean_prob_threshold must be in [0, 1].")

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

    return np.where(pvals >= minimum, np.clip(weights, 0.0, 1.0), 0.0)


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


def threshold_for_removed_fraction(
    probs, X, target_removed_frac: float, min_probability: float = 0.5,
) -> tuple[float, float]:
    r = min(max(float(target_removed_frac), 0.0), 1.0)
    if not np.isfinite(min_probability) or not 0 <= min_probability <= 1:
        raise ValueError("min_probability must be in [0, 1].")
    if r == 0.0:
        return np.inf, 0.0

    counts, pvals = expressed_counts_and_probabilities(probs, X)
    if counts.size == 0:
        raise ValueError("No expressed transcripts found (X > 0).")

    total = float(counts.sum())
    target = r * total
    eligible = pvals >= min_probability
    if not np.any(eligible):
        return np.inf, 0.0
    eligible_counts = counts[eligible]
    eligible_pvals = pvals[eligible]
    order = np.argsort(eligible_pvals)[::-1]
    p_sorted = eligible_pvals[order]
    c_sorted = eligible_counts[order]
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
        return np.inf, 0.0

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
            min_probability=getattr(args, "clean_prob_threshold", 0.5),
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


def median_cell_count_summaries(X) -> tuple[float, float]:
    """Return median detected genes and median transcript count per cell."""
    if sparse.issparse(X):
        genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
        transcripts_per_cell = np.asarray(X.sum(axis=1)).ravel()
    else:
        X_array = np.asarray(X)
        genes_per_cell = np.count_nonzero(X_array > 0, axis=1)
        transcripts_per_cell = X_array.sum(axis=1)

    return float(np.median(genes_per_cell)), float(np.median(transcripts_per_cell))


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
        "median_genes_per_cell": np.nan,
        "median_transcripts_per_cell": np.nan,
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
        row["n_cells_before_qc"] = int(X_raw.shape[0])
        if args.separation_min_counts > 0 or args.separation_min_genes > 0:
            if sparse.issparse(X_raw):
                cell_counts = np.asarray(X_raw.sum(axis=1)).ravel()
                cell_genes = np.asarray((X_raw > 0).sum(axis=1)).ravel()
            else:
                cell_counts = np.asarray(X_raw).sum(axis=1)
                cell_genes = np.count_nonzero(np.asarray(X_raw) > 0, axis=1)
            keep = ((cell_counts >= args.separation_min_counts)
                    & (cell_genes >= args.separation_min_genes))
            X_raw = X_raw[keep]
            labels_prepared = labels_prepared[keep]
        row["n_cells_dropped_qc"] = row["n_cells_before_qc"] - int(X_raw.shape[0])
        unique_labels = pd.unique(labels_prepared)
        row["n_cells"] = int(X_raw.shape[0])
        row["n_genes"] = int(X_raw.shape[1])
        row["n_labels"] = int(len(unique_labels))
        (
            row["median_genes_per_cell"],
            row["median_transcripts_per_cell"],
        ) = median_cell_count_summaries(X_raw)
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
    needs_admix_probs = not args.only_additional_counts or args.additional_counts_zero_high_prob
    if needs_admix_probs and args.admix_probs_layer not in adata.layers:
        row = empty_metric_row(tissue, source_file, "all")
        row["error"] = f"Missing admixture probability layer {args.admix_probs_layer!r}."
        return [row]

    labels = adata.obs[args.label_key]
    exclude_labels = set(parse_str_list(args.exclude_labels))
    if exclude_labels:
        keep = ~labels.astype(str).isin(exclude_labels)
        adata = adata[keep].copy()
        labels = adata.obs[args.label_key]

    rows = [] if args.only_additional_counts else [
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

    admix_probs = adata.layers[args.admix_probs_layer] if needs_admix_probs else None
    target_fractions = [] if args.only_additional_counts else parse_float_list(args.target_removed_fractions)
    for target_removed_fraction in target_fractions:
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


def filtered_tcell_adata(adata: ad.AnnData, args: argparse.Namespace) -> tuple[ad.AnnData, dict[str, int]]:
    valid = np.ones(adata.n_obs, dtype=bool)
    dropped: dict[str, int] = {}

    exclude_labels = set(parse_str_list(args.tcell_exclude_labels) or parse_str_list(args.exclude_labels))

    label_keys = [args.tcell_cell_type_key]
    if args.cell_type_specific_removal:
        removal_group_key = args.removal_group_key or args.label_key
        if removal_group_key not in label_keys:
            label_keys.append(removal_group_key)

    for key in label_keys:
        if key not in adata.obs:
            continue
        values = adata.obs[key]
        text = values.astype(str).str.strip()
        key_valid = values.notna().to_numpy()
        key_valid &= text.str.lower().ne("nan").to_numpy()
        if exclude_labels:
            key_valid &= ~text.isin(exclude_labels).to_numpy()
        dropped[f"{key}_label"] = int((valid & ~key_valid).sum())
        valid &= key_valid

    coords = np.asarray(adata.obsm[args.coord_key], dtype=float)
    coord_valid = np.isfinite(coords).all(axis=1)
    dropped["coordinates"] = int((valid & ~coord_valid).sum())
    valid &= coord_valid

    dropped["total"] = int((~valid).sum())
    if valid.all():
        return adata, dropped
    return adata[valid].copy(), dropped


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


def parse_cell_type_map(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in parse_str_list(value):
        if "=" in item:
            left, right = item.split("=", 1)
        elif ":" in item:
            left, right = item.split(":", 1)
        else:
            raise ValueError(
                "Cell type map entries must use spatial=reference or spatial:reference syntax."
            )
        mapping[left.strip()] = right.strip()
    return mapping


def selected_tcell_types(cell_types: Sequence[str], args: argparse.Namespace) -> set[str]:
    exact = set(parse_str_list(args.tcell_cosine_cell_types))
    if exact:
        return {cell_type for cell_type in cell_types if cell_type in exact}

    pattern = args.tcell_cosine_cell_type_pattern
    if not pattern:
        return set(cell_types)

    regex = re.compile(pattern)
    return {cell_type for cell_type in cell_types if regex.search(str(cell_type))}


def select_matrix_columns(X, column_indices: np.ndarray):
    if sparse.issparse(X):
        return X[:, column_indices]

    column_indices = np.asarray(column_indices)
    try:
        order = np.argsort(column_indices)
        sorted_columns = column_indices[order]
        selected = X[:, sorted_columns]
        if hasattr(selected, "to_memory"):
            selected = selected.to_memory()
        if not sparse.issparse(selected):
            selected = np.asarray(selected)
        return selected[:, np.argsort(order)]
    except Exception:
        return np.asarray(X)[:, column_indices]


def pseudobulk_log_profiles(
    adata: ad.AnnData,
    *,
    cell_type_key: str,
    genes: Sequence[str],
    counts_layer: str | None,
    min_cells: int,
    target_sum: float,
) -> tuple[pd.DataFrame, pd.Series]:
    if cell_type_key not in adata.obs:
        raise KeyError(f"Missing cell type key {cell_type_key!r}.")

    genes = [str(gene) for gene in genes if gene in adata.var_names]
    if not genes:
        raise ValueError("No shared genes available for pseudo-bulk profiles.")

    gene_indices = adata.var_names.get_indexer(genes)
    X = select_matrix_columns(counts_from_layer(adata, counts_layer), gene_indices)
    cell_types = adata.obs[cell_type_key].astype(str).to_numpy()

    rows = []
    index = []
    n_cells: dict[str, int] = {}
    for cell_type in pd.unique(cell_types):
        row_idx = np.flatnonzero(cell_types == cell_type)
        if row_idx.size < min_cells:
            continue
        X_group = X[row_idx]
        if sparse.issparse(X_group):
            summed = np.asarray(X_group.sum(axis=0)).ravel()
        else:
            summed = np.asarray(X_group).sum(axis=0)
        rows.append(summed)
        index.append(str(cell_type))
        n_cells[str(cell_type)] = int(row_idx.size)

    if not rows:
        return (
            pd.DataFrame(columns=genes, index=pd.Index([], name=cell_type_key)),
            pd.Series(dtype=int, name="n_cells"),
        )

    values = np.vstack(rows).astype(float, copy=False)
    totals = values.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    values = np.log1p(values / totals * target_sum)

    return (
        pd.DataFrame(
            values,
            index=pd.Index(index, name=cell_type_key),
            columns=genes,
        ),
        pd.Series(n_cells, name="n_cells"),
    )


def cached_reference_pseudobulk_log_profiles(
    reference_adata: ad.AnnData,
    *,
    cell_type_key: str,
    genes: Sequence[str],
    counts_layer: str | None,
    min_cells: int,
    target_sum: float,
    cache: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    cache_key = (
        tuple(str(gene) for gene in genes),
        cell_type_key,
        counts_layer or "X",
        int(min_cells),
        float(target_sum),
    )
    if cache_key not in cache:
        cache[cache_key] = pseudobulk_log_profiles(
            reference_adata,
            cell_type_key=cell_type_key,
            genes=genes,
            counts_layer=counts_layer,
            min_cells=min_cells,
            target_sum=target_sum,
        )
    return cache[cache_key]


def cosine_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return np.nan
    return float(np.dot(x, y) / denom)


def compute_tcell_cosine_similarity(
    adata: ad.AnnData,
    reference_adata: ad.AnnData,
    *,
    spatial_counts_layer: str | None,
    tissue: str,
    source_file: str,
    condition: str,
    args: argparse.Namespace,
    reference_profile_cache: dict,
) -> pd.DataFrame:
    reference_cell_type_key = args.reference_cell_type_key or args.tcell_cell_type_key
    reference_counts_layer = args.reference_counts_layer or None
    spatial_layer_label = spatial_counts_layer or "X"
    reference_layer_label = reference_counts_layer or "X"

    base_error = {
        "tissue": tissue,
        "source_file": source_file,
        "condition": condition,
        "spatial_counts_layer": spatial_layer_label,
        "reference_counts_layer": reference_layer_label,
        "reference_cell_type_key": reference_cell_type_key,
        "cosine_similarity": np.nan,
        "n_spatial_cells": np.nan,
        "n_reference_cells": np.nan,
        "n_shared_genes": np.nan,
    }

    try:
        shared_genes = adata.var_names.intersection(reference_adata.var_names).astype(str)
        if len(shared_genes) == 0:
            raise ValueError("No shared genes between spatial AnnData and reference AnnData.")

        spatial_pb, spatial_n = pseudobulk_log_profiles(
            adata,
            cell_type_key=args.tcell_cell_type_key,
            genes=shared_genes,
            counts_layer=spatial_counts_layer,
            min_cells=args.tcell_cosine_min_spatial_cells,
            target_sum=args.library_size_target_sum,
        )
        reference_pb, reference_n = cached_reference_pseudobulk_log_profiles(
            reference_adata,
            cell_type_key=reference_cell_type_key,
            genes=shared_genes,
            counts_layer=reference_counts_layer,
            min_cells=args.tcell_cosine_min_reference_cells,
            target_sum=args.library_size_target_sum,
            cache=reference_profile_cache,
        )
        if spatial_pb.empty:
            raise ValueError("No spatial cell type pseudo-bulk profiles passed filters.")
        if reference_pb.empty:
            raise ValueError("No reference cell type pseudo-bulk profiles passed filters.")

        cell_type_map = parse_cell_type_map(args.tcell_cosine_cell_type_map)
        tcell_types = selected_tcell_types(spatial_pb.index.astype(str).tolist(), args)
        if not tcell_types:
            raise ValueError("No spatial cell types matched the T-cell cosine selection.")

        rows = []
        for spatial_cell_type in spatial_pb.index.astype(str):
            if spatial_cell_type not in tcell_types:
                continue
            reference_cell_type = cell_type_map.get(spatial_cell_type, spatial_cell_type)
            row = {
                **base_error,
                "spatial_cell_type": spatial_cell_type,
                "reference_cell_type": reference_cell_type,
                "n_shared_genes": int(len(shared_genes)),
                "error": "",
            }
            if reference_cell_type not in reference_pb.index:
                row["error"] = f"Missing reference cell type {reference_cell_type!r}."
                rows.append(row)
                continue

            row["cosine_similarity"] = cosine_between_vectors(
                spatial_pb.loc[spatial_cell_type].to_numpy(),
                reference_pb.loc[reference_cell_type].to_numpy(),
            )
            row["n_spatial_cells"] = int(spatial_n.loc[spatial_cell_type])
            row["n_reference_cells"] = int(reference_n.loc[reference_cell_type])
            rows.append(row)

        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame([{**base_error, "error": repr(exc)}])


def safe_layer_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def run_tcell_contamination(
    adata: ad.AnnData,
    tissue: str,
    source_file: str,
    args: argparse.Namespace,
    reference_adata: ad.AnnData | None = None,
    reference_profile_cache: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty_df = pd.DataFrame()
    if args.tcell_cell_type_key not in adata.obs:
        return (
            pd.DataFrame(
                [
                    {
                        "tissue": tissue,
                        "source_file": source_file,
                        "condition": "all",
                        "error": f"Missing T-cell label key {args.tcell_cell_type_key!r}.",
                    }
                ]
            ),
            empty_df,
            empty_df,
            empty_df,
        )
    if args.admix_probs_layer not in adata.layers:
        return (
            pd.DataFrame(
                [
                    {
                        "tissue": tissue,
                        "source_file": source_file,
                        "condition": "all",
                        "error": f"Missing admixture probability layer {args.admix_probs_layer!r}.",
                    }
                ]
            ),
            empty_df,
            empty_df,
            empty_df,
        )

    try:
        ensure_tcell_inputs(adata, tissue, args)
    except Exception as exc:
        return (
            pd.DataFrame(
                [
                    {
                        "tissue": tissue,
                        "source_file": source_file,
                        "condition": "all",
                        "error": repr(exc),
                    }
                ]
            ),
            empty_df,
            empty_df,
            empty_df,
        )

    if args.filter_tcell_invalid_cells:
        adata, dropped = filtered_tcell_adata(adata, args)
        print(
            "> Filtered T-cell metric cells: "
            f"dropped {dropped['total']:,}; remaining {adata.n_obs:,}; details={dropped}"
        )
        if adata.n_obs == 0:
            return (
                pd.DataFrame(
                    [
                        {
                            "tissue": tissue,
                            "source_file": source_file,
                            "condition": "all",
                            "error": "No cells remain after T-cell invalid-cell filtering.",
                        }
                    ]
                ),
                empty_df,
                empty_df,
                empty_df,
            )

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
    cosine_results: list[pd.DataFrame] = []
    specificity_layers: list[dict] = [] if args.only_additional_counts else [
        {
            "condition": args.uncorrected_condition_label,
            "layer": lognorm_uncorrected,
            "counts_layer": None,
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
            return (
                pd.DataFrame(
                    [
                        {
                            "tissue": tissue,
                            "source_file": source_file,
                            "condition": "cell_type_specific_removal",
                            "error": f"Missing removal group key {removal_group_key!r}.",
                        }
                    ]
                ),
                empty_df,
                empty_df,
                empty_df,
            )
        removal_labels = adata.obs[removal_group_key]

    admix_probs = adata.layers[args.admix_probs_layer]
    target_fractions = [] if args.only_additional_counts else parse_float_list(args.target_removed_fractions)
    for target_removed_fraction in target_fractions:
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
                    "counts_layer": raw_layer,
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
                    "counts_layer": args.additional_counts_layer,
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
                                "counts_layer": raw_layer,
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
        if (
            reference_adata is not None
            and not args.skip_tcell_cosine_similarity
        ):
            if reference_profile_cache is None:
                reference_profile_cache = {}
            cosine = compute_tcell_cosine_similarity(
                adata,
                reference_adata,
                spatial_counts_layer=layer_info.get("counts_layer"),
                tissue=tissue,
                source_file=source_file,
                condition=condition,
                args=args,
                reference_profile_cache=reference_profile_cache,
            )
            if not cosine.empty:
                cosine["target_removed_fraction"] = layer_info["target_removed_fraction"]
                cosine["admix_prob_threshold"] = layer_info["admix_prob_threshold"]
                cosine["achieved_removed_fraction"] = layer_info["achieved_removed_fraction"]
                cosine["cleaning_mode"] = args.cleaning_mode
                cosine["probability_transform"] = (
                    args.probability_transform
                    if args.cleaning_mode == "probability_scale"
                    else ""
                )
                cosine["probability_removal_scale"] = layer_info["probability_removal_scale"]
                cosine["cell_type_specific_removal"] = args.cell_type_specific_removal
                cosine["group_admix_prob_thresholds"] = layer_info[
                    "group_admix_prob_thresholds"
                ]
                cosine["group_probability_removal_scales"] = layer_info[
                    "group_probability_removal_scales"
                ]
                cosine["group_achieved_removed_fractions"] = layer_info[
                    "group_achieved_removed_fractions"
                ]
                cosine_results.append(cosine)

        if args.skip_tcell_neighborhood_metrics:
            continue

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

    cosine_df = (
        pd.concat(cosine_results, ignore_index=True)
        if cosine_results
        else pd.DataFrame()
    )
    if not specificity_results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), cosine_df

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
    return specificity_df, coefficient_df, profile_df, cosine_df


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

    metadata_cols = [
        col
        for col in [
            "sample_replicate",
            "dropout_label",
            "dropout_fraction",
            "input_layer",
        ]
        if col in metric_rows.columns
    ]
    group_cols = [
        "tissue",
        "source_file",
        *metadata_cols,
        "condition",
        "target_cell_type_i",
        "contaminating_cell_type_j",
    ]
    index_cols_list = [
        "tissue",
        "source_file",
        *metadata_cols,
        "target_cell_type_i",
        "contaminating_cell_type_j",
    ]

    effect = (
        metric_rows.groupby(group_cols, dropna=False)["marker_overlap_n"]
        .mean()
        .reset_index()
        .pivot(
            index=index_cols_list,
            columns="condition",
            values="marker_overlap_n",
        )
        .reset_index()
    )

    index_cols = set(index_cols_list)
    condition_cols = [col for col in effect.columns if col not in index_cols]
    if args.uncorrected_condition_label in condition_cols:
        baseline_cols = [args.uncorrected_condition_label]
    else:
        baseline_cols = [
            col for col in condition_cols if str(col).endswith("_uncorrected")
        ]
    for uncorrected in baseline_cols:
        if uncorrected not in condition_cols:
            continue
        safe_uncorrected = safe_layer_suffix(str(uncorrected)).lower()
        for condition in condition_cols:
            if condition == uncorrected:
                continue
            safe_condition = safe_layer_suffix(str(condition)).lower()
            effect[f"delta_{safe_condition}_minus_{safe_uncorrected}"] = (
                effect[condition] - effect[uncorrected]
            )

    return effect


def add_layered_metadata(
    data,
    *,
    input_layer: str,
    evaluation_method: str,
    probability_layer: str = "",
    counts_layer: str = "",
    sample_replicate: str = "",
    dropout_label: str = "",
    dropout_fraction=np.nan,
):
    if isinstance(data, list):
        for row in data:
            row["input_layer"] = input_layer
            row["evaluation_method"] = evaluation_method
            row["probability_layer"] = probability_layer
            row["counts_layer"] = counts_layer
            row["sample_replicate"] = sample_replicate
            row["dropout_label"] = dropout_label
            row["dropout_fraction"] = dropout_fraction
        return data

    if data is None or data.empty:
        return data
    data = data.copy()
    data["input_layer"] = input_layer
    data["evaluation_method"] = evaluation_method
    data["probability_layer"] = probability_layer
    data["counts_layer"] = counts_layer
    data["sample_replicate"] = sample_replicate
    data["dropout_label"] = dropout_label
    data["dropout_fraction"] = dropout_fraction
    return data


def layered_work_adata(
    adata: ad.AnnData,
    counts_layer: str,
    *,
    probability_layer: str | None = None,
    split_counts_layer: str | None = None,
    args: argparse.Namespace,
) -> ad.AnnData:
    if counts_layer != "X" and counts_layer not in adata.layers:
        raise KeyError(f"Missing count layer {counts_layer!r}.")

    work = ad.AnnData(
        X=sanitize_numeric_matrix(counts_from_layer(adata, counts_layer).copy()),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )

    if args.coord_key in adata.obsm:
        work.obsm[args.coord_key] = adata.obsm[args.coord_key].copy()

    if probability_layer is not None:
        if probability_layer not in adata.layers:
            raise KeyError(f"Missing probability layer {probability_layer!r}.")
        work.layers["_metrics_admix_probs"] = adata.layers[probability_layer].copy()

    if split_counts_layer is not None:
        if split_counts_layer not in adata.layers:
            raise KeyError(f"Missing SPLIT counts layer {split_counts_layer!r}.")
        work.layers["_metrics_split_counts"] = sanitize_numeric_matrix(
            adata.layers[split_counts_layer].copy()
        )

    return work


def missing_layer_metric_row(
    tissue: str,
    source_file: str,
    *,
    case_name: str,
    layer_key: str,
    input_layer: str,
    evaluation_method: str,
    sample_replicate: str,
    dropout_label: str = "",
    dropout_fraction=np.nan,
    probability_layer: str = "",
    counts_layer: str = "",
) -> dict:
    row = empty_metric_row(tissue, source_file, case_name)
    row["error"] = f"Missing layer {layer_key!r}."
    return add_layered_metadata(
        [row],
        input_layer=input_layer,
        evaluation_method=evaluation_method,
        probability_layer=probability_layer,
        counts_layer=counts_layer,
        sample_replicate=sample_replicate,
        dropout_label=dropout_label,
        dropout_fraction=dropout_fraction,
    )[0]


def missing_layer_tcell_frame(
    tissue: str,
    source_file: str,
    *,
    condition: str,
    layer_key: str,
    input_layer: str,
    evaluation_method: str,
    sample_replicate: str,
    dropout_label: str = "",
    dropout_fraction=np.nan,
    probability_layer: str = "",
    counts_layer: str = "",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tissue": tissue,
                "source_file": source_file,
                "condition": condition,
                "error": f"Missing layer {layer_key!r}.",
                "input_layer": input_layer,
                "evaluation_method": evaluation_method,
                "probability_layer": probability_layer,
                "counts_layer": counts_layer,
                "sample_replicate": sample_replicate,
                "dropout_label": dropout_label,
                "dropout_fraction": dropout_fraction,
            }
        ]
    )


def score_split_cell_type_case(
    adata: ad.AnnData,
    split_counts,
    tissue: str,
    source_file: str,
    case_name: str,
    args: argparse.Namespace,
) -> list[dict]:
    if args.label_key not in adata.obs:
        row = empty_metric_row(tissue, source_file, case_name)
        row["error"] = f"Missing label key {args.label_key!r}."
        return [row]

    labels = adata.obs[args.label_key]
    exclude_labels = set(parse_str_list(args.exclude_labels))
    if exclude_labels:
        keep = ~labels.astype(str).isin(exclude_labels)
        split_counts = split_counts[keep.to_numpy()]
        labels = labels[keep]

    return [score_case(tissue, source_file, case_name, split_counts, labels, args)]


def run_layered_evaluations_for_file(
    adata: ad.AnnData,
    tissue: str,
    source_file: str,
    sample_replicate: str,
    dropout_label: str,
    dropout_fraction: float,
    args: argparse.Namespace,
    skip_zero_dropout_layers: bool = False,
    reference_adata: ad.AnnData | None = None,
    reference_profile_cache: dict | None = None,
) -> tuple[
    list[dict],
    list[pd.DataFrame],
    list[pd.DataFrame],
    list[pd.DataFrame],
    list[pd.DataFrame],
]:
    cell_type_rows: list[dict] = []
    tcell_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    profile_frames: list[pd.DataFrame] = []
    cosine_frames: list[pd.DataFrame] = []

    layers = parse_str_list(args.evaluation_layers)
    if not layers:
        return cell_type_rows, tcell_frames, coefficient_frames, profile_frames, cosine_frames

    for input_layer in layers:
        if skip_zero_dropout_layers and input_layer == "X":
            continue
        layer_dropout_label, layer_dropout_fraction = layer_dropout_metadata(
            input_layer,
            dropout_label,
            dropout_fraction,
        )
        baseline_label = args.uncorrected_condition_label
        baseline_counts_layer = input_layer
        if input_layer != "X" and input_layer not in adata.layers:
            if not args.skip_cell_type_separation:
                cell_type_rows.append(
                    missing_layer_metric_row(
                        tissue,
                        source_file,
                        case_name=f"{baseline_label}_missing_input",
                        layer_key=input_layer,
                        input_layer=input_layer,
                        evaluation_method="baseline",
                        counts_layer=baseline_counts_layer,
                        sample_replicate=sample_replicate,
                        dropout_label=layer_dropout_label,
                        dropout_fraction=layer_dropout_fraction,
                    )
                )
            if not args.skip_tcell_contamination:
                missing_baseline = missing_layer_tcell_frame(
                    tissue,
                    source_file,
                    condition=f"{baseline_label}_missing_input",
                    layer_key=input_layer,
                    input_layer=input_layer,
                    evaluation_method="baseline",
                    counts_layer=baseline_counts_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
                tcell_frames.append(missing_baseline)
                cosine_frames.append(missing_baseline.copy())
            continue

        baseline_counts = sanitize_numeric_matrix(
            counts_from_layer(adata, baseline_counts_layer).copy()
        )
        if not args.skip_cell_type_separation:
            baseline_rows = score_split_cell_type_case(
                adata,
                baseline_counts,
                tissue,
                source_file,
                baseline_label,
                args,
            )
            cell_type_rows.extend(
                add_layered_metadata(
                    baseline_rows,
                    input_layer=input_layer,
                    evaluation_method="baseline",
                    counts_layer=baseline_counts_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )

        if not args.skip_tcell_contamination:
            baseline_args = copy.copy(args)
            baseline_args.target_removed_fractions = ""
            baseline_args.admix_probs_layer = "_metrics_zero_probs"
            baseline_args.additional_counts_layer = ""
            baseline_args.uncorrected_condition_label = baseline_label
            baseline_work = ad.AnnData(
                X=baseline_counts.copy(),
                obs=adata.obs.copy(),
                var=adata.var.copy(),
            )
            if args.coord_key in adata.obsm:
                baseline_work.obsm[args.coord_key] = adata.obsm[args.coord_key].copy()
            baseline_work.layers["_metrics_zero_probs"] = sparse.csr_matrix(baseline_work.shape)
            tcell_df, coefficient_df, profile_df, cosine_df = run_tcell_contamination(
                baseline_work,
                tissue,
                source_file,
                baseline_args,
                reference_adata=reference_adata,
                reference_profile_cache=reference_profile_cache,
            )
            tcell_frames.append(
                add_layered_metadata(
                    tcell_df,
                    input_layer=input_layer,
                    evaluation_method="baseline",
                    counts_layer=baseline_counts_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )
            coefficient_frames.append(
                add_layered_metadata(
                    coefficient_df,
                    input_layer=input_layer,
                    evaluation_method="baseline",
                    counts_layer=baseline_counts_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )
            profile_frames.append(
                add_layered_metadata(
                    profile_df,
                    input_layer=input_layer,
                    evaluation_method="baseline",
                    counts_layer=baseline_counts_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )
            cosine_frames.append(
                add_layered_metadata(
                    cosine_df,
                    input_layer=input_layer,
                    evaluation_method="baseline",
                    counts_layer=baseline_counts_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )

        if args.layered_baseline_only:
            continue

        for method, prob_template, label in [
            ("syndri", args.syndri_probs_template, args.syndri_case_label),
            ("mlp", args.mlp_probs_template, args.mlp_case_label),
        ]:
            probability_layer = format_layer_template(prob_template, input_layer)
            probability_fallbacks = ()
            if input_layer == "X":
                probability_fallbacks = (
                    "admix_probs" if method == "syndri" else "mlp_admix_probs",
                )
            resolved_probability_layer = resolve_layer_key(
                adata,
                probability_layer,
                fallback_keys=probability_fallbacks,
            )
            method_args = copy.copy(args)
            if method == "mlp":
                method_args.clean_prob_threshold = 0.0
            method_args.admix_probs_layer = "_metrics_admix_probs"
            method_args.additional_counts_layer = ""
            method_args.uncorrected_condition_label = f"{label}_uncorrected"

            if input_layer != "X" and input_layer not in adata.layers:
                cell_type_rows.append(
                    missing_layer_metric_row(
                        tissue,
                        source_file,
                        case_name=f"{label}_missing_input",
                        layer_key=input_layer,
                        input_layer=input_layer,
                        evaluation_method=method,
                        probability_layer=probability_layer,
                        sample_replicate=sample_replicate,
                        dropout_label=layer_dropout_label,
                        dropout_fraction=layer_dropout_fraction,
                    )
                )
                continue
            if resolved_probability_layer is None:
                cell_type_rows.append(
                    missing_layer_metric_row(
                        tissue,
                        source_file,
                        case_name=f"{label}_missing_probability",
                        layer_key=probability_layer,
                        input_layer=input_layer,
                        evaluation_method=method,
                        probability_layer=probability_layer,
                        counts_layer=input_layer,
                        sample_replicate=sample_replicate,
                        dropout_label=layer_dropout_label,
                        dropout_fraction=layer_dropout_fraction,
                    )
                )
                tcell_frames.append(
                    missing_layer_tcell_frame(
                        tissue,
                        source_file,
                        condition=f"{label}_missing_probability",
                        layer_key=probability_layer,
                        input_layer=input_layer,
                        evaluation_method=method,
                        probability_layer=probability_layer,
                        counts_layer=input_layer,
                        sample_replicate=sample_replicate,
                        dropout_label=layer_dropout_label,
                        dropout_fraction=layer_dropout_fraction,
                    )
                )
                cosine_frames.append(
                    missing_layer_tcell_frame(
                        tissue,
                        source_file,
                        condition=f"{label}_missing_probability",
                        layer_key=probability_layer,
                        input_layer=input_layer,
                        evaluation_method=method,
                        probability_layer=probability_layer,
                        counts_layer=input_layer,
                        sample_replicate=sample_replicate,
                        dropout_label=layer_dropout_label,
                        dropout_fraction=layer_dropout_fraction,
                    )
                )
                continue

            work = layered_work_adata(
                adata,
                input_layer,
                probability_layer=resolved_probability_layer,
                args=method_args,
            )

            if method == "syndri" and args.syndri_reference_cosine:
                # Preserve reference provenance for automatic cache matching.
                work.uns = copy.deepcopy(adata.uns)
                adaptive = run_reference_cosine_evaluation(
                    work, tissue, source_file, method_args, input_layer=input_layer,
                    reference_adata=reference_adata, reference_profile_cache=reference_profile_cache)
                for result, destination in zip(adaptive, [cell_type_rows, tcell_frames,
                        coefficient_frames, profile_frames, cosine_frames]):
                    result = add_layered_metadata(
                        result, input_layer=input_layer, evaluation_method="syndri",
                        probability_layer=resolved_probability_layer, counts_layer=input_layer,
                        sample_replicate=sample_replicate, dropout_label=layer_dropout_label,
                        dropout_fraction=layer_dropout_fraction)
                    if isinstance(result, list):
                        destination.extend(result)
                    else:
                        destination.append(result)

            if not args.skip_cell_type_separation:
                rows = run_cell_type_separation(work, tissue, source_file, method_args)
                rows = [row for row in rows if row.get("case") != "admixed"]
                for row in rows:
                    if str(row.get("case", "")).startswith("cleaned_admixed_"):
                        row["case"] = str(row["case"]).replace("cleaned_admixed", f"{label}_cleaned", 1)
                cell_type_rows.extend(
                    add_layered_metadata(
                        rows,
                        input_layer=input_layer,
                        evaluation_method=method,
                        probability_layer=resolved_probability_layer,
                        counts_layer=input_layer,
                        sample_replicate=sample_replicate,
                        dropout_label=layer_dropout_label,
                        dropout_fraction=layer_dropout_fraction,
                    )
                )

            if not args.skip_tcell_contamination:
                tcell_df, coefficient_df, profile_df, cosine_df = run_tcell_contamination(
                    work,
                    tissue,
                    source_file,
                    method_args,
                    reference_adata=reference_adata,
                    reference_profile_cache=reference_profile_cache,
                )
                for frame in [tcell_df, coefficient_df, profile_df, cosine_df]:
                    if frame is not None and not frame.empty and "condition" in frame:
                        frame.drop(
                            frame.index[
                                frame["condition"].astype(str)
                                == method_args.uncorrected_condition_label
                            ],
                            inplace=True,
                        )
                        frame["condition"] = frame["condition"].astype(str).replace(
                            {
                                method_args.uncorrected_condition_label: f"{label}_uncorrected",
                            }
                        )
                        frame["condition"] = frame["condition"].str.replace(
                            "cleaned_admixed", f"{label}_cleaned", regex=False
                        )
                tcell_frames.append(
                    add_layered_metadata(
                        tcell_df,
                        input_layer=input_layer,
                        evaluation_method=method,
                        probability_layer=resolved_probability_layer,
                        counts_layer=input_layer,
                        sample_replicate=sample_replicate,
                        dropout_label=layer_dropout_label,
                        dropout_fraction=layer_dropout_fraction,
                    )
                )
                coefficient_frames.append(
                    add_layered_metadata(
                        coefficient_df,
                        input_layer=input_layer,
                        evaluation_method=method,
                        probability_layer=resolved_probability_layer,
                        counts_layer=input_layer,
                        sample_replicate=sample_replicate,
                        dropout_label=layer_dropout_label,
                        dropout_fraction=layer_dropout_fraction,
                    )
                )
                profile_frames.append(
                    add_layered_metadata(
                        profile_df,
                        input_layer=input_layer,
                        evaluation_method=method,
                        probability_layer=resolved_probability_layer,
                        counts_layer=input_layer,
                        sample_replicate=sample_replicate,
                        dropout_label=layer_dropout_label,
                        dropout_fraction=layer_dropout_fraction,
                    )
                )
                cosine_frames.append(
                    add_layered_metadata(
                        cosine_df,
                        input_layer=input_layer,
                        evaluation_method=method,
                        probability_layer=resolved_probability_layer,
                        counts_layer=input_layer,
                        sample_replicate=sample_replicate,
                        dropout_label=layer_dropout_label,
                        dropout_fraction=layer_dropout_fraction,
                    )
                )

        split_layer = format_layer_template(args.split_counts_template, input_layer)
        split_label = args.split_case_label
        split_method = "split"
        if input_layer != "X" and input_layer not in adata.layers:
            cell_type_rows.append(
                missing_layer_metric_row(
                    tissue,
                    source_file,
                    case_name=f"{split_label}_missing_input",
                    layer_key=input_layer,
                    input_layer=input_layer,
                    evaluation_method=split_method,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )
            continue
        if split_layer not in adata.layers:
            cell_type_rows.append(
                missing_layer_metric_row(
                    tissue,
                    source_file,
                    case_name=f"{split_label}_missing_counts",
                    layer_key=split_layer,
                    input_layer=input_layer,
                    evaluation_method=split_method,
                    counts_layer=split_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )
            tcell_frames.append(
                missing_layer_tcell_frame(
                    tissue,
                    source_file,
                    condition=f"{split_label}_missing_counts",
                    layer_key=split_layer,
                    input_layer=input_layer,
                    evaluation_method=split_method,
                    counts_layer=split_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )
            cosine_frames.append(
                missing_layer_tcell_frame(
                    tissue,
                    source_file,
                    condition=f"{split_label}_missing_counts",
                    layer_key=split_layer,
                    input_layer=input_layer,
                    evaluation_method=split_method,
                    counts_layer=split_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )
            continue

        split_work = layered_work_adata(
            adata,
            input_layer,
            split_counts_layer=split_layer,
            args=args,
        )
        if not args.skip_cell_type_separation:
            split_rows = score_split_cell_type_case(
                split_work,
                counts_from_layer(split_work, "_metrics_split_counts"),
                tissue,
                source_file,
                split_label,
                args,
            )
            cell_type_rows.extend(
                add_layered_metadata(
                    split_rows,
                    input_layer=input_layer,
                    evaluation_method=split_method,
                    counts_layer=split_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )

        if not args.skip_tcell_contamination:
            split_args = copy.copy(args)
            split_args.target_removed_fractions = ""
            split_args.admix_probs_layer = "_metrics_zero_probs"
            split_args.additional_counts_layer = "_metrics_split_counts"
            split_args.purified_condition_label = split_label
            split_args.additional_counts_case_label = split_label
            split_args.uncorrected_condition_label = f"{split_label}_input_counts"
            split_work.layers["_metrics_zero_probs"] = sparse.csr_matrix(split_work.shape)
            tcell_df, coefficient_df, profile_df, cosine_df = run_tcell_contamination(
                split_work,
                tissue,
                source_file,
                split_args,
                reference_adata=reference_adata,
                reference_profile_cache=reference_profile_cache,
            )
            for frame_name, frame in [
                ("tcell", tcell_df),
                ("coefficients", coefficient_df),
                ("profiles", profile_df),
                ("cosine", cosine_df),
            ]:
                if frame is not None and not frame.empty and "condition" in frame:
                    keep = frame["condition"].astype(str) == split_label
                    if frame_name == "tcell":
                        tcell_df = frame.loc[keep].copy()
                    elif frame_name == "coefficients":
                        coefficient_df = frame.loc[keep].copy()
                    elif frame_name == "profiles":
                        profile_df = frame.loc[keep].copy()
                    else:
                        cosine_df = frame.loc[keep].copy()

            tcell_frames.append(
                add_layered_metadata(
                    tcell_df,
                    input_layer=input_layer,
                    evaluation_method=split_method,
                    counts_layer=split_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )
            coefficient_frames.append(
                add_layered_metadata(
                    coefficient_df,
                    input_layer=input_layer,
                    evaluation_method=split_method,
                    counts_layer=split_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )
            profile_frames.append(
                add_layered_metadata(
                    profile_df,
                    input_layer=input_layer,
                    evaluation_method=split_method,
                    counts_layer=split_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )
            cosine_frames.append(
                add_layered_metadata(
                    cosine_df,
                    input_layer=input_layer,
                    evaluation_method=split_method,
                    counts_layer=split_layer,
                    sample_replicate=sample_replicate,
                    dropout_label=layer_dropout_label,
                    dropout_fraction=layer_dropout_fraction,
                )
            )

    return cell_type_rows, tcell_frames, coefficient_frames, profile_frames, cosine_frames


def run_reference_cosine_evaluation(
    adata, tissue, source_file, args, *, input_layer="X",
    reference_adata=None, reference_profile_cache=None,
):
    from syndri_reference_cleaning import prepare_reference_cleaned

    cleaned, achieved, profile_path = prepare_reference_cleaned(
        adata, args, source_file, input_layer=input_layer)
    name = "syndri_reference_cosine"
    adaptive_args = copy.copy(args)
    adaptive_args.target_removed_fractions = ""
    adaptive_args.additional_counts_layer = ""
    adaptive_args.cleaning_mode = "zero"
    adaptive_args.cell_type_specific_removal = False
    adaptive_args.uncorrected_condition_label = name
    adaptive_args.admix_probs_layer = "_metrics_zero_probs"
    work = ad.AnnData(X=cleaned, obs=adata.obs.copy(), var=adata.var.copy())
    if args.coord_key in adata.obsm:
        work.obsm[args.coord_key] = adata.obsm[args.coord_key].copy()
    work.layers[adaptive_args.admix_probs_layer] = sparse.csr_matrix(work.shape)
    rows = []
    if not args.skip_cell_type_separation:
        rows = run_cell_type_separation(work, tissue, source_file, adaptive_args)
        for row in rows:
            row["case"] = name
    frames = [pd.DataFrame() for _ in range(4)]
    if not args.skip_tcell_contamination:
        frames = list(run_tcell_contamination(
            work, tissue, source_file, adaptive_args, reference_adata=reference_adata,
            reference_profile_cache=reference_profile_cache))
    metadata = dict(
        cleaning_mode="zero",
        threshold_selection="cell_type_smallest_near_optimal_k",
        reference_profiles=profile_path,
        achieved_removed_fraction=achieved,
        probability_transform="",
        cell_type_specific_removal=True,
        syndri_probability_ordering="cell_type_expressed_mean",
        syndri_one_se_multiplier=getattr(args, "syndri_one_se_multiplier", 1.0),
    )
    for row in rows:
        row.update(metadata)
    for frame in frames:
        if not frame.empty:
            frame["condition"] = name
            for key, value in metadata.items():
                frame[key] = value
    return (rows, *frames)


def output_identity_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "source_file",
        "tissue",
        "sample_replicate",
        "dropout_label",
        "dropout_fraction",
        "input_layer",
        "evaluation_method",
        "probability_layer",
        "counts_layer",
        "case",
        "condition",
        "target_removed_fraction",
        "threshold_selection",
        "cosine_window",
        "reference_profiles",
        "target_cell_type_i",
        "contaminating_cell_type_j",
        "spatial_cell_type",
        "reference_cell_type",
        "spatial_counts_layer",
        "reference_counts_layer",
        "gene",
        "cell_type",
        "layer",
    ]
    return [col for col in candidates if col in df.columns]


def remove_layer_count_baselines(
    df: pd.DataFrame,
    uncorrected_condition_label: str,
) -> pd.DataFrame:
    label_columns = [col for col in ["case", "condition"] if col in df.columns]
    if not label_columns:
        return df

    keep = pd.Series(True, index=df.index)
    for col in label_columns:
        values = df[col].astype(str)
        remove = values.str.endswith("_uncorrected") & (values != uncorrected_condition_label)
        if {"input_layer", "counts_layer"}.issubset(df.columns):
            remove |= (
                (values == uncorrected_condition_label)
                & df["input_layer"].notna()
                & df["counts_layer"].notna()
                & (df["input_layer"].astype(str) != df["counts_layer"].astype(str))
            )
        keep &= ~remove
    return df.loc[keep].copy()


def write_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    append: bool = False,
    replace_layer_count_baselines: bool = False,
    uncorrected_condition_label: str = "uncorrected",
    replace_labels: tuple[str, ...] = (),
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.copy()
    if append and path.exists():
        existing = pd.read_csv(path)
        if replace_labels:
            replace = pd.Series(False, index=existing.index)
            for column in [col for col in ["case", "condition"] if col in existing]:
                replace |= existing[column].astype(str).isin(replace_labels)
            existing = existing.loc[~replace].copy()
        if replace_layer_count_baselines:
            existing = remove_layer_count_baselines(existing, uncorrected_condition_label)
        output = pd.concat([existing, output], ignore_index=True, sort=False)
        identity = output_identity_columns(output)
        if identity:
            output = output.drop_duplicates(subset=identity, keep="last")
    output.to_csv(path, index=False)
    print(f"Wrote {len(output):,} rows: {path}")
    return output


def remove_stale_tcell_cosine_errors(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "error" not in df.columns:
        return df
    errors = df["error"].fillna("").astype(str)
    stale = errors.str.contains("No spatial cell types matched the T-cell cosine selection", regex=False)
    if "spatial_cell_type" in df.columns:
        stale &= df["spatial_cell_type"].isna() | (df["spatial_cell_type"].astype(str) == "")
    return df.loc[~stale].copy()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not np.isfinite(args.separation_min_counts) or args.separation_min_counts < 0:
        parser.error("--separation-min-counts must be finite and nonnegative")
    if args.separation_min_genes < 0:
        parser.error("--separation-min-genes must be nonnegative")

    if args.syndri_cosine_window < 1:
        parser.error("--syndri-cosine-window must be positive")
    if not np.isfinite(args.syndri_one_se_multiplier) or args.syndri_one_se_multiplier < 0:
        parser.error("--syndri-one-se-multiplier must be finite and nonnegative")
    if args.only_additional_counts and not args.additional_counts_layer:
        parser.error("--only-additional-counts requires --additional-counts-layer")
    if args.only_additional_counts and parse_str_list(args.evaluation_layers):
        parser.error("--only-additional-counts is not supported in layered mode")
    if args.only_syndri_reference_cosine and not args.syndri_reference_cosine:
        parser.error("--only-syndri-reference-cosine requires --syndri-reference-cosine")
    if args.only_syndri_reference_cosine and args.only_additional_counts:
        parser.error("The two single-case evaluation modes are mutually exclusive")
    if args.only_syndri_reference_cosine and parse_str_list(args.evaluation_layers):
        parser.error("--only-syndri-reference-cosine is not supported in layered mode")

    files = discover_input_files(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / args.config_output
    config_path.write_text(
        json.dumps({**vars(args), "input_files": [str(f) for f in files]}, indent=2) + "\n"
    )
    print(f"Wrote config: {config_path}")
    print(f"Found {len(files)} input file(s). sample_frac={args.sample_frac}")

    reference_adata = None
    reference_profile_cache: dict = {}
    if not args.skip_tcell_contamination and not args.skip_tcell_cosine_similarity:
        if args.reference_data:
            reference_path = Path(args.reference_data).expanduser().resolve()
            if not reference_path.exists():
                raise FileNotFoundError(f"Reference AnnData does not exist: {reference_path}")
            reference_adata = ad.read_h5ad(reference_path, backed="r")
            print(f"Loaded T-cell cosine reference: {reference_path}")
        else:
            print("Skipping T-cell cosine similarity: --reference-data was not provided.")

    cell_type_rows: list[dict] = []
    tcell_frames: list[pd.DataFrame] = []
    tcell_coefficient_frames: list[pd.DataFrame] = []
    cell_type_profile_frames: list[pd.DataFrame] = []
    tcell_cosine_frames: list[pd.DataFrame] = []
    zero_dropout_replicates_seen: set[str] = set()

    for path in tqdm(files, desc="Spatial files"):
        tissue = tissue_name_from_path(path, args.tissue_name_regex)
        sample_replicate = sample_replicate_from_path(path, args.sample_replicate_regex)
        dropout_label, dropout_fraction = dropout_from_path(path)
        print(f"\nReading {path}")
        adata = ad.read_h5ad(path)
        adata.obs[args.source_file_key] = str(path)
        adata.obs["_metrics_tissue"] = tissue
        adata.obs["_metrics_sample_replicate"] = sample_replicate
        adata.obs["_metrics_dropout_label"] = dropout_label
        adata.obs["_metrics_dropout_fraction"] = dropout_fraction
        adata.X = sanitize_numeric_matrix(adata.X)
        adata = sample_adata(adata, args)

        if parse_str_list(args.evaluation_layers):
            skip_zero_dropout_layers = (
                args.deduplicate_zero_dropout_inputs
                and "X" in parse_str_list(args.evaluation_layers)
                and sample_replicate in zero_dropout_replicates_seen
            )
            print(
                "Computing layered metrics: "
                f"{tissue} layers={parse_str_list(args.evaluation_layers)}"
                + (" (skipping duplicate X/0pct)" if skip_zero_dropout_layers else "")
            )
            (
                layered_cell_rows,
                layered_tcell_frames,
                layered_coefficient_frames,
                layered_profile_frames,
                layered_cosine_frames,
            ) = run_layered_evaluations_for_file(
                adata,
                tissue,
                str(path),
                sample_replicate,
                dropout_label,
                dropout_fraction,
                args,
                skip_zero_dropout_layers=skip_zero_dropout_layers,
                reference_adata=reference_adata,
                reference_profile_cache=reference_profile_cache,
            )
            if "X" in parse_str_list(args.evaluation_layers):
                zero_dropout_replicates_seen.add(sample_replicate)
            cell_type_rows.extend(layered_cell_rows)
            tcell_frames.extend(layered_tcell_frames)
            tcell_coefficient_frames.extend(layered_coefficient_frames)
            cell_type_profile_frames.extend(layered_profile_frames)
            tcell_cosine_frames.extend(layered_cosine_frames)
            continue

        if not args.skip_cell_type_separation and not args.only_syndri_reference_cosine:
            print(f"Computing cell-type separation metrics: {tissue}")
            cell_type_rows.extend(run_cell_type_separation(adata, tissue, str(path), args))

        if not args.skip_tcell_contamination and not args.only_syndri_reference_cosine:
            print(f"Computing T-cell contamination metrics: {tissue}")
            tcell_df, coefficient_df, profile_df, cosine_df = run_tcell_contamination(
                adata.copy(),
                tissue,
                str(path),
                args,
                reference_adata=reference_adata,
                reference_profile_cache=reference_profile_cache,
            )
            tcell_frames.append(tcell_df)
            tcell_coefficient_frames.append(coefficient_df)
            cell_type_profile_frames.append(profile_df)
            tcell_cosine_frames.append(cosine_df)

        if args.syndri_reference_cosine and not args.only_additional_counts:
            adaptive_rows, adaptive_tcell, adaptive_coefficients, adaptive_profiles, adaptive_cosines = (
                run_reference_cosine_evaluation(
                    adata, tissue, str(path), args, reference_adata=reference_adata,
                    reference_profile_cache=reference_profile_cache))
            cell_type_rows.extend(adaptive_rows)
            tcell_frames.append(adaptive_tcell)
            tcell_coefficient_frames.append(adaptive_coefficients)
            cell_type_profile_frames.append(adaptive_profiles)
            tcell_cosine_frames.append(adaptive_cosines)

    replace_labels = (
        ("syndri_reference_cosine",) if args.only_syndri_reference_cosine else ()
    )

    if not args.skip_cell_type_separation:
        cell_type_df = pd.DataFrame(cell_type_rows)
        write_csv(
            cell_type_df,
            output_dir / args.cell_type_output,
            append=args.append_output,
            replace_layer_count_baselines=args.replace_layer_count_baselines,
            uncorrected_condition_label=args.uncorrected_condition_label,
            replace_labels=replace_labels,
        )

    if not args.skip_tcell_contamination:
        tcell_df = (
            pd.concat(tcell_frames, ignore_index=True)
            if tcell_frames
            else pd.DataFrame()
        )
        tcell_df = write_csv(
            tcell_df,
            output_dir / args.tcell_output,
            append=args.append_output,
            replace_layer_count_baselines=args.replace_layer_count_baselines,
            uncorrected_condition_label=args.uncorrected_condition_label,
            replace_labels=replace_labels,
        )
        write_csv(summarize_tcell_effects(tcell_df, args), output_dir / args.tcell_effect_output)
        tcell_coefficient_df = (
            pd.concat(tcell_coefficient_frames, ignore_index=True)
            if tcell_coefficient_frames
            else pd.DataFrame()
        )
        write_csv(
            tcell_coefficient_df,
            output_dir / args.tcell_lr_gene_output,
            append=args.append_output,
            replace_layer_count_baselines=args.replace_layer_count_baselines,
            uncorrected_condition_label=args.uncorrected_condition_label,
            replace_labels=replace_labels,
        )
        cell_type_profile_df = (
            pd.concat(cell_type_profile_frames, ignore_index=True)
            if cell_type_profile_frames
            else pd.DataFrame()
        )
        write_csv(
            cell_type_profile_df,
            output_dir / args.cell_type_profile_output,
            append=args.append_output,
            replace_layer_count_baselines=args.replace_layer_count_baselines,
            uncorrected_condition_label=args.uncorrected_condition_label,
            replace_labels=replace_labels,
        )
        if not args.skip_tcell_cosine_similarity and reference_adata is not None:
            tcell_cosine_df = (
                pd.concat(tcell_cosine_frames, ignore_index=True)
                if tcell_cosine_frames
                else pd.DataFrame()
            )
            if args.append_output:
                tcell_cosine_df = remove_stale_tcell_cosine_errors(tcell_cosine_df)
                cosine_path = output_dir / args.tcell_cosine_output
                if cosine_path.exists():
                    existing_cosine = remove_stale_tcell_cosine_errors(pd.read_csv(cosine_path))
                    existing_cosine.to_csv(cosine_path, index=False)
            write_csv(
                tcell_cosine_df,
                output_dir / args.tcell_cosine_output,
                append=args.append_output,
                replace_layer_count_baselines=args.replace_layer_count_baselines,
                uncorrected_condition_label=args.uncorrected_condition_label,
                replace_labels=replace_labels,
            )


if __name__ == "__main__":
    main()
