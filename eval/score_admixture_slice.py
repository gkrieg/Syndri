#!/usr/bin/env python3
"""Calculate spatial admixture scores for one AnnData tissue slice.

This is the production form of notebooks/seg_cleaner/ScoreAdmixture.ipynb.
It estimates a cell-type adjacency matrix from the slice, derives expression
profiles from a single-cell reference, and scores raw plus Syndri-cleaned
spatial counts at requested transcript-removal fractions.

Version 2 uses count-weighted, background-adjusted cellAdmix scores and
log-normalized expression only for correlation filtering. Cell-centroid
adjacency is an intentional adaptation of the authors' molecule-based tutorial.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


MODULE_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "seg_cleaner"
sys.path.insert(0, str(MODULE_DIR))

from admix_scores import (  # noqa: E402
    estimate_cell_adjacency,
    estimate_cell_type_adjacency,
    estimate_correlation_preservation,
)
from sc_score_from_st import get_scrna_admixture_scores_from_spatial_K  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-h5ad", required=True, type=Path)
    parser.add_argument(
        "--reference-h5ad",
        type=Path,
        default=None,
        help=(
            "Single-cell reference. By default, infer it from "
            "uns['rctd_run_metadata[_X]']['reference_file']."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spatial-cell-type-key", required=True)
    parser.add_argument("--reference-cell-type-key", required=True)
    parser.add_argument("--reference-layer", default="X", help="Reference count matrix (not log expression).")
    parser.add_argument("--coord-x", required=True)
    parser.add_argument("--coord-y", required=True)
    parser.add_argument("--fov-key", default="")
    parser.add_argument("--probability-layer", default="syndri_admix_probs_X")
    parser.add_argument(
        "--split-layer",
        default="split_purified_counts_X",
        help="SPLIT purified-count layer to score; use an empty string to skip SPLIT.",
    )
    parser.add_argument(
        "--target-removed-fractions",
        default="0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5",
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.2)
    parser.add_argument("--skip-correlation-filter", action="store_true")
    parser.add_argument("--adjust", action=argparse.BooleanOptionalAction, default=True,
                        help="Subtract the reference-derived null score and clip at zero (default: enabled).")
    parser.add_argument("--min-expr-frac-specific", type=float, default=0.05)
    parser.add_argument("--min-expr-frac-contam", type=float, default=0.05)
    parser.add_argument("--signal-threshold", type=float, default=0.4)
    parser.add_argument("--max-prob-high", type=float, default=0.1)
    parser.add_argument("--contamination-prior", type=float, default=0.25)
    parser.add_argument("--edge-max-mad", type=float, default=4.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--syndri-reference-cosine", action="store_true")
    parser.add_argument(
        "--syndri-reference-profiles",
        default="/home/jwinchel/Syndri/data/cache/rctd_profiles",
    )
    parser.add_argument("--syndri-cell-type-key", default="")
    parser.add_argument("--syndri-one-se-multiplier", type=float, default=1.0)
    parser.add_argument("--syndri-profile-min-umi", type=int, default=10)
    parser.add_argument("--syndri-profile-max-umi", type=int, default=2000)
    parser.add_argument("--syndri-profile-cell-min", type=int, default=25)
    parser.add_argument("--syndri-profile-max-cells", type=int, default=10000)
    parser.add_argument("--syndri-profile-cell-type-col", default="cell_type")
    parser.add_argument(
        "--label-map-json",
        default="{}",
        help='Exact label replacements, e.g. \'{"pDC":"DC","mDC":"DC"}\'.',
    )
    parser.add_argument(
        "--collapse-prefix",
        action="append",
        default=[],
        help="Collapse labels beginning with PREFIX to LABEL, as PREFIX=LABEL.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["python", "celladmix"], default="python")
    parser.add_argument("--celladmix-repo", type=Path, default=Path("/home/jwinchel/cellAdmix"))
    parser.add_argument("--celladmix-notebooks-repo", type=Path, default=Path("/home/jwinchel/cellAdmix-Notebooks"))
    parser.add_argument("--rscript", default="/home/jwinchel/miniconda3/envs/r_env/bin/Rscript")
    return parser.parse_args()


def normalize_labels(values: pd.Series, args: argparse.Namespace) -> pd.Series:
    labels = values.astype("string")
    labels = labels.mask(labels.str.strip().str.lower().isin(["", "nan", "none", "<na>"]))
    replacements = json.loads(args.label_map_json)
    if not isinstance(replacements, dict):
        raise ValueError("--label-map-json must decode to an object.")
    labels = labels.replace(replacements)
    for specification in args.collapse_prefix:
        if "=" not in specification:
            raise ValueError("--collapse-prefix must use PREFIX=LABEL syntax.")
        prefix, replacement = specification.split("=", 1)
        mask = labels.str.startswith(prefix, na=False)
        labels.loc[mask] = replacement
    return labels


def infer_reference_path(spatial: ad.AnnData) -> Path:
    """Recover the RCTD reference path recorded in spatial `.uns`."""
    candidates = []
    for key, value in spatial.uns.items():
        if not str(key).startswith("rctd_run_metadata") or not isinstance(value, dict):
            continue
        reference = value.get("reference_file")
        if reference:
            candidates.append(Path(str(reference)).expanduser())

    if not candidates:
        # Older RCTD exports may only retain a list of source H5AD files.
        spatial_source = str(spatial.uns.get("split_purification", {}).get("source_h5ad", ""))
        for key, value in spatial.uns.items():
            if not str(key).startswith("source_h5_files"):
                continue
            for source in np.atleast_1d(value):
                path = Path(str(source)).expanduser()
                if path.suffix == ".h5ad" and str(path) != spatial_source:
                    candidates.append(path)

    existing = []
    for path in candidates:
        resolved = path.resolve()
        if resolved.exists() and resolved not in existing:
            existing.append(resolved)
    if len(existing) == 1:
        return existing[0]
    if not existing:
        raise ValueError(
            "No existing RCTD reference file could be inferred from spatial .uns; "
            "pass --reference-h5ad explicitly."
        )
    raise ValueError(
        f"Multiple possible RCTD references were found in spatial .uns: {existing}"
    )


def expressed_counts_and_probabilities(X, probabilities):
    if sparse.issparse(X):
        coo = X.tocoo()
        counts = coo.data.astype(float)
        probs = np.asarray(probabilities[coo.row, coo.col]).ravel().astype(float)
    else:
        counts_array = np.asarray(X)
        probabilities = (
            probabilities.toarray() if sparse.issparse(probabilities)
            else np.asarray(probabilities)
        )
        nonzero = counts_array > 0
        counts = counts_array[nonzero].astype(float)
        probs = probabilities[nonzero].astype(float)
    valid = np.isfinite(counts) & np.isfinite(probs) & (counts > 0)
    return counts[valid], probs[valid]


def threshold_for_removed_fraction(X, probabilities, fraction: float):
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction == 0:
        return np.inf, 0.0
    counts, probs = expressed_counts_and_probabilities(X, probabilities)
    if counts.size == 0:
        raise ValueError("No positive, finite transcript counts were found.")
    order = np.argsort(probs)[::-1]
    sorted_probs = probs[order]
    sorted_counts = counts[order]
    index = min(
        int(np.searchsorted(np.cumsum(sorted_counts), fraction * sorted_counts.sum())),
        len(sorted_probs) - 1,
    )
    threshold = float(sorted_probs[index])
    achieved = float(sorted_counts[sorted_probs >= threshold].sum() / sorted_counts.sum())
    return threshold, achieved


def zero_counts_at_probability(X, probabilities, threshold: float):
    if sparse.issparse(X):
        output = X.tocsr(copy=True)
        row = np.repeat(np.arange(output.shape[0]), np.diff(output.indptr))
        probs = np.asarray(probabilities[row, output.indices]).ravel()
        output.data[probs >= threshold] = 0
        output.eliminate_zeros()
        return output
    probabilities = probabilities.toarray() if sparse.issparse(probabilities) else probabilities
    return np.where(np.asarray(probabilities) >= threshold, 0, np.asarray(X))


def build_adjacency(spatial: ad.AnnData, args: argparse.Namespace) -> pd.DataFrame:
    required = [args.spatial_cell_type_key, args.coord_x, args.coord_y]
    if args.fov_key:
        required.append(args.fov_key)
    missing = [key for key in required if key not in spatial.obs]
    if missing:
        raise KeyError(f"Spatial obs is missing: {missing}")

    cells = pd.DataFrame({
        "x": pd.to_numeric(spatial.obs[args.coord_x], errors="coerce"),
        "y": pd.to_numeric(spatial.obs[args.coord_y], errors="coerce"),
        "cell_type": normalize_labels(spatial.obs[args.spatial_cell_type_key], args),
        "cell": spatial.obs_names.astype(str),
        "fov": spatial.obs[args.fov_key].to_numpy() if args.fov_key else "slice",
    }).dropna()

    edges = []
    for _, group in cells.groupby("fov", sort=True, observed=True):
        if group["cell"].nunique() < 4:
            continue
        result = estimate_cell_adjacency(
            group[["x", "y", "cell_type", "cell"]],
            random_shift=1e-3,
            edge_max_mad=args.edge_max_mad,
            n_jobs=args.n_jobs,
            random_state=args.random_seed,
        )
        if result is not None and not result.empty:
            edges.append(result)
    if not edges:
        raise RuntimeError("No spatial adjacency edges could be calculated.")

    adjacency = estimate_cell_type_adjacency(pd.concat(edges, ignore_index=True))
    types = sorted(cells["cell_type"].dropna().unique())
    return adjacency.reindex(index=types, columns=types, fill_value=0.0)


def align_and_prepare(
    spatial: ad.AnnData,
    reference: ad.AnnData,
    args: argparse.Namespace,
) -> tuple[ad.AnnData, ad.AnnData]:
    if args.probability_layer not in spatial.layers:
        raise KeyError(f"Spatial layer {args.probability_layer!r} is missing.")
    if args.reference_cell_type_key not in reference.obs:
        raise KeyError(f"Reference obs key {args.reference_cell_type_key!r} is missing.")
    if not spatial.obs_names.is_unique:
        raise ValueError("Spatial cell IDs must be unique for per-cell scoring.")
    if args.reference_layer != "X":
        reference.X = reference.layers[args.reference_layer].copy()

    if not spatial.var_names.is_unique or not reference.var_names.is_unique:
        raise ValueError("Spatial and reference gene names must be unique.")
    common = spatial.var_names[spatial.var_names.isin(reference.var_names)]
    if common.empty:
        raise ValueError("Spatial and reference data have no shared genes.")
    # The tutorial normalizes before intersecting panels. Preserve full-panel
    # cell totals so reference-only genes still contribute to library sizes.
    spatial_totals = np.asarray(spatial.X.sum(axis=1)).ravel()
    reference_totals = np.asarray(reference.X.sum(axis=1)).ravel()
    spatial = spatial[:, common].copy()
    reference = reference[:, common].copy()
    spatial.obs['_admixture_library_size'] = spatial_totals
    reference.obs['_admixture_library_size'] = reference_totals

    spatial.obs[args.spatial_cell_type_key] = normalize_labels(
        spatial.obs[args.spatial_cell_type_key], args
    )
    reference.obs[args.reference_cell_type_key] = normalize_labels(
        reference.obs[args.reference_cell_type_key], args
    )
    reference = reference[reference.obs[args.reference_cell_type_key].notna()].copy()
    if reference.n_obs == 0:
        raise ValueError("No labeled reference cells remain.")
    # Like the authors' tutorial, build K using only represented cell types.
    supported = spatial.obs[args.spatial_cell_type_key].isin(
        reference.obs[args.reference_cell_type_key].unique()
    )
    dropped = spatial.obs.loc[~supported, args.spatial_cell_type_key].value_counts(dropna=False)
    if len(dropped):
        print(f"Excluding spatial cells without a matching reference label: {dropped.to_dict()}")
    spatial = spatial[supported].copy()
    if spatial.n_obs == 0:
        raise ValueError("No spatial cells match the reference labels; check the label keys/map.")
    for name, matrix in [("spatial X", spatial.X), ("reference counts", reference.X)]:
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} must contain finite, nonnegative counts.")
    return spatial, reference


def log_normalize_counts(X, totals=None):
    """Seurat-style LogNormalize: log1p(count / cell total * 10,000)."""
    matrix = sparse.csr_matrix(X, dtype=np.float64, copy=True)
    totals = (np.asarray(matrix.sum(axis=1)).ravel() if totals is None
              else np.asarray(totals, dtype=float))
    scales = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    matrix = matrix.multiply(scales[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def correlation_exclusions(spatial, reference, args):
    if args.skip_correlation_filter:
        return [], pd.DataFrame(columns=["gene", "correlation", "excluded"])
    spatial_genes_by_cells = log_normalize_counts(
        spatial.X, spatial.obs.get('_admixture_library_size')
    ).T.tocsr()
    reference_genes_by_cells = log_normalize_counts(
        reference.X, reference.obs.get('_admixture_library_size')
    ).T.tocsr()
    expressed = (
        (np.asarray(spatial_genes_by_cells.sum(axis=1)).ravel() > 0)
        & (np.asarray(reference_genes_by_cells.sum(axis=1)).ravel() > 0)
    )
    retained_genes = spatial.var_names[expressed].tolist()
    correlations = estimate_correlation_preservation(
        spatial_genes_by_cells[expressed],
        reference_genes_by_cells[expressed],
        gene_names=retained_genes,
    )
    table = correlations.reindex(spatial.var_names).rename_axis("gene").reset_index(name="correlation")
    table["excluded"] = table["correlation"].lt(args.correlation_threshold) | table["correlation"].isna()
    table["reason"] = np.select(
        [~expressed, table["correlation"].isna(), table["correlation"].lt(args.correlation_threshold)],
        ["not_expressed_in_both", "undefined_correlation", "below_threshold"],
        default="retained",
    )
    return table.loc[table["excluded"], "gene"].tolist(), table


def score_counts(reference, spatial, adjacency, excluded_genes, args, X, setting, target, achieved, threshold):
    work = ad.AnnData(X=X, obs=spatial.obs.copy(), var=spatial.var.copy())
    result, gene_sets = get_scrna_admixture_scores_from_spatial_K(
        reference,
        work,
        adjacency,
        cell_type_key=args.reference_cell_type_key,
        spatial_cell_type_key=args.spatial_cell_type_key,
        adjust=args.adjust,
        return_gene_sets=True,
        min_expr_frac_specific=args.min_expr_frac_specific,
        min_expr_frac_contam=args.min_expr_frac_contam,
        signal_thres=args.signal_threshold,
        max_prob_high=args.max_prob_high,
        p_c=args.contamination_prior,
        exclude_genes=excluded_genes,
    )
    result["setting"] = setting
    result["removed_frac_target"] = target
    result["removed_frac"] = achieved
    result["removed_pct"] = achieved * 100
    result["threshold"] = threshold
    totals = pd.Series(np.asarray(X.sum(axis=1)).ravel(), index=spatial.obs_names)
    result["total_counts"] = result["cell"].map(totals)
    result["score_status"] = np.where(result["total_counts"] > 0, "ok", "zero_counts")
    result["score_version"] = "2_count_weighted"
    return result, gene_sets


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.syndri_one_se_multiplier) or args.syndri_one_se_multiplier < 0:
        raise ValueError("--syndri-one-se-multiplier must be finite and nonnegative")
    if args.backend == "celladmix":
        from score_admixture_celladmix import run
        run(args)
        return
    output = args.output_dir.expanduser().resolve()
    score_path = output / "admixture_scores.csv"
    if score_path.exists() and not args.overwrite:
        raise FileExistsError(f"{score_path} exists; pass --overwrite to replace it.")
    output.mkdir(parents=True, exist_ok=True)

    spatial = ad.read_h5ad(args.spatial_h5ad.expanduser().resolve())
    reference_path = (
        args.reference_h5ad.expanduser().resolve()
        if args.reference_h5ad is not None
        else infer_reference_path(spatial)
    )
    print(f"Using reference: {reference_path}")
    if args.syndri_reference_cosine:
        from syndri_reference_cleaning import prepare_reference_cleaned

        # Adapter compatibility: profile-cache matching uses the original RCTD
        # reference recorded in spatial.uns, independently of the p_admix
        # reference used below for scoring.
        args.reference_data = ""
        args.label_key = args.spatial_cell_type_key
        args.admix_probs_layer = args.probability_layer
        adaptive_counts, adaptive_achieved, adaptive_profile_path = prepare_reference_cleaned(
            spatial, args, str(args.spatial_h5ad.expanduser().resolve()), input_layer="X"
        )
        spatial.layers["_syndri_reference_cosine_counts"] = adaptive_counts

    reference = ad.read_h5ad(reference_path)
    spatial, reference = align_and_prepare(spatial, reference, args)
    if args.syndri_reference_cosine:
        original_total = float(spatial.X.sum())
        adaptive_total = float(spatial.layers["_syndri_reference_cosine_counts"].sum())
        adaptive_achieved = (
            (original_total - adaptive_total) / original_total if original_total else 0.0
        )
    adjacency = build_adjacency(spatial, args)
    excluded_genes, correlations = correlation_exclusions(spatial, reference, args)
    probabilities = spatial.layers[args.probability_layer]

    frames = []
    gene_sets = {}
    baseline, baseline_sets = score_counts(
        reference, spatial, adjacency, excluded_genes, args, spatial.X,
        "raw", 0.0, 0.0, np.nan,
    )
    frames.append(baseline)
    gene_sets["raw"] = baseline_sets

    if args.split_layer:
        if args.split_layer not in spatial.layers:
            raise KeyError(f"Spatial SPLIT layer {args.split_layer!r} is missing.")
        split_scores, split_sets = score_counts(
            reference,
            spatial,
            adjacency,
            excluded_genes,
            args,
            spatial.layers[args.split_layer],
            "SPLIT",
            np.nan,
            np.nan,
            np.nan,
        )
        frames.append(split_scores)
        gene_sets["SPLIT"] = split_sets

    if args.syndri_reference_cosine:
        adaptive_scores, adaptive_sets = score_counts(
            reference,
            spatial,
            adjacency,
            excluded_genes,
            args,
            spatial.layers["_syndri_reference_cosine_counts"],
            "syndri_reference_cosine",
            np.nan,
            adaptive_achieved,
            np.nan,
        )
        adaptive_scores["threshold_selection"] = "cell_type_smallest_near_optimal_k"
        adaptive_scores["syndri_probability_ordering"] = "cell_type_expressed_mean"
        adaptive_scores["syndri_one_se_multiplier"] = args.syndri_one_se_multiplier
        adaptive_scores["reference_profiles"] = adaptive_profile_path
        frames.append(adaptive_scores)
        gene_sets["syndri_reference_cosine"] = adaptive_sets

    for fraction in parse_csv(args.target_removed_fractions):
        target = float(fraction)
        threshold, achieved = threshold_for_removed_fraction(spatial.X, probabilities, target)
        cleaned = zero_counts_at_probability(spatial.X, probabilities, threshold)
        setting = f"syndri_remove_{target:g}"
        scored, sets = score_counts(
            reference, spatial, adjacency, excluded_genes, args, cleaned,
            setting, target, achieved, threshold,
        )
        frames.append(scored)
        gene_sets[setting] = sets

    scores = pd.concat(frames, ignore_index=True)
    scores.insert(0, "slice", args.spatial_h5ad.stem)
    scores.insert(0, "source_file", str(args.spatial_h5ad.expanduser().resolve()))
    scores.to_csv(score_path, index=False)
    adjacency.to_csv(output / "cell_type_adjacency.csv")
    correlations.to_csv(output / "gene_correlation_preservation.csv", index=False)
    with (output / "admixture_gene_sets.pkl").open("wb") as handle:
        pickle.dump(gene_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    saved_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    saved_config["resolved_reference_h5ad"] = str(reference_path)
    saved_config["adjacency_source"] = "spatial_cell_coordinates"
    saved_config["score_version"] = "2_count_weighted"
    saved_config["score_weighting"] = "counts"
    saved_config["correlation_preprocessing"] = "log1p(counts/full_panel_cell_total*10000), then shared genes"
    saved_config["zero_count_policy"] = "NaN"
    saved_config["n_shared_genes"] = spatial.n_vars
    saved_config["n_excluded_genes"] = len(excluded_genes)
    saved_config["n_scored_cells"] = spatial.n_obs
    (output / "admixture_score_config.json").write_text(
        json.dumps(saved_config, indent=2) + "\n"
    )
    print(f"Wrote {len(scores):,} score rows to {score_path}")
    print(f"Excluded {len(excluded_genes):,} genes below correlation threshold.")


if __name__ == "__main__":
    main()
