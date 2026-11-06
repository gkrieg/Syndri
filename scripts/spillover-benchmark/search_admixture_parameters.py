#!/usr/bin/env python
"""Grid search admixture parameters against matched spatial CosMx counts."""

from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import cdist, pdist


REPO_DIR = Path("/home/jwinchel/Syndri")
TRAINING_PATH = Path(
    "/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/"
    "spillover-benchmark/train_val.h5ad"
)
SPATIAL_DIR = Path(
    "/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/results/final"
)
DEFAULT_OUTPUT_DIR = Path(
    "/work/magroup/skrieger/Syndri/data/datasets/nsclc_cosmx/"
    "spillover-benchmark/parameter-search"
)

K_VALUES = (3, 4, 5, 6, 7, 8, 9)
TAU_MIN_VALUES = (0.0,)
TAU_MAX_VALUES = (0.65, 0.75, 0.85, 0.95)
SPATIAL_COLS = ("CenterX_global_px", "CenterY_global_px")
CELL_TYPE_COL = "cell_type_harmonized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=TRAINING_PATH)
    parser.add_argument("--spatial-dir", type=Path, default=SPATIAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-query-per-cell-type",
        type=int,
        default=200,
        help="Fixed, reproducible query sample cap for each cell type and slice.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds used for each parameter configuration.",
    )
    parser.add_argument("--min-source-counts", type=int, default=400)
    parser.add_argument(
        "--min-sampled-transcript-threshold", type=int, default=10
    )
    return parser.parse_args()


def normalized_log_counts(matrix, target_sum: float = 10_000.0) -> np.ndarray:
    """Return dense log1p library-size-normalized expression vectors."""
    x = csr_matrix(matrix, dtype=np.float64)
    row_sums = np.asarray(x.sum(axis=1)).ravel()
    scale = np.divide(
        target_sum,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float64),
        where=row_sums > 0,
    )
    x = diags(scale) @ x
    x.data = np.log1p(x.data)
    return x.toarray().astype(np.float32, copy=False)


def rbf_bandwidth_squared(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.vstack([x, y]).astype(np.float64, copy=False)
    distances = pdist(pooled, metric="sqeuclidean")
    positive = distances[distances > 0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def rbf_mmd_squared(x: np.ndarray, y: np.ndarray, bandwidth_squared: float) -> float:
    denominator = 2.0 * bandwidth_squared
    k_xx = np.exp(-cdist(x, x, metric="sqeuclidean") / denominator)
    k_yy = np.exp(-cdist(y, y, metric="sqeuclidean") / denominator)
    k_xy = np.exp(-cdist(x, y, metric="sqeuclidean") / denominator)
    # Biased MMD^2 estimator is stable for the small, fixed per-class samples.
    value = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(max(0.0, value))


def load_slice_pair(
    training, spatial_dir: Path, slice_name: str, max_query_per_cell_type: int
):
    mapped = training[
        training.obs["spatial_slice"].astype(str) == slice_name
    ].copy()
    if mapped.n_obs == 0:
        raise ValueError(f"No training pseudo-spatial cells for {slice_name}")

    if CELL_TYPE_COL not in mapped.obs:
        raise KeyError(f"Training data is missing obs[{CELL_TYPE_COL!r}]")
    mapped = mapped[mapped.obs[CELL_TYPE_COL].astype(str) != "Treg"].copy()

    spatial_path = spatial_dir / f"st_integrated_slice-{slice_name}.h5ad"
    if not spatial_path.is_file():
        raise FileNotFoundError(f"Missing spatial reference slice: {spatial_path}")
    spatial = ad.read_h5ad(spatial_path)
    spatial = spatial[spatial.obs[CELL_TYPE_COL].astype(str) != "Treg"].copy()

    prefix = f"{slice_name}:"
    mapped_ids = mapped.obs_names.astype(str).tolist()
    if not all(cell_id.startswith(prefix) for cell_id in mapped_ids):
        raise ValueError(
            f"Expected mapped obs_names prefixed with {prefix!r} for {slice_name}"
        )
    spatial_ids = [cell_id[len(prefix) :] for cell_id in mapped_ids]
    spatial_index = pd.Index(spatial.obs_names.astype(str))
    spatial_row_indices = spatial_index.get_indexer(spatial_ids)
    if np.any(spatial_row_indices < 0):
        missing = [
            spatial_ids[i]
            for i in np.flatnonzero(spatial_row_indices < 0)[:10]
        ]
        raise ValueError(
            f"Mapped cells could not be matched to {slice_name} spatial rows: {missing}"
        )
    spatial = spatial[spatial_row_indices].copy()

    mapped_labels = mapped.obs[CELL_TYPE_COL].astype(str).to_numpy()
    spatial_labels = spatial.obs[CELL_TYPE_COL].astype(str).to_numpy()
    if not np.array_equal(mapped_labels, spatial_labels):
        mismatches = int(np.count_nonzero(mapped_labels != spatial_labels))
        raise ValueError(
            f"{mismatches} mapped/spatial labels differ in {slice_name}; "
            "the paired cells must have matching annotations for this benchmark."
        )

    spatial_gene_set = set(spatial.var_names.astype(str))
    shared_genes = [
        gene
        for gene in mapped.var_names.astype(str)
        if gene in spatial_gene_set
    ]
    if not shared_genes:
        raise ValueError(f"No shared genes between mapped data and {slice_name}")
    mapped = mapped[:, shared_genes].copy()
    spatial = spatial[:, shared_genes].copy()

    # Use the same sampled query cells for every K, tau range, and seed.
    rng = np.random.default_rng(91827)
    query_ids = []
    for cell_type in sorted(np.unique(mapped_labels)):
        ids = mapped.obs_names[mapped_labels == cell_type].astype(str).to_numpy()
        if len(ids) < 2:
            continue
        n_take = min(len(ids), max_query_per_cell_type)
        query_ids.extend(rng.choice(ids, size=n_take, replace=False).tolist())
    if not query_ids:
        raise ValueError(f"No cell types have enough query cells in {slice_name}")

    query = mapped[query_ids].copy()
    query_positions = mapped.obs_names.astype(str).get_indexer(query_ids)
    target = spatial[query_positions].copy()
    query_labels = query.obs[CELL_TYPE_COL].astype(str).to_numpy()
    baseline_expression = normalized_log_counts(query.X)
    target_expression = normalized_log_counts(target.X)

    metric_by_cell_type = {}
    for cell_type in sorted(np.unique(query_labels)):
        mask = query_labels == cell_type
        x_base = baseline_expression[mask]
        y_real = target_expression[mask]
        metric_by_cell_type[cell_type] = {
            "mask": mask,
            "x_base": x_base,
            "y_real": y_real,
            "bandwidth_squared": rbf_bandwidth_squared(x_base, y_real),
            "n_query": int(mask.sum()),
        }

    return mapped, query_ids, query_labels, metric_by_cell_type


def main() -> None:
    args = parse_args()
    if args.max_query_per_cell_type < 2:
        raise ValueError("--max-query-per-cell-type must be at least 2")
    if not args.seeds:
        raise ValueError("At least one random seed is required")

    heimdall_root = REPO_DIR / "Heimdall"
    sys.path.insert(0, str(heimdall_root))
    from Heimdall.admixture_sampling import (
        build_neighbor_dict,
        sample_admixture_from_neighbor_dict,
    )

    training = ad.read_h5ad(args.training)
    required_obs = {"split", "spatial_slice", CELL_TYPE_COL}
    missing_obs = required_obs - set(training.obs.columns)
    if missing_obs:
        raise KeyError(f"Training AnnData is missing columns: {sorted(missing_obs)}")
    training = training[training.obs["split"].astype(str) == "train"].copy()
    training = training[
        training.obs[CELL_TYPE_COL].astype(str) != "Treg"
    ].copy()
    if training.n_obs == 0:
        raise ValueError("No training observations found in the input AnnData")

    slice_names = sorted(training.obs["spatial_slice"].astype(str).unique())
    slice_data = {}
    for slice_name in slice_names:
        mapped, query_ids, query_labels, metric_by_cell_type = load_slice_pair(
            training,
            args.spatial_dir,
            slice_name,
            args.max_query_per_cell_type,
        )
        slice_data[slice_name] = {
            "mapped": mapped,
            "query_ids": query_ids,
            "query_labels": query_labels,
            "metric_by_cell_type": metric_by_cell_type,
        }
        print(
            f"Loaded {slice_name}: {len(query_ids)} queries, "
            f"{len(metric_by_cell_type)} cell types",
            flush=True,
        )
        gc.collect()

    run_dir = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    replicate_path = run_dir / "parameter_search_replicates.csv"
    cell_burden_path = run_dir / "cell_admixture_burdens.csv.gz"
    summary_path = run_dir / "parameter_search_summary.csv"
    print(f"Training input: {args.training}", flush=True)
    print(f"Training slices: {', '.join(slice_names)}", flush=True)
    print(f"Output: {run_dir}", flush=True)

    # Hold only one K's neighbor dictionaries at a time.
    for k_neighbors in K_VALUES:
        neighbor_dicts = {}
        for slice_name, data in slice_data.items():
            neighbor_adata = build_neighbor_dict(
                data["mapped"],
                k_neighbors=k_neighbors,
                coord_cols=SPATIAL_COLS,
                tissue_col=None,
                cell_type_col=CELL_TYPE_COL,
                min_source_counts=args.min_source_counts,
            )
            neighbor_dicts[slice_name] = neighbor_adata.uns["neighbor_dict"]
            del neighbor_adata
            gc.collect()

        for tau_min in TAU_MIN_VALUES:
            for tau_max in TAU_MAX_VALUES:
                if tau_min > tau_max:
                    continue
                for seed in args.seeds:
                    rows = []
                    for slice_name, data in slice_data.items():
                        full_neighbors = neighbor_dicts[slice_name]
                        query_ids = data["query_ids"]
                        query_neighbors = {
                            query_id: full_neighbors.get(query_id, [])
                            for query_id in query_ids
                        }
                        neighbor_ids = list(
                            dict.fromkeys(
                                neighbor_id
                                for ids in query_neighbors.values()
                                for neighbor_id in ids
                            )
                        )
                        selected_ids = list(dict.fromkeys(query_ids + neighbor_ids))
                        sampling_data = data["mapped"][selected_ids].copy()
                        sampling_data.uns["neighbor_dict"] = query_neighbors

                        x_admixed, _ = sample_admixture_from_neighbor_dict(
                            sampling_data,
                            tau_range=(tau_min, tau_max),
                            n_repeats=1,
                            min_sampled_transcript_threshold=(
                                args.min_sampled_transcript_threshold
                            ),
                            only_sample_unexpressed_genes=True,
                            rng=np.random.default_rng(seed),
                            return_sparse=True,
                        )
                        n_query = len(query_ids)
                        synthetic_counts = (
                            csr_matrix(sampling_data.X[:n_query])
                            + x_admixed[:n_query]
                        )
                        synthetic_expression = normalized_log_counts(synthetic_counts)
                        base_totals = np.asarray(
                            csr_matrix(sampling_data.X[:n_query]).sum(axis=1)
                        ).ravel()
                        added_totals = np.asarray(
                            x_admixed[:n_query].sum(axis=1)
                        ).ravel()
                        achieved_burden = np.divide(
                            added_totals,
                            base_totals + added_totals,
                            out=np.zeros_like(added_totals, dtype=float),
                            where=(base_totals + added_totals) > 0,
                        )
                        labels = sampling_data.obs[CELL_TYPE_COL].astype(str).to_numpy()[:n_query]
                        burden_rows = pd.DataFrame(
                            {
                                "cell_id": query_ids,
                                "slice": slice_name,
                                "cell_type": labels,
                                "K": k_neighbors,
                                "tau_min": tau_min,
                                "tau_max": tau_max,
                                "seed": seed,
                                "source_counts": base_totals,
                                "admixture_counts": added_totals,
                                "achieved_admixture_burden": achieved_burden,
                            }
                        )
                        burden_rows.to_csv(
                            cell_burden_path,
                            mode="a",
                            compression="gzip",
                            header=not cell_burden_path.exists(),
                            index=False,
                        )

                        for cell_type, metric in data["metric_by_cell_type"].items():
                            mask = labels == cell_type
                            synthetic_ct = synthetic_expression[mask]
                            mmd2 = rbf_mmd_squared(
                                synthetic_ct,
                                metric["y_real"],
                                metric["bandwidth_squared"],
                            )
                            rows.append(
                                {
                                    "slice": slice_name,
                                    "cell_type": cell_type,
                                    "K": k_neighbors,
                                    "tau_min": tau_min,
                                    "tau_max": tau_max,
                                    "seed": seed,
                                    "n_query": int(mask.sum()),
                                    "mmd2": mmd2,
                                    "mean_achieved_admixture_burden": float(
                                        achieved_burden[mask].mean()
                                    ),
                                }
                            )

                        del sampling_data, x_admixed, synthetic_counts, synthetic_expression
                        gc.collect()

                    pd.DataFrame(rows).to_csv(
                        replicate_path,
                        mode="a",
                        header=not replicate_path.exists(),
                        index=False,
                    )
                    print(
                        f"Finished K={k_neighbors}, tau=({tau_min:.3f}, "
                        f"{tau_max:.3f}), seed={seed}",
                        flush=True,
                    )

        del neighbor_dicts
        gc.collect()

    replicates = pd.read_csv(replicate_path)
    seed_scores = (
        replicates.groupby(["K", "tau_min", "tau_max", "seed"], as_index=False)
        .agg(
            seed_mean_mmd2=("mmd2", "mean"),
            seed_mean_achieved_burden=(
                "mean_achieved_admixture_burden", "mean"
            ),
            n_stratum_evaluations=("mmd2", "size"),
        )
    )
    summary = (
        seed_scores.groupby(["K", "tau_min", "tau_max"], as_index=False)
        .agg(
            mean_mmd2=("seed_mean_mmd2", "mean"),
            sd_mmd2_across_seeds=("seed_mean_mmd2", "std"),
            mean_achieved_admixture_burden=(
                "seed_mean_achieved_burden", "mean"
            ),
            n_seeds=("seed", "nunique"),
            n_stratum_seed_evaluations=("n_stratum_evaluations", "sum"),
        )
        .sort_values("mean_mmd2", ascending=True)
    )
    summary.to_csv(summary_path, index=False)
    print("Best parameter configurations:", flush=True)
    print(summary.head(10).to_string(index=False), flush=True)
    print(f"Replicates: {replicate_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
