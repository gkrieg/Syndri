#!/usr/bin/env python
"""Prepare tissue-specific AnnData files for a context responsiveness experiment.

This samples query cells and diverse observed neighborhood contexts from cells
of the requested type, then creates their Cartesian product. Expression always
comes from the query cell; only its neighborhood cell-type context changes.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Heimdall.admixture_sampling import build_neighbor_dict, build_neighbor_celltype_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Tissue-specific input .h5ad files.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cell-type", required=True, help="Exact cell-type value to use as focal cells.")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--coord-cols", default="CenterX_global_px,CenterY_global_px")
    parser.add_argument("--tissue-col", default=None)
    parser.add_argument("--k-neighbors", type=int, default=3)
    parser.add_argument("--min-source-counts", type=int, default=400)
    parser.add_argument(
        "--use-same-cell-type-neighbors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--split-value", default="val")
    parser.add_argument("--suffix", default="responsiveness_contexts")
    parser.add_argument("--n-contexts", type=int, default=None)
    parser.add_argument("--n-query-cells", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--include-original-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ensure every sampled query has its own original neighborhood as a baseline.",
    )
    parser.add_argument(
        "--max-output-rows",
        type=int,
        default=5_000_000,
        help="Safety limit for the quadratic output size; use 0 to disable.",
    )
    return parser.parse_args()


def _cell_type_dict(adata: ad.AnnData, cell_type_col: str) -> dict[str, int]:
    existing = adata.uns.get("cell_type_dict")
    if isinstance(existing, dict) and existing:
        return dict(existing)
    cell_types = sorted(adata.obs[cell_type_col].astype(str).unique())
    return {cell_type: -(idx + 1) for idx, cell_type in enumerate(cell_types)}


def _farthest_point_sample(vectors: np.ndarray, n_select: int, rng) -> np.ndarray:
    """Select diverse observed contexts using farthest-point sampling."""
    n_rows = len(vectors)
    if n_select >= n_rows:
        return np.arange(n_rows, dtype=int)
    selected = np.empty(n_select, dtype=int)
    selected[0] = int(rng.integers(n_rows))
    available = np.ones(n_rows, dtype=bool)
    available[selected[0]] = False
    min_distance = np.sum((vectors - vectors[selected[0]]) ** 2, axis=1)

    for out_idx in range(1, n_select):
        candidate_indices = np.flatnonzero(available)
        candidate_distances = min_distance[candidate_indices]
        max_distance = candidate_distances.max()
        tied = candidate_indices[np.isclose(candidate_distances, max_distance)]
        chosen = int(rng.choice(tied))
        selected[out_idx] = chosen
        available[chosen] = False
        distance = np.sum((vectors - vectors[chosen]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return selected


def prepare_tissue(
    input_path: Path,
    output_path: Path,
    *,
    cell_type: str,
    cell_type_col: str,
    coord_cols: tuple[str, str],
    tissue_col: str | None,
    k_neighbors: int,
    min_source_counts: int,
    use_same_cell_type_neighbors: bool,
    split_col: str,
    split_value: str,
    n_contexts: int | None = None,
    n_query_cells: int | None = None,
    n_repeats: int = 1,
    random_seed: int = 0,
    include_original_context: bool = True,
    max_output_rows: int = 5_000_000,
) -> ad.AnnData:
    adata = ad.read_h5ad(input_path)
    if not adata.obs_names.is_unique:
        raise ValueError(f"obs_names must be unique: {input_path}")
    if cell_type_col not in adata.obs:
        raise KeyError(f"{cell_type_col!r} not found in {input_path}")

    adata.uns["cell_type_dict"] = _cell_type_dict(adata, cell_type_col)
    with_neighbors = build_neighbor_dict(
        adata,
        k_neighbors=k_neighbors,
        coord_cols=coord_cols,
        tissue_col=tissue_col,
        cell_type_col=cell_type_col,
        min_source_counts=min_source_counts,
        use_same_cell_type_neighbors=use_same_cell_type_neighbors,
    )
    with_neighbors = build_neighbor_celltype_dict(
        with_neighbors,
        tissue_col=tissue_col,
        cell_type_col=cell_type_col,
    )

    target_mask = with_neighbors.obs[cell_type_col].astype(str).eq(cell_type).to_numpy()
    target_positions = np.flatnonzero(target_mask)
    if len(target_positions) == 0:
        raise ValueError(f"No cells with {cell_type_col}={cell_type!r} in {input_path}")

    target_ids = with_neighbors.obs_names[target_positions].astype(str).to_numpy()
    n_target = len(target_positions)
    n_contexts = n_target if n_contexts is None else min(n_contexts, n_target)
    n_query_cells = n_target if n_query_cells is None else min(n_query_cells, n_target)
    if n_contexts <= 0 or n_query_cells <= 0 or n_repeats <= 0:
        raise ValueError("n_contexts, n_query_cells, and n_repeats must be positive")

    source_contexts = with_neighbors.uns["neighbor_celltype_dict"]
    composition_cell_types = sorted(with_neighbors.obs[cell_type_col].astype(str).unique())
    cell_type_to_col = {name: idx for idx, name in enumerate(composition_cell_types)}
    target_context_counts = np.zeros((n_target, len(composition_cell_types)), dtype=np.int16)
    for row, source_id in enumerate(target_ids):
        for neighbor_type in source_contexts[source_id]:
            target_context_counts[row, cell_type_to_col[str(neighbor_type)]] += 1
    denominators = target_context_counts.sum(axis=1, keepdims=True)
    target_context_proportions = np.divide(
        target_context_counts,
        denominators,
        out=np.zeros_like(target_context_counts, dtype=np.float32),
        where=denominators != 0,
    )

    focal_local_blocks = []
    context_local_blocks = []
    repeat_blocks = []
    context_number_blocks = []
    representative_blocks = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(random_seed + repeat)
        if n_query_cells == n_target:
            query_local = np.arange(n_target, dtype=int)
        else:
            query_local = np.sort(rng.choice(n_target, size=n_query_cells, replace=False))
        context_local = _farthest_point_sample(target_context_proportions, n_contexts, rng)

        focal_local_blocks.append(np.repeat(query_local, len(context_local)))
        context_local_blocks.append(np.tile(context_local, len(query_local)))
        repeat_blocks.append(np.full(len(query_local) * len(context_local), repeat, dtype=int))
        context_number_blocks.append(np.tile(np.arange(len(context_local), dtype=int), len(query_local)))
        representative_blocks.append(np.ones(len(query_local) * len(context_local), dtype=bool))

        if include_original_context:
            missing_original = query_local[~np.isin(query_local, context_local)]
            if len(missing_original):
                focal_local_blocks.append(missing_original)
                context_local_blocks.append(missing_original)
                repeat_blocks.append(np.full(len(missing_original), repeat, dtype=int))
                context_number_blocks.append(np.full(len(missing_original), -1, dtype=int))
                representative_blocks.append(np.zeros(len(missing_original), dtype=bool))

    focal_local = np.concatenate(focal_local_blocks)
    context_local = np.concatenate(context_local_blocks)
    sampling_repeat = np.concatenate(repeat_blocks)
    context_number = np.concatenate(context_number_blocks)
    is_representative_context = np.concatenate(representative_blocks)
    n_rows = len(focal_local)
    if max_output_rows > 0 and n_rows > max_output_rows:
        raise ValueError(
            f"The sampled design would create {n_rows:,} rows, exceeding "
            f"--max-output-rows={max_output_rows:,}. Increase the limit or pass 0 to disable it.",
        )
    focal_positions = target_positions[focal_local]
    focal_ids = target_ids[focal_local]
    context_source_ids = target_ids[context_local]
    is_original = focal_ids == context_source_ids

    # Duplicate source rows intentionally; replace their repeated names
    # immediately with unique inference-row IDs.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Observation names are not unique")
        result = with_neighbors[focal_positions, :].copy()
    result.obs_names = pd.Index(
        [
            f"responsiveness_{row:012d}"
            for row in range(n_rows)
        ],
    )
    result.obs["focal_cell_id"] = focal_ids
    result.obs["context_source_cell_id"] = context_source_ids
    result.obs["context_kind"] = np.where(is_original, "original", "representative")
    result.obs["is_original"] = is_original
    result.obs["is_representative_context"] = is_representative_context
    result.obs["sampling_repeat"] = sampling_repeat
    result.obs["context_number"] = context_number
    result.obs[split_col] = split_value

    result_contexts = {
        query_id: list(source_contexts[source_id])
        for query_id, source_id in zip(result.obs_names.astype(str), context_source_ids)
    }
    result.uns.pop("neighbor_dict", None)
    result.uns.pop("neighbor_dict_config", None)
    result.uns["neighbor_celltype_dict"] = result_contexts
    result.uns["responsiveness_config"] = {
        "source_path": str(input_path),
        "cell_type": cell_type,
        "cell_type_col": cell_type_col,
        "coord_cols": list(coord_cols),
        "tissue_col": tissue_col or "",
        "k_neighbors": int(k_neighbors),
        "min_source_counts": int(min_source_counts),
        "use_same_cell_type_neighbors": bool(use_same_cell_type_neighbors),
        "n_contexts": int(n_contexts),
        "n_query_cells": int(n_query_cells),
        "n_repeats": int(n_repeats),
        "random_seed": int(random_seed),
        "include_original_context": bool(include_original_context),
        "representative_context_method": "farthest_point_composition",
    }

    counts = np.zeros((n_rows, len(composition_cell_types)), dtype=np.int16)
    for row, neighbor_types in enumerate(result_contexts.values()):
        for neighbor_type in neighbor_types:
            counts[row, cell_type_to_col[str(neighbor_type)]] += 1
    denominators = counts.sum(axis=1, keepdims=True)
    proportions = np.divide(
        counts,
        denominators,
        out=np.zeros_like(counts, dtype=np.float32),
        where=denominators != 0,
    )
    result.obsm["neighborhood_counts"] = pd.DataFrame(
        counts,
        index=result.obs_names,
        columns=composition_cell_types,
    )
    result.obsm["neighborhood_proportions"] = pd.DataFrame(
        proportions,
        index=result.obs_names,
        columns=composition_cell_types,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.writing")
    result.write_h5ad(temporary_path)
    temporary_path.replace(output_path)
    print(
        f"> Wrote {output_path}: {n_query_cells:,} queries x {n_contexts:,} representative contexts "
        f"x {n_repeats:,} repeats, plus requested original baselines = {n_rows:,} rows",
    )
    return result


def main() -> None:
    args = parse_args()
    coord_cols = tuple(part.strip() for part in args.coord_cols.split(",") if part.strip())
    if len(coord_cols) != 2:
        raise ValueError("--coord-cols must contain exactly two comma-separated columns")

    for input_path in args.inputs:
        output_path = args.output_dir / f"{input_path.stem}_{args.suffix}.h5ad"
        prepare_tissue(
            input_path,
            output_path,
            cell_type=args.cell_type,
            cell_type_col=args.cell_type_col,
            coord_cols=coord_cols,
            tissue_col=args.tissue_col,
            k_neighbors=args.k_neighbors,
            min_source_counts=args.min_source_counts,
            use_same_cell_type_neighbors=args.use_same_cell_type_neighbors,
            split_col=args.split_col,
            split_value=args.split_value,
            n_contexts=args.n_contexts,
            n_query_cells=args.n_query_cells,
            n_repeats=args.n_repeats,
            random_seed=args.random_seed,
            include_original_context=args.include_original_context,
            max_output_rows=args.max_output_rows,
        )


if __name__ == "__main__":
    main()
