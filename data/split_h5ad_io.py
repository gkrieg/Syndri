"""Prepare SPLIT inputs from RCTD-annotated AnnData files and merge SPLIT outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite


def sanitize_key(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value))
    return re.sub(r"__+", "_", text).strip("_") or "unnamed"


def sample_id_for(h5ad_path: Path, layer: str) -> str:
    return f"{sanitize_key(h5ad_path.stem)}__{sanitize_key(layer)}"


def parse_layers(value: str) -> list[str]:
    layers = [layer.strip() for layer in value.split(",") if layer.strip()]
    if not layers:
        raise ValueError("--layers must contain at least one layer name")
    return layers


def matrix_to_csr(matrix: Any) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def get_count_matrix(adata: ad.AnnData, layer: str) -> sparse.csr_matrix:
    if layer == "X":
        return matrix_to_csr(adata.X)
    if layer not in adata.layers:
        raise KeyError(f"Layer {layer!r} not found. Available layers: {sorted(adata.layers.keys())}")
    return matrix_to_csr(adata.layers[layer])


def rctd_key(adata: ad.AnnData, base: str, layer: str, axis: str) -> str:
    suffix = sanitize_key(layer)
    candidates = [f"{base}_{suffix}"]
    if layer == "X":
        candidates.append(base)

    container = adata.obsm if axis == "obsm" else adata.uns
    for key in candidates:
        if key in container:
            return key
    raise KeyError(f"Could not find {base} for layer {layer!r}. Tried: {', '.join(candidates)}")


def get_reference_path(
    adata: ad.AnnData,
    layer: str,
    explicit_reference: str | None,
) -> Path:
    if explicit_reference:
        return Path(explicit_reference)

    metadata_candidates = [
        f"rctd_run_metadata_{sanitize_key(layer)}",
        "rctd_run_metadata" if layer == "X" else "",
    ]
    for key in metadata_candidates:
        if key and key in adata.uns:
            metadata = adata.uns[key]
            if "reference_file" in metadata:
                return Path(str(metadata["reference_file"]))

    raise ValueError(
        "Reference AnnData was not provided and could not be inferred from "
        f"rctd_run_metadata for layer {layer!r}."
    )


def write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]], gzip_output: bool = False) -> None:
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def reference_profiles(
    reference_path: Path,
    genes: pd.Index,
    cell_types: Sequence[str],
    cell_type_col: str,
    reference_layer: str,
) -> pd.DataFrame:
    reference = ad.read_h5ad(reference_path)
    missing_genes = genes.difference(reference.var_names)
    if len(missing_genes) > 0:
        preview = ", ".join(map(str, missing_genes[:20]))
        raise ValueError(f"Reference {reference_path} is missing {len(missing_genes)} genes. First missing: {preview}")
    if cell_type_col not in reference.obs:
        raise KeyError(f"Reference obs column {cell_type_col!r} not found in {reference_path}")

    ref_gene_idx = reference.var_names.get_indexer(genes)
    if reference_layer == "X":
        matrix = reference.X[:, ref_gene_idx]
    else:
        if reference_layer not in reference.layers:
            raise KeyError(f"Reference layer {reference_layer!r} not found in {reference_path}")
        matrix = reference.layers[reference_layer][:, ref_gene_idx]
    matrix = matrix_to_csr(matrix)

    labels = reference.obs[cell_type_col].astype(str).to_numpy()
    rows: list[np.ndarray] = []
    missing_cell_types: list[str] = []
    for cell_type in cell_types:
        mask = labels == str(cell_type)
        if not np.any(mask):
            missing_cell_types.append(str(cell_type))
            continue
        mean_values = matrix[mask, :].mean(axis=0)
        rows.append(np.asarray(mean_values).ravel())

    if missing_cell_types:
        raise ValueError(
            f"Reference {reference_path} has no cells for {len(missing_cell_types)} RCTD cell type(s): "
            + ", ".join(missing_cell_types[:20])
        )

    return pd.DataFrame(rows, index=pd.Index(cell_types, name="cell_type"), columns=genes)


def prepare_layer(
    adata: ad.AnnData,
    h5ad_path: Path,
    layer: str,
    output_root: Path,
    reference_h5ad: str | None,
    reference_layer: str,
    reference_cell_type_col: str,
    include_zero_weight_cells: bool,
    overwrite: bool,
) -> dict[str, Any]:
    sample_id = sample_id_for(h5ad_path, layer)
    sample_dir = output_root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    expected = [
        sample_dir / "counts.mtx",
        sample_dir / "cells.txt",
        sample_dir / "genes.txt",
        sample_dir / "deconvolution_weights.tsv",
        sample_dir / "primary_cell_type.tsv",
        sample_dir / "reference.tsv.gz",
    ]
    if not overwrite and all(path.exists() for path in expected):
        return {"sample_id": sample_id, "layer": layer, "status": "skipped", "sample_dir": str(sample_dir)}

    weights_key = rctd_key(adata, "rctd_weights", layer, "obsm")
    cell_type_key = rctd_key(adata, "rctd_cell_type_names", layer, "uns")
    count_matrix = get_count_matrix(adata, layer)
    weights = np.asarray(adata.obsm[weights_key], dtype=np.float64)
    cell_types = [str(value) for value in list(adata.uns[cell_type_key])]

    if weights.shape != (adata.n_obs, len(cell_types)):
        raise ValueError(
            f"{weights_key} has shape {weights.shape}, expected ({adata.n_obs}, {len(cell_types)})"
        )

    row_sums = weights.sum(axis=1)
    keep_mask = np.isfinite(row_sums)
    if not include_zero_weight_cells:
        keep_mask &= row_sums > 0
    if not np.any(keep_mask):
        raise ValueError(f"No cells remain for layer {layer!r} after filtering zero/non-finite RCTD weights")

    reference_path = get_reference_path(adata, layer, reference_h5ad)
    ref = ad.read_h5ad(reference_path, backed="r")
    common_genes = adata.var_names.intersection(ref.var_names)
    ref.file.close()
    if len(common_genes) == 0:
        raise ValueError(f"No shared genes between {h5ad_path} and reference {reference_path}")

    adata_gene_idx = adata.var_names.get_indexer(common_genes)
    counts_subset = count_matrix[keep_mask, :][:, adata_gene_idx].tocsr()
    obs_names = pd.Index(adata.obs_names[keep_mask].astype(str))

    weights_subset = weights[keep_mask, :]
    positive_sums = weights_subset.sum(axis=1)
    valid = positive_sums > 0
    weights_subset[valid, :] = weights_subset[valid, :] / positive_sums[valid, None]
    primary = np.asarray(cell_types, dtype=object)[np.argmax(weights_subset, axis=1)]

    ref_profiles = reference_profiles(
        reference_path=reference_path,
        genes=common_genes,
        cell_types=cell_types,
        cell_type_col=reference_cell_type_col,
        reference_layer=reference_layer,
    )

    mmwrite(sample_dir / "counts.mtx", counts_subset)
    (sample_dir / "cells.txt").write_text("\n".join(obs_names) + "\n", encoding="utf-8")
    (sample_dir / "genes.txt").write_text("\n".join(map(str, common_genes)) + "\n", encoding="utf-8")

    weights_df = pd.DataFrame(weights_subset, index=obs_names, columns=cell_types)
    weights_df.index.name = "cell_id"
    weights_df.to_csv(sample_dir / "deconvolution_weights.tsv", sep="\t")

    primary_df = pd.DataFrame({"cell_id": obs_names, "primary_cell_type": primary})
    primary_df.to_csv(sample_dir / "primary_cell_type.tsv", sep="\t", index=False)

    write_tsv(
        sample_dir / "reference.tsv.gz",
        ["cell_type", *map(str, common_genes)],
        ([cell_type, *ref_profiles.loc[cell_type].to_numpy()] for cell_type in cell_types),
        gzip_output=True,
    )

    metadata = {
        "sample_id": sample_id,
        "source_h5ad": str(h5ad_path),
        "layer": layer,
        "reference_h5ad": str(reference_path),
        "reference_layer": reference_layer,
        "reference_cell_type_col": reference_cell_type_col,
        "weights_key": weights_key,
        "cell_type_key": cell_type_key,
        "n_cells": int(counts_subset.shape[0]),
        "n_genes": int(counts_subset.shape[1]),
        "n_cell_types": len(cell_types),
        "n_zero_weight_cells_excluded": int((row_sums <= 0).sum()),
    }
    (sample_dir / "split_input_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "status": "completed", "sample_dir": str(sample_dir)}


def prepare(args: argparse.Namespace) -> None:
    h5ad_path = Path(args.h5ad)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)
    adata = ad.read_h5ad(h5ad_path)

    rows = [
        prepare_layer(
            adata=adata,
            h5ad_path=h5ad_path,
            layer=layer,
            output_root=output_root,
            reference_h5ad=args.reference_h5ad,
            reference_layer=args.reference_layer,
            reference_cell_type_col=args.reference_cell_type_col,
            include_zero_weight_cells=args.include_zero_weight_cells,
            overwrite=args.overwrite,
        )
        for layer in layers
    ]

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_root / "split_h5ad_prepare_manifest.csv", index=False)
    print(f"Wrote prepare manifest: {output_root / 'split_h5ad_prepare_manifest.csv'}")


def read_split_counts(sample_dir: Path) -> tuple[sparse.csr_matrix, pd.Index, pd.Index]:
    counts_path = sample_dir / "split_purified_counts.mtx"
    genes_path = sample_dir / "split_purified_counts_genes.txt"
    cells_path = sample_dir / "split_purified_counts_cells.txt"
    for path in [counts_path, genes_path, cells_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing SPLIT export file: {path}")

    counts_genes_by_cells = matrix_to_csr(mmread(counts_path))
    genes = pd.Index(genes_path.read_text(encoding="utf-8").splitlines())
    cells = pd.Index(cells_path.read_text(encoding="utf-8").splitlines())
    if counts_genes_by_cells.shape != (len(genes), len(cells)):
        raise ValueError(f"{counts_path} shape does not match exported gene/cell lists")
    return counts_genes_by_cells.T.tocsr(), genes, cells


def merge_metadata(adata: ad.AnnData, sample_dir: Path, layer: str) -> dict[str, Any]:
    metadata_path = sample_dir / "split_cell_metadata.csv"
    if not metadata_path.exists():
        return {}

    metadata = pd.read_csv(metadata_path)
    if "cell_id" not in metadata.columns:
        return {}

    cell_index = pd.Index(metadata["cell_id"].astype(str))
    obs_index = pd.Index(adata.obs_names.astype(str))
    row_index = obs_index.get_indexer(cell_index)
    keep = row_index >= 0
    added: dict[str, str] = {}
    for column in metadata.columns:
        if column == "cell_id":
            continue
        out_col = f"split_{sanitize_key(layer)}_{sanitize_key(column)}"
        if pd.api.types.is_numeric_dtype(metadata[column]):
            values = pd.Series(np.nan, index=adata.obs_names, dtype="float64")
            values.iloc[row_index[keep]] = metadata.loc[keep, column].astype(float).to_numpy()
        else:
            values = pd.Series("", index=adata.obs_names, dtype="object")
            values.iloc[row_index[keep]] = metadata.loc[keep, column].astype(str).to_numpy()
        adata.obs[out_col] = values
        added[column] = out_col
    return added


def merge_layer(
    adata: ad.AnnData,
    h5ad_path: Path,
    layer: str,
    split_output_root: Path,
    result_layer_prefix: str,
    fill_missing: str,
    overwrite: bool,
) -> dict[str, Any]:
    sample_id = sample_id_for(h5ad_path, layer)
    sample_dir = split_output_root / sample_id
    result_layer = f"{result_layer_prefix}_{sanitize_key(layer)}"
    if result_layer in adata.layers and not overwrite:
        raise FileExistsError(f"Layer {result_layer!r} already exists; use --overwrite to replace it")

    purified, genes, cells = read_split_counts(sample_dir)
    obs_index = pd.Index(adata.obs_names.astype(str))
    var_index = pd.Index(adata.var_names.astype(str))
    row_index = obs_index.get_indexer(cells)
    col_index = var_index.get_indexer(genes)
    if (row_index < 0).any():
        raise ValueError(f"SPLIT output contains cells not present in target AnnData for {sample_id}")
    if (col_index < 0).any():
        raise ValueError(f"SPLIT output contains genes not present in target AnnData for {sample_id}")

    if fill_missing == "original":
        output = get_count_matrix(adata, layer).tolil()
    else:
        output = sparse.lil_matrix(adata.shape, dtype=purified.dtype)
    output[np.ix_(row_index, col_index)] = purified
    adata.layers[result_layer] = output.tocsr()

    added_obs_columns = merge_metadata(adata, sample_dir, layer)
    return {
        "sample_id": sample_id,
        "layer": layer,
        "result_layer": result_layer,
        "sample_dir": str(sample_dir),
        "n_cells": int(purified.shape[0]),
        "n_genes": int(purified.shape[1]),
        "fill_missing": fill_missing,
        "obs_columns": added_obs_columns,
    }


def merge(args: argparse.Namespace) -> None:
    h5ad_path = Path(args.h5ad)
    output_h5ad = Path(args.output_h5ad) if args.output_h5ad else h5ad_path
    layers = parse_layers(args.layers)
    adata = ad.read_h5ad(h5ad_path)

    rows = [
        merge_layer(
            adata=adata,
            h5ad_path=h5ad_path,
            layer=layer,
            split_output_root=Path(args.split_output_root),
            result_layer_prefix=args.result_layer_prefix,
            fill_missing=args.fill_missing,
            overwrite=args.overwrite,
        )
        for layer in layers
    ]

    adata.uns["split_purification"] = {
        "source_h5ad": str(h5ad_path),
        "split_output_root": str(args.split_output_root),
        "layer_results": {sanitize_key(row["layer"]): row for row in rows},
    }

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    if output_h5ad == h5ad_path:
        tmp_path = output_h5ad.with_suffix(output_h5ad.suffix + f".tmp-{os.getpid()}")
        adata.write_h5ad(tmp_path)
        os.replace(tmp_path, output_h5ad)
    else:
        adata.write_h5ad(output_h5ad)
    print(f"Wrote merged AnnData: {output_h5ad}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Export SPLIT intermediate files from an RCTD-annotated h5ad")
    prep.add_argument("--h5ad", required=True)
    prep.add_argument("--output-root", required=True)
    prep.add_argument("--layers", required=True, help="Comma-separated layer names, using X for adata.X")
    prep.add_argument("--reference-h5ad", default=None, help="Optional reference h5ad; otherwise inferred from RCTD metadata")
    prep.add_argument("--reference-layer", default="X", help="Reference count layer to average; use X for adata.X")
    prep.add_argument("--reference-cell-type-col", default="cell_type")
    prep.add_argument("--include-zero-weight-cells", action="store_true")
    prep.add_argument("--overwrite", action="store_true")
    prep.set_defaults(func=prepare)

    merge_parser = subparsers.add_parser("merge", help="Merge exported SPLIT purified counts into an h5ad")
    merge_parser.add_argument("--h5ad", required=True)
    merge_parser.add_argument("--split-output-root", required=True)
    merge_parser.add_argument("--layers", required=True, help="Comma-separated layer names, using X for adata.X")
    merge_parser.add_argument("--output-h5ad", default=None, help="Defaults to in-place replacement of --h5ad")
    merge_parser.add_argument("--result-layer-prefix", default="split_purified_counts")
    merge_parser.add_argument("--fill-missing", choices=["original", "zero"], default="original")
    merge_parser.add_argument("--overwrite", action="store_true")
    merge_parser.set_defaults(func=merge)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
