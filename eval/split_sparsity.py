"""Cache RCTD reference profiles and measure cell-type sparsity in merged SPLIT h5ad files."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "data"))
from rctd_annotation import build_reference, Reference, sanitize_key
from split_h5ad_io import get_reference_path, rctd_key


def cached_profiles(reference_path, spatial_genes, cache_dir, **settings):
    """Subset genes BEFORE UMI filtering/normalization, exactly as annotation does."""
    for name, parameter in inspect.signature(build_reference).parameters.items():
        if name != "reference_adata":
            settings.setdefault(name, parameter.default)
    reference_path = Path(reference_path).resolve()
    stat = reference_path.stat()
    ref = ad.read_h5ad(reference_path, backed="r")
    try:
        if not ref.var_names.is_unique or not spatial_genes.is_unique:
            raise ValueError("Reference and spatial gene names must be unique")
        genes = spatial_genes.intersection(ref.var_names)
    finally:
        ref.file.close()
    if not len(genes):
        raise ValueError("No shared reference/spatial genes")
    metadata = dict(reference_file=str(reference_path), size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns, genes=list(genes), settings=settings,
                    implementation=hashlib.sha256(
                        (inspect.getsource(build_reference) + inspect.getsource(Reference)).encode()
                    ).hexdigest())
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.h5ad"
    # Array tasks sharing a reference and gene panel build the cache only once.
    with (cache_dir / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            ref = ad.read_h5ad(reference_path)
            obj = build_reference(ref[:, genes].copy(), **settings)
            profiles = ad.AnnData(
                X=obj.profiles.T,
                obs=pd.DataFrame(index=pd.Index(obj.cell_type_names, name="cell_type")),
                var=pd.DataFrame(index=pd.Index(obj.gene_names, name="gene")),
            )
            profiles.uns["cache_metadata_json"] = json.dumps(metadata, sort_keys=True)
            tmp = path.with_suffix(f".tmp-{os.getpid()}.h5ad")
            try:
                profiles.write_h5ad(tmp)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
            print(f"Built profile cache: {path}", flush=True)
        else:
            print(f"Reusing profile cache: {path}", flush=True)
        return ad.read_h5ad(path), path


def sparsity_stats(matrix):
    """Exact zeros; also report zeros in the cell-type mean expression vector."""
    matrix = sparse.csr_matrix(matrix, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Expression must be finite and nonnegative")
    n_cells, n_genes = matrix.shape
    entries = n_cells * n_genes
    return dict(n_cells=n_cells, n_genes=n_genes, n_entries=entries,
                n_nonzero=matrix.nnz, n_zero=entries - matrix.nnz,
                zero_fraction=(entries - matrix.nnz) / entries if entries else np.nan,
                mean_profile_zero_fraction=(np.count_nonzero(
                    np.asarray(matrix.sum(axis=0)).ravel() == 0) / n_genes
                    if n_cells and n_genes else np.nan))


def spatial_labels(adata, layer, column=None):
    if column:
        labels = adata.obs[column]
        keep = labels.notna().to_numpy() & labels.astype(str).ne("").to_numpy()
        return labels.astype(str).to_numpy(), keep, f"obs:{column}"
    names = np.asarray(adata.uns[rctd_key(adata, "rctd_cell_type_names", layer, "uns")], dtype=str)
    weights = np.asarray(adata.obsm[rctd_key(adata, "rctd_weights", layer, "obsm")])
    if weights.shape != (adata.n_obs, len(names)) or not len(names):
        raise ValueError("RCTD weights do not match cells and cell-type names")
    keep = np.isfinite(weights).all(axis=1) & (weights >= 0).all(axis=1) & (weights.sum(axis=1) > 0)
    return names[np.argmax(weights, axis=1)], keep, "original_rctd_weights_argmax"


def run(args):
    spatial_path = Path(args.spatial).resolve()
    adata = ad.read_h5ad(spatial_path)
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Spatial cell and gene names must be unique")
    rows = []
    for layer in args.layers.split(","):
        layer = layer.strip()
        after_layer = f"{args.result_layer_prefix}_{sanitize_key(layer)}"
        before = adata.X if layer == "X" else adata.layers[layer]
        after = adata.layers[after_layer]
        reference_path = get_reference_path(adata, layer, args.reference)
        profiles, cache_path = cached_profiles(
            reference_path, adata.var_names, args.cache_dir,
            cell_type_col=args.cell_type_col, cell_min=args.cell_min,
            n_max_cells=args.n_max_cells, min_UMI=args.reference_min_umi,
            max_UMI=args.reference_max_umi,
        )
        genes = adata.var_names.get_indexer(profiles.var_names)
        labels, keep, label_source = spatial_labels(adata, layer, args.spatial_cell_type_col)
        before = sparse.csr_matrix(before)[keep][:, genes]
        after = sparse.csr_matrix(after)[keep][:, genes]
        labels = labels[keep]
        common = dict(spatial_file=str(spatial_path), input_layer=layer,
                      result_layer=after_layer, profile_cache=str(cache_path),
                      label_source=label_source, n_excluded_cells=int((~keep).sum()),
                      cell_scope="merged_h5ad_valid_labels")
        for cell_type in sorted(set(profiles.obs_names) | set(labels)):
            if cell_type in profiles.obs_names:
                rows.append(dict(common, cell_type=cell_type, stage="scrna_profile",
                                 **sparsity_stats(profiles[cell_type].X)))
            mask = labels == cell_type
            for stage, matrix in [("spatial_before", before), ("spatial_after", after)]:
                rows.append(dict(common, cell_type=cell_type, stage=stage,
                                 **sparsity_stats(matrix[mask])))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_id = hashlib.sha256(str(spatial_path).encode()).hexdigest()[:12]
    output = output_dir / f"{spatial_path.stem}.{file_id}.sparsity.csv"
    tmp = output.with_suffix(f".tmp-{os.getpid()}.csv")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, output)
    print(f"Saved statistics: {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", required=True, help="Merged h5ad with original and SPLIT layers")
    parser.add_argument("--reference", help="Defaults to original RCTD run metadata")
    parser.add_argument("--layers", default="X", help="Comma-separated original layers")
    parser.add_argument("--result-layer-prefix", default="split_purified_counts")
    parser.add_argument("--cell-type-col", default="cell_type", help="Reference obs label column")
    parser.add_argument("--spatial-cell-type-col", help="Override RCTD argmax with this obs column")
    parser.add_argument("--cell-min", type=int, default=25)
    parser.add_argument("--n-max-cells", type=int, default=10000)
    parser.add_argument("--reference-min-umi", type=int, default=10,
                        help="Inclusive minimum shared-panel UMI count for reference cells.")
    parser.add_argument("--reference-max-umi", type=int, default=2000,
                        help="Inclusive maximum shared-panel UMI count for reference cells.")
    parser.add_argument("--cache-dir", default=str(ROOT / "data/cache/rctd_profiles"))
    parser.add_argument("--output-dir", default=str(ROOT / "eval/split_sparsity"))
    args = parser.parse_args()
    if args.reference_min_umi < 1 or args.reference_max_umi < args.reference_min_umi:
        parser.error("Reference UMI bounds must satisfy 1 <= minimum <= maximum")
    return args


if __name__ == "__main__":
    run(parse_args())
