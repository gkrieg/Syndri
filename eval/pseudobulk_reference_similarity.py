import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def _get_X(adata, layer=None):
    if layer is None:
        return adata.X
    return adata.layers[layer]


def pseudobulk_by_cell_type(
    adata,
    cell_type_key,
    layer=None,
    genes=None,
    min_cells=1,
):
    """
    Sum gene expression across all cells assigned to each cell type.

    Returns
    -------
    pb : pd.DataFrame
        Rows = cell types
        Columns = genes
        Values = summed counts/expression
    n_cells : pd.Series
        Number of cells per cell type
    """
    if genes is not None:
        adata = adata[:, genes].copy()

    X = _get_X(adata, layer)
    cell_types = adata.obs[cell_type_key].astype(str)

    rows = []
    index = []
    n_cells = {}

    for ct, idx in cell_types.groupby(cell_types).groups.items():
        idx = np.asarray(list(idx))
        pos = adata.obs_names.get_indexer(idx)

        if len(pos) < min_cells:
            continue

        X_ct = X[pos]

        if sparse.issparse(X_ct):
            summed = np.asarray(X_ct.sum(axis=0)).ravel()
        else:
            summed = np.asarray(X_ct).sum(axis=0)

        rows.append(summed)
        index.append(ct)
        n_cells[ct] = len(pos)

    pb = pd.DataFrame(
        np.vstack(rows),
        index=pd.Index(index, name=cell_type_key),
        columns=adata.var_names.astype(str),
    )

    n_cells = pd.Series(n_cells, name="n_cells")
    return pb, n_cells


def log_normalize_pseudobulk(pb, target_sum=1e4):
    """
    Library-size normalize each pseudobulk profile, then log1p transform.
    """
    values = pb.to_numpy(dtype=float)

    lib_size = values.sum(axis=1, keepdims=True)
    lib_size[lib_size == 0] = 1.0

    values = values / lib_size * target_sum
    values = np.log1p(values)

    return pd.DataFrame(
        values,
        index=pb.index,
        columns=pb.columns,
    )


def cell_type_pseudobulk_cosine_similarity(
    spatial_adata,
    chromium_adata,
    spatial_cell_type_key="cell_type",
    chromium_cell_type_key="cell_type",
    spatial_layer=None,
    chromium_layer=None,
    target_sum=1e4,
    min_cells_spatial=1,
    min_cells_chromium=1,
    cell_type_map=None,
):
    """
    Compare spatial pseudobulk profiles to Chromium scRNA-seq pseudobulk profiles.

    For each cell type:
        1. Sum expression across all cells of that type.
        2. Log-normalize the pseudobulk profile.
        3. Compute cosine similarity between matched spatial and Chromium profiles.

    Parameters
    ----------
    cell_type_map : dict or None
        Optional mapping from spatial cell type names to Chromium cell type names.
        Example: {"malignant cell": "tumor", "T cell": "T cell"}

    Returns
    -------
    similarities : pd.DataFrame
        One row per matched cell type.
    spatial_pb_log : pd.DataFrame
        Log-normalized spatial pseudobulk profiles.
    chromium_pb_log : pd.DataFrame
        Log-normalized Chromium pseudobulk profiles.
    """
    shared_genes = spatial_adata.var_names.intersection(chromium_adata.var_names)

    if len(shared_genes) == 0:
        raise ValueError("No shared genes between spatial_adata and chromium_adata.")

    spatial_pb, spatial_n = pseudobulk_by_cell_type(
        spatial_adata,
        cell_type_key=spatial_cell_type_key,
        layer=spatial_layer,
        genes=shared_genes,
        min_cells=min_cells_spatial,
    )

    chromium_pb, chromium_n = pseudobulk_by_cell_type(
        chromium_adata,
        cell_type_key=chromium_cell_type_key,
        layer=chromium_layer,
        genes=shared_genes,
        min_cells=min_cells_chromium,
    )

    spatial_pb_log = log_normalize_pseudobulk(spatial_pb, target_sum=target_sum)
    chromium_pb_log = log_normalize_pseudobulk(chromium_pb, target_sum=target_sum)

    rows = []

    for spatial_ct in spatial_pb_log.index:
        if cell_type_map is None:
            chromium_ct = spatial_ct
        else:
            chromium_ct = cell_type_map.get(spatial_ct)

        if chromium_ct is None:
            continue

        if chromium_ct not in chromium_pb_log.index:
            continue

        x = spatial_pb_log.loc[[spatial_ct]].to_numpy()
        y = chromium_pb_log.loc[[chromium_ct]].to_numpy()

        sim = cosine_similarity(x, y)[0, 0]

        rows.append(
            {
                "spatial_cell_type": spatial_ct,
                "chromium_cell_type": chromium_ct,
                "cosine_similarity": sim,
                "n_spatial_cells": int(spatial_n.loc[spatial_ct]),
                "n_chromium_cells": int(chromium_n.loc[chromium_ct]),
                "n_shared_genes": int(len(shared_genes)),
            }
        )

    similarities = (
        pd.DataFrame(rows)
        .sort_values("cosine_similarity", ascending=False)
        .reset_index(drop=True)
    )

    return similarities, spatial_pb_log, chromium_pb_log