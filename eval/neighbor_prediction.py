import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
import warnings

import scanpy as sc


def add_log_normalized_layer(
    adata,
    counts_layer=None,
    out_layer="lognorm",
    target_sum=1e4,
):
    """
    Add sparse-friendly log-normalized counts layer.

    If counts_layer is None, uses adata.X.
    """
    X = adata.layers[counts_layer] if counts_layer is not None else adata.X
    X = X.copy()

    if sparse.issparse(X):
        X = X.tocsr()
        totals = np.asarray(X.sum(axis=1)).ravel()
        totals[totals == 0] = 1.0
        scale = target_sum / totals
        X = sparse.diags(scale).dot(X)
        X.data = np.log1p(X.data)
    else:
        X = np.asarray(X, dtype=np.float64)
        totals = X.sum(axis=1)
        totals[totals == 0] = 1.0
        X = X * (target_sum / totals)[:, None]
        X = np.log1p(X)

    adata.layers[out_layer] = X
    return adata


def _get_X(adata, layer=None):
    return adata.layers[layer] if layer is not None else adata.X


def _get_coords(adata, coord_key="spatial"):
    if isinstance(coord_key, str):
        if coord_key in adata.obsm:
            return np.asarray(adata.obsm[coord_key], dtype=float)
        if coord_key in adata.obs:
            arr = adata.obs[coord_key].to_numpy()
            return np.asarray(arr, dtype=float)
        raise KeyError(f"{coord_key!r} not found in adata.obsm or adata.obs.")

    # For coord_key=("x", "y") style input.
    return adata.obs[list(coord_key)].to_numpy(dtype=float)


def compute_sample_markers_wilcoxon(
    adata,
    sample_key,
    cell_type_key,
    layer=None,
    top_n=20,
    min_cells_per_type=10,
):
    """
    For each sample, define markers for each cell type as the top_n genes
    by Wilcoxon rank-sum statistic comparing that cell type against all others.

    Returns
    -------
    markers : dict
        (sample, cell_type) -> set(marker_genes)
    marker_table : pd.DataFrame
        Long-form table of marker genes.
    """
    markers = {}
    rows = []

    sample_values = adata.obs[sample_key].astype(str).to_numpy()

    for sample in pd.unique(sample_values):
        sample_mask = sample_values == sample
        ad_s = adata[sample_mask].copy()

        if layer is not None:
            ad_s.X = _get_X(ad_s, layer).copy()

        ad_s.obs["_ct_marker"] = ad_s.obs[cell_type_key].astype(str).to_numpy()

        ct_counts = ad_s.obs["_ct_marker"].value_counts()
        valid_cell_types = ct_counts[ct_counts >= min_cells_per_type].index.tolist()

        if len(valid_cell_types) < 2:
            continue

        ad_s = ad_s[ad_s.obs["_ct_marker"].isin(valid_cell_types)].copy()

        sc.tl.rank_genes_groups(
            ad_s,
            groupby="_ct_marker",
            method="wilcoxon",
            n_genes=top_n,
            use_raw=False,
        )

        for ct in valid_cell_types:
            try:
                df = sc.get.rank_genes_groups_df(ad_s, group=ct)
            except Exception:
                continue

            top_genes = df["names"].astype(str).head(top_n).tolist()
            markers[(str(sample), str(ct))] = set(top_genes)

            for rank, gene in enumerate(top_genes, start=1):
                rows.append(
                    {
                        "sample": str(sample),
                        "cell_type": str(ct),
                        "marker_rank": rank,
                        "gene": gene,
                        # "gene_symbol
                    }
                )

    marker_table = pd.DataFrame(rows)
    return markers, marker_table


def _fit_logistic_regression(
    X,
    y,
    max_iter=500,
    exact_zscore="auto",
    dense_size_limit_entries=1e8,
    random_state=0,
):
    """
    Fit logistic regression using z-scored features.

    exact_zscore:
        True  -> densify sparse matrix and use exact mean-centering.
        False -> keep sparse matrix sparse; scale by std only.
        "auto" -> exact if matrix is small enough, otherwise sparse-safe scaling.

    Note:
        For sparse matrices, exact centering makes the matrix dense.
    """
    n_entries = X.shape[0] * X.shape[1]
    is_sparse = sparse.issparse(X)

    if exact_zscore is True:
        if is_sparse:
            X = X.toarray()
        scaler = StandardScaler(with_mean=True)
        zscore_mode = "exact"

    elif exact_zscore == "auto":
        if is_sparse and n_entries <= dense_size_limit_entries:
            X = X.toarray()
            scaler = StandardScaler(with_mean=True)
            zscore_mode = "exact"
        elif is_sparse:
            scaler = StandardScaler(with_mean=False)
            zscore_mode = "sparse_scale_only"
        else:
            scaler = StandardScaler(with_mean=True)
            zscore_mode = "exact"

    else:
        scaler = StandardScaler(with_mean=False if is_sparse else True)
        zscore_mode = "sparse_scale_only" if is_sparse else "exact"

    model = make_pipeline(
        scaler,
        LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state,
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)

    coef = model.named_steps["logisticregression"].coef_.ravel()
    return coef, zscore_mode


def run_specificity_contamination_assessment(
    adata,
    sample_key="sample_id",
    cell_type_key="cell_type",
    coord_key="spatial",
    layer=None,
    radius_um=15.0,
    top_n=20,
    min_cells_per_label=100,
    downsample_if_n_cells=50_000,
    max_cells_per_label=25_000,
    exact_zscore="auto",
    dense_size_limit_entries=1e8,
    random_state=0,
):
    """
    Reimplementation of the contamination/specificity analysis.

    For every sample and every ordered pair of distinct cell types (i, j):

        i = target cell type
        j = possible contaminating adjacent cell type

    Among cells of type i:
        y = 1 if spatially adjacent within radius_um to at least one cell of type j
        y = 0 otherwise

    Then train logistic regression:
        y ~ z-scored log-normalized expression

    Genes are ranked by abs(logistic coefficient). The output reports how many
    of the top_n ranked genes are also top_n markers of contaminating cell type j.

    Higher marker_overlap_n means stronger evidence that expression in target
    cell type i contains signal associated with nearby cell type j.
    """
    rng = np.random.default_rng(random_state)

    X_all = _get_X(adata, layer)
    genes = np.asarray(adata.var_names.astype(str))

    coords = _get_coords(adata, coord_key)
    samples = adata.obs[sample_key].astype(str).to_numpy()
    cell_types = adata.obs[cell_type_key].astype(str).to_numpy()

    markers, marker_table = compute_sample_markers_wilcoxon(
        adata=adata,
        sample_key=sample_key,
        cell_type_key=cell_type_key,
        layer=layer,
        top_n=top_n,
    )

    result_rows = []
    coefficient_rows = []

    for sample in pd.unique(samples):
        sample = str(sample)
        sample_idx = np.flatnonzero(samples == sample)
        sample_cell_types = pd.unique(cell_types[sample_idx])

        # Precompute cell indices and neighbor models for each contaminating type j.
        idx_by_type = {
            ct: sample_idx[cell_types[sample_idx] == ct]
            for ct in sample_cell_types
        }

        nn_by_type = {}
        for ct, idx in idx_by_type.items():
            if len(idx) > 0:
                nn = NearestNeighbors(radius=radius_um, metric="euclidean")
                nn.fit(coords[idx])
                nn_by_type[ct] = nn

        for target_ct in sample_cell_types:
            target_idx = idx_by_type[target_ct]
            if len(target_idx) == 0:
                continue

            target_coords = coords[target_idx]

            for contam_ct in sample_cell_types:
                if contam_ct == target_ct:
                    continue

                contam_idx = idx_by_type[contam_ct]
                if len(contam_idx) == 0:
                    continue

                nn = nn_by_type[contam_ct]

                nearby = nn.radius_neighbors(
                    target_coords,
                    return_distance=False,
                )

                y = np.fromiter(
                    (len(neighbors) > 0 for neighbors in nearby),
                    dtype=np.int8,
                    count=len(target_idx),
                )

                n_proximal = int((y == 1).sum())
                n_distal = int((y == 0).sum())

                # Exclude pairs with fewer than 100 cells in either proximal/distal class.
                if n_proximal < min_cells_per_label or n_distal < min_cells_per_label:
                    continue

                pos_local = np.flatnonzero(y == 1)
                neg_local = np.flatnonzero(y == 0)

                # Downsample very large target sets.
                if len(target_idx) > downsample_if_n_cells:
                    if len(pos_local) > max_cells_per_label:
                        pos_local = rng.choice(
                            pos_local,
                            size=max_cells_per_label,
                            replace=False,
                        )
                    if len(neg_local) > max_cells_per_label:
                        neg_local = rng.choice(
                            neg_local,
                            size=max_cells_per_label,
                            replace=False,
                        )

                keep_local = np.concatenate([pos_local, neg_local])
                rng.shuffle(keep_local)

                fit_idx = target_idx[keep_local]
                fit_y = y[keep_local]

                X_fit = X_all[fit_idx]

                coef, zscore_mode = _fit_logistic_regression(
                    X_fit,
                    fit_y,
                    max_iter=500,
                    exact_zscore=exact_zscore,
                    dense_size_limit_entries=dense_size_limit_entries,
                    random_state=random_state,
                )

                top_coef_idx = np.argsort(np.abs(coef))[::-1][:top_n]
                top_coef_genes = genes[top_coef_idx].tolist()

                contam_markers = markers.get((sample, str(contam_ct)), set())
                overlap_genes = [g for g in top_coef_genes if g in contam_markers]

                result_rows.append(
                    {
                        "sample": sample,
                        "target_cell_type_i": str(target_ct),
                        "contaminating_cell_type_j": str(contam_ct),
                        "n_target_cells_total": int(len(target_idx)),
                        "n_proximal_to_j": n_proximal,
                        "n_distal_to_j": n_distal,
                        "n_fit_cells": int(len(fit_idx)),
                        "top_n": top_n,
                        "marker_overlap_n": int(len(overlap_genes)),
                        "marker_overlap_frac": float(len(overlap_genes) / top_n),
                        "overlap_genes": overlap_genes,
                        "top_coef_genes": top_coef_genes,
                        "contaminating_type_markers": sorted(contam_markers),
                        "zscore_mode": zscore_mode,
                    }
                )

                for rank, gene_idx in enumerate(top_coef_idx, start=1):
                    coefficient_rows.append(
                        {
                            "sample": sample,
                            "target_cell_type_i": str(target_ct),
                            "contaminating_cell_type_j": str(contam_ct),
                            "coef_rank": rank,
                            "gene": genes[gene_idx],
                            "coef": float(coef[gene_idx]),
                            "abs_coef": float(abs(coef[gene_idx])),
                            "is_marker_of_contaminating_type": genes[gene_idx] in contam_markers,
                        }
                    )

    results = pd.DataFrame(result_rows)
    coefficients = pd.DataFrame(coefficient_rows)

    return results, coefficients, marker_table