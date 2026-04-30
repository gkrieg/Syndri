import os
import re
import numpy as np
import anndata as ad
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

def _sanitize_name(name):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(name))


def integrate_single_st_file_with_harmony(
    st_file,
    sc_data_orig,
    tissue_filter="tumor",
    n_pcs=30,
    tissue_name=None,
):
    """
    Read one spatial AnnData file and one single-cell reference AnnData, preprocess,
    run PCA + Harmony integration, and store the integrated embeddings under a
    tissue-specific obsm key.

    Parameters
    ----------
    st_file : str
        Path to one spatial .h5ad file.
    sc_data_orig : AnnData
        Single-cell reference AnnData.
    tissue_filter : str, default="tumor"
        Value in sc_data_orig.obs['Tissue'] used to subset the single-cell data.
    n_pcs : int, default=30
        Number of PCA components to compute before Harmony.
    tissue_name : str or None, default=None
        Name used in the obsm key. If None, inferred from the ST filename.

    Returns
    -------
    sc_data : AnnData
        Filtered single-cell AnnData with tissue-specific Harmony embedding in obsm.
    st_data : AnnData
        Spatial AnnData with tissue-specific Harmony embedding in obsm.
    embedding_key : str
        Tissue-specific embedding key used in obsm.
    """
    if tissue_name is None:
        tissue_name = os.path.splitext(os.path.basename(st_file))[0]
    tissue_name = _sanitize_name(tissue_name)
    embedding_key = f"X_pca_harmony_{tissue_name}"

    # filter single-cell
    sc_data = sc_data_orig[sc_data_orig.obs["Tissue"] == tissue_filter, :].copy()
    sc_data.obs["batch"] = "sc"

    # read spatial
    st_data = ad.read_h5ad(st_file)
    st_data.obs["batch"] = "st"

    # concatenate and preprocess
    adata = sc_data.concatenate(
        st_data,
        batch_key="batch",
        batch_categories=["sc", "st"],
    )

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=n_pcs)

    print(adata.obs.columns.tolist())
    print(adata.obs["batch"].dtype)
    print(adata.obs["batch"].value_counts(dropna=False))
    print(adata.obsm["X_pca"].shape)

    # Harmony integration in PCA space
    sc.external.pp.harmony_integrate(adata, "batch", basis="X_pca")

    is_st = adata.obs["batch"] == "st"
    is_sc = adata.obs["batch"] == "sc"

    # split Harmony embeddings back out using tissue-specific key
    sc_data.obsm[embedding_key] = adata.obsm["X_pca_harmony"][is_sc, :]
    st_data.obsm[embedding_key] = adata.obsm["X_pca_harmony"][is_st, :]

    return sc_data, st_data, embedding_key


import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_sc_tissue_mask(sc_data, tissue_name, tissue_col="harmony_spatial_tissue"):
    """
    Boolean mask for SC rows corresponding to one spatial tissue.
    """
    if tissue_col not in sc_data.obs:
        raise KeyError(f"{tissue_col!r} not found in sc_data.obs")
    return sc_data.obs[tissue_col].astype(str).to_numpy() == str(tissue_name)


def get_tissue_name_from_st(
    st_data,
    tissue_name=None,
    tissue_col="harmony_spatial_tissue",
):
    """
    Resolve the tissue/sample name for the current ST object.

    Priority:
      1. explicit tissue_name argument
      2. unique value in st_data.obs[tissue_col]
    """
    if tissue_name is not None:
        return str(tissue_name)

    if tissue_col not in st_data.obs:
        raise ValueError(
            f"tissue_name not provided and {tissue_col!r} not found in st_data.obs"
        )

    vals = st_data.obs[tissue_col].astype(str).unique()
    if len(vals) != 1:
        raise ValueError(
            f"Expected exactly one unique tissue in st_data.obs[{tissue_col!r}], got {vals}"
        )
    return vals[0]


def compute_top_st_matches(
    sc_emb,
    st_emb,
    st_ids,
    st_coords,
    n_matches=3,
):
    """
    Compute top-k nearest ST matches for each SC embedding row.

    Returns
    -------
    result : dict
        {
          "ids":   (n_sc, n_matches) array of ST IDs
          "dists": (n_sc, n_matches) array of distances
          "x":     (n_sc, n_matches) array of x coords
          "y":     (n_sc, n_matches) array of y coords
          "indices": (n_sc, n_matches) array of integer neighbor indices
        }
    """
    if sc_emb.shape[1] != st_emb.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: sc_emb.shape={sc_emb.shape}, st_emb.shape={st_emb.shape}"
        )

    nn = NearestNeighbors(n_neighbors=n_matches, metric="euclidean")
    nn.fit(st_emb)
    distances, indices = nn.kneighbors(sc_emb)

    return {
        "ids": st_ids[indices].astype(str),
        "dists": distances,
        "x": st_coords[indices, 0],
        "y": st_coords[indices, 1],
        "indices": indices,
    }


def compute_match_weights(distances, top1_ids=None, eps=1e-8):
    """
    Convert distances to weights and optionally penalize reuse based on the
    frequency of top-1 assignments.

    Parameters
    ----------
    distances : np.ndarray
        Shape (n_cells, k)
    top1_ids : array-like or None
        Length n_cells. If provided, reuse penalties are derived from the
        frequency of these IDs.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    w : np.ndarray
        Shape (n_cells, k), row-normalized weights.
    """
    tau = np.median(distances)
    w = np.exp(-(distances ** 2) / (tau ** 2 + eps))

    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    w = w / row_sums

    if top1_ids is not None:
        unique_ids, counts_arr = np.unique(np.asarray(top1_ids).astype(str), return_counts=True)
        counts = dict(zip(unique_ids, counts_arr))

        # penalties based on how frequent each candidate's current top1 is
        # to preserve your original logic, caller should pass the actual match ids separately
        return w, counts

    return w


def apply_reuse_penalty(weights, match_ids, counts, eps=1e-6):
    """
    Apply per-candidate reuse penalties to weights.

    Parameters
    ----------
    weights : np.ndarray
        Shape (n_cells, k)
    match_ids : np.ndarray
        Shape (n_cells, k), string IDs for candidate ST matches
    counts : dict
        Frequency dictionary, usually based on top-1 IDs
    eps : float
        Numerical stability constant.

    Returns
    -------
    w : np.ndarray
        Penalized and re-normalized weights.
    """
    w = weights.copy()
    penalties = np.vectorize(lambda x: 1.0 / (counts.get(str(x), 1) + eps))(match_ids)
    w *= penalties

    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    w = w / row_sums
    return w


def sample_match_indices(weights, rng=None):
    """
    Sample one column index per row from a row-stochastic weight matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = weights.shape[1]
    choices = np.arange(k)
    return np.array([rng.choice(choices, p=weights[i]) for i in range(weights.shape[0])])


def build_sampled_match_df(match_result, selected_idx):
    """
    Turn selected match indices into per-cell sampled outputs.
    """
    n = match_result["ids"].shape[0]
    rows = np.arange(n)
    return {
        "sampled_st_id": match_result["ids"][rows, selected_idx],
        "sampled_st_x": match_result["x"][rows, selected_idx],
        "sampled_st_y": match_result["y"][rows, selected_idx],
        "selected_match_idx": selected_idx,
    }


def initialize_match_columns(sc_data, n_matches=3, prefix="st_match"):
    """
    Ensure match columns exist in sc_data.obs.
    """
    for j in range(1, n_matches + 1):
        for suffix in ["id", "dist", "x", "y"]:
            col = f"{prefix}_{j}_{suffix}"
            if col not in sc_data.obs.columns:
                sc_data.obs[col] = np.nan

    for col in ["sampled_st_id", "sampled_st_x", "sampled_st_y", "selected_match_idx"]:
        if col not in sc_data.obs.columns:
            sc_data.obs[col] = np.nan

    return sc_data


def write_match_results_to_obs(sc_data, mask, match_result, sampled_result, prefix="st_match"):
    """
    Write match results for the masked subset back into full sc_data.obs.
    """
    mask = np.asarray(mask)
    row_idx = np.where(mask)[0]

    for j in range(match_result["ids"].shape[1]):
        sc_data.obs.iloc[row_idx, sc_data.obs.columns.get_loc(f"{prefix}_{j+1}_id")] = match_result["ids"][:, j]
        sc_data.obs.iloc[row_idx, sc_data.obs.columns.get_loc(f"{prefix}_{j+1}_dist")] = match_result["dists"][:, j]
        sc_data.obs.iloc[row_idx, sc_data.obs.columns.get_loc(f"{prefix}_{j+1}_x")] = match_result["x"][:, j]
        sc_data.obs.iloc[row_idx, sc_data.obs.columns.get_loc(f"{prefix}_{j+1}_y")] = match_result["y"][:, j]

    for key, values in sampled_result.items():
        if key in sc_data.obs.columns:
            sc_data.obs.iloc[row_idx, sc_data.obs.columns.get_loc(key)] = values

    return sc_data


def map_sc_subset_to_st(
    sc_data,
    st_data,
    embedding_key="X_pca_harmony",
    tissue_name=None,
    tissue_col="harmony_spatial_tissue",
    spatial_cols=("CenterX_global_px", "CenterY_global_px"),
    n_matches=3,
    rng=None,
    prefix="st_match",
):
    """
    Map only the SC rows corresponding to the current tissue slice onto the given ST data.

    Parameters
    ----------
    sc_data : AnnData
        Full SC AnnData containing duplicated rows for multiple tissues.
    st_data : AnnData
        One ST AnnData for a specific tissue slice.
    embedding_key : str
        Key in `.obsm` containing the concatenated Harmony embedding.
    tissue_name : str or None
        Tissue name to map. If None, inferred from st_data.obs[tissue_col].
    tissue_col : str
        Column in both sc_data.obs and optionally st_data.obs indicating tissue identity.
    spatial_cols : tuple[str, str]
        Names of x/y coordinate columns in st_data.obs.
    n_matches : int
        Number of top ST matches per SC row.
    rng : np.random.Generator or None
        Random generator for weighted sampling.
    prefix : str
        Prefix for stored top-match columns in sc_data.obs.

    Returns
    -------
    sc_data : AnnData
        Full SC object with only the relevant subset updated in `.obs`.
    coverage : float
        Fraction of unique sampled ST IDs among all ST cells for this tissue.
    match_result : dict
        Raw match outputs for the mapped subset.
    """
    if embedding_key not in sc_data.obsm:
        raise KeyError(f"{embedding_key!r} not found in sc_data.obsm")
    if embedding_key not in st_data.obsm:
        raise KeyError(f"{embedding_key!r} not found in st_data.obsm")

    xcol, ycol = spatial_cols
    for col in spatial_cols:
        if col not in st_data.obs:
            raise KeyError(f"{col!r} not found in st_data.obs")

    tissue_name = get_tissue_name_from_st(
        st_data,
        tissue_name=tissue_name,
        tissue_col=tissue_col,
    )

    mask = get_sc_tissue_mask(sc_data, tissue_name, tissue_col=tissue_col)
    if mask.sum() == 0:
        raise ValueError(f"No SC rows found for tissue {tissue_name!r}")

    sc_data = sc_data.copy()
    sc_data = initialize_match_columns(sc_data, n_matches=n_matches, prefix=prefix)

    sc_emb = np.asarray(sc_data.obsm[embedding_key])[mask]
    st_emb = np.asarray(st_data.obsm[embedding_key])

    st_ids = st_data.obs_names.to_numpy().astype(str)
    st_coords = st_data.obs[[xcol, ycol]].to_numpy()

    match_result = compute_top_st_matches(
        sc_emb=sc_emb,
        st_emb=st_emb,
        st_ids=st_ids,
        st_coords=st_coords,
        n_matches=n_matches,
    )

    base_weights, counts = compute_match_weights(
        match_result["dists"],
        top1_ids=match_result["ids"][:, 0],
    )
    weights = apply_reuse_penalty(
        base_weights,
        match_ids=match_result["ids"],
        counts=counts,
    )

    selected_idx = sample_match_indices(weights, rng=rng)
    sampled_result = build_sampled_match_df(match_result, selected_idx)

    sc_data = write_match_results_to_obs(
        sc_data,
        mask=mask,
        match_result=match_result,
        sampled_result=sampled_result,
        prefix=prefix,
    )

    sampled_ids = np.asarray(sampled_result["sampled_st_id"]).astype(str)
    coverage = np.unique(sampled_ids).shape[0] / st_data.n_obs

    return sc_data, coverage, match_result


