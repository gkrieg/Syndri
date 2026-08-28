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
    max_iter_harmony=30,
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
    if tissue_filter is not None:
        sc_data = sc_data_orig[sc_data_orig.obs["Tissue"] == tissue_filter, :].copy()
    else:
        sc_data = sc_data_orig.copy()
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
    sc.external.pp.harmony_integrate(adata, "batch", basis="X_pca",max_iter_harmony=max_iter_harmony)

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


def compute_top_sc_matches(
    st_emb,
    sc_emb,
    sc_ids,
    n_matches=3,
):
    """
    Compute top-k nearest SC matches for each ST/Xenium embedding row.

    Returns
    -------
    result : dict
        {
          "ids":   (n_st, n_matches) array of SC IDs
          "dists": (n_st, n_matches) array of distances
          "indices": (n_st, n_matches) array of integer neighbor indices
        }
    """
    if st_emb.shape[1] != sc_emb.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: st_emb.shape={st_emb.shape}, sc_emb.shape={sc_emb.shape}"
        )

    nn = NearestNeighbors(n_neighbors=n_matches, metric="euclidean")
    nn.fit(sc_emb)
    distances, indices = nn.kneighbors(st_emb)

    return {
        "ids": sc_ids[indices].astype(str),
        "dists": distances,
        "indices": indices,
    }


def compute_top_sc_matches_by_cell_type(
    st_emb,
    sc_emb,
    sc_ids,
    st_cell_types,
    sc_cell_types,
    n_matches=3,
):
    """
    Compute top-k nearest SC matches for each ST/Xenium row, restricted to
    SC rows with the same cell type as the ST row.
    """
    if st_emb.shape[1] != sc_emb.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: st_emb.shape={st_emb.shape}, sc_emb.shape={sc_emb.shape}"
        )

    st_cell_types = np.asarray(st_cell_types).astype(str)
    sc_cell_types = np.asarray(sc_cell_types).astype(str)
    if st_cell_types.shape[0] != st_emb.shape[0]:
        raise ValueError("st_cell_types must have one value per ST embedding row.")
    if sc_cell_types.shape[0] != sc_emb.shape[0]:
        raise ValueError("sc_cell_types must have one value per SC embedding row.")

    n_st = st_emb.shape[0]
    ids = np.full((n_st, n_matches), "", dtype=object)
    dists = np.full((n_st, n_matches), np.inf, dtype=float)
    indices = np.full((n_st, n_matches), -1, dtype=int)

    for cell_type in np.unique(st_cell_types):
        st_idx = np.flatnonzero(st_cell_types == cell_type)
        sc_idx = np.flatnonzero(sc_cell_types == cell_type)
        if sc_idx.size == 0:
            raise ValueError(
                f"No SC reference cells found for spatial cell type {cell_type!r}."
            )

        k = min(n_matches, sc_idx.size)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(sc_emb[sc_idx])
        distances, local_indices = nn.kneighbors(st_emb[st_idx])
        global_indices = sc_idx[local_indices]

        ids[np.ix_(st_idx, np.arange(k))] = sc_ids[global_indices].astype(str)
        dists[np.ix_(st_idx, np.arange(k))] = distances
        indices[np.ix_(st_idx, np.arange(k))] = global_indices

    return {
        "ids": ids,
        "dists": dists,
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
    distances = np.asarray(distances, dtype=float)
    finite = np.isfinite(distances)
    if not finite.any():
        raise ValueError("No finite match distances available.")

    tau = np.median(distances[finite])
    w = np.zeros_like(distances, dtype=float)
    w[finite] = np.exp(-(distances[finite] ** 2) / (tau ** 2 + eps))

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


def apply_reuse_penalty(weights, match_ids, counts, reuse_penalty=1.0, eps=1e-6):
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
    reuse_penalty : float
        Strength of the reuse penalty. Current default behavior is 1.0,
        applying penalties as 1 / count. Set to 0.0 to disable the
        penalty, use values between 0 and 1 to soften it, and values
        greater than 1 to penalize reused ST cells more aggressively.
    eps : float
        Numerical stability constant.

    Returns
    -------
    w : np.ndarray
        Penalized and re-normalized weights.
    """
    if reuse_penalty < 0:
        raise ValueError("reuse_penalty must be non-negative.")

    w = weights.copy()
    if reuse_penalty == 0:
        penalties = 1.0
    else:
        penalties = np.vectorize(lambda x: 1.0 / ((counts.get(str(x), 1) + eps) ** reuse_penalty))(match_ids)
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


def sample_match_indices_with_reuse_cap(
    weights,
    match_ids,
    max_reuse_per_st,
    rng=None,
    shuffle_order=True,
    on_exhausted="best_available",
):
    """
    Sample one column index per row while enforcing a hard cap per ST ID.

    Parameters
    ----------
    weights : np.ndarray
        Shape (n_cells, k), row-normalized candidate weights.
    match_ids : np.ndarray
        Shape (n_cells, k), string IDs for candidate ST matches.
    max_reuse_per_st : int
        Maximum number of SC rows that can sample the same ST ID.
    rng : np.random.Generator or None
        Random generator for weighted sampling.
    shuffle_order : bool
        If True, process SC rows in random order to reduce row-order bias.
    on_exhausted : {"best_available", "raise"}
        Behavior when all candidate ST IDs for a row are already at capacity.
        "best_available" picks the highest-weight candidate even though it
        exceeds the cap. "raise" raises a ValueError.

    Returns
    -------
    selected_idx : np.ndarray
        Length n_cells, selected candidate column index per row.
    """
    if rng is None:
        rng = np.random.default_rng()
    if max_reuse_per_st is None:
        return sample_match_indices(weights, rng=rng)
    if max_reuse_per_st < 1:
        raise ValueError("max_reuse_per_st must be at least 1.")
    if on_exhausted not in {"best_available", "raise"}:
        raise ValueError("on_exhausted must be either 'best_available' or 'raise'.")

    weights = np.asarray(weights, dtype=float)
    match_ids = np.asarray(match_ids).astype(str)
    if weights.shape != match_ids.shape:
        raise ValueError(f"weights and match_ids must have the same shape; got {weights.shape} and {match_ids.shape}.")

    n_cells, k = weights.shape
    choices = np.arange(k)
    selected_idx = np.full(n_cells, -1, dtype=int)
    reuse_counts = {}

    order = np.arange(n_cells)
    if shuffle_order:
        order = rng.permutation(order)

    for row in order:
        valid_candidate = match_ids[row] != ""
        available = valid_candidate & np.array(
            [reuse_counts.get(st_id, 0) < max_reuse_per_st for st_id in match_ids[row]]
        )

        if not available.any():
            if not valid_candidate.any():
                raise ValueError("No valid match candidates available for at least one row.")
            if on_exhausted == "raise":
                raise ValueError(
                    "All candidate ST IDs are already at capacity for at least one SC row. "
                    "Increase max_reuse_per_st or n_matches.",
                )
            row_weights = weights[row].copy()
            row_weights[~valid_candidate] = -np.inf
            selected = int(np.argmax(row_weights))
        else:
            row_weights = weights[row].copy()
            row_weights[~available] = 0.0
            total = row_weights.sum()
            if total <= 0:
                row_weights = available.astype(float) / available.sum()
            else:
                row_weights = row_weights / total
            selected = int(rng.choice(choices, p=row_weights))

        selected_idx[row] = selected
        selected_id = match_ids[row, selected]
        reuse_counts[selected_id] = reuse_counts.get(selected_id, 0) + 1

    return selected_idx


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
    new_data = sc_data.copy()
    for j in range(1, n_matches + 1):
        for suffix in ["id", "dist", "x", "y"]:
            col = f"{prefix}_{j}_{suffix}"
            if col not in new_data.obs.columns:
                new_data.obs[col] = np.nan

    for col in ["sampled_st_id", "sampled_st_x", "sampled_st_y", "selected_match_idx"]:
        if col not in new_data.obs.columns:
            new_data.obs[col] = np.nan

    return new_data


def _prepare_obs_column_for_assignment(obs, col):
    """
    Convert categorical columns to object so new IDs/values can be assigned.
    """
    if str(obs[col].dtype) == "category":
        obs[col] = obs[col].astype(object)


def write_match_results_to_obs(sc_data, mask, match_result, sampled_result, prefix="st_match"):
    """
    Write match results for the masked subset back into full sc_data.obs.
    """
    mask = np.asarray(mask)
    row_idx = np.where(mask)[0]

    for j in range(match_result["ids"].shape[1]):
        for suffix, values in [
            ("id", match_result["ids"][:, j]),
            ("dist", match_result["dists"][:, j]),
            ("x", match_result["x"][:, j]),
            ("y", match_result["y"][:, j]),
        ]:
            col = f"{prefix}_{j+1}_{suffix}"
            _prepare_obs_column_for_assignment(sc_data.obs, col)
            sc_data.obs.loc[sc_data.obs.index[row_idx], col] = values

    for key, values in sampled_result.items():
        if key in sc_data.obs.columns:
            _prepare_obs_column_for_assignment(sc_data.obs, key)
            sc_data.obs.loc[sc_data.obs.index[row_idx], key] = values

    return sc_data


def _get_unique_spatial_rows(st_data, spatial_cols):
    """
    Return first row indices for all unique spatial coordinate pairs.
    """
    for col in spatial_cols:
        if col not in st_data.obs:
            raise KeyError(f"{col!r} not found in st_data.obs")

    coords = st_data.obs.loc[:, list(spatial_cols)]
    unique_mask = ~coords.duplicated(keep="first")
    unique_row_idx = np.flatnonzero(unique_mask.to_numpy())
    return unique_row_idx, coords.iloc[unique_row_idx].to_numpy()


def ensure_all_sc_indices_used(
    selected_idx,
    match_result,
    sc_emb,
    st_emb,
    sc_ids,
    sc_cell_types=None,
    st_cell_types=None,
):
    """
    Force every SC row to be selected at least once when there are enough
    spatial rows. Missing SC rows replace duplicate selections at their best
    eligible spatial row.
    """
    selected_idx = np.asarray(selected_idx, dtype=int).copy()
    indices = match_result["indices"]
    n_spatial = indices.shape[0]
    n_sc = sc_emb.shape[0]
    if n_sc > n_spatial:
        return selected_idx, 0

    rows = np.arange(n_spatial)
    selected_sc_indices = indices[rows, selected_idx]
    if np.any(selected_sc_indices < 0):
        raise ValueError("Selected match indices include invalid candidates.")

    counts = np.bincount(selected_sc_indices, minlength=n_sc)
    missing_sc_indices = np.flatnonzero(counts == 0)
    if missing_sc_indices.size == 0:
        return selected_idx, 0

    if sc_cell_types is not None or st_cell_types is not None:
        if sc_cell_types is None or st_cell_types is None:
            raise ValueError(
                "Both sc_cell_types and st_cell_types are required for cell-type-aware coverage."
            )
        sc_cell_types = np.asarray(sc_cell_types).astype(str)
        st_cell_types = np.asarray(st_cell_types).astype(str)

    for sc_idx in missing_sc_indices:
        replaceable = np.flatnonzero(counts[selected_sc_indices] > 1)
        if sc_cell_types is not None:
            replaceable = replaceable[st_cell_types[replaceable] == sc_cell_types[sc_idx]]

        if replaceable.size == 0:
            raise ValueError(
                "Cannot ensure every SC reference cell is used at least once with the "
                "current candidate constraints."
            )

        distances = np.linalg.norm(st_emb[replaceable] - sc_emb[sc_idx], axis=1)
        best_idx = int(np.argmin(distances))
        row = replaceable[best_idx]
        col = selected_idx[row]
        old_sc_idx = selected_sc_indices[row]

        counts[old_sc_idx] -= 1
        counts[sc_idx] += 1
        selected_sc_indices[row] = sc_idx

        match_result["ids"][row, col] = str(sc_ids[sc_idx])
        match_result["dists"][row, col] = float(distances[best_idx])
        match_result["indices"][row, col] = int(sc_idx)

    return selected_idx, int(missing_sc_indices.size)


def build_spatially_mapped_sc_adata(
    sc_data,
    unique_row_idx,
    match_result,
    selected_idx,
    st_data,
    spatial_cols,
    prefix="sc_match",
):
    """
    Build an SC AnnData with one selected SC row per unique ST/Xenium coordinate.
    """
    rows = np.arange(match_result["ids"].shape[0])
    selected_sc_indices = match_result["indices"][rows, selected_idx]
    selected_sc_ids = match_result["ids"][rows, selected_idx]

    mapped_sc_data = sc_data[selected_sc_indices, :].copy()
    mapped_sc_data.obs_names = st_data.obs_names[unique_row_idx].astype(str)

    mapped_sc_data.obs["source_sc_id"] = selected_sc_ids
    mapped_sc_data.obs["sampled_sc_id"] = selected_sc_ids
    mapped_sc_data.obs["matched_st_id"] = st_data.obs_names[unique_row_idx].astype(str)
    mapped_sc_data.obs["matched_st_index"] = unique_row_idx
    mapped_sc_data.obs["selected_match_idx"] = selected_idx

    for i, col in enumerate(spatial_cols):
        mapped_sc_data.obs[col] = match_result["coords"][:, i]

    if len(spatial_cols) >= 2:
        mapped_sc_data.obsm["spatial"] = match_result["coords"][:, :2]

    for j in range(match_result["ids"].shape[1]):
        mapped_sc_data.obs[f"{prefix}_{j+1}_id"] = match_result["ids"][:, j]
        mapped_sc_data.obs[f"{prefix}_{j+1}_dist"] = match_result["dists"][:, j]
        mapped_sc_data.obs[f"{prefix}_{j+1}_index"] = match_result["indices"][:, j]

    return mapped_sc_data


def map_sc_reference_to_unique_xenium_cells(
    sc_data,
    st_data,
    embedding_key="X_pca_harmony",
    spatial_cols=("x_centroid", "y_centroid"),
    n_matches=3,
    reuse_penalty=1.0,
    ensure_all_sc_used=True,
    match_cell_type=False,
    sc_cell_type_col="cell_type",
    st_cell_type_col="cell_type",
    max_reuse_per_sc=None,
    shuffle_capped_sampling_order=True,
    capped_sampling_on_exhausted="best_available",
    rng=None,
    prefix="sc_match",
):
    """
    Map one single-cell reference row to each unique Xenium spatial coordinate.

    Parameters
    ----------
    sc_data : AnnData
        Single-cell reference AnnData with embeddings in `.obsm[embedding_key]`.
    st_data : AnnData
        Spatial/Xenium AnnData with embeddings in `.obsm[embedding_key]`.
    embedding_key : str
        Key in both `.obsm` matrices used for nearest-neighbor matching.
    spatial_cols : tuple[str, str]
        Names of x/y coordinate columns in st_data.obs. Unique coordinate
        pairs define the Xenium cells to map.
    n_matches : int
        Number of top SC reference candidates per unique Xenium coordinate.
    reuse_penalty : float
        Strength of the SC-cell reuse penalty used when sampling among
        candidate matches. The default 1.0 penalizes by 1 / count.
        Set to 0.0 to disable reuse penalization.
    ensure_all_sc_used : bool
        If True and the number of SC reference rows is less than or equal to
        the number of unique spatial coordinates, force every SC reference
        row to be selected at least once.
    match_cell_type : bool
        If True, only use SC reference candidates whose cell type matches the
        spatial cell type for each unique Xenium coordinate.
    sc_cell_type_col : str
        Column in sc_data.obs containing SC reference cell type labels when
        match_cell_type=True.
    st_cell_type_col : str
        Column in st_data.obs containing spatial cell type labels when
        match_cell_type=True.
    max_reuse_per_sc : int or None
        If provided, use capped sequential sampling so each SC reference cell
        can be sampled at most this many times, unless all candidates for a
        row are exhausted and capped_sampling_on_exhausted="best_available".
    shuffle_capped_sampling_order : bool
        If True, process Xenium coordinates in random order when
        max_reuse_per_sc is set.
    capped_sampling_on_exhausted : {"best_available", "raise"}
        Behavior when all candidate SC IDs for a coordinate are at capacity.
    rng : np.random.Generator or None
        Random generator for weighted sampling.
    prefix : str
        Prefix for stored top-match columns in the returned `.obs`.

    Returns
    -------
    mapped_sc_data : AnnData
        Copy of selected rows from sc_data, allowing duplicate SC source rows,
        with one row per unique spatial coordinate in st_data. The output
        `.obs_names` are the corresponding st_data row names, while
        `.obs["source_sc_id"]` and `.obs["sampled_sc_id"]` identify the
        original SC row copied into each spatial position.
    """
    if embedding_key not in sc_data.obsm:
        raise KeyError(f"{embedding_key!r} not found in sc_data.obsm")
    if embedding_key not in st_data.obsm:
        raise KeyError(f"{embedding_key!r} not found in st_data.obsm")

    if n_matches < 1:
        raise ValueError("n_matches must be at least 1.")
    if n_matches > sc_data.n_obs:
        raise ValueError(
            f"n_matches={n_matches} exceeds the number of SC reference cells ({sc_data.n_obs})."
        )
    if match_cell_type:
        if sc_cell_type_col not in sc_data.obs:
            raise KeyError(f"{sc_cell_type_col!r} not found in sc_data.obs")
        if st_cell_type_col not in st_data.obs:
            raise KeyError(f"{st_cell_type_col!r} not found in st_data.obs")

    unique_row_idx, unique_coords = _get_unique_spatial_rows(st_data, spatial_cols)
    if len(unique_row_idx) == 0:
        raise ValueError("No unique spatial coordinates found in st_data.obs.")

    sc_emb = np.asarray(sc_data.obsm[embedding_key])
    st_emb = np.asarray(st_data.obsm[embedding_key])[unique_row_idx]
    sc_ids = sc_data.obs_names.to_numpy().astype(str)
    sc_cell_types = None
    st_cell_types = None
    if match_cell_type:
        sc_cell_types = sc_data.obs[sc_cell_type_col].astype(str).to_numpy()
        st_cell_types = st_data.obs[st_cell_type_col].astype(str).to_numpy()[unique_row_idx]

    if match_cell_type:
        match_result = compute_top_sc_matches_by_cell_type(
            st_emb=st_emb,
            sc_emb=sc_emb,
            sc_ids=sc_ids,
            st_cell_types=st_cell_types,
            sc_cell_types=sc_cell_types,
            n_matches=n_matches,
        )
    else:
        match_result = compute_top_sc_matches(
            st_emb=st_emb,
            sc_emb=sc_emb,
            sc_ids=sc_ids,
            n_matches=n_matches,
        )
    match_result["coords"] = unique_coords
    match_result["st_indices"] = unique_row_idx

    base_weights, counts = compute_match_weights(
        match_result["dists"],
        top1_ids=match_result["ids"][:, 0],
    )
    weights = apply_reuse_penalty(
        base_weights,
        match_ids=match_result["ids"],
        counts=counts,
        reuse_penalty=reuse_penalty,
    )

    if max_reuse_per_sc is None:
        selected_idx = sample_match_indices(weights, rng=rng)
    else:
        selected_idx = sample_match_indices_with_reuse_cap(
            weights,
            match_ids=match_result["ids"],
            max_reuse_per_st=max_reuse_per_sc,
            rng=rng,
            shuffle_order=shuffle_capped_sampling_order,
            on_exhausted=capped_sampling_on_exhausted,
        )

    n_forced_sc_assignments = 0
    if ensure_all_sc_used:
        selected_idx, n_forced_sc_assignments = ensure_all_sc_indices_used(
            selected_idx=selected_idx,
            match_result=match_result,
            sc_emb=sc_emb,
            st_emb=st_emb,
            sc_ids=sc_ids,
            sc_cell_types=sc_cell_types,
            st_cell_types=st_cell_types,
        )

    mapped_sc_data = build_spatially_mapped_sc_adata(
        sc_data=sc_data,
        unique_row_idx=unique_row_idx,
        match_result=match_result,
        selected_idx=selected_idx,
        st_data=st_data,
        spatial_cols=spatial_cols,
        prefix=prefix,
    )

    selected_sc_indices = match_result["indices"][
        np.arange(match_result["indices"].shape[0]), selected_idx
    ]
    mapped_sc_data.uns["sc_reference_to_spatial_mapping"] = {
        "embedding_key": embedding_key,
        "spatial_cols": tuple(spatial_cols),
        "n_matches": int(n_matches),
        "reuse_penalty": float(reuse_penalty),
        "ensure_all_sc_used": bool(ensure_all_sc_used),
        "match_cell_type": bool(match_cell_type),
        "sc_cell_type_col": sc_cell_type_col if match_cell_type else None,
        "st_cell_type_col": st_cell_type_col if match_cell_type else None,
        "max_reuse_per_sc": max_reuse_per_sc,
        "n_unique_spatial_coordinates": int(len(unique_row_idx)),
        "n_sc_reference_cells": int(sc_data.n_obs),
        "n_forced_sc_coverage_assignments": int(n_forced_sc_assignments),
        "sc_coverage": float(np.unique(selected_sc_indices).shape[0] / sc_data.n_obs),
    }

    return mapped_sc_data


def map_sc_subset_to_st(
    sc_data,
    st_data,
    embedding_key="X_pca_harmony",
    tissue_name=None,
    tissue_col="harmony_spatial_tissue",
    spatial_cols=("CenterX_global_px", "CenterY_global_px"),
    n_matches=3,
    reuse_penalty=1.0,
    max_reuse_per_st=None,
    shuffle_capped_sampling_order=True,
    capped_sampling_on_exhausted="best_available",
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
    reuse_penalty : float
        Strength of the ST-cell reuse penalty used when sampling among
        candidate matches. The default 1.0 preserves the previous behavior
        of penalizing by 1 / count. Set to 0.0 to disable reuse penalization,
        use values between 0 and 1 to soften it, and values greater than 1
        to penalize reused ST cells more aggressively.
    max_reuse_per_st : int or None
        If provided, use capped sequential sampling so each ST ID can be
        sampled at most this many times, unless all candidates for a row are
        exhausted and capped_sampling_on_exhausted="best_available".
    shuffle_capped_sampling_order : bool
        If True, process SC rows in random order when max_reuse_per_st is set
        to reduce row-order bias.
    capped_sampling_on_exhausted : {"best_available", "raise"}
        Behavior when all candidate ST IDs for a row are already at capacity.
        "best_available" picks the highest-weight candidate even though it
        exceeds the cap. "raise" raises a ValueError.
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
        reuse_penalty=reuse_penalty,
    )

    if max_reuse_per_st is None:
        selected_idx = sample_match_indices(weights, rng=rng)
    else:
        selected_idx = sample_match_indices_with_reuse_cap(
            weights,
            match_ids=match_result["ids"],
            max_reuse_per_st=max_reuse_per_st,
            rng=rng,
            shuffle_order=shuffle_capped_sampling_order,
            on_exhausted=capped_sampling_on_exhausted,
        )
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
