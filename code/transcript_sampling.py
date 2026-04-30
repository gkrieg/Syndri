import numpy as np
from scipy.sparse import csr_matrix
from tqdm.notebook import tqdm


def sample_transcripts_sparse_row(
    source_indices,
    source_counts,
    query_nz_set,
    frac=0.05,
    rng=None,
    min_n=10,
):
    """
    Sample transcripts without replacement from a sparse source row,
    restricted to genes not expressed in the query cell.

    Parameters
    ----------
    source_indices : 1D np.ndarray
        Gene indices with nonzero counts in the source cell.
    source_counts : 1D np.ndarray
        Counts for those genes.
    query_nz_set : set[int]
        Genes expressed by the query cell.
    frac : float
        Fraction of eligible transcripts to sample.
    rng : np.random.Generator or None
    min_n : int
        Return empty if sampled transcript count would be < min_n.

    Returns
    -------
    sampled_gene_ids : np.ndarray
    sampled_counts : np.ndarray
        Sparse representation of sampled transcript vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    # keep only genes absent from the query cell
    keep = np.fromiter(
        (g not in query_nz_set for g in source_indices),
        dtype=bool,
        count=len(source_indices),
    )

    if not np.any(keep):
        return np.array([], dtype=int), np.array([], dtype=int)

    eligible_genes = source_indices[keep]
    eligible_counts = source_counts[keep].astype(np.int64, copy=False)

    total_eligible = int(eligible_counts.sum())
    n = int(round(frac * total_eligible))

    if total_eligible == 0 or n < min_n:
        return np.array([], dtype=int), np.array([], dtype=int)

    sampled = rng.multivariate_hypergeometric(eligible_counts, n)
    nz = sampled > 0
    return eligible_genes[nz], sampled[nz]


def sample_admixture_transcripts(
    adata,
    k_neighbors=3,
    frac=0.05,
    n_repeats=10,
    min_sampled_transcript_threshold=10,
):
    """
    Sample admixture transcripts for each query cell using preassigned neighbors
    recorded in ADM_ID_0, ..., ADM_ID_{k_neighbors-1}.

    For each query cell and each repeat:
      - read source neighbors from ADM_ID_* columns
      - sample transcripts from each source cell
      - only sample genes absent from the query cell
      - accumulate sampled counts into X_admixed[rep, query_idx, :]

    Parameters
    ----------
    adata : AnnData
        AnnData with count matrix in `.X` and neighbor IDs in `.obs[ADM_ID_*]`.
        ADM_ID values should match `adata.obs_names`.
    k_neighbors : int, default=3
        Number of ADM_ID columns to use: ADM_ID_0 ... ADM_ID_{k_neighbors-1}.
    frac : float, default=0.05
        Fraction of eligible transcripts to sample from each source cell.
    n_repeats : int, default=10
        Number of independent admixture replicates.
    min_sampled_transcript_threshold : int, default=10
        Minimum sampled transcript count required for a source to contribute.

    Returns
    -------
    X_admixed : np.ndarray
        Array of shape (n_repeats, n_cells, n_genes) containing sampled admixture counts.
    adata : AnnData
        Unmodified input AnnData object.
    """
    X = adata.X
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    rng = np.random.default_rng()

    n_cells, n_genes = X.shape
    X_admixed = np.zeros((n_repeats, n_cells, n_genes), dtype=np.float32)

    # verify ADM_ID columns exist
    adm_cols = [f"ADM_ID_{j}" for j in range(k_neighbors)]
    missing_cols = [c for c in adm_cols if c not in adata.obs.columns]
    if missing_cols:
        raise KeyError(f"Missing required neighbor columns in adata.obs: {missing_cols}")

    obs_names = np.asarray(adata.obs_names).astype(str)
    name_to_idx = {name: i for i, name in enumerate(obs_names)}

    indptr = X.indptr
    indices = X.indices
    data = X.data

    def get_row_sparse(i):
        s, e = indptr[i], indptr[i + 1]
        return indices[s:e], data[s:e]

    # precompute nonzero gene sets for query cells
    query_nz_sets = []
    for i in range(n_cells):
        row_idx, _ = get_row_sparse(i)
        query_nz_sets.append(set(row_idx))

    # resolve ADM_ID columns to source indices once
    # shape: (n_cells, k_neighbors), entries are source row indices or -1
    neighbor_idx_mat = np.full((n_cells, k_neighbors), -1, dtype=int)

    for j, col in enumerate(adm_cols):
        raw_ids = adata.obs[col].to_numpy()
        for i, val in enumerate(raw_ids):
            if val is None:
                continue
            if isinstance(val, float) and np.isnan(val):
                continue

            key = str(val)

            # exact obs_name match
            if key in name_to_idx:
                neighbor_idx_mat[i, j] = name_to_idx[key]
                continue

            # fallback: handle values like "<obs_name>_0", "<obs_name>_1", etc.
            # only strip trailing "_<digits>" if direct match failed
            if "_" in key:
                maybe_base, maybe_rep = key.rsplit("_", 1)
                if maybe_rep.isdigit() and maybe_base in name_to_idx:
                    neighbor_idx_mat[i, j] = name_to_idx[maybe_base]

    for rep in range(n_repeats):
        for query_idx in tqdm(range(n_cells), desc=f"rep {rep}"):
            query_nz_set = query_nz_sets[query_idx]
            admix_vec = np.zeros(n_genes, dtype=np.int32)

            for j in range(k_neighbors):
                surr_idx = neighbor_idx_mat[query_idx, j]
                if surr_idx < 0 or surr_idx == query_idx:
                    continue

                src_genes, src_counts = get_row_sparse(surr_idx)

                sampled_genes, sampled_counts = sample_transcripts_sparse_row(
                    src_genes,
                    src_counts,
                    query_nz_set=query_nz_set,
                    frac=frac,
                    rng=rng,
                    min_n=min_sampled_transcript_threshold,
                )

                if len(sampled_genes):
                    admix_vec[sampled_genes] += sampled_counts

            X_admixed[rep, query_idx] = admix_vec

    return X_admixed, adata