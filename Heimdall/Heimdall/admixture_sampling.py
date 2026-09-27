import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import warnings


def validate_precomputed_neighbor_dict(
    adata,
    uns_key="neighbor_dict",
    id_col=None,
    max_neighbors=None,
):
    """Validate a user-supplied neighbor dictionary without changing it.

    Precomputed dictionaries may contain only the query cells that will be
    inferred, but every query and neighbor ID they contain must resolve to a
    row in ``adata``.
    """
    if uns_key not in adata.uns:
        raise KeyError(
            f"use_precomputed_neighbor=True requires adata.uns[{uns_key!r}]",
        )

    neighbor_dict = adata.uns[uns_key]
    if not isinstance(neighbor_dict, dict) or not neighbor_dict:
        raise ValueError(
            f"adata.uns[{uns_key!r}] must be a non-empty dictionary when "
            "use_precomputed_neighbor=True",
        )

    if id_col is None:
        valid_ids = set(adata.obs_names.astype(str))
        id_name = "obs_names"
    else:
        if id_col not in adata.obs:
            raise KeyError(f"{id_col!r} not found in adata.obs")
        ids = adata.obs[id_col].astype(str)
        if ids.duplicated().any():
            raise ValueError(f"Values in {id_col!r} must be unique.")
        valid_ids = set(ids)
        id_name = id_col

    referenced_ids = set()
    for query_id, neighbor_ids in neighbor_dict.items():
        query_id = str(query_id)
        referenced_ids.add(query_id)
        if isinstance(neighbor_ids, (str, bytes)) or not hasattr(neighbor_ids, "__iter__"):
            raise TypeError(
                f"Neighbors for query {query_id!r} must be a sequence of cell IDs, "
                f"got {type(neighbor_ids).__name__}.",
            )
        if max_neighbors is not None and len(neighbor_ids) > max_neighbors:
            raise ValueError(
                f"Query {query_id!r} has {len(neighbor_ids)} precomputed neighbors, "
                f"but k_neighbors={max_neighbors}.",
            )
        referenced_ids.update(str(neighbor_id) for neighbor_id in neighbor_ids)

    missing_ids = sorted(referenced_ids - valid_ids)
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} IDs from {uns_key} were not found in adata.{id_name}. "
            f"First 20 missing IDs: {missing_ids[:20]}",
        )

    return neighbor_dict


def validate_precomputed_neighbor_celltype_dict(
    adata,
    uns_key="neighbor_celltype_dict",
    max_neighbors=None,
):
    """Validate precomputed neighbor cell types for query-only AnnData files."""
    if uns_key not in adata.uns:
        raise KeyError(
            "use_precomputed_neighbor=True requires either "
            "adata.uns['neighbor_dict'] or adata.uns['neighbor_celltype_dict']",
        )

    neighbor_celltype_dict = adata.uns[uns_key]
    if not isinstance(neighbor_celltype_dict, dict) or not neighbor_celltype_dict:
        raise ValueError(f"adata.uns[{uns_key!r}] must be a non-empty dictionary")

    valid_query_ids = set(adata.obs_names.astype(str))
    missing_queries = sorted(set(map(str, neighbor_celltype_dict)) - valid_query_ids)
    if missing_queries:
        raise ValueError(
            f"{len(missing_queries)} query IDs from {uns_key} were not found in adata.obs_names. "
            f"First 20 missing IDs: {missing_queries[:20]}",
        )

    known_cell_types = set(map(str, adata.uns.get("cell_type_dict", {})))
    for query_id, neighbor_types in neighbor_celltype_dict.items():
        if isinstance(neighbor_types, (str, bytes)) or not hasattr(neighbor_types, "__iter__"):
            raise TypeError(f"Neighbor cell types for query {query_id!r} must be a sequence.")
        if max_neighbors is not None and len(neighbor_types) > max_neighbors:
            raise ValueError(
                f"Query {query_id!r} has {len(neighbor_types)} precomputed neighbors, "
                f"but k_neighbors={max_neighbors}.",
            )
        if known_cell_types:
            unknown = sorted(set(map(str, neighbor_types)) - known_cell_types)
            if unknown:
                raise ValueError(
                    f"Query {query_id!r} contains cell types absent from cell_type_dict: {unknown[:20]}",
                )

    return neighbor_celltype_dict






def build_neighbor_dict(
    adata,
    k_neighbors=3,
    coord_cols=("sampled_st_x", "sampled_st_y"),
    tissue_col="harmony_spatial_tissue",
    cell_type_col="cell_type",
    min_source_counts=400,
    use_same_cell_type_neighbors=False,
    id_col=None,
    uns_key="neighbor_dict",
    initial_query_k=None,
    query_k_growth=2,
    batch_size=2048,
):
    """
    Build a neighbor dictionary and store it in `adata.uns[uns_key]`:

        adata.uns[uns_key] = {
            query_id: [neighbor_id_1, ..., neighbor_id_m]
        }

    where m <= k_neighbors if fewer than k eligible neighbors exist.

    Neighbors are the true nearest Euclidean neighbors in coordinate space
    satisfying:
      - same tissue, if tissue_col is provided
      - different cell type, unless use_same_cell_type_neighbors is True
      - not self
      - source total counts > min_source_counts

    This version fits one nearest-neighbor index per tissue and does batched
    neighbor queries without repeated refitting.

    Parameters
    ----------
    adata : AnnData
    k_neighbors : int
    coord_cols : tuple[str, str]
    tissue_col : str or None
        If None, all cells with valid coordinates are treated as one tissue.
    cell_type_col : str
    min_source_counts : int
    use_same_cell_type_neighbors : bool
        If True, allow source neighbors with the same cell type as the query
        cell. If False, same-cell-type source neighbors are skipped.
    id_col : str or None
        If None, use adata.obs_names. Otherwise use adata.obs[id_col].
    uns_key : str
    initial_query_k : int or None
        Initial number of candidate neighbors to request. If None, uses
        max(32, 8 * k_neighbors).
    query_k_growth : int
        Multiplicative growth factor for candidate expansion.
    batch_size : int
        Number of query cells to process together.

    Returns
    -------
    adata : AnnData
        Copy with neighbor dict stored in `.uns[uns_key]`.
    """
    adata = adata.copy()
    use_same_cell_type_neighbors = bool(use_same_cell_type_neighbors)

    xcol, ycol = coord_cols
    tissue_col = tissue_col or None
    required_obs_cols = [xcol, ycol, cell_type_col]
    if tissue_col is not None:
        required_obs_cols.append(tissue_col)
    for col in required_obs_cols:
        if col not in adata.obs:
            raise KeyError(f"{col!r} not found in adata.obs")

    if id_col is None:
        all_ids = adata.obs_names.astype(str).to_numpy()
        id_name = "obs_names"
    else:
        if id_col not in adata.obs:
            raise KeyError(f"{id_col!r} not found in adata.obs")
        all_ids = adata.obs[id_col].astype(str).to_numpy()
        id_name = id_col

    if pd.Series(all_ids).duplicated().any():
        raise ValueError(f"Values in {id_name!r} must be unique.")

    coords_df = adata.obs[[xcol, ycol]]
    valid_coord_mask = coords_df.notna().all(axis=1).to_numpy()
    if valid_coord_mask.sum() == 0:
        raise ValueError(f"No rows have valid coordinates in {coord_cols}.")

    row_sums = np.asarray(adata.X.sum(axis=1)).ravel()
    if tissue_col is None:
        tissues = np.full(adata.n_obs, "all_cells", dtype=object)
    else:
        tissues = adata.obs[tissue_col].astype(str).to_numpy()
    cell_types = adata.obs[cell_type_col].astype(str).to_numpy()

    if initial_query_k is None:
        initial_query_k = max(32, 8 * k_neighbors)

    neighbor_dict = {}
    insufficient_count = 0

    for i in np.where(valid_coord_mask)[0]:
        neighbor_dict[all_ids[i]] = []

    for tissue in np.unique(tissues[valid_coord_mask]):
        tissue_mask = valid_coord_mask & (tissues == tissue)
        tissue_idx = np.where(tissue_mask)[0]
        if len(tissue_idx) == 0:
            continue

        tissue_coords = adata.obs.iloc[tissue_idx][[xcol, ycol]].to_numpy(dtype=float)
        tissue_cell_types = cell_types[tissue_idx]
        tissue_row_sums = row_sums[tissue_idx]
        tissue_ids = all_ids[tissue_idx]
        n_in_tissue = len(tissue_idx)

        nn = NearestNeighbors(metric="euclidean")
        nn.fit(tissue_coords)

        for start in range(0, n_in_tissue, batch_size):
            stop = min(start + batch_size, n_in_tissue)
            batch_local_idx = np.arange(start, stop)
            batch_coords = tissue_coords[batch_local_idx]

            chosen_lists = [[] for _ in range(len(batch_local_idx))]
            done = np.zeros(len(batch_local_idx), dtype=bool)

            query_k = min(n_in_tissue, max(1, initial_query_k))

            while True:
                _, inds = nn.kneighbors(batch_coords, n_neighbors=query_k)

                for b, local_i in enumerate(batch_local_idx):
                    if done[b]:
                        continue

                    query_ct = tissue_cell_types[local_i]
                    chosen = []

                    for local_j in inds[b]:
                        if local_j == local_i:
                            continue
                        if not use_same_cell_type_neighbors and tissue_cell_types[local_j] == query_ct:
                            continue
                        if tissue_row_sums[local_j] <= min_source_counts:
                            continue

                        chosen.append(tissue_ids[local_j])

                        if len(chosen) == k_neighbors:
                            break

                    chosen_lists[b] = chosen
                    if len(chosen) >= k_neighbors or query_k == n_in_tissue:
                        done[b] = True

                if done.all():
                    break

                if query_k == n_in_tissue:
                    break

                query_k = min(n_in_tissue, max(query_k + 1, query_k * query_k_growth))

            for b, local_i in enumerate(batch_local_idx):
                query_id = tissue_ids[local_i]
                neighbor_dict[query_id] = chosen_lists[b]
                if len(chosen_lists[b]) < k_neighbors:
                    insufficient_count += 1

    if insufficient_count > 0:
        warnings.warn(
            f"{insufficient_count} cells had fewer than {k_neighbors} eligible neighbors.",
            stacklevel=2,
        )

    adata.uns[uns_key] = neighbor_dict
    adata.uns[f"{uns_key}_config"] = {
        "k_neighbors": int(k_neighbors),
        "coord_cols": list(coord_cols),
        "tissue_col": tissue_col,
        "cell_type_col": cell_type_col,
        "min_source_counts": int(min_source_counts),
        "use_same_cell_type_neighbors": use_same_cell_type_neighbors,
    }
    return adata

def build_neighbor_celltype_dict(
    adata,
    uns_key="neighbor_dict",
    id_col=None,
    tissue_col=None,
    cell_type_col="cell_type",
    output_key="neighbor_celltype_dict",
):
    """
    Build a dictionary mapping each query cell_ID to a list of neighbor cell types.
    Also validates that all IDs in neighbor_dict exist in adata.

    Parameters
    ----------
    adata : AnnData
    uns_key : str
        Key in adata.uns containing neighbor dict (cell_ID -> list of neighbor IDs)
    id_col : str or None
        Column in adata.obs containing cell IDs. If None, use adata.obs.index.
    tissue_col : str or None
        Accepted for API symmetry with build_neighbor_dict. This function does
        not use tissue information.
    cell_type_col : str
        Column in adata.obs containing cell types
    output_key : str
        Key to store the resulting dictionary in adata.uns

    Returns
    -------
    dict
        Mapping: query_cell_id -> list of neighbor cell types
    """
    if uns_key not in adata.uns:
        raise KeyError(f"{uns_key} not found in adata.uns")

    neighbor_dict = adata.uns[uns_key]

    # Build ID -> cell_type mapping
    if id_col is None:
        id_series = adata.obs.index
    else:
        id_series = adata.obs[id_col]

    id_to_type = dict(zip(id_series, adata.obs[cell_type_col]))

    # ----------------------------
    # Validation: check all IDs exist
    # ----------------------------
    all_ids_in_dict = set(neighbor_dict.keys())
    for neighbor_ids in neighbor_dict.values():
        all_ids_in_dict.update(neighbor_ids)

    missing_ids = all_ids_in_dict - set(id_to_type.keys())

    if missing_ids:
        missing_preview = list(missing_ids)[:20]
        raise ValueError(
            f"{len(missing_ids)} IDs from neighbor_dict not found in adata ({'obs.index' if id_col is None else id_col}). "
            f"First 20 missing IDs: {missing_preview}"
        )

    # ----------------------------
    # Build output
    # ----------------------------
    neighbor_celltype_dict = {}

    for query_id, neighbor_ids in neighbor_dict.items():
        neighbor_types = [id_to_type[n_id] for n_id in neighbor_ids]
        neighbor_celltype_dict[query_id] = neighbor_types

    adata.uns[output_key] = neighbor_celltype_dict

    return adata


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
    """
    if rng is None:
        rng = np.random.default_rng()

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

def sample_tau_admixture_from_neighbors(
    neighbor_gene_counts,
    query_total,
    tau,
    rng=None,
    min_n=10,
):
    """
    Select whole admixed genes from pooled neighbor counts until

        admixed_total / (admixed_total + query_total) >= tau

    This implies:

        target added count = tau / (1 - tau) * query_total

    The returned fraction may exceed tau because counts are added one whole
    gene at a time. If all eligible genes cannot reach target, use all genes.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(neighbor_gene_counts) == 0 or query_total <= 0 or tau <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if tau >= 1:
        raise ValueError("tau must be less than 1.")

    genes = np.fromiter(neighbor_gene_counts.keys(), dtype=int)
    counts = np.fromiter(neighbor_gene_counts.values(), dtype=np.int64)

    total_available = int(counts.sum())
    target_n = int(round((tau / (1.0 - tau)) * query_total))

    if target_n < min_n:
        return np.array([], dtype=int), np.array([], dtype=int)

    if total_available <= target_n:
        return genes, counts

    order = rng.permutation(len(genes))
    genes = genes[order]
    counts = counts[order]

    csum = np.cumsum(counts)
    k = np.searchsorted(csum, target_n, side="left") + 1

    return genes[:k], counts[:k]

def sample_admixture_from_neighbor_dict(
    adata,
    tau_range=(0.0,0.15),
    n_repeats=1,
    min_sampled_transcript_threshold=10,
    uns_key="neighbor_dict",
    id_col=None,
    only_sample_unexpressed_genes=True,
    mask_out_random_genes=False,
    mask_out_frac=0.0,
    rng=None,
    return_sparse=False,
):
    """
    Sample admixture transcripts for each query cell using neighbors stored in
    `adata.uns[uns_key]`.

    Expected format:
        adata.uns[uns_key] = {
            query_cell_id_long: [neighbor_cell_id_long_1, neighbor_cell_id_long_2, ...]
        }

    Parameters
    ----------
    adata : AnnData
    frac : float
        Fraction of eligible transcripts to sample from each neighbor.
    n_repeats : int
    min_sampled_transcript_threshold : int
    uns_key : str
    id_col : str or None
    only_sample_unexpressed_genes : bool, default=True
        If True, only sample neighbor transcripts for genes not expressed in
        the query cell. If False, all source-neighbor expressed genes are
        eligible.
    mask_out_random_genes : bool, default=False
        If True, randomly zero out `mask_out_frac` fraction of admixed genes
        per cell per repeat, considering only genes with nonzero admixture.
    mask_out_frac : float, default=0.0
        Fraction of nonzero admixed genes to zero out per cell per repeat.
        Must be in [0, 1].
    rng : np.random.Generator or None
        Random number generator used for transcript sampling and masking.

    Returns
    -------
    X_admixed : np.ndarray
        Shape (n_repeats, n_cells, n_genes)
    adata : AnnData
    """
    if uns_key not in adata.uns:
        raise KeyError(f"{uns_key!r} not found in adata.uns")

    neighbor_dict = adata.uns[uns_key]
    if not isinstance(neighbor_dict, dict):
        raise TypeError(f"adata.uns[{uns_key!r}] must be a dict")

    if not (0 <= mask_out_frac <= 1):
        raise ValueError("mask_out_frac must be between 0 and 1.")
    only_sample_unexpressed_genes = bool(only_sample_unexpressed_genes)

    X = adata.X
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    if id_col is None:
        all_ids = np.asarray(adata.obs_names).astype(str)
        id_name = "obs_names"
    else:
        if id_col not in adata.obs:
            raise KeyError(f"{id_col!r} not found in adata.obs")
        all_ids = adata.obs[id_col].astype(str).to_numpy()
        id_name = id_col

    if pd.Series(all_ids).duplicated().any():
        raise ValueError(f"Values in {id_name!r} must be unique.")

    id_to_idx = {cell_id: i for i, cell_id in enumerate(all_ids)}

    if rng is None:
        rng = np.random.default_rng()

    n_cells, n_genes = X.shape
    if return_sparse and n_repeats != 1:
        raise ValueError("return_sparse=True currently supports n_repeats=1 only.")

    if not return_sparse:
        X_admixed = np.zeros((n_repeats, n_cells, n_genes), dtype=np.float32)
    else:
        sparse_rows = []
        sparse_cols = []
        sparse_data = []

    indptr = X.indptr
    indices = X.indices
    data = X.data
    row_sums = np.asarray(X.sum(axis=1)).ravel()

    def get_row_sparse(i):
        s, e = indptr[i], indptr[i + 1]
        return indices[s:e], data[s:e]

    query_nz_sets = None
    if only_sample_unexpressed_genes:
        query_nz_sets = []
        for i in range(n_cells):
            row_idx, _ = get_row_sparse(i)
            query_nz_sets.append(set(row_idx))

    # resolve neighbor IDs to row indices once
    neighbor_idx_lists = []
    for query_id in all_ids:
        raw_neighbors = neighbor_dict.get(query_id, [])
        resolved = []

        for nbr_id in raw_neighbors:
            key = str(nbr_id)
            if key in id_to_idx:
                resolved.append(id_to_idx[key])

        neighbor_idx_lists.append(resolved)

    for rep in range(n_repeats):
        print('beginning admixture sampling...')
        for query_idx in range(n_cells):
            query_nz_set = query_nz_sets[query_idx] if only_sample_unexpressed_genes else None
            query_total = int(row_sums[query_idx])
            neighbor_gene_counts = {}

            for surr_idx in neighbor_idx_lists[query_idx]:
                if surr_idx == query_idx:
                    continue

                src_genes, src_counts = get_row_sparse(surr_idx)

                if only_sample_unexpressed_genes:
                    keep = np.fromiter(
                        (g not in query_nz_set for g in src_genes),
                        dtype=bool,
                        count=len(src_genes),
                    )
                else:
                    keep = np.ones(len(src_genes), dtype=bool)

                if not np.any(keep):
                    continue

                for g, c in zip(src_genes[keep], src_counts[keep]):
                    neighbor_gene_counts[int(g)] = neighbor_gene_counts.get(int(g), 0) + int(c)

            tau = rng.uniform(tau_range[0], tau_range[1])

            sampled_genes, sampled_counts = sample_tau_admixture_from_neighbors(
                neighbor_gene_counts,
                query_total=query_total,
                tau=tau,
                rng=rng,
                min_n=min_sampled_transcript_threshold,
            )

            if len(sampled_genes):
                if mask_out_random_genes:
                    n_mask = int(round(mask_out_frac * len(sampled_genes)))
                    if n_mask > 0:
                        keep_mask = np.ones(len(sampled_genes), dtype=bool)
                        mask_idx = rng.choice(len(sampled_genes), size=n_mask, replace=False)
                        keep_mask[mask_idx] = False
                        sampled_genes = sampled_genes[keep_mask]
                        sampled_counts = sampled_counts[keep_mask]

            if return_sparse:
                if len(sampled_genes):
                    sparse_rows.extend([query_idx] * len(sampled_genes))
                    sparse_cols.extend(sampled_genes.tolist())
                    sparse_data.extend(sampled_counts.astype(np.float32).tolist())
            else:
                admix_vec = np.zeros(n_genes, dtype=np.int32)
                if len(sampled_genes):
                    admix_vec[sampled_genes] += sampled_counts
                X_admixed[rep, query_idx] = admix_vec
        print('admixture sampled!')

    if return_sparse:
        return csr_matrix((sparse_data, (sparse_rows, sparse_cols)), shape=(n_cells, n_genes), dtype=np.float32), adata

    if n_repeats == 1:
        return X_admixed[0], adata
    return X_admixed, adata
