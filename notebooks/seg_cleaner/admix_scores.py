# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

from scipy.spatial import Delaunay
from scipy.stats import median_abs_deviation
from scipy import sparse

# ---------- Helpers ----------

def _mad_normal(x: np.ndarray) -> float:
    """
    R's mad(x) uses scale='normal' (≈1.4826 * median absolute deviation).
    """
    return float(median_abs_deviation(x, scale='normal', nan_policy='omit'))

def _ensure_csr(x) -> sparse.csr_matrix:
    if sparse.isspmatrix_csr(x):
        return x
    return sparse.csr_matrix(x)

def _cut_indices(vals: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Similar to R's cut(..., n_bins) → integer bin labels 1..n_bins.
    """
    bins = np.linspace(vals.min(), vals.max(), num=n_bins+1, endpoint=True)
    # right-closed bins like R default; map to 1..n_bins
    idx = np.clip(np.digitize(vals, bins[1:-1], right=True) + 1, 1, n_bins)
    return idx

# ---------- 1) estimate_cell_adjacency ----------

def estimate_cell_adjacency(
    df_spatial: pd.DataFrame,
    random_shift: float = 1e-3,
    edge_max_mad: float = 4.0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Python version of estimate_cell_adjacency().
    Required columns: ['x', 'y', 'cell_type', 'cell'].
    Returns a DataFrame with unique, undirected cell–cell edges:
        columns: ['cell_s','cell_e','cts','cte']
    """
    required = {'x', 'y', 'cell_type', 'cell'}
    if not required.issubset(df_spatial.columns):
        missing = required - set(df_spatial.columns)
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    # drop NA cell_type
    cdf = df_spatial.loc[df_spatial['cell_type'].notna()].copy()

    # jitter
    rng = np.random.default_rng()
    cdf['x'] = cdf['x'] + rng.uniform(-random_shift, random_shift, size=len(cdf))
    cdf['y'] = cdf['y'] + rng.uniform(-random_shift, random_shift, size=len(cdf))

    # grid split
    n_grid_cells = int(np.ceil(np.sqrt(n_jobs) * 2)) if n_jobs and n_jobs > 1 else 1
    if n_grid_cells > 1:
        gx = _cut_indices(cdf['x'].to_numpy(), n_grid_cells)
        gy = _cut_indices(cdf['y'].to_numpy(), n_grid_cells)
        cdf['grid_id'] = (gx - 1) * n_grid_cells + gy
    else:
        cdf['grid_id'] = 1

    def _edges_from_delaunay(block: pd.DataFrame) -> pd.DataFrame:
        if len(block) < 4:
            # need at least 3 points for a triangle; <4 makes degenerate edges likely
            return pd.DataFrame(columns=['xs','xe','ys','ye','cts','cte','cell_s','cell_e','dist'])

        pts = block[['x','y']].to_numpy()
        try:
            tri = Delaunay(pts)
        except Exception:
            # degenerate points, return empty
            return pd.DataFrame(columns=['xs','xe','ys','ye','cts','cte','cell_s','cell_e','dist'])

        simplices = tri.simplices  # (m, 3) int indices (local to block)
        e12 = simplices[:, [0, 1]]
        e13 = simplices[:, [0, 2]]
        e23 = simplices[:, [1, 2]]
        edges = np.vstack([e12, e13, e23])
        # undirected unique pairs
        vmin = edges.min(axis=1)
        vmax = edges.max(axis=1)
        edges = np.unique(np.stack([vmin, vmax], axis=1), axis=0)

        # build rows
        xs = pts[edges[:, 0], 0]
        ys = pts[edges[:, 0], 1]
        xe = pts[edges[:, 1], 0]
        ye = pts[edges[:, 1], 1]

        # map to metadata
        block_reset = block.reset_index(drop=True)
        cts = block_reset.loc[edges[:, 0], 'cell_type'].to_numpy()
        cte = block_reset.loc[edges[:, 1], 'cell_type'].to_numpy()
        cs  = block_reset.loc[edges[:, 0], 'cell'].to_numpy()
        ce  = block_reset.loc[edges[:, 1], 'cell'].to_numpy()

        dist = np.sqrt((xs - xe) ** 2 + (ys - ye) ** 2)

        return pd.DataFrame({
            'xs': xs, 'ys': ys, 'xe': xe, 'ye': ye,
            'cts': cts, 'cte': cte, 'cell_s': cs, 'cell_e': ce, 'dist': dist
        })

    # compute per grid
    groups = [g for _, g in cdf.groupby('grid_id') if len(g) > 3]
    if not groups:
        return pd.DataFrame(columns=['cell_s','cell_e','cts','cte'])

    if n_jobs and n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            parts = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_edges_from_delaunay)(g) for g in groups
            )
        except Exception:
            parts = [_edges_from_delaunay(g) for g in groups]
    else:
        parts = [_edges_from_delaunay(g) for g in groups]

    adj_df = pd.concat(parts, axis=0, ignore_index=True)
    if adj_df.empty:
        return pd.DataFrame(columns=['cell_s','cell_e','cts','cte'])

    # prune long edges by MAD rule (median + edge_max_mad * MAD_normal)
    med = float(np.median(adj_df['dist'].to_numpy()))
    mad = _mad_normal(adj_df['dist'].to_numpy())
    keep_mask = adj_df['dist'] < (med + edge_max_mad * mad)
    adj_df = adj_df.loc[keep_mask].copy()

    # keep unique cell-cell (exclude self), drop geometry cols
    adj_df = (
        adj_df.loc[adj_df['cell_s'] != adj_df['cell_e'], ['cell_s','cell_e','cts','cte']]
        .drop_duplicates()
    )

    # add reverse edges & unique
    rev = adj_df.rename(columns={'cell_s': 'cell_e', 'cell_e': 'cell_s', 'cts': 'cte', 'cte': 'cts'})
    adj_df = pd.concat([adj_df, rev], axis=0, ignore_index=True).drop_duplicates()

    return adj_df


# ---------- 2) estimate_cell_type_adjacency ----------

def estimate_cell_type_adjacency(cell_adj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Python version of estimate_cell_type_adjacency().
    Input columns: ['cell_s','cell_e','cts','cte'].
    Returns a square DataFrame (matrix) K with rows=cte, cols=cts (filled with 0 where NA).
    """
    # Count edges per (cell_s, cts, cte); then average counts per (cts, cte)
    # (This mirrors group_by(cell_s, cts, cte) %>% count() %>% group_by(cts, cte) %>% summarise(n=mean(n)))
    tmp = (
        cell_adj_df
        .groupby(['cell_s', 'cts', 'cte'], as_index=False)
        .size()
        .rename(columns={'size': 'n'})
        .groupby(['cts', 'cte'], as_index=False)['n']
        .mean()
    )
    K = tmp.pivot(index='cte', columns='cts', values='n').fillna(0.0)

    # Ensure same ordering for rows/cols (cte/cts share the same set of types)
    types = sorted(set(K.index).union(K.columns))
    K = K.reindex(index=types, columns=types, fill_value=0.0)
    return K


# ---------- 3) estimate_gene_prob_per_type ----------

def estimate_gene_prob_per_type(
    cm: Union[sparse.spmatrix, np.ndarray],
    cell_annot: pd.Series,
    expr_thres: float = 0.0,
    use_counts: bool = False,
) -> pd.DataFrame:
    """
    Python version of estimate_gene_prob_per_type().
    cm: genes x cells (sparse CSR recommended).
    cell_annot: Series indexed by cell-id (matching cm columns) mapping to cell_type.
    Returns DataFrame with index=cell_types, columns=genes (types x genes).
    """
    X = _ensure_csr(cm)
    # ensure cell order matches the annotation index order
    if hasattr(X, 'shape') and X.shape[1] != len(cell_annot):
        raise ValueError("Number of columns in cm must equal length of cell_annot.")

    # group cells by type
    types = cell_annot.values
    unique_types = pd.Index(sorted(np.unique(types)))
    # Build a mapping from type to column indices
    type_to_idx = {t: np.where(types == t)[0] for t in unique_types}

    out = []
    for t in unique_types:
        cols = type_to_idx[t]
        Xi = X[:, cols]
        if use_counts:
            # rowSums over those cols, then normalize by colSums across types (R does ./. colSums(.))
            rs = np.asarray(Xi.sum(axis=1)).ravel()  # genes
            out.append(rs)
        else:
            # fraction of cells with expression > expr_thres (rowMeans boolean)
            if expr_thres == 0:
                # fast path for > 0
                nnz_per_row = np.diff(Xi.indptr)  # number of non-zeros per row per column? (No, CSR is row-major; here Xi is genes x cells)
                # Above is not correct; we need per-row counts of nonzeros across all selected columns.
                # Simpler: (Xi > expr_thres).mean(axis=1)
            bool_mat = Xi.copy()
            bool_mat.data = (bool_mat.data > expr_thres).astype(np.float64)
            frac = np.asarray(bool_mat.mean(axis=1)).ravel()  # mean across columns
            out.append(frac)

    M = np.vstack(out)  # shape: n_types x n_genes (each row corresponds to a type)
    if use_counts:
        # divide each column by sum over types (colSums in R on probs matrix before transpose)
        colsum = M.sum(axis=0)
        # avoid division by zero
        colsum[colsum == 0] = 1.0
        M = M / colsum

    df = pd.DataFrame(M, index=unique_types)  # columns will be 0..n_genes-1
    # assign gene names if available (row names of cm)
    # We assume cm has an attribute .shape only; caller should wrap with DataFrame if wanting names.
    # If gene names exist separately, pass them after the call via df.columns = gene_names.
    return df


# ---------- 4) estimate_gene_scores ----------

def estimate_gene_scores(
    prob_gene_per_type: pd.DataFrame,  # types x genes
    cell_type_adj_mat: pd.DataFrame,   # rows & cols: types
    cell_type: str
) -> pd.DataFrame:
    """
    Returns DataFrame with index=genes, columns=['cont','ref'].
    cont = sum_{v != u} p(g|t_v) * (k_{v,u} / sum_{x != u} k_{x,u})
    ref  = p(g|t_u)
    """
    if cell_type not in cell_type_adj_mat.columns:
        raise KeyError(f"{cell_type} not found in adjacency matrix columns.")

    col = cell_type_adj_mat[cell_type]
    cont_fracs = col.drop(index=cell_type, errors='ignore')
    cont_fracs = cont_fracs / cont_fracs.sum()

    # Align rows (types) and columns (genes)
    common_types = cont_fracs.index.intersection(prob_gene_per_type.index)
    if len(common_types) == 0:
        raise ValueError("No overlapping types between adjacency and probability matrix.")

    # Weighted sum across contaminating types → per-gene contamination score
    # prob_gene_per_type: types x genes
    weighted = prob_gene_per_type.loc[common_types].multiply(cont_fracs.loc[common_types], axis=0)
    scores_cont = weighted.sum(axis=0)  # genes

    # ref = p(g|t_u)
    if cell_type not in prob_gene_per_type.index:
        raise KeyError(f"{cell_type} not found in prob_gene_per_type index.")
    scores_ref = prob_gene_per_type.loc[cell_type, scores_cont.index]

    return pd.DataFrame({'cont': scores_cont, 'ref': scores_ref})


# ---------- 5) estimate_gene_contamination_probabilities ----------

def estimate_gene_contamination_probabilities(
    gene_scores: pd.DataFrame,  # columns: cont, ref; index=genes
    p_c: Optional[float] = None,
    prior_cont_prob: Optional[float] = None
) -> pd.Series:
    """
    p(m_i ∈ C | g_i, ...) = p_cu * cont / (p_cu * cont + (1 - p_cu) * ref)
    If p_c is None → p_cu = 0.5; else p_cu = p_c * prior_cont_prob
    """
    if p_c is None:
        p_cu = 0.5
    else:
        if prior_cont_prob is None:
            raise ValueError("prior_cont_prob must be provided when p_c is not None.")
        p_cu = p_c * prior_cont_prob

    num = p_cu * gene_scores['cont'].to_numpy()
    den = num + (1.0 - p_cu) * gene_scores['ref'].to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.divide(num, den, where=(den != 0))
    out = np.nan_to_num(out, nan=0.0)
    return pd.Series(out, index=gene_scores.index)


# ---------- 6) estimate_cell_admixture_fractions ----------

def estimate_cell_admixture_fractions(
    gene_cont_probs: pd.Series,             # index=genes
    cell_type_cm: Union[sparse.spmatrix, np.ndarray],  # genes x cells
    mask: Optional[pd.Series] = None        # optional boolean Series indexed by gene
) -> pd.Series:
    """
    E[f_c | ...] for each cell in the given cell-type submatrix.
    Returns a Series indexed by column position (0..n_cells-1).
    """
    X = _ensure_csr(cell_type_cm)
    genes = np.arange(X.shape[0])

    # If we have named genes, align; otherwise assume 1:1 order
    # Expectation: caller aligns gene_cont_probs to cm rows beforehand.
    if mask is not None:
        # set prob=0 for genes where mask[gene]==False
        submask = mask.reindex(gene_cont_probs.index)
        to_zero = submask.index[~submask.fillna(False)]
        gene_cont_probs = gene_cont_probs.copy()
        gene_cont_probs.loc[to_zero] = 0.0

    # Map gene_cont_probs to row order; if not labeled, assume already aligned
    # If the Series has integer RangeIndex equal to row order, this is fine.
    if isinstance(gene_cont_probs.index, pd.RangeIndex) and len(gene_cont_probs) == X.shape[0]:
        w = gene_cont_probs.to_numpy()
    else:
        # If the caller has a mapping of row-name → gene, build it and align.
        # Here we assume gene_cont_probs.index order matches the cm rows order externally.
        w = gene_cont_probs.to_numpy()

    # Numerator: sum over genes of (counts * gene_prob)
    # Efficiently: scale rows by w, then sum across rows.
    # For CSR genes x cells: multiply each row i by w[i]
    Xw = X.multiply(w[:, None])  # genes x cells
    num = np.asarray(Xw.sum(axis=0)).ravel()
    den = np.asarray(X.sum(axis=0)).ravel()
    den[den == 0] = 1.0
    fracs = num / den
    return pd.Series(fracs)


# ---------- 7) mask_out_genes ----------

def mask_out_genes(
    cont_probs: pd.DataFrame,          # genes x types (columns are ct), values probabilities
    genes: Optional[Union[Iterable[str], Dict[str, Iterable[str]]]],
    all_name: str = 'all'
) -> pd.DataFrame:
    """
    Zero out specified genes globally or per-type.
    - If genes is None: no change
    - If genes is an iterable: zero out those rows across all columns
    - If genes is a dict: keys are type names (or 'all'), values are iterables of genes to zero
    """
    if genes is None:
        return cont_probs

    cp = cont_probs.copy()

    if not isinstance(genes, dict):
        # iterable → zero all these genes in all columns
        cols = list(set(cont_probs.columns).intersection(set(genes)))
        if cols:
            cp.loc[:, cols] = 0.0
        return cp

    # dict case
    for k, gs in genes.items():
        rows = list(set(cont_probs.index).intersection(set(gs)))
        if not rows:
            continue
        if k == all_name:
            cp.loc[rows, :] = 0.0
        else:
            if k in cp.columns:
                cp.loc[rows, k] = 0.0
    return cp


# ---------- 8) estimate_contamination_scores ----------

@dataclass
class ContaminationResult:
    cont_probs_per_type: pd.DataFrame   # genes x types
    doublet_scores: pd.Series           # concatenated per-type scores (same behavior as R: unlisted)


def estimate_contamination_scores(
    cm_rna: Union[sparse.spmatrix, np.ndarray],
    cm_spatial: Union[sparse.spmatrix, np.ndarray],
    annot_rna: pd.Series,            # cells → type
    annot_spatial: pd.Series,        # cells → type
    cell_type_adj_mat: pd.DataFrame, # rows/cols: types
    p_c: float = 0.25,
    signal_thres: float = 0.25,
    min_expr_frac: float = 0.05,
    max_prob_high: float = 0.1,
    exclude_cell_types: Optional[Iterable[str]] = None,
    exclude_genes: Optional[Union[Iterable[str], Dict[str, Iterable[str]]]] = None,
    use_counts: bool = False,
    adjust: bool = True,
    gene_names: Optional[List[str]] = None
) -> ContaminationResult:
    """
    Python version of estimate_contamination_scores().
    Returns:
        - cont_probs_per_type: DataFrame (genes x types)
        - doublet_scores: Series (concatenated per-type vectors)
    """
    X_rna = _ensure_csr(cm_rna)
    X_sp  = _ensure_csr(cm_spatial)

    # (Optional) set gene names, else use integer indices 0..n_genes-1
    if gene_names is None:
        gene_names = [str(i) for i in range(X_rna.shape[0])]

    # Intersect genes by row index; here we assume same ordering; if not, caller should reorder.
    if X_rna.shape[0] != X_sp.shape[0]:
        raise ValueError("cm_rna and cm_spatial must have same number/order of genes (rows).")

    # 1) gene probabilities per type (types x genes)
    prob_gene_per_type = estimate_gene_prob_per_type(
        X_rna, annot_rna, expr_thres=0.0, use_counts=use_counts
    )
    prob_gene_per_type.columns = gene_names

    # prior contamination probs: 1 - diag(K) / colSums(K)
    colsum = cell_type_adj_mat.sum(axis=0)
    diag = np.diag(cell_type_adj_mat.reindex(index=cell_type_adj_mat.index, columns=cell_type_adj_mat.columns, fill_value=0.0).to_numpy())
    prior_cont_probs = pd.Series(1.0 - (diag / colsum.to_numpy()), index=cell_type_adj_mat.columns)

    # expr.frac.mask: max over types per gene > min_expr_frac
    expr_frac_mask = (prob_gene_per_type.max(axis=0) > min_expr_frac)

    # high-expression matrix at threshold 1
    prob_gene_per_type_high = estimate_gene_prob_per_type(
        X_rna, annot_rna, expr_thres=1.0, use_counts=False
    )
    prob_gene_per_type_high.columns = gene_names

    # build contamination probabilities per type (genes x types)
    types_all = [t for t in cell_type_adj_mat.columns]
    if exclude_cell_types is not None:
        types_use = [t for t in types_all if t not in set(exclude_cell_types)]
    else:
        types_use = types_all

    cont_cols = {}
    for ct in types_use:
        gs = estimate_gene_scores(prob_gene_per_type, cell_type_adj_mat, cell_type=ct)
        cont = estimate_gene_contamination_probabilities(gs, p_c=p_c, prior_cont_prob=prior_cont_probs[ct])
        # signal & expression mask
        cont.loc[~((cont > signal_thres) & expr_frac_mask.loc[cont.index])] = 0.0
        cont_cols[ct] = cont

    cont_probs_per_type = pd.DataFrame(cont_cols)  # genes x types (columns present only for types_use)
    # include absent types as empty columns if needed for shape consistency
    for t in types_all:
        if t not in cont_probs_per_type.columns:
            cont_probs_per_type[t] = 0.0
    cont_probs_per_type = cont_probs_per_type[types_all]  # reorder

    # mask out genes globally or per-type
    cont_probs_per_type = mask_out_genes(cont_probs_per_type, exclude_genes)

    # compute null fractions per type (cont.fracs.null)
    cont_fracs_null = {}
    for ct in cont_probs_per_type.columns:
        p = prob_gene_per_type.loc[ct, :]
        # sum(cont_probs * p) / sum(p)
        denom = p.sum()
        denom = denom if denom > 0 else 1.0
        cont_fracs_null[ct] = float((cont_probs_per_type[ct] * p).sum() / denom)
    cont_fracs_null = pd.Series(cont_fracs_null)

    # score each type against the spatial cells of that type
    # (warn on genes highly expressed in reference type among contamination set)
    doubles = []
    for ct in cont_probs_per_type.columns:
        # subset spatial columns for this type
        ct_cols = np.where(annot_spatial.values == ct)[0]
        if ct_cols.size == 0:
            continue
        cm_ct = X_sp[:, ct_cols]

        cont_probs = cont_probs_per_type[ct]
        cont_genes = cont_probs.index[cont_probs.values > signal_thres]

        if len(cont_genes) > 0:
            high_mask = (prob_gene_per_type_high.loc[ct, cont_genes] > max_prob_high)
            if np.any(high_mask):
                bad = list(cont_genes[high_mask.values])
                # replicate R's warning (here: print)
                print(
                    f"Warning: Genes with high expression in '{ct}' detected in contamination: "
                    + ", ".join([repr(g) for g in bad])
                    + ". Consider improving annotation or masking these genes."
                )

        scores = estimate_cell_admixture_fractions(cont_probs, cm_ct)
        if adjust:
            scores = np.maximum(scores - cont_fracs_null[ct], 0.0)
        # keep as a Series; when concatenated, this mirrors R's unlist() behavior
        doubles.append(pd.Series(scores.values, index=[(ct, i) for i in range(len(scores))]))

    doublet_scores = pd.concat(doubles) if doubles else pd.Series(dtype=float)

    return ContaminationResult(
        cont_probs_per_type=cont_probs_per_type,
        doublet_scores=doublet_scores
    )


# ---------- 9) sparse_cor (column-wise) ----------

def sparse_cor(x: Union[sparse.spmatrix, np.ndarray]) -> np.ndarray:
    """
    Python version of sparse_cor() for a genes x cells matrix x.
    Returns the correlation matrix between columns (cells x cells), as dense ndarray.
    """
    X = _ensure_csr(x)
    n = X.shape[0]

    # column means
    col_means = np.asarray(X.mean(axis=0)).ravel()

    # crossprod and cov
    XtX = (X.T @ X).astype(np.float64)  # cells x cells
    # cov = (XtX - 2 * (cmean %o% colSums) + n * (cmean %o% cmean)) / (n-1)
    col_sums = np.asarray(X.sum(axis=0)).ravel()
    # outer helpers
    outer_cmean_colsums = np.outer(col_means, col_sums)
    outer_cmean_cmean   = np.outer(col_means, col_means)

    cov = (XtX.toarray()
           - 2.0 * outer_cmean_colsums
           + n * outer_cmean_cmean) / max(n - 1, 1)

    sd = np.sqrt(np.diag(cov))
    sd[sd == 0] = np.nan
    denom = np.outer(sd, sd)
    cor = cov / denom
    return cor


# ---------- 10) estimate_correlation_preservation ----------

def estimate_correlation_preservation(
    cm1: Union[sparse.spmatrix, np.ndarray],
    cm2: Union[sparse.spmatrix, np.ndarray],
    gene_names: Optional[List[str]] = None
) -> pd.Series:
    """
    Python version of estimate_correlation_preservation().
    For common genes with non-zero rows in both matrices:
      compute column-vs-column correlation matrices (cells x cells) per dataset,
      zero-out diagonals, and return per-gene (interpreted as 'per-column') correlation
      of corresponding columns across the two correlation matrices.
    NOTE: The original R code names the output with `common.genes`, but it
          actually loops over columns of the correlation matrices. Here, we
          keep the same behavior but name entries 0..ncols-1 unless provided.
    """
    X1 = _ensure_csr(cm1)
    X2 = _ensure_csr(cm2)

    if X1.shape[0] != X2.shape[0]:
        raise ValueError("cm1 and cm2 must have the same number/order of genes (rows).")

    # filter genes that are all-zero in either matrix
    nz1 = np.ravel(np.asarray(X1.sum(axis=1))) > 0
    nz2 = np.ravel(np.asarray(X2.sum(axis=1))) > 0
    keep_rows = nz1 & nz2
    if not np.any(keep_rows):
        return pd.Series(dtype=float)

    X1f = X1[keep_rows, :]
    X2f = X2[keep_rows, :]

    cors1 = sparse_cor(X1f.T)  # (cells x cells); but R did t(cm) first, then sparse_cor → same
    cors2 = sparse_cor(X2f.T)

    # remove self-correlations
    np.fill_diagonal(cors1, np.nan)
    np.fill_diagonal(cors2, np.nan)

    # per-column correlation, using pairwise-complete observations
    n_cols = cors1.shape[1]
    out = np.empty(n_cols, dtype=float)
    for i in range(n_cols):
        a = cors1[:, i]
        b = cors2[:, i]
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() < 2:
            out[i] = np.nan
        else:
            out[i] = np.corrcoef(a[mask], b[mask])[0, 1]

    if gene_names is None:
        idx = pd.RangeIndex(n_cols)
    else:
        # keep length-consistency if caller supplied names
        idx = pd.Index(gene_names[:n_cols])
    return pd.Series(out, index=idx)
