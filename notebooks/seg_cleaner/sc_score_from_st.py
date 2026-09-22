from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Dict, Iterable, Optional, Union, Tuple, List
from admix_scores import *
# Assumes you already have these functions from your port:
# - estimate_gene_prob_per_type
# - estimate_gene_scores
# - estimate_gene_contamination_probabilities
# - estimate_cell_admixture_fractions
# - mask_out_genes
# (or you can keep using estimate_contamination_scores, but this wrapper avoids tuple-index outputs.)

def _ensure_csr(x) -> sparse.csr_matrix:
    return x if sparse.isspmatrix_csr(x) else sparse.csr_matrix(x)

def _get_counts_matrix_from_adata(
    adata,
    layer: Optional[str] = None,
    use_raw: bool = False,
) -> sparse.csr_matrix:
    """
    Return genes x cells raw counts as CSR.
    - If use_raw=True, uses adata.raw.X (and adata.raw.var_names).
    - Else if layer provided, uses adata.layers[layer].
    - Else uses adata.X.
    """
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None.")
        X = adata.raw.X
    else:
        X = adata.layers[layer] if layer is not None else adata.X

    X = _ensure_csr(X)
    # AnnData stores cells x genes, but our scoring expects genes x cells
    return X.T.tocsr()

def _var_names_from_adata(adata, use_raw: bool = False) -> List[str]:
    return list(adata.raw.var_names) if use_raw else list(adata.var_names)

def _obs_names_from_adata(adata) -> List[str]:
    return list(adata.obs_names)

def get_cell_type_specific_genes(
    prob_gene_per_type: pd.DataFrame,  # types x genes
    *,
    min_expr_frac: float = 0.05,
    specificity_ratio: float = 5.0,
    top_n: Optional[int] = None,
) -> Dict[str, List[str]]:
    # types x genes -> genes x types
    P = prob_gene_per_type.T

    # top1 and top2 per gene
    top1_type = P.idxmax(axis=1)
    top1_val = P.max(axis=1)

    # second best: mask the max then take max again
    P2 = P.copy()
    P2.values[np.arange(P2.shape[0]), P.values.argmax(axis=1)] = -np.inf
    top2_val = P2.max(axis=1)

    ratio = top1_val / np.maximum(top2_val, 1e-12)
    ok = (top1_val >= min_expr_frac) & (ratio >= specificity_ratio)

    specific = {}
    for ct in prob_gene_per_type.index:
        genes = P.index[ok & (top1_type == ct)].tolist()
        if top_n is not None:
            # rank within ct by top1_val (desc)
            genes = (
                pd.Series(top1_val.loc[genes])
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
        specific[ct] = genes
    return specific

def get_scrna_admixture_scores_from_spatial_K(
    adata, # scRNA,
    spatial_adata, # spatial
    cell_type_adj_mat: pd.DataFrame,
    *,
    cell_type_key: str = "cell_type",
    layer: Optional[str] = None,          # e.g., "counts" if you stored counts in a layer
    use_raw: bool = False,                # use adata.raw.X / adata.raw.var_names if True
    spatial_layer: Optional[str] = None,  # e.g., "counts" if you stored counts in a layer for spatial data
    spatial_use_raw: bool = False,        # use spatial_adata.raw.X / spatial_adata.raw.var_names if True
    spatial_cell_type_key: str = "cell_type",  # key in spatial_adata.obs for cell types
    # score params (mirrors your R list order)
    signal_thres: float = 0.25,
    exclude_genes: Optional[Union[Iterable[str], Dict[str, Iterable[str]]]] = None,
    p_c: float = 0.25,
    min_expr_frac_specific: float = 0.1,
    min_expr_frac_contam: float = 0.05,
    neighbor_top_k: int = 3,   # or use a weight threshold instead
    exclude_cell_types: Optional[Iterable[str]] = None,
    max_prob_high: float = 0.1,
    use_counts: bool = False,
    adjust: bool = True,
    intersect_with_spatial_genes: bool = True,
    return_gene_sets: bool = False,
) -> pd.DataFrame:
    """
    Compute count-weighted admixture scores on spatial_adata using reference
    detection frequencies from adata and the spatial adjacency matrix K.

    This is equivalent in spirit to:
      estimate_contamination_scores_seurat(so_rna, so_rna, cell.type.adj.mat=K, ...)

    Returns a DataFrame with columns:
      ['cell', 'cell_type', 'scores', 'dataset']
    """
    # ---- pull scRNA counts + names ----
    X_sc = _get_counts_matrix_from_adata(adata, layer=layer, use_raw=use_raw)  # genes x cells
    sc_gene_names = _var_names_from_adata(adata, use_raw=use_raw)
    sc_cell_names = _obs_names_from_adata(adata)

    # ---- pull spatial counts + names ----
    X_sp = _get_counts_matrix_from_adata(spatial_adata, layer=spatial_layer, use_raw=spatial_use_raw)  # genes x spatial_obs
    sp_gene_names = _var_names_from_adata(spatial_adata, use_raw=spatial_use_raw)
    sp_obs_names  = _obs_names_from_adata(spatial_adata)

    if cell_type_key not in adata.obs.columns:
        raise KeyError(f"{cell_type_key!r} not found in adata.obs columns.")
    annot = pd.Series(adata.obs[cell_type_key].values, index=sc_cell_names, name="cell_type")

    # ---- optionally intersect genes with spatial transcript universe ----
    if intersect_with_spatial_genes:
        common = pd.Index(sc_gene_names).intersection(pd.Index(sp_gene_names))
        if len(common) == 0:
            raise ValueError("No overlapping genes between scRNA adata and spatial_adata.")

        # keep scRNA order
        keep = [g for g in sc_gene_names if g in set(common)]

        # subset scRNA
        sc_g2i = {g: i for i, g in enumerate(sc_gene_names)}
        X_sc = X_sc[np.array([sc_g2i[g] for g in keep], dtype=int), :]

        # subset spatial to same gene order
        sp_g2i = {g: i for i, g in enumerate(sp_gene_names)}
        X_sp = X_sp[np.array([sp_g2i[g] for g in keep], dtype=int), :]

        gene_names = keep
    else:
        if sc_gene_names != sp_gene_names:
            raise ValueError("Reference and spatial genes must match in order.")
        gene_names = sc_gene_names

    # ---- drop cells with missing types ----
    good_cells = annot.notna()
    if not bool(good_cells.all()):
        keep_cells = np.where(good_cells.values)[0]
        X_sc = X_sc[:, keep_cells]
        cell_names = [sc_cell_names[i] for i in keep_cells]
        annot = annot.loc[cell_names]

    # ---- choose which types to score ----
    types_all = list(cell_type_adj_mat.columns)
    if exclude_cell_types is not None:
        excl = set(exclude_cell_types)
        types_use = [t for t in types_all if t not in excl]
    else:
        types_use = types_all

    # ---- compute gene probabilities per type from scRNA ----
    # estimate_gene_prob_per_type expects cell_annot length == n_cells, aligned to columns\
    annot_sc = annot

    prob_gene_per_type = estimate_gene_prob_per_type(
        X_sc, cell_annot=annot_sc, expr_thres=0.0, use_counts=use_counts
    )
    prob_gene_per_type.columns = gene_names
    # prob_gene_per_type = estimate_gene_prob_per_type(
    #     X_sc, cell_annot=annot, expr_thres=0.0, use_counts=use_counts
    # )
    # prob_gene_per_type.columns = gene_names  # types x genes

    # prior.cont.probs = 1 - diag(K)/colSums(K)
    colsum = cell_type_adj_mat.sum(axis=0).to_numpy()
    diag = np.diag(cell_type_adj_mat.to_numpy())
    # avoid /0
    colsum_safe = np.where(colsum == 0, 1.0, colsum)
    prior_cont_probs = pd.Series(1.0 - (diag / colsum_safe), index=cell_type_adj_mat.columns)

    # expr.frac.mask: max over types for each gene
    expr_frac_mask = (prob_gene_per_type.max(axis=0) > min_expr_frac_contam)

    # high-expression matrix used for warnings
    prob_gene_per_type_high = estimate_gene_prob_per_type(
        X_sc, cell_annot=annot_sc, expr_thres=1.0, use_counts=False
    )
    prob_gene_per_type_high.columns = gene_names

    # ---- contamination probabilities per gene per type (genes x types) ----
    cont_cols = {}
    for ct in types_use:
        if ct not in prob_gene_per_type.index:
            # if K includes a type not present in scRNA labels, skip
            continue
        gs = estimate_gene_scores(prob_gene_per_type, cell_type_adj_mat, cell_type=ct)  # genes x {cont,ref}
        cont = estimate_gene_contamination_probabilities(
            gs, p_c=p_c, prior_cont_prob=float(prior_cont_probs.loc[ct])
        )
        # threshold + expression mask
        cont.loc[~((cont > signal_thres) & expr_frac_mask.loc[cont.index])] = 0.0
        cont_cols[ct] = cont

    cont_probs_per_type = pd.DataFrame(cont_cols)  # genes x scored types

    # add missing K types as zeros for consistency
    for t in types_all:
        if t not in cont_probs_per_type.columns:
            cont_probs_per_type[t] = 0.0
    cont_probs_per_type = cont_probs_per_type[types_all]

    # mask out genes (global or per-type)
    cont_probs_per_type = mask_out_genes(cont_probs_per_type, exclude_genes)

    # ---- null expected fraction per type ----
    cont_fracs_null = {}
    for ct in cont_probs_per_type.columns:
        if ct not in prob_gene_per_type.index:
            cont_fracs_null[ct] = 0.0
            continue
        p = prob_gene_per_type.loc[ct, :]
        denom = float(p.sum()) if float(p.sum()) > 0 else 1.0
        cont_fracs_null[ct] = float((cont_probs_per_type[ct] * p).sum() / denom)
    cont_fracs_null = pd.Series(cont_fracs_null)


    # Match cellAdmix: weight probabilities by counts, not log expression.

    spatial_annot = pd.Series(
        spatial_adata.obs[spatial_cell_type_key].values,
        index=sp_obs_names,
        name="cell_type"
    )
    rows = []
    obs_to_col = {o: i for i, o in enumerate(sp_obs_names)}

    celltype_specific = get_cell_type_specific_genes(
        prob_gene_per_type,
        min_expr_frac=min_expr_frac_specific,
        specificity_ratio=3.0,
        top_n=50,
        )
    
    cont_genes_by_type = {}
    bad_genes_by_type = {}

    for ct in cont_probs_per_type.columns:
        
        # cells of this type
        ct_obs = spatial_annot.index[spatial_annot.values == ct].tolist()
        if not ct_obs:
            continue

        cols = np.array([obs_to_col[c] for c in ct_obs], dtype=int)
        cm_ct = X_sp[:, cols]  # genes x spatial cells, original count scale

        cont_probs = cont_probs_per_type[ct]
        cont_genes = cont_probs.index[cont_probs.values > signal_thres].tolist()

        if len(cont_genes) > 0 and ct in prob_gene_per_type_high.index:
            bad_mask = (prob_gene_per_type_high.loc[ct, cont_genes] > max_prob_high)
            bad_genes = list(pd.Index(cont_genes)[bad_mask.values])
            bad_genes_by_type[ct] = bad_genes

            # auto-mask them for this ct
            if bad_genes:
                cont_probs = cont_probs.copy()
                # cont_probs.loc[bad_genes] = 0.0

                
        
        # recompute cont_genes AFTER masking
        cont_genes = cont_probs.index[cont_probs.values > signal_thres].tolist()

        cont_genes_by_type[ct] = cont_genes
        if cont_genes and ct in prob_gene_per_type_high.index:
            high_mask = (prob_gene_per_type_high.loc[ct, cont_genes] > max_prob_high)
            if bool(np.any(high_mask)):
                bad = list(pd.Index(cont_genes)[high_mask.values])
                print(
                    f"Warning: Genes with high expression in '{ct}' detected in contamination: "
                    + ", ".join(map(repr, bad))
                    + ". Consider improving annotation or masking these genes."
                )

        scores = estimate_cell_admixture_fractions(cont_probs, cm_ct)  # Series length == len(ct_cells)
        if adjust:
            scores = np.maximum(scores.to_numpy() - float(cont_fracs_null.loc[ct]), 0.0)
        else:
            scores = scores.to_numpy()

        for c, s in zip(ct_obs, scores):
            rows.append((c, ct, float(s)))
        

    out = pd.DataFrame(rows, columns=["cell", "cell_type", "scores"])
    out["dataset"] = "spatial"

    if return_gene_sets:
        return out, {
            "celltype_specific_genes": celltype_specific,      # optional: for inspection
            "contamination_genes": cont_genes_by_type,         # actually used in score
            "high_expression_warning_genes": bad_genes_by_type,  # warnings only
            "prob_gene_per_type": prob_gene_per_type,          # optional: for inspection
            "cont_probs_per_type": cont_probs_per_type,        # optional: for inspection
        }

    return out
