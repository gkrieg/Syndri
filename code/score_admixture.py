# --------------------------------------------------
# p_admix
# --------------------------------------------------
import pandas as pd
import anndata as ad
import numpy as np
from admix_scores import estimate_cell_adjacency, estimate_cell_type_adjacency

# read in spatial test set (or synthetically admixed scRNA-seq train/val set)
st_data = ad.read_h5ad('cosmx_nsclc_Lung5-Rep1_adata_annotated_med_with_admix_probs.h5ad')

# read in scRNA-seq set
sc_data = ad.concat([ad.read_h5ad('/compute/oven-0-13/jwinchel/spatial_adata/heimdall_inference/GSE127465_tumor_integrated_val_med_with_admix_probs.h5ad'),
                    ad.read_h5ad('/compute/oven-0-13/jwinchel/spatial_adata/heimdall_inference/GSE127465_tumor_integrated_train_med_with_admix_probs.h5ad')],
                    join='outer')

def build_cell_df_from_st_data(
    st_data,
    *,
    cell_type_col="cell_type",
    fov_col="fov",
    # Choose ONE of these coordinate sources:
    x_col=None,  # e.g. "CenterX_global_px" or "x_global_px"
    y_col=None,  # e.g. "CenterY_global_px" or "y_global_px"
    obsm_key="spatial",  # fallback if x_col/y_col not present
    # optional unit conversion
    px_to_um=None,       # e.g. 0.18 if 180nm per pixel
):
    obs = st_data.obs.copy()

    if cell_type_col not in obs.columns:
        raise KeyError(f"{cell_type_col!r} not found in st_data.obs")
    if fov_col not in obs.columns:
        raise KeyError(f"{fov_col!r} not found in st_data.obs")

    # cell ids (use obs_names so they match other outputs)
    obs["cell"] = st_data.obs_names.astype(str)

    # coordinates
    if x_col is not None and y_col is not None:
        if x_col not in obs.columns or y_col not in obs.columns:
            raise KeyError(f"x_col/y_col not found in obs: {x_col}, {y_col}")
        obs["x"] = obs[x_col].astype(float)
        obs["y"] = obs[y_col].astype(float)
    else:
        if obsm_key not in st_data.obsm:
            raise KeyError(
                f"Neither x_col/y_col provided nor st_data.obsm[{obsm_key!r}] found."
            )
        xy = np.asarray(st_data.obsm[obsm_key])
        if xy.shape[1] != 2:
            raise ValueError(f"st_data.obsm[{obsm_key!r}] must be (n,2), got {xy.shape}")
        obs["x"] = xy[:, 0].astype(float)
        obs["y"] = xy[:, 1].astype(float)

    # optional unit conversion
    if px_to_um is not None:
        obs["x"] *= float(px_to_um)
        obs["y"] *= float(px_to_um)

    # keep only what adjacency needs + fov for grouping
    cell_df = obs[["x", "y", cell_type_col, "cell", fov_col]].rename(
        columns={cell_type_col: "cell_type", fov_col: "fov"}
    )

    # drop missing
    cell_df = cell_df.dropna(subset=["x", "y", "cell", "cell_type", "fov"]).reset_index(drop=True)
    return cell_df


cell_df = build_cell_df_from_st_data(
    st_data,
    cell_type_col="cell_type",
    fov_col="fov",
    # x_col='CenterX_global_px',
    # y_col='CenterY_global_px',
    x_col='sampled_st_x',
    y_col='sampled_st_y',
    obsm_key="spatial",
    px_to_um=None,   # set to 0.18 if those coords are pixels and you want microns
)

# ----------------------------
# 2) Compute cell adjacency per FOV, concatenate, then compute type adjacency once
# ----------------------------
edge_kwargs = dict(
    random_shift=1e-3,
    edge_max_mad=4.0,
    n_jobs=1,
)

all_edges = []
for fov, g in cell_df.groupby("fov", sort=True):
    g = g[["x", "y", "cell_type", "cell"]].copy()
    if g["cell"].nunique() < 4:
        continue

    adj = estimate_cell_adjacency(g, **edge_kwargs)
    if adj is None or len(adj) == 0:
        continue

    all_edges.append(adj)

if len(all_edges) == 0:
    raise RuntimeError("No adjacency edges computed (check coords/cell counts per FOV).")

cell_adj_df = pd.concat(all_edges, axis=0, ignore_index=True)

# One global K from all cell-cell edges across all FOVs
cell_type_adj = estimate_cell_type_adjacency(cell_adj_df)

# Ensure square + consistent ordering
all_types = sorted(cell_df["cell_type"].dropna().unique().tolist())
cell_type_adj = (
    pd.DataFrame(cell_type_adj)
    .reindex(index=all_types, columns=all_types)
    .fillna(0.0)
)

st_data.obs['cell_type'] = st_data.obs['cell_type'].astype(str)
st_data.obs.loc[st_data.obs['cell_type'].str.startswith('tumor'),'cell_type'] = 'tumor'

sc_data.obs['cell_type'] = sc_data.obs['cell_type'].astype(str)
sc_data.obs.loc[sc_data.obs['cell_type'].str.startswith('tumor'),'cell_type'] = 'tumor'

st_data.layers['admix_probs'] = st_data.layers['admix_probs'].todense()

from admix_scores import estimate_correlation_preservation

cor_cors = estimate_correlation_preservation(st_data.X.T,sc_data.X.T)
uncor_gene_ids = np.argwhere(cor_cors.values < 0.2).squeeze()
uncor_genes = st_data.var_names.to_numpy()[uncor_gene_ids]


import numpy as np
import pandas as pd
import importlib

import sc_score_from_st
import admix_scores
importlib.reload(sc_score_from_st)
importlib.reload(admix_scores)
from sc_score_from_st import get_scrna_admixture_scores_from_spatial_K


def make_clean_spatial_by_threshold(st_data, thr, probs_layer="admix_probs"):
    """Return a copy of st_data with X entries zeroed where admix_probs > thr."""
    st_clean = st_data.copy()
    probs = st_clean.layers[probs_layer]
    mask = probs > thr
    st_clean.X[mask] = 0.0
    return st_clean, mask


import numpy as np
from scipy import sparse

def threshold_for_removed_fraction(probs, X, target_removed_frac: float):
    """
    Choose thr so that removing entries with probs >= thr removes ~target_removed_frac
    of TOTAL transcript counts (sum of counts), considering only X>0 entries.

    Returns
    -------
    thr : float
    achieved : float  (fraction of transcript counts removed using probs >= thr)
    """
    r = float(target_removed_frac)
    r = min(max(r, 0.0), 1.0)
    if r == 0.0:
        return np.inf, 0.0
    if r == 1.0:
        return -np.inf, 1.0

    # Pull (pvals, counts) only where X>0
    if sparse.issparse(X):
        Xcoo = X.tocoo()
        counts = Xcoo.data.astype(float)
        pvals = np.asarray(probs[Xcoo.row, Xcoo.col]).ravel().astype(float)
    else:
        X = np.asarray(X)
        probs = np.asarray(probs)
        nz = X > 0
        counts = X[nz].astype(float)
        pvals = probs[nz].astype(float)

    good = np.isfinite(pvals) & np.isfinite(counts) & (counts > 0)
    pvals = pvals[good]
    counts = counts[good]
    if counts.size == 0:
        raise ValueError("No expressed transcripts found (X>0).")

    total = float(counts.sum())
    target = r * total

    # Sort by probability descending, accumulate transcript counts
    order = np.argsort(pvals)[::-1]
    p_sorted = pvals[order]
    c_sorted = counts[order]
    cum = np.cumsum(c_sorted)

    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(p_sorted) - 1)
    thr = float(p_sorted[idx])

    # Achieved fraction if you remove probs >= thr (important!)
    removed = float(c_sorted[p_sorted >= thr].sum())
    achieved = removed / total

    return thr, achieved


# --- choose your target transcript-removal percentages here ---
target_removed_fracs = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]  # 0% to 20%

# pull probs once from the original st_data
probs = st_data.layers["admix_probs"]

all_dfs = []
gene_sets_by_setting = {}  # optional: keep gene_sets per threshold run

# always compute the baseline (unclean) once
base_df, base_gene_sets = get_scrna_admixture_scores_from_spatial_K(
    sc_data, st_data, cell_type_adj, adjust=False, return_gene_sets=True,
    min_expr_frac_specific=0.05,
    min_expr_frac_contam=0.05,
    signal_thres=0.4,
    max_prob_high=0.1,
    p_c=0.25,
    # exclude_genes=uncor_gene_ids.tolist(),
)
base_df["setting"] = "unclean"
base_df["removed_frac"] = 0.0
base_df["removed_pct"] = 0.0
base_df["threshold"] = np.nan
all_dfs.append(base_df)
gene_sets_by_setting["unclean"] = base_gene_sets

# now run for each removal fraction > 0
for r in target_removed_fracs:
    if r <= 0:
        continue

    thr, achieved = threshold_for_removed_fraction(probs, st_data.X, r)
    st_clean, mask = make_clean_spatial_by_threshold(st_data, thr, probs_layer="admix_probs")

    df, gs = get_scrna_admixture_scores_from_spatial_K(
        sc_data, st_clean, 
        cell_type_adj, adjust=False, 
        return_gene_sets=True,
        min_expr_frac_specific=0.05,
        min_expr_frac_contam=0.05,
        signal_thres=0.4,
        max_prob_high=0.1,
        p_c=0.25,
        # exclude_genes=uncor_gene_ids.tolist(),

    )

    df["setting"] = f"remove_{int(round(r*100))}pct"
    df["removed_frac_target"] = float(r)
    df["removed_frac"] = achieved
    df["removed_pct"] = achieved * 100.0
    df["threshold"] = thr

    all_dfs.append(df)
    gene_sets_by_setting[df["setting"].iloc[0]] = gs

score_df = pd.concat(all_dfs, axis=0, ignore_index=True)

score_df.head()



## scRNA ref version:
import numpy as np
import pandas as pd
import importlib

import sc_score_from_st
import admix_scores
importlib.reload(sc_score_from_st)
importlib.reload(admix_scores)
from sc_score_from_st import get_scrna_admixture_scores_from_spatial_K


def make_corrected_from_admix_probs(
    adata,
    thr: float,
    probs_layer: str = "admix_probs",
    copy_layer_name: str = "X_corrected",
):
    """
    Return a copy of `adata` with X entries zeroed where admix_probs > thr.
    Also stores the corrected dense/sparse matrix in `.layers[copy_layer_name]`.
    """
    ad = adata.copy()
    probs = ad.layers[probs_layer]
    mask = np.asarray(probs) > thr
    ad.X[mask] = 0.0
    ad.layers[copy_layer_name] = ad.X.copy()  # optional snapshot
    return ad, mask

import numpy as np
from scipy import sparse

def threshold_for_removed_fraction(probs, X, target_removed_frac: float):
    """
    Choose thr so that removing entries with probs >= thr removes ~target_removed_frac
    of TOTAL transcript counts (sum of counts), considering only X>0 entries.

    Returns
    -------
    thr : float
    achieved : float  (fraction of transcript counts removed using probs >= thr)
    """
    r = float(target_removed_frac)
    r = min(max(r, 0.0), 1.0)
    if r == 0.0:
        return np.inf, 0.0
    if r == 1.0:
        return -np.inf, 1.0

    # Pull (pvals, counts) only where X>0
    if sparse.issparse(X):
        Xcoo = X.tocoo()
        counts = Xcoo.data.astype(float)
        pvals = np.asarray(probs[Xcoo.row, Xcoo.col]).ravel().astype(float)
    else:
        X = np.asarray(X)
        probs = np.asarray(probs)
        nz = X > 0
        counts = X[nz].astype(float)
        pvals = probs[nz].astype(float)

    good = np.isfinite(pvals) & np.isfinite(counts) & (counts > 0)
    pvals = pvals[good]
    counts = counts[good]
    if counts.size == 0:
        raise ValueError("No expressed transcripts found (X>0).")

    total = float(counts.sum())
    target = r * total

    # Sort by probability descending, accumulate transcript counts
    order = np.argsort(pvals)[::-1]
    p_sorted = pvals[order]
    c_sorted = counts[order]
    cum = np.cumsum(c_sorted)

    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(p_sorted) - 1)
    thr = float(p_sorted[idx])

    # Achieved fraction if you remove probs >= thr (important!)
    removed = float(c_sorted[p_sorted >= thr].sum())
    achieved = removed / total

    return thr, achieved


def _prep_sc_for_scoring(sc_like, corrected_layer: str | None = None):
    """
    Helper: if corrected_layer is provided, pass it to scoring via `layer=...`.
    This assumes your `_get_counts_matrix_from_adata` supports pulling from layers by name.
    """
    return dict(layer=corrected_layer, use_raw=False)


# -------------------------
# SETTINGS YOU CONTROL
# -------------------------
target_removed_fracs = [0.01, 0.02, 0.05, 0.1]  # choose any grid
probs_layer = "admix_probs"

# sc_data: original scRNA reference (uncorrupted)
# sc_data_admixed: synthetically admixed scRNA (has .X + .layers['admixed'])
# sc_data_admixed.layers['admix_probs']: model probabilities on the admixed data
sc_data_admixed = sc_data.copy()  # <-- replace with your actual admixed data with probs in layers[probs_layer]
sc_data_admixed.X += sc_data_admixed.layers['admixed']  # <-- if your admixed data is additive, add it to the original counts
#
# NOTE: you'll need to define sc_data_admixed in your notebook/environment.
# e.g. sc_data_admixed = sc_data.copy(); sc_data_admixed.layers['admixed']=...; sc_data_admixed.X = sc_data_admixed.layers['admixed']

# -------------------------
# RUN COMPARISON
# -------------------------
all_dfs = []
gene_sets_by_setting = {}  # optional

# 1) Baseline: original scRNA (sc_data) scored against *spatial* st_data
df_ref, gs_ref = get_scrna_admixture_scores_from_spatial_K(
    sc_data, sc_data, cell_type_adj,
    adjust=False,
    return_gene_sets=True,
    min_expr_frac_specific=0.05,
    min_expr_frac_contam=0.05,
    signal_thres=0.4,
    max_prob_high=0.1,
    p_c=0.25,
    #exclude_genes=uncor_gene_ids.tolist(),
)
df_ref["setting"] = "sc_reference"
df_ref["removed_frac_target"] = 0.0
df_ref["removed_frac"] = 0.0
df_ref["removed_pct"] = 0.0
df_ref["threshold"] = np.nan
all_dfs.append(df_ref)
gene_sets_by_setting["sc_reference"] = gs_ref

# 2) Synthetically admixed scRNA (use .layers['admixed'] as the counts matrix)
#    We score the *cells from sc_data_admixed* by passing `layer="admixed"`
df_syn, gs_syn = get_scrna_admixture_scores_from_spatial_K(
    sc_data, sc_data_admixed, cell_type_adj,
    adjust=False,
    return_gene_sets=True,
    #layer="admixed",          # <-- key change: use synthetic mixture counts
    use_raw=False,
    min_expr_frac_specific=0.05,
    min_expr_frac_contam=0.05,
    signal_thres=0.4,
    max_prob_high=0.1,
    p_c=0.25,
    # exclude_genes=uncor_gene_ids.tolist(),
)
df_syn["setting"] = "sc_synthetic_admixed"
df_syn["removed_frac_target"] = 0.0
df_syn["removed_frac"] = 0.0
df_syn["removed_pct"] = 0.0
df_syn["threshold"] = np.nan
all_dfs.append(df_syn)
gene_sets_by_setting["sc_synthetic_admixed"] = gs_syn

# 3) Corrected admixed scRNA: zero-out entries in sc_data_admixed where admix_probs > thr
probs = sc_data_admixed.layers[probs_layer]

for r in target_removed_fracs:
    if r <= 0:
        continue

    thr, achieved = threshold_for_removed_fraction(probs, sc_data_admixed.X, r)
    sc_corr, mask = make_corrected_from_admix_probs(
        sc_data_admixed, thr, probs_layer=probs_layer, copy_layer_name=f"X_corrected_{int(round(r*100))}pct"
    )

    print(achieved)

    df_corr, gs_corr = get_scrna_admixture_scores_from_spatial_K(
        sc_data, sc_corr, cell_type_adj,
        adjust=False,
        return_gene_sets=True,
        # IMPORTANT: corrected is now in sc_corr.X, so no layer needed
        min_expr_frac_specific=0.05,
        min_expr_frac_contam=0.05,
        signal_thres=0.4,
        max_prob_high=0.1,
        p_c=0.25,
        # exclude_genes=uncor_gene_ids.tolist(),
    )

    df_corr["setting"] = f"sc_admixed_corrected ({int(round(achieved*100))}%)"
    df_corr["removed_frac_target"] = float(r)
    df_corr["removed_frac"] = achieved
    df_corr["removed_pct"] = achieved * 100.0
    df_corr["threshold"] = float(thr)

    all_dfs.append(df_corr)
    gene_sets_by_setting[f"sc_admixed_corrected_remove_{int(round(r*100))}pct"] = gs_corr

score_df = pd.concat(all_dfs, axis=0, ignore_index=True)

# Optional: quick sanity summaries
summary = (
    score_df.groupby(["setting", "removed_pct", "cell_type"])["scores"]
    .agg(["mean", "median", "count"])
    .reset_index()
)
score_df.head(), summary.head()