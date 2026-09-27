import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tifffile

def _normalize01(x, p_low=1, p_high=99):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1)

def two_channel_rgb(ch_purple, ch_green, purple_gain=1.0, green_gain=1.0,
                    p_low=1, p_high=99):
    """Return HxWx3 float RGB in [0,1], with ch_purple -> (R,B), ch_green -> G."""
    p = _normalize01(ch_purple, p_low=p_low, p_high=p_high) * purple_gain
    g = _normalize01(ch_green,  p_low=p_low, p_high=p_high) * green_gain
    p = np.clip(p, 0, 1)
    g = np.clip(g, 0, 1)

    rgb = np.zeros((*p.shape, 3), dtype=np.float32)
    rgb[..., 0] = p
    rgb[..., 1] = g
    rgb[..., 2] = p
    return rgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import sparse

# def plot_cell_crops_with_tx(
#     img_df: pd.DataFrame,     # dataframe of image planes/files
#     cell_df: pd.DataFrame,
#     tx_df: pd.DataFrame,
#     *,
#     # ---- linkage between cells/transcripts and images ----
#     cell_fov_col: str = "fov",     # column in cell_df giving FOV id
#     tx_fov_col: str = "fov",       # column in tx_df giving FOV id (if present)
#     # kwargs passed into load_z_projected_image
#     z_col: str = "z",
#     fov_col_img: str = "fov",
#     path_col: str = "filepath",
#     z_selection="all",
#     projection: str = "max",
#     extra_query: str | None = None,

#     # ---- cell/transcript columns ----
#     cell_id_col="cell_ID",
#     center_x_col="CenterX_local_px",
#     center_y_col="CenterY_local_px",
#     tx_x_col="x_local_px",
#     tx_y_col="y_local_px",
#     y_origin="bottom",   # "bottom" if your y increases bottom->top, else "top"

#     # ---- sampling / layout ----
#     n_cells=16,
#     crop_size=150,
#     random_state=1,
#     ncols=None,
#     figsize=(10, 10),
#     dpi=150,

#     # ---- composite settings (ignored if image already RGB) ----
#     ch_purple=4,
#     ch_green=0,
#     purple_gain=1.0,
#     green_gain=1.0,
#     p_low=1,
#     p_high=99,

#     # ---- scatter settings ----
#     scatter=True,
#     label_col="CellComp",
#     palette=None,
#     markers=None,
#     s=10,
#     legend=True,
#     legend_loc="upper center",
#     legend_bbox=(0.5, 0.95),

#     # ---- perf / behavior ----
#     cache_images: bool = True,   # cache loaded FOV images
#     only_plot_tx_in_cell_fov: bool = True,  # filter tx to same FOV if tx_fov_col exists
# ):
#     """
#     Plot N random cell crops in a grid, with tx scatter overlay colored/styled by label_col.
#     Loads the correct image per cell using load_z_projected_image(img_df, fov=...).

#     Returns (fig, axes).
#     """

#     # --- sanity checks ---
#     if cell_fov_col not in cell_df.columns:
#         raise KeyError(f"{cell_fov_col!r} not found in cell_df columns.")
#     if cell_id_col not in cell_df.columns:
#         raise KeyError(f"{cell_id_col!r} not found in cell_df columns.")

#     # --- sample cells ---
#     sample_cells = cell_df.sample(n=min(n_cells, len(cell_df)), random_state=random_state).reset_index(drop=True)

#     # --- optionally pre-merge transcripts (keeps your old behavior) ---
#     sample_tx = sample_cells[[cell_id_col]].merge(tx_df, on=cell_id_col, how="left")

#     # --- determine category order + palette/markers globally for consistency ---
#     if scatter and label_col is not None and label_col in tx_df.columns:
#         labels = sorted(tx_df[label_col].dropna().unique().tolist())
#     else:
#         labels = []

#     if palette is None and labels:
#         palette = dict(zip(labels, sns.color_palette("tab10", n_colors=len(labels))))
#     if markers is None and labels:
#         default_markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "H"]
#         if len(labels) > len(default_markers):
#             default_markers = (default_markers * (len(labels) // len(default_markers) + 1))[:len(labels)]
#         markers = {lab: mk for lab, mk in zip(labels, default_markers)}

#     # --- grid ---
#     if ncols is None:
#         ncols = int(np.ceil(np.sqrt(len(sample_cells))))
#     nrows = int(np.ceil(len(sample_cells) / ncols))

#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
#     axes = np.array(axes).reshape(-1)

#     half = crop_size // 2
#     last_handles, last_labels = None, None

#     # --- image cache per FOV ---
#     fov_to_img = {}

#     def get_fov_img(fov_val):
#         if cache_images and fov_val in fov_to_img:
#             return fov_to_img[fov_val]

#         img_chw = load_z_projected_image(
#             img_df,
#             fov=fov_val,
#             z_col=z_col,
#             fov_col=fov_col_img,
#             path_col=path_col,
#             z_selection=z_selection,
#             projection=projection,
#             extra_query=extra_query,
#         )  # (C,H,W)

#         # convert to RGB for consistent downstream behavior
#         if img_chw.ndim == 3 and img_chw.shape[-1] == 3:
#             rgb = img_chw
#         elif img_chw.ndim == 3:
#             rgb = two_channel_rgb(
#                 img_chw[ch_purple], img_chw[ch_green],
#                 purple_gain=purple_gain, green_gain=green_gain,
#                 p_low=p_low, p_high=p_high
#             )
#         else:
#             raise ValueError(f"Unexpected projected image shape: {img_chw.shape}")

#         if cache_images:
#             fov_to_img[fov_val] = rgb
#         return rgb

#     # --- main loop ---
#     for i, ax in enumerate(axes):
#         ax.axis("off")
#         if i >= len(sample_cells):
#             continue

#         cell = sample_cells.iloc[i]
#         cell_id = cell[cell_id_col]
#         fov_val = cell[cell_fov_col]

#         rgb_img = get_fov_img(fov_val)
#         H, W = rgb_img.shape[:2]

#         cx = int(cell[center_x_col])
#         cy_in = int(cell[center_y_col])

#         if y_origin == "bottom":
#             cy = (H - 1) - cy_in
#         elif y_origin == "top":
#             cy = cy_in
#         else:
#             raise ValueError("y_origin must be 'bottom' or 'top'")

#         y0 = max(0, cy - half)
#         y1 = min(H, cy + half)
#         x0 = max(0, cx - half)
#         x1 = min(W, cx + half)

#         crop = rgb_img[y0:y1, x0:x1]

#         # you currently display grayscale; keep the same behavior
#         ax.imshow(crop.max(axis=2), cmap="gray")

#         # --- scatter transcripts ---
#         if scatter:
#             cell_tx = sample_tx[sample_tx[cell_id_col] == cell_id].copy()
#             cell_tx = cell_tx.dropna(subset=[tx_x_col, tx_y_col])

#             # optionally restrict to tx in the same FOV (if tx_df has a fov column)
#             if only_plot_tx_in_cell_fov and (tx_fov_col in cell_tx.columns):
#                 cell_tx = cell_tx[cell_tx[tx_fov_col] == fov_val]

#             if len(cell_tx) > 0:
#                 tx_x = cell_tx[tx_x_col].to_numpy() - x0
#                 if y_origin == "bottom":
#                     tx_y = (H - 1 - cell_tx[tx_y_col].to_numpy()) - y0
#                 else:
#                     tx_y = cell_tx[tx_y_col].to_numpy() - y0

#                 if label_col is not None and label_col in cell_tx.columns:
#                     sns.scatterplot(
#                         x=tx_x,
#                         y=tx_y,
#                         hue=cell_tx[label_col],
#                         style=cell_tx[label_col],
#                         hue_order=labels if labels else None,
#                         style_order=labels if labels else None,
#                         palette=palette,
#                         markers=markers,
#                         s=s,
#                         ax=ax,
#                         legend=True,
#                     )
#                     if ax.legend_ is not None:
#                         last_handles, last_labels = ax.get_legend_handles_labels()
#                         ax.legend_.remove()
#                 else:
#                     ax.scatter(tx_x, tx_y, s=s)

#     # --- global legend ---
#     if legend and last_handles is not None and last_labels is not None:
#         fig.legend(last_handles, last_labels, loc=legend_loc, bbox_to_anchor=legend_bbox)

#     return fig, axes

def plot_cell_crops_with_tx(
    img,                 # raw tif array: (C,H,W) or already RGB: (H,W,3)
    cell_df,
    tx_df,
    *,
    cell_id_col="cell_ID",
    center_x_col="CenterX_local_px",
    center_y_col="CenterY_local_px",
    tx_x_col="x_local_px",
    tx_y_col="y_local_px",
    y_origin="bottom",   # "bottom" if your y increases bottom->top, else "top"
    n_cells=16,
    crop_size=150,
    random_state=1,
    ncols=None,          # if None, choose square-ish grid
    figsize=(10, 10),
    dpi=150,
    # composite settings (ignored if img is already RGB)
    ch_purple=4,
    ch_green=0,
    purple_gain=1.0,
    green_gain=1.0,
    p_low=1,
    p_high=99,
    # scatter settings
    scatter=True,
    label_col="CellComp",   # column in tx_df for hue/style
    palette=None,           # dict or seaborn palette; if None computed from label_col
    markers=None,           # dict mapping label->marker; if None auto-assigned
    s=10,
    legend=True,
    legend_loc="upper right",
    legend_bbox=(0.5, 0.95),
):
    """
    Plot N random cell crops in a grid, with tx scatter overlay colored/styled by label_col.
    Returns (fig, axes).
    """

    # --- Build RGB image ---
    if img.ndim == 3 and img.shape[-1] == 3:
        rgb_img = img
    elif img.ndim == 3:
        # assume (C,H,W)
        rgb_img = two_channel_rgb(
            img[ch_purple], img[ch_green],
            purple_gain=purple_gain, green_gain=green_gain,
            p_low=p_low, p_high=p_high
        )
    else:
        raise ValueError(f"Unexpected img shape: {img.shape}. Expected (C,H,W) or (H,W,3).")

    H, W = rgb_img.shape[:2]
    half = crop_size // 2

    # --- Sample cells & join transcripts ---
    sample_cells = cell_df.sort_values(by=['cell_ID']).head(n_cells)
    sample_tx = sample_cells[[cell_id_col]].merge(tx_df, on=cell_id_col, how="left")

    # --- Determine category order + palette/markers globally for consistency ---
    if scatter and label_col is not None and label_col in tx_df.columns:
        labels = sorted(sample_tx[label_col].dropna().unique().tolist())
    else:
        labels = []

    if palette is None and labels:
        palette = dict(zip(labels, sns.color_palette("tab10", n_colors=len(labels))))
    if markers is None and labels:
        default_markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "H"]
        if len(labels) > len(default_markers):
            # repeat markers if many labels
            default_markers = (default_markers * (len(labels) // len(default_markers) + 1))[:len(labels)]
        markers = {lab: mk for lab, mk in zip(labels, default_markers)}

    # --- Grid ---
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(n_cells)))
    nrows = int(np.ceil(n_cells / ncols))
    print(ncols, nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    # axes = np.array(axes).reshape(-1)  # flatten robustly

    last_handles, last_labels = None, None

    for i in range(nrows * ncols):
        ax = axes.flat[i % nrows + i // nrows]  # works even if axes is 1D or 2D
        ax.axis("off")
        # if (i % nrows) == 0:
        #     ax.set_title(f'{sample_cells.iloc[i]["cell_type"]}', fontsize=12)

        if i >= n_cells:
            continue

        cell = sample_cells.iloc[i]
        cell_id = cell[cell_id_col]
        cx = int(cell[center_x_col])
        cy_in = int(cell[center_y_col])

        # convert y to row coords if needed
        if y_origin == "bottom":
            cy = (H - 1) - cy_in
        elif y_origin == "top":
            cy = cy_in
        else:
            raise ValueError("y_origin must be 'bottom' or 'top'")

        y0 = max(0, cy - half)
        y1 = min(H, cy + half)
        x0 = max(0, cx - half)
        x1 = min(W, cx + half)

        crop = rgb_img[y0:y1, x0:x1]
        # ax.imshow(crop.max(axis=2),cmap='gray')
        ax.imshow(crop)

        if scatter:
            cell_tx = sample_tx[sample_tx[cell_id_col] == cell_id]
            cell_tx = cell_tx.dropna(subset=[tx_x_col, tx_y_col])

            if len(cell_tx) > 0:
                tx_x = cell_tx[tx_x_col].to_numpy() - x0

                if y_origin == "bottom":
                    tx_y = (H - 1 - cell_tx[tx_y_col].to_numpy()) - y0
                else:
                    tx_y = cell_tx[tx_y_col].to_numpy() - y0

                if label_col is not None and label_col in cell_tx.columns:
                    sns.scatterplot(
                        x=tx_x,
                        y=tx_y,
                        hue=cell_tx[label_col],
                        style=cell_tx[label_col],
                        hue_order=labels if labels else None,
                        style_order=labels if labels else None,
                        palette=palette,
                        markers=markers,
                        s=s,
                        ax=ax,
                        legend=True,  # we’ll remove per-axes legend and make a global one
                    )
                    if ax.legend_ is not None:
                        # capture handles for global legend
                        last_handles, last_labels = ax.get_legend_handles_labels()
                        ax.legend_.remove()
                else:
                    ax.scatter(tx_x, tx_y, s=s)

    # Global legend
    if legend and last_handles is not None and last_labels is not None:
        fig.legend(last_handles, last_labels, loc=legend_loc, bbox_to_anchor=legend_bbox)
    
    return fig, axes


def load_z_projected_image(
    img_df: pd.DataFrame,
    *,
    fov,
    z_col: str = "z",
    fov_col: str = "fov",
    path_col: str = "filepath",
    z_selection="all",
    projection: str = "max",
    # Optional extra filters (e.g. channel set, timepoint, etc.)
    extra_query: str | None = None,
) -> np.ndarray:
    """
    Returns a (C,H,W) image after projecting across selected z-planes.

    Assumes each row in img_df corresponds to ONE z-plane image file,
    and each file loads as (C,H,W) or (H,W) (single-channel).
    """

    df = img_df[img_df[fov_col] == fov].copy()
    if extra_query:
        df = df.query(extra_query)

    if z_col not in df.columns:
        raise ValueError(f"`{z_col}` column not found in img_df. Available: {list(df.columns)}")

    # Decide which z-planes to use
    zs_all = sorted(df[z_col].unique().tolist())
    print(zs_all)
    if z_selection == "all":
        zs = zs_all
    elif isinstance(z_selection, (list, tuple, set, np.ndarray)):
        if isinstance(z_selection, tuple) and len(z_selection) == 2 and all(isinstance(x, (int, np.integer)) for x in z_selection):
            # treat as a (start, stop) range
            start, stop = z_selection
            zs = [z for z in zs_all if (z >= start and z < stop)]
        else:
            zs = [z for z in zs_all if z in set(z_selection)]
    elif callable(z_selection):
        zs = [z for z in zs_all if z_selection(z)]
    else:
        raise ValueError("z_selection must be 'all', a list/set/tuple, a (start, stop) tuple, or a callable.")

    if len(zs) == 0:
        raise ValueError(f"No z-planes matched z_selection={z_selection}. Available z: {zs_all}")

    # Load planes (each -> (C,H,W))
    planes = []
    for z in zs:
        row = df[df[z_col] == z].iloc[0]
        arr = tifffile.imread(row[path_col])

        # Normalize to (C,H,W)
        if arr.ndim == 2:
            arr = arr[None, ...]  # (1,H,W)
        elif arr.ndim == 3:
            pass  # (C,H,W) is fine
        else:
            raise ValueError(f"Unexpected image ndim={arr.ndim} from {row[path_col]}. Expected 2 or 3.")

        planes.append(arr)

    stack = np.stack(planes, axis=0)  # (Z,C,H,W)

    if projection == "max":
        proj = stack.max(axis=0)
    elif projection == "mean":
        proj = stack.mean(axis=0)
    else:
        raise ValueError("projection must be 'max' or 'mean'")

    return proj  # (C,H,W)


import numpy as np
import pandas as pd
import scanpy as sc

def add_marker_set_scores(
    adata,
    genes_by_type: dict,
    *,
    score_prefix: str = "ctspec",
    use_raw: bool = False,
    layer: str | None = None,
):
    """
    Adds one obs column per cell-type gene set:
      f"{score_prefix}_{ct}" = mean expression of that ct's specific genes in each cell.
    Uses scanpy.tl.score_genes which is robust + fast.
    """
    for ct, genes in genes_by_type.items():
        genes_use = [g for g in genes if g in (adata.raw.var_names if (use_raw and adata.raw is not None) else adata.var_names)]
        if len(genes_use) == 0:
            continue
        sc.tl.score_genes(
            adata,
            gene_list=genes_use,
            score_name=f"{score_prefix}_{ct}",
            use_raw=use_raw,
        )

def marker_set_dotplot(adata, *, cell_type_key="cell_type", score_prefix="ctspec"):
    score_cols = [c for c in adata.obs.columns if c.startswith(score_prefix + "_")]
    # nicer labels
    var_names = [c.replace(score_prefix + "_", "") for c in score_cols]
    return sc.pl.dotplot(
        adata,
        var_names=score_cols,
        groupby=cell_type_key,
        standard_scale="var",   # compares columns (gene sets) on same scale
        swap_axes=True,         # columns become rows if you prefer; toggle as desired
        show=True,
        return_fig=True,
    )