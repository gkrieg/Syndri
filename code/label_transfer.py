from __future__ import annotations

from typing import Optional, Sequence, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from sklearn.neighbors import KNeighborsClassifier


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _get_obs_indices(adata: AnnData, obs_names: Sequence[str]) -> np.ndarray:
    """
    Convert AnnData observation names to integer row indices.
    """
    return np.array([adata.obs.index.get_loc(i) for i in obs_names], dtype=int)


def _get_embedding_rows(
    adata: AnnData,
    obs_names: Sequence[str],
    embedding_key: str,
) -> np.ndarray:
    """
    Extract embedding rows from adata.obsm[embedding_key] for a set of obs names.
    """
    idx = _get_obs_indices(adata, obs_names)
    return adata.obsm[embedding_key][idx, :]


# ---------------------------------------------------------------------
# 1) Initial scRNA-seq cell type normalization
# ---------------------------------------------------------------------
DEFAULT_CELL_TYPE_MAP = {
    "Neutrophils": "neutrophil",
    "B cells": "B-cell",
    "tumor": "tumor",
    "NK cells": "NK",
    "Plasma cells": "plasmablast",
    "Monocytes": "monocyte",
    "Mast cells": "mast",
    "Fibroblasts": "fibroblast",
    "Endothelial cells": "endothelial",
    "Type II cells": "epithelial",
    "pDC": "DC",
    "Ciliated cells": "epithelial",
    "Club cells": "epithelial",
    "Type I cells": "epithelial",
    "T cells": "T cells",
}


def normalize_sc_cell_types(
    sc_data: AnnData,
    source_col: str = "Major cell type",
    target_col: str = "cell_type",
    mapping: Optional[Mapping[str, str]] = None,
    unknown_prefix: str = "orig-",
) -> AnnData:
    """
    Normalize coarse scRNA-seq cell type labels into a standard target column.

    Steps:
    - strip leading 'b' and 't' from the source labels
    - convert labels matching '^Patient\\d+.*' to 'tumor'
    - map known labels using `mapping`
    - assign unknown labels as `unknown_prefix + original_label`
    """
    if mapping is None:
        mapping = DEFAULT_CELL_TYPE_MAP

    raw = sc_data.obs[source_col].astype(str).copy()
    cleaned = (
        raw.str.replace(r"^b", "", regex=True)
           .str.replace(r"^t", "", regex=True)
           .str.replace(r"^Patient\d+.*", "tumor", regex=True)
    )

    mapped = cleaned.map(mapping)
    sc_data.obs[target_col] = mapped.fillna(unknown_prefix + cleaned)

    return sc_data


# ---------------------------------------------------------------------
# 2) Transfer T-cell subtypes from ST -> scRNA-seq
# ---------------------------------------------------------------------
def transfer_tcell_subtypes_from_st(
    sc_data: AnnData,
    st_data: AnnData,
    *,
    sc_base_tcell_col: str = "cell_type_simplified",
    sc_base_tcell_value: str = "T cells",
    st_selector_col: str = "cell_type",
    st_selector_prefix: str = "T",
    st_label_col: str = "cell_type_merged",
    target_col: str = "cell_type",
    embedding_key: str = "X_pca_harmony",
    n_neighbors: int = 30,
) -> AnnData:
    """
    Train a kNN classifier on ST T-cell labels and transfer those labels to
    scRNA-seq cells identified as coarse T cells.
    """
    st_mask = st_data.obs[st_selector_col].astype(str).str.startswith(st_selector_prefix)
    st_tcell_ids = st_data.obs.index[st_mask]

    if len(st_tcell_ids) == 0:
        raise ValueError("No ST T cells found for training.")

    sc_mask = sc_data.obs[sc_base_tcell_col].astype(str).eq(sc_base_tcell_value)
    sc_tcell_ids = sc_data.obs.index[sc_mask]

    if len(sc_tcell_ids) == 0:
        return sc_data

    X_train = _get_embedding_rows(st_data, st_tcell_ids, embedding_key)
    y_train = st_data.obs.loc[st_tcell_ids, st_label_col].to_numpy()

    X_pred = _get_embedding_rows(sc_data, sc_tcell_ids, embedding_key)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    sc_data.obs.loc[sc_tcell_ids, target_col] = knn.predict(X_pred)
    return sc_data


# ---------------------------------------------------------------------
# 3) Transfer labels to unlabeled / orig-* scRNA-seq cells
# ---------------------------------------------------------------------
def transfer_unmapped_sc_labels_from_st(
    sc_data: AnnData,
    st_data: AnnData,
    *,
    sc_label_col: str = "cell_type",
    st_label_col: str = "cell_type",
    target_col: str = "cell_type",
    embedding_key: str = "X_pca_harmony",
    unknown_prefix: str = "orig-",
    n_neighbors: int = 30,
    sketch_size: int = 30000,
    random_state: int = 0,
) -> AnnData:
    """
    Predict labels for scRNA-seq cells whose labels still begin with `unknown_prefix`,
    using only ST labels as the training label space.
    """
    sc_unmapped_mask = sc_data.obs[sc_label_col].astype(str).str.startswith(unknown_prefix)
    sc_unmapped_ids = sc_data.obs.index[sc_unmapped_mask]

    if len(sc_unmapped_ids) == 0:
        return sc_data

    if len(st_data) == 0:
        raise ValueError("st_data has no observations.")
    if sketch_size <= 0:
        raise ValueError("sketch_size must be positive.")

    frac = min(1.0, sketch_size / len(st_data))

    st_subset = (
        st_data.obs
        .groupby(st_label_col, group_keys=False)
        .sample(frac=frac, replace=False, random_state=random_state)
        .index
    )

    X_train = _get_embedding_rows(st_data, st_subset, embedding_key)
    y_train = st_data.obs.loc[st_subset, st_label_col].to_numpy()

    X_pred = _get_embedding_rows(sc_data, sc_unmapped_ids, embedding_key)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    sc_data.obs.loc[sc_unmapped_ids, target_col] = knn.predict(X_pred)
    return sc_data

# ---------------------------------------------------------------------
# 4) Mapping / correspondence table and heatmap
# ---------------------------------------------------------------------
def make_label_correspondence_table(
    sc_data: AnnData,
    predicted_col: str = "cell_type",
    original_col: str = "Major cell type",
) -> pd.DataFrame:
    """
    Build a contingency table:
    rows   = predicted / transferred labels
    cols   = original scRNA-seq labels
    """
    obs = sc_data.obs[[predicted_col, original_col]].dropna()
    ct = pd.crosstab(obs[predicted_col], obs[original_col])
    return ct


def sort_contingency_columns_by_row_argmax(ct: pd.DataFrame) -> pd.DataFrame:
    """
    Sort columns so that columns whose maximum occurs in earlier rows appear first.
    Useful for a visually cleaner heatmap.
    """
    if ct.shape[0] == 0 or ct.shape[1] == 0:
        return ct.copy()

    max_row_idx = ct.values.argmax(axis=0)
    col_order = np.argsort(max_row_idx)
    return ct.iloc[:, col_order]


def plot_label_correspondence_heatmap(
    ct: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (14, 8),
    dpi: int = 200,
    cmap: str = "magma",
    title: str = "Correspondence of scRNA cell types to transferred ST labels",
    xlabel: str = "scRNA-seq cell type",
    ylabel: str = "Predicted ST cell type",
    log_transform: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap of label correspondence counts.
    """
    M = sort_contingency_columns_by_row_argmax(ct)

    values = np.log1p(M).astype(float) if log_transform else M.astype(float)
    cbar_label = "count (log1p)" if log_transform else "count"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.heatmap(
        values,
        cmap=cmap,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------
# 5) End-to-end preprocessing wrapper
# ---------------------------------------------------------------------
def preprocess_and_transfer_cell_types(
    sc_data: AnnData,
    st_data: AnnData,
    *,
    source_major_type_col: str = "Major cell type",
    initial_target_col: str = "cell_type",
    sc_tcell_selector_col: str = "cell_type",
    sc_tcell_selector_value: str = "T cells",
    st_tcell_selector_col: str = "cell_type",
    st_tcell_selector_prefix: str = "T",
    st_merged_label_col: str = "cell_type",
    embedding_key: str = "X_pca_harmony",
    cell_type_map: Optional[Mapping[str, str]] = None,
    unknown_prefix: str = "orig-",
    n_neighbors: int = 30,
    sketch_size: int = 30000,
    random_state: int = 0,
) -> AnnData:
    normalize_sc_cell_types(
        sc_data,
        source_col=source_major_type_col,
        target_col=initial_target_col,
        mapping=cell_type_map,
        unknown_prefix=unknown_prefix,
    )

    transfer_tcell_subtypes_from_st(
        sc_data,
        st_data,
        sc_base_tcell_col=sc_tcell_selector_col,
        sc_base_tcell_value=sc_tcell_selector_value,
        st_selector_col=st_tcell_selector_col,
        st_selector_prefix=st_tcell_selector_prefix,
        st_label_col=st_merged_label_col,
        target_col=initial_target_col,
        embedding_key=embedding_key,
        n_neighbors=n_neighbors,
    )

    transfer_unmapped_sc_labels_from_st(
        sc_data,
        st_data,
        sc_label_col=initial_target_col,
        st_label_col=st_merged_label_col,
        target_col=initial_target_col,
        embedding_key=embedding_key,
        unknown_prefix=unknown_prefix,
        n_neighbors=n_neighbors,
        sketch_size=sketch_size,
        random_state=random_state,
    )

    return sc_data
