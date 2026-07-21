from __future__ import annotations

from typing import Any, Optional

import numpy as np
from anndata import AnnData
from sklearn.neighbors import KNeighborsClassifier


def _get_representation(adata: AnnData, embedding_key: Optional[str]) -> Any:
    """
    Return the matrix used for nearest-neighbor matching.
    """
    if embedding_key is None:
        return adata.X

    if embedding_key not in adata.obsm:
        raise KeyError(f"{embedding_key!r} not found in adata.obsm")

    return np.asarray(adata.obsm[embedding_key])


def transfer_cell_type_labels_nn(
    sc_data: AnnData,
    st_data: AnnData,
    *,
    sc_label_col: str = "cell_type",
    embedding_key: Optional[str] = "X_pca_harmony",
    n_neighbors: int = 1,
    metric: str = "euclidean",
) -> list[str]:
    """
    Transfer cell type labels from single-cell reference data to ST cells using kNN.

    Parameters
    ----------
    sc_data : AnnData
        Single-cell reference AnnData containing labels in `sc_label_col`.
    st_data : AnnData
        Spatial transcriptomics AnnData to label. Output order matches
        `st_data.obs_names`.
    sc_label_col : str
        Column in `sc_data.obs` containing reference cell type labels.
    embedding_key : str or None
        Key in `.obsm` for both AnnData objects. If None, use `.X`.
    n_neighbors : int
        Number of nearest SC reference cells used for label prediction.
        The default 1 is direct nearest-neighbor label transfer.
    metric : str
        Distance metric passed to sklearn's KNeighborsClassifier.

    Returns
    -------
    labels : list[str]
        Predicted cell type labels in the same row order as `st_data`.
    """
    if sc_label_col not in sc_data.obs:
        raise KeyError(f"{sc_label_col!r} not found in sc_data.obs")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")

    labels = sc_data.obs[sc_label_col]
    valid_mask = labels.notna().to_numpy()
    if valid_mask.sum() == 0:
        raise ValueError(f"No non-null labels found in sc_data.obs[{sc_label_col!r}].")
    if n_neighbors > valid_mask.sum():
        raise ValueError(
            f"n_neighbors={n_neighbors} exceeds the number of labeled SC cells ({valid_mask.sum()})."
        )

    X_sc = _get_representation(sc_data, embedding_key)
    X_st = _get_representation(st_data, embedding_key)
    if X_sc.shape[1] != X_st.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: sc_data has {X_sc.shape[1]} columns, "
            f"st_data has {X_st.shape[1]} columns."
        )

    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    classifier.fit(X_sc[valid_mask], labels[valid_mask].astype(str).to_numpy())

    return classifier.predict(X_st).tolist()
