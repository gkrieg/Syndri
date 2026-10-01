"""Loading and plotting helpers for Syndri's sampled-context experiment."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse


PREDICTION_SUFFIX = "_responsiveness_predictions.h5ad"


def prediction_files(directory: str | Path, cell_type: str, tissues: list[str] | None = None) -> list[Path]:
    """Find completed predictions for an experiment suffix such as ``cytotoxic``."""
    directory = Path(directory)
    files = sorted(directory.glob(f"*_{cell_type}{PREDICTION_SUFFIX}"))
    if tissues is not None:
        files = [path for path in files if any(tissue in path.name for tissue in tissues)]
    if not files:
        raise FileNotFoundError(f"No {cell_type!r} prediction files matched in {directory}")
    return files


def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def load_results(
    files: list[str | Path],
    *,
    max_focal_cells_per_tissue: int | None = 30,
    max_repeats: int | None = 2,
    seed: int = 42,
) -> ad.AnnData:
    """Sample query cells in each file before combining the AnnData objects.

    All contexts for a chosen query and repeat are retained, including its
    original baseline. Passing ``None`` for both limits loads every row.
    """
    if not files:
        raise ValueError("Provide at least one prediction file")
    rng = np.random.default_rng(seed)
    sampled = []
    for filename in files:
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        tissue = path.name.removesuffix(PREDICTION_SUFFIX)
        source = ad.read_h5ad(path, backed="r")
        try:
            required = {"focal_cell_id", "context_source_cell_id", "is_original", "sampling_repeat"}
            missing = required - set(source.obs)
            if missing:
                raise KeyError(f"{path.name} lacks obs columns {sorted(missing)}")

            focal = source.obs["focal_cell_id"].astype(str)
            candidates = np.sort(focal.unique())
            if max_focal_cells_per_tissue is not None and len(candidates) > max_focal_cells_per_tissue:
                candidates = rng.choice(candidates, max_focal_cells_per_tissue, replace=False)
            repeats = np.sort(source.obs["sampling_repeat"].unique())
            if max_repeats is not None:
                repeats = repeats[:max_repeats]
            keep = (focal.isin(candidates) & source.obs["sampling_repeat"].isin(repeats)).to_numpy()
            if not keep.any():
                raise ValueError(f"Sampling selected no rows from {path}")
            subset = source[keep, :].to_memory(copy=True)
        finally:
            source.file.close()

        # These dictionaries are large and are unnecessary once the aligned
        # neighborhood proportions have been materialized in obsm.
        subset.uns.pop("neighbor_celltype_dict", None)
        subset.uns.pop("neighbor_dict", None)
        subset.obs_names = pd.Index(tissue + "::" + subset.obs_names.astype(str))
        for column in ("focal_cell_id", "context_source_cell_id"):
            subset.obs[column] = tissue + "::" + subset.obs[column].astype(str)
        subset.obs["tissue"] = tissue
        sampled.append(subset)

    all_types = sorted({
        cell_type
        for part in sampled
        for cell_type in part.obsm["neighborhood_proportions"].columns
    })
    for part in sampled:
        frame = part.obsm["neighborhood_proportions"]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Expected neighborhood_proportions to be a DataFrame")
        part.obsm["neighborhood_proportions"] = frame.reindex(
            columns=all_types, fill_value=0
        ).astype(np.float32)

    result = ad.concat(sampled, join="inner", merge="same", index_unique=None)
    if result.n_vars == 0:
        raise ValueError("The prediction files have no genes in common")
    if not result.obs_names.is_unique:
        raise ValueError("Inference-row IDs are not unique after adding tissue prefixes")
    return result


def add_gene_annotations(adata: ad.AnnData, excel_path: str | Path) -> None:
    """Attach gene annotations while preserving the AnnData variable index."""
    annotations = pd.read_excel(excel_path, sheet_name="ST 4", skiprows=1)
    annotations = annotations.dropna(subset=["Ensembl_ID"]).drop_duplicates("Ensembl_ID")
    annotations = annotations.set_index("Ensembl_ID")
    if "id" not in adata.var:
        raise KeyError("Expected adata.var['id'] for the Ensembl annotation join")
    ids = adata.var["id"].astype(str)
    for column in ("Annotation", "Annotation_DB"):
        if column in annotations:
            adata.var[column] = ids.map(annotations[column])


def analyze_results(
    adata: ad.AnnData,
    *,
    genes: list[str] | None = None,
    n_auto_genes: int = 12,
    probability_layer: str = "admix_probs",
    logit: bool = False,
    min_contexts: int = 10,
    min_prevalence: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-row responses and per-query composition slopes."""
    if probability_layer not in adata.layers:
        raise KeyError(f"Missing adata.layers[{probability_layer!r}]")
    obs = adata.obs.reset_index(drop=True).copy()
    obs["focal_cell_id"] = obs["focal_cell_id"].astype(str)
    obs["context_source_cell_id"] = obs["context_source_cell_id"].astype(str)
    obs["sampling_repeat"] = obs["sampling_repeat"].astype(int)
    obs["is_original"] = obs["is_original"].astype(bool)
    if "is_representative_context" not in obs:
        obs["is_representative_context"] = ~obs["is_original"]
    obs["is_representative_context"] = obs["is_representative_context"].astype(bool)

    var_names = adata.var_names.astype(str).to_numpy()
    gene_names = adata.var.get("feature_name", pd.Series(var_names, index=adata.var_names))
    gene_names = gene_names.fillna("").astype(str).to_numpy()
    gene_names = np.where(gene_names == "", var_names, gene_names)

    raw_probabilities = _dense(adata.layers[probability_layer]).astype(np.float32)
    if genes:
        wanted = set(genes)
        gene_idx = np.flatnonzero(np.isin(gene_names, list(wanted)) | np.isin(var_names, list(wanted)))
        if not len(gene_idx):
            raise ValueError(f"None of the requested genes were found: {genes}")
    else:
        representative = obs["is_representative_context"].to_numpy()
        if not representative.any():
            representative = np.ones(len(obs), dtype=bool)
        variance = np.nanvar(raw_probabilities[representative], axis=0)
        gene_idx = np.argsort(variance)[-min(n_auto_genes, adata.n_vars):][::-1]
    labels = gene_names[gene_idx]
    probabilities = raw_probabilities[:, gene_idx]
    if logit:
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        probabilities = np.log(probabilities / (1 - probabilities))

    keys = ["focal_cell_id", "sampling_repeat"]
    original = obs.loc[obs["is_original"], keys].copy()
    original["row"] = np.flatnonzero(obs["is_original"].to_numpy())
    if original.duplicated(keys).any():
        raise ValueError("More than one original row exists for a query and repeat")
    baseline = obs[keys].merge(original, on=keys, how="left", validate="many_to_one")["row"]
    if baseline.isna().any():
        raise ValueError("Some query/repeat groups have no original context")
    baseline_rows = baseline.to_numpy(dtype=int)
    delta = probabilities - probabilities[baseline_rows]

    original_expression = _dense(adata.X[original["row"].to_numpy(), :])[:, gene_idx] > 0
    expressed_at_original = np.zeros_like(probabilities, dtype=bool)
    expressed_at_original[original["row"].to_numpy()] = original_expression
    expressed = expressed_at_original[baseline_rows]

    long = obs[keys + ["tissue", "context_source_cell_id", "is_original", "is_representative_context"]]
    long = long.loc[long.index.repeat(len(gene_idx))].reset_index(drop=True)
    long["gene"] = np.tile(labels, adata.n_obs)
    long["value"] = probabilities.reshape(-1)
    long["delta_from_original"] = delta.reshape(-1)
    long["expressed_at_original"] = expressed.reshape(-1)

    composition = adata.obsm["neighborhood_proportions"].reset_index(drop=True)
    context_array = composition.to_numpy(dtype=np.float32)
    records = []
    groups = obs.groupby(keys, observed=True).indices
    is_representative = obs["is_representative_context"].to_numpy()
    sources = obs["context_source_cell_id"].to_numpy()
    for (focal_id, repeat), positions in groups.items():
        positions = np.asarray(positions, dtype=int)
        positions = positions[is_representative[positions]]
        _, first = np.unique(sources[positions], return_index=True)
        positions = positions[np.sort(first)]
        if len(positions) < min_contexts:
            continue
        C, Y = context_array[positions], probabilities[positions]
        finite = np.isfinite(C).all(axis=1) & np.isfinite(Y).all(axis=1)
        C, Y = C[finite], Y[finite]
        if len(C) < min_contexts:
            continue
        Cc, Yc = C - C.mean(axis=0), Y - Y.mean(axis=0)
        denominator = (Cc ** 2).sum(axis=0)
        eligible = (denominator > 1e-10) & ((C > 0).mean(axis=0) >= min_prevalence)
        original_row = int(original.loc[(original["focal_cell_id"] == focal_id) & (original["sampling_repeat"] == repeat), "row"].iloc[0])
        for j in np.flatnonzero(eligible):
            slope = 0.1 * (Cc[:, j] @ Yc) / denominator[j]
            records.extend(
                (focal_id, int(repeat), obs.loc[original_row, "tissue"], composition.columns[j], gene, float(value), len(C))
                for gene, value, is_expressed in zip(labels, slope, expressed_at_original[original_row])
                if is_expressed
            )
    slopes = pd.DataFrame(records, columns=[
        "focal_cell_id", "sampling_repeat", "tissue", "neighbor_cell_type", "gene", "delta_per_10pct", "n_contexts"
    ])
    if "Annotation_DB" in adata.var:
        annotation = pd.Series(adata.var["Annotation_DB"].to_numpy(), index=gene_names)
        slopes["Annotation_DB"] = slopes["gene"].map(annotation)
    return long, slopes


def plot_response_distribution(long: pd.DataFrame, *, logit: bool = False):
    data = long.loc[
        long["is_representative_context"] & ~long["is_original"] & long["expressed_at_original"]
    ]
    if data.empty:
        raise ValueError("No expressed genes in non-original representative contexts")
    fig, ax = plt.subplots(figsize=(max(8, data["gene"].nunique() * 0.85), 5))
    sns.boxplot(data=data, x="gene", y="delta_from_original", showfliers=False, color="#4C78A8", ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set(xlabel="Gene", ylabel=f"Change in {'logit' if logit else 'probability'} from original")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    return fig, ax


def plot_slope_clustermap(
    slopes: pd.DataFrame,
    *,
    genes: list[str] | None = None,
    annotation_col: str = "Annotation_DB",
    logit: bool = False,
):
    if slopes.empty:
        raise ValueError("No slopes were calculated")
    matrix = slopes.groupby(["neighbor_cell_type", "gene"], observed=True)["delta_per_10pct"].median().unstack("gene")
    if genes:
        matrix = matrix.reindex(columns=[gene for gene in genes if gene in matrix])
    matrix = matrix.fillna(0)
    if matrix.empty:
        raise ValueError("No selected genes have slope estimates")

    colors = None
    handles = []
    if annotation_col in slopes:
        annotation = slopes[["gene", annotation_col]].drop_duplicates()
        if annotation["gene"].duplicated().any():
            raise ValueError(f"Some genes have multiple {annotation_col} values")
        annotation = annotation.set_index("gene")[annotation_col].reindex(matrix.columns).fillna("Unannotated").astype(str)
        categories = sorted(annotation.unique())
        palette = dict(zip(categories, sns.color_palette("tab20", n_colors=len(categories))))
        colors = annotation.map(palette)
        handles = [Patch(facecolor=palette[name], label=name) for name in categories]

    grid = sns.clustermap(
        matrix,
        col_colors=colors,
        cmap="vlag",
        center=0,
        row_cluster=len(matrix) > 1,
        col_cluster=matrix.shape[1] > 1,
        figsize=(max(8, matrix.shape[1] * 0.85), max(5, len(matrix) * 0.45)),
        cbar_kws={"label": f"Median Δ {'logit' if logit else 'probability'} per +10% neighbors"},
    )
    grid.ax_heatmap.set(xlabel="Gene", ylabel="Neighbor cell type")
    plt.setp(grid.ax_heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if handles:
        grid.fig.legend(handles=handles, title=annotation_col, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return grid
