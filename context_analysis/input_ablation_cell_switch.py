"""Compare per-cell cleaning and metric-matched Leiden clusters for k10 visible1/10."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scib_metrics.metrics import _nmi_ari as nmi_ari_module


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval"))
from metrics_eval import (  # noqa: E402
    cleaned_counts_for_removed_fraction_maybe_grouped,
    neighbors_for_scib_leiden_metrics,
    parse_str_list,
    prepare_matrix_and_labels,
    preprocess_counts_for_metrics,
    sample_adata,
    sanitize_numeric_matrix,
)


DATA_ROOT = Path(
    "/work/magroup/skrieger/Syndri/data/datasets/breast_xenium/results/"
    "breast_xenium_10neighbor_token_sweep"
)
DEFAULT_OUTPUT = REPO_ROOT / "context_analysis/results/input_ablations_cell_switch"
TARGET_FRACTION = 0.5
CASE_NAME = f"cleaned_admixed_removed_frac_{TARGET_FRACTION:g}"
MODELS = (1, 10)


# Same families as the color groups in Figure2_UMAPs_Xenium.ipynb.
CELL_TYPE_FAMILIES = {
    **dict.fromkeys(['CD4-positive, alpha-beta memory T cell', 'cytotoxic T cell',
                     'natural T-regulatory cell', 'natural killer cell', 'plasma cell'], 'Lymphoid'),
    **dict.fromkeys(['classical monocyte', 'macrophage', 'mast cell',
                     'plasmacytoid dendritic cell'], 'Myeloid / innate'),
    **dict.fromkeys(['capillary endothelial cell', 'fibroblast of breast',
                     'fibroblast of lung', 'pericyte', 'smooth muscle cell'], 'Stromal / vascular'),
    'malignant cell': 'Malignant',
}


def family_switch_labels(cells):
    """Label clusters by their unique predominant annotated type, then its family."""
    unknown = set(cells['cell_type']) - set(CELL_TYPE_FAMILIES)
    if unknown:
        raise ValueError(f'Cell types without Xenium family mapping: {sorted(unknown)}')
    result = cells.copy()
    result['annotated_family'] = result['cell_type'].map(CELL_TYPE_FAMILIES)
    for model in (1, 10):
        counts = result.groupby([f'cluster_{model}', 'cell_type']).size().unstack(fill_value=0)
        tied = counts.eq(counts.max(axis=1), axis=0).sum(axis=1).gt(1)
        types = counts.idxmax(axis=1).mask(tied)
        result[f'predominant_type_{model}'] = result[f'cluster_{model}'].map(types)
        result[f'family_{model}'] = result[f'predominant_type_{model}'].map(CELL_TYPE_FAMILIES)
    result['switch_eligible'] = result[['family_1', 'family_10']].notna().all(axis=1)
    result['switched'] = (result['switch_eligible'] & result['family_1'].ne(result['family_10']))
    return result


def _matrix_hash(X) -> str:
    """Verify that each model received exactly the same raw count matrix."""
    digest = hashlib.sha256()
    if sparse.issparse(X):
        matrix = X.tocsr()
        matrix.sort_indices()
        for array in (matrix.indptr, matrix.indices, matrix.data):
            digest.update(np.asarray(array).tobytes())
    else:
        digest.update(np.ascontiguousarray(X).tobytes())
    return digest.hexdigest()


def _metric_inputs(sample: str, visible_tokens: int) -> tuple[Path, Path, Path]:
    model_dir = DATA_ROOT / f"breast_xenium_k10_visible{visible_tokens}"
    data_path = model_dir / f"{sample}.h5ad"
    metrics_dir = model_dir / "metrics/non_tcell" / sample
    return data_path, metrics_dir / "non_tcell_metrics_config.json", metrics_dir / "non_tcell_metrics.csv"


def _analyze_model(sample: str, visible_tokens: int) -> dict:
    data_path, config_path, metrics_path = _metric_inputs(sample, visible_tokens)
    for path in (data_path, config_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    args = argparse.Namespace(**json.loads(config_path.read_text()))
    if args.sample_frac != 1.0:
        raise ValueError(f"Expected full-cell metrics, got sample_frac={args.sample_frac}")
    if args.leiden_optimize_resolution:
        raise ValueError("This analysis expects a fixed Leiden resolution")
    if args.cleaning_mode != "probability_scale" or not args.cell_type_specific_removal:
        raise ValueError("Saved metrics use a different cleaning setup")

    adata = ad.read_h5ad(data_path)
    adata.X = sanitize_numeric_matrix(adata.X)
    adata = sample_adata(adata, args)
    labels = adata.obs[args.label_key]
    excluded = set(parse_str_list(args.exclude_labels))
    if excluded:
        keep = ~labels.astype(str).isin(excluded)
        adata = adata[keep].copy()
        labels = adata.obs[args.label_key]

    group_key = args.removal_group_key or args.label_key
    cleaned, _, achieved, _, _, _, _ = cleaned_counts_for_removed_fraction_maybe_grouped(
        adata.X,
        adata.layers[args.admix_probs_layer],
        TARGET_FRACTION,
        args,
        labels=adata.obs[group_key],
    )
    cleaned_for_metric, labels_prepared = prepare_matrix_and_labels(cleaned, labels)
    valid = ~labels.isna().to_numpy()
    raw_for_metric = adata.X[valid]
    cells = adata.obs.iloc[np.flatnonzero(valid)]
    gene_names = adata.var_names.astype(str).to_numpy()
    raw_hash = _matrix_hash(raw_for_metric)

    embedding = preprocess_counts_for_metrics(cleaned_for_metric, args)
    neighbors = neighbors_for_scib_leiden_metrics(embedding, args)
    # scib_metrics.nmi_ari_cluster_labels_leiden calls this exact private
    # clustering function on the same connectivities in fixed-resolution mode.
    # Its fixed-resolution branch does not forward `seed`, so the helper's
    # default (currently 42) is the seed actually used by the saved metrics.
    metric_seed = inspect.signature(
        nmi_ari_module._compute_nmi_ari_cluster_labels
    ).parameters["seed"].default
    clusters = nmi_ari_module._compute_clustering_leiden(
        neighbors.knn_graph_connectivities,
        args.leiden_resolution,
        metric_seed,
    )
    observed_nmi = normalized_mutual_info_score(
        labels_prepared, clusters, average_method="arithmetic"
    )
    observed_ari = adjusted_rand_score(labels_prepared, clusters)

    metric_rows = pd.read_csv(metrics_path)
    saved = metric_rows.loc[metric_rows["case"].eq(CASE_NAME)]
    if len(saved) != 1 or pd.notna(saved.iloc[0].get("error")):
        raise ValueError(f"Expected one successful {CASE_NAME} row in {metrics_path}")
    saved = saved.iloc[0]
    if len(clusters) != int(saved["n_cells"]):
        raise ValueError(f"{sample} visible{visible_tokens}: cell count differs from saved metrics")
    if not np.isclose(achieved, saved["achieved_removed_fraction"], rtol=0, atol=1e-8):
        raise ValueError(f"{sample} visible{visible_tokens}: removal fraction differs from saved metrics")
    metrics_match = bool(
        np.isclose(observed_nmi, saved["leiden_nmi_vs_label"], rtol=0, atol=1e-6)
        and np.isclose(observed_ari, saved["leiden_adjusted_rand_vs_label"], rtol=0, atol=1e-6)
    )
    if not metrics_match:
        warnings.warn(
            f"{sample} visible{visible_tokens}: reproduced Leiden scores differ from saved "
            f"metrics (NMI {observed_nmi:.6f} vs {saved['leiden_nmi_vs_label']:.6f}; "
            f"ARI {observed_ari:.6f} vs {saved['leiden_adjusted_rand_vs_label']:.6f}). "
            "Using reproduced cell-level labels as requested.",
            stacklevel=2,
        )

    removed = raw_for_metric - cleaned_for_metric
    if sparse.issparse(removed):
        removed = removed.tocsr()
        removed.eliminate_zeros()
        if removed.nnz and removed.data.min() < -1e-8:
            raise ValueError("Cleaned expression exceeds raw expression")
        removed.data = np.maximum(removed.data, 0.0)
        modified_genes = np.diff(removed.indptr)
        removed_per_cell = np.asarray(removed.sum(axis=1)).ravel()
        raw_per_cell = np.asarray(raw_for_metric.sum(axis=1)).ravel()
        detected_before = np.asarray((raw_for_metric > 0).sum(axis=1)).ravel()
        detected_after = np.asarray((cleaned_for_metric > 0).sum(axis=1)).ravel()
    else:
        removed = np.maximum(np.asarray(removed), 0.0)
        modified_genes = np.count_nonzero(removed > 0, axis=1)
        removed_per_cell = removed.sum(axis=1)
        raw_per_cell = np.asarray(raw_for_metric).sum(axis=1)
        detected_before = np.count_nonzero(np.asarray(raw_for_metric) > 0, axis=1)
        detected_after = np.count_nonzero(np.asarray(cleaned_for_metric) > 0, axis=1)

    return {
        "cell_ids": cells.index.astype(str).to_numpy(),
        "cell_types": cells[args.label_key].astype(str).to_numpy(),
        "x": pd.to_numeric(cells.get("x_centroid"), errors="coerce").to_numpy(),
        "y": pd.to_numeric(cells.get("y_centroid"), errors="coerce").to_numpy(),
        "genes": gene_names,
        "raw_hash": raw_hash,
        "clusters": clusters,
        "modified_genes": modified_genes,
        "zeroed_genes": detected_before - detected_after,
        "removed_per_cell": removed_per_cell,
        "raw_per_cell": raw_per_cell,
        "removed_matrix": removed,
        "nmi": observed_nmi,
        "ari": observed_ari,
        "saved_nmi": float(saved["leiden_nmi_vs_label"]),
        "saved_ari": float(saved["leiden_adjusted_rand_vs_label"]),
        "metrics_match": metrics_match,
        "achieved_fraction": achieved,
        "leiden_seed": metric_seed,
        "leiden_resolution": args.leiden_resolution,
    }


def analyze_slice(sample: str, output_root: Path = DEFAULT_OUTPUT, *, switching="cluster") -> Path:
    """Write paired cell, gene, transition, and slice-level diagnostics."""
    if switching not in {"cluster", "family"}:
        raise ValueError(f"Unknown switching definition: {switching}")
    first = _analyze_model(sample, MODELS[0])
    last = _analyze_model(sample, MODELS[1])
    for key in ("cell_ids", "cell_types", "genes", "raw_per_cell"):
        if not np.array_equal(first[key], last[key]):
            raise ValueError(f"1-token and 10-token {key} are misaligned for {sample}")
    if first["raw_hash"] != last["raw_hash"]:
        raise ValueError(f"Raw count matrices differ for {sample}")

    cluster_1 = np.asarray(first["clusters"], dtype=int)
    cluster_10 = np.asarray(last["clusters"], dtype=int)
    contingency = pd.crosstab(
        pd.Index(cluster_1, name="cluster_1"),
        pd.Index(cluster_10, name="cluster_10"),
    ).rename_axis(index="cluster_1", columns="cluster_10")
    row_idx, col_idx = linear_sum_assignment(-contingency.to_numpy())
    mapping = {
        int(contingency.columns[col]): int(contingency.index[row])
        for row, col in zip(row_idx, col_idx)
    }
    aligned_10 = np.array([mapping.get(int(cluster), -1) for cluster in cluster_10])
    switched = aligned_10 != cluster_1
    cluster_ari = adjusted_rand_score(cluster_1, cluster_10)
    cluster_nmi = normalized_mutual_info_score(cluster_1, cluster_10)

    cells = pd.DataFrame({
        "slice": sample,
        "cell_id": first["cell_ids"],
        "cell_type": first["cell_types"],
        "x_centroid": first["x"],
        "y_centroid": first["y"],
        "cluster_1": cluster_1,
        "cluster_10": cluster_10,
        "cluster_10_aligned": aligned_10,
        "switched": switched,
        "raw_transcripts": first["raw_per_cell"],
        "modified_genes_1": first["modified_genes"],
        "modified_genes_10": last["modified_genes"],
        "zeroed_genes_1": first["zeroed_genes"],
        "zeroed_genes_10": last["zeroed_genes"],
        "removed_transcripts_1": first["removed_per_cell"],
        "removed_transcripts_10": last["removed_per_cell"],
    })
    cells['cluster_switched'] = cells['switched']
    matched_cluster_1 = set(mapping.values())
    matched_cluster_10 = set(mapping.keys())
    cells['cluster_match_eligible'] = (
        cells['cluster_1'].isin(matched_cluster_1)
        & cells['cluster_10'].isin(matched_cluster_10)
    )
    if switching == 'family':
        cells = family_switch_labels(cells)
        # Use only cells assigned to clusters participating in the one-to-one
        # Hungarian match. The family labels themselves remain based on each
        # model's cluster-predominant annotated type.
        cells['switch_eligible'] &= cells['cluster_match_eligible']
        cells['switched'] = (
            cells['switch_eligible']
            & cells['family_1'].ne(cells['family_10'])
        )
        switched = cells['switched'].to_numpy()
        eligible = cells['switch_eligible'].to_numpy()
    else:
        eligible = np.ones(len(cells), dtype=bool)
        cells['switch_eligible'] = eligible
    stable = eligible & ~switched
    cells["delta_modified_genes_10_minus_1"] = (
        cells["modified_genes_10"] - cells["modified_genes_1"]
    )
    cells["delta_removed_transcripts_10_minus_1"] = (
        cells["removed_transcripts_10"] - cells["removed_transcripts_1"]
    )

    gene_rows = {"slice": sample, "gene": first["genes"]}
    for visible, result in ((1, first), (10, last)):
        removed = result["removed_matrix"]
        gene_rows[f"removed_transcripts_{visible}"] = np.asarray(
            removed.sum(axis=0)
        ).ravel()
        gene_rows[f"modified_cells_{visible}"] = (
            np.asarray((removed > 0).sum(axis=0)).ravel()
        )
        gene_rows[f"removed_transcripts_switched_{visible}"] = np.asarray(
            removed[switched].sum(axis=0)
        ).ravel()
        gene_rows[f"removed_transcripts_stable_{visible}"] = np.asarray(
            removed[stable].sum(axis=0)
        ).ravel()
    genes = pd.DataFrame(gene_rows)
    n_cells = len(cells)
    n_switched = int(switched.sum())
    n_eligible = int(eligible.sum())
    n_stable = int(stable.sum())
    for group, denominator in (("switched", n_switched), ("stable", n_stable)):
        for visible in MODELS:
            genes[f"mean_removed_{group}_{visible}"] = (
                genes[f"removed_transcripts_{group}_{visible}"] / denominator
                if denominator else np.nan
            )
        genes[f"delta_mean_removed_{group}_10_minus_1"] = (
            genes[f"mean_removed_{group}_10"] - genes[f"mean_removed_{group}_1"]
        )
    genes["delta_removed_transcripts_10_minus_1"] = (
        genes["removed_transcripts_10"] - genes["removed_transcripts_1"]
    )
    genes["delta_modified_cells_10_minus_1"] = (
        genes["modified_cells_10"] - genes["modified_cells_1"]
    )

    transitions = contingency.stack().rename("n_cells").reset_index()
    transitions["cluster_10_aligned"] = transitions["cluster_10"].map(mapping).fillna(-1).astype(int)
    transitions["switched"] = transitions["cluster_1"] != transitions["cluster_10_aligned"]
    summary = {
        "slice": sample,
        "target_removed_fraction": TARGET_FRACTION,
        "n_cells": n_cells,
        "n_genes": len(genes),
        "n_clusters_1": int(contingency.shape[0]),
        "n_clusters_10": int(contingency.shape[1]),
        "n_switched": n_switched,
        "switch_fraction": n_switched / n_eligible if n_eligible else None,
        "switching_definition": (
            "family_after_hungarian_cluster_matching"
            if switching == "family" else switching
        ),
        "n_switch_eligible": n_eligible,
        "n_ambiguous": int(
            (cells["cluster_match_eligible"] & ~cells["switch_eligible"]).sum()
        ),
        "n_unmatched_cluster_cells": int((~cells['cluster_match_eligible']).sum()),
        "n_hungarian_cluster_matches": len(mapping),
        "cluster_ari_1_vs_10": float(cluster_ari),
        "cluster_nmi_1_vs_10": float(cluster_nmi),
        "metric_nmi_1": float(first["nmi"]),
        "metric_nmi_10": float(last["nmi"]),
        "metric_ari_1": float(first["ari"]),
        "metric_ari_10": float(last["ari"]),
        "saved_metric_nmi_1": float(first["saved_nmi"]),
        "saved_metric_nmi_10": float(last["saved_nmi"]),
        "saved_metric_ari_1": float(first["saved_ari"]),
        "saved_metric_ari_10": float(last["saved_ari"]),
        "reproduced_scores_match_saved_1": bool(first["metrics_match"]),
        "reproduced_scores_match_saved_10": bool(last["metrics_match"]),
        "achieved_removed_fraction_1": float(first["achieved_fraction"]),
        "achieved_removed_fraction_10": float(last["achieved_fraction"]),
        "leiden_resolution": float(first["leiden_resolution"]),
        "leiden_seed_used_by_metric_library": int(first["leiden_seed"]),
    }

    if switching == 'family':
        family_pairs = cells.loc[cells.switch_eligible].groupby(
            ['family_1', 'family_10']).size().rename('n_cells').reset_index()
        cluster_1_families = cells.drop_duplicates('cluster_1').set_index('cluster_1')
        cluster_10_families = cells.drop_duplicates('cluster_10').set_index('cluster_10')
        cluster_matches = pd.DataFrame([
            {
                'cluster_1': cluster_1,
                'cluster_10': cluster_10,
                'cluster_10_aligned': cluster_1,
                'n_overlap_cells': int(contingency.loc[cluster_1, cluster_10]),
                'predominant_type_1': cluster_1_families.loc[cluster_1, 'predominant_type_1'],
                'family_1': cluster_1_families.loc[cluster_1, 'family_1'],
                'predominant_type_10': cluster_10_families.loc[cluster_10, 'predominant_type_10'],
                'family_10': cluster_10_families.loc[cluster_10, 'family_10'],
                'family_switched': bool(
                    pd.notna(cluster_1_families.loc[cluster_1, 'family_1'])
                    and pd.notna(cluster_10_families.loc[cluster_10, 'family_10'])
                    and cluster_1_families.loc[cluster_1, 'family_1']
                    != cluster_10_families.loc[cluster_10, 'family_10']
                ),
            }
            for cluster_10, cluster_1 in mapping.items()
        ])
    output = Path(output_root) / sample
    output.mkdir(parents=True, exist_ok=True)
    if switching == 'family':
        family_pairs.to_csv(output / 'family_transitions.csv', index=False)
        cluster_matches.to_csv(output / 'hungarian_cluster_matches.csv', index=False)
    cells.to_csv(output / "cells.csv.gz", index=False, compression="gzip")
    genes.to_csv(output / "genes.csv", index=False)
    transitions.to_csv(output / "cluster_transitions.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{sample}: {n_switched:,}/{n_cells:,} cells switch ({(summary['switch_fraction'] if n_eligible else float('nan')):.1%})")
    return output


def recompute_family_switch_from_saved(sample: str, input_root: Path, output_root: Path) -> Path:
    """Re-align saved Leiden labels and recompute family-level switch outputs.

    The saved per-cell rows retain the reproduced Leiden cluster assignments
    and removal counts, so this can update the Hungarian alignment and all
    family-derived summaries without rerunning preprocessing and Leiden.
    """
    source = Path(input_root) / sample
    output = Path(output_root) / sample
    cells = pd.read_csv(source / "cells.csv.gz", low_memory=False)
    old_summary = json.loads((source / "summary.json").read_text())
    cluster_1 = cells["cluster_1"].astype(int).to_numpy()
    cluster_10 = cells["cluster_10"].astype(int).to_numpy()
    contingency = pd.crosstab(
        pd.Index(cluster_1, name="cluster_1"),
        pd.Index(cluster_10, name="cluster_10"),
    ).rename_axis(index="cluster_1", columns="cluster_10")
    row_idx, col_idx = linear_sum_assignment(-contingency.to_numpy())
    mapping = {
        int(contingency.columns[col]): int(contingency.index[row])
        for row, col in zip(row_idx, col_idx)
    }
    cells["cluster_10_aligned"] = [mapping.get(cluster, -1) for cluster in cluster_10]
    cells["cluster_switched"] = cells["cluster_10_aligned"].ne(cells["cluster_1"])
    matched_cluster_1 = set(mapping.values())
    matched_cluster_10 = set(mapping)
    cells["cluster_match_eligible"] = (
        cells["cluster_1"].isin(matched_cluster_1)
        & cells["cluster_10"].isin(matched_cluster_10)
    )
    cells = family_switch_labels(cells)
    cells["switch_eligible"] &= cells["cluster_match_eligible"]
    cells["switched"] = (
        cells["switch_eligible"] & cells["family_1"].ne(cells["family_10"])
    )
    cells["delta_modified_genes_10_minus_1"] = (
        cells["modified_genes_10"] - cells["modified_genes_1"]
    )
    cells["delta_removed_transcripts_10_minus_1"] = (
        cells["removed_transcripts_10"] - cells["removed_transcripts_1"]
    )
    eligible = cells["switch_eligible"]
    switched = cells["switched"]
    family_pairs = (
        cells.loc[eligible].groupby(["family_1", "family_10"]).size()
        .rename("n_cells").reset_index()
    )

    type_1 = cells.drop_duplicates("cluster_1").set_index("cluster_1")
    type_10 = cells.drop_duplicates("cluster_10").set_index("cluster_10")
    matches = pd.DataFrame([
        {
            "cluster_1": cluster_1_id,
            "cluster_10": cluster_10_id,
            "cluster_10_aligned": cluster_1_id,
            "n_overlap_cells": int(contingency.loc[cluster_1_id, cluster_10_id]),
            "predominant_type_1": type_1.loc[cluster_1_id, "predominant_type_1"],
            "family_1": type_1.loc[cluster_1_id, "family_1"],
            "predominant_type_10": type_10.loc[cluster_10_id, "predominant_type_10"],
            "family_10": type_10.loc[cluster_10_id, "family_10"],
            "family_switched": bool(
                pd.notna(type_1.loc[cluster_1_id, "family_1"])
                and pd.notna(type_10.loc[cluster_10_id, "family_10"])
                and type_1.loc[cluster_1_id, "family_1"]
                != type_10.loc[cluster_10_id, "family_10"]
            ),
        }
        for cluster_10_id, cluster_1_id in mapping.items()
    ])
    transitions = contingency.stack().rename("n_cells").reset_index()
    transitions["cluster_10_aligned"] = transitions["cluster_10"].map(mapping).fillna(-1).astype(int)
    transitions["switched"] = transitions["cluster_1"] != transitions["cluster_10_aligned"]

    n_cells = len(cells)
    n_eligible = int(eligible.sum())
    n_unmatched = int((~cells["cluster_match_eligible"]).sum())
    n_ambiguous = int((cells["cluster_match_eligible"] & ~eligible).sum())
    summary = dict(old_summary)
    summary.update(
        switching_definition="family_after_hungarian_cluster_matching",
        n_cells=n_cells,
        n_clusters_1=int(contingency.shape[0]),
        n_clusters_10=int(contingency.shape[1]),
        n_hungarian_cluster_matches=len(mapping),
        n_switch_eligible=n_eligible,
        n_unmatched_cluster_cells=n_unmatched,
        n_ambiguous=n_ambiguous,
        n_switched=int(switched.sum()),
        switch_fraction=float(switched.sum() / n_eligible) if n_eligible else None,
    )

    output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(output / "cells.csv.gz", index=False, compression="gzip")
    family_pairs.to_csv(output / "family_transitions.csv", index=False)
    matches.to_csv(output / "hungarian_cluster_matches.csv", index=False)
    transitions.to_csv(output / "cluster_transitions.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Preserve only per-gene totals, which do not depend on the switch labels.
    gene_path = source / "genes.csv"
    if gene_path.exists():
        genes = pd.read_csv(gene_path, low_memory=False)
        keep = [
            column for column in genes.columns
            if column in {
                "slice", "gene", "removed_transcripts_1", "removed_transcripts_10",
                "modified_cells_1", "modified_cells_10",
                "delta_removed_transcripts_10_minus_1", "delta_modified_cells_10_minus_1",
            }
        ]
        genes[keep].to_csv(output / "genes.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", help="Slice basename without .h5ad")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    options = parser.parse_args()
    analyze_slice(options.sample, options.output_root)


if __name__ == "__main__":
    main()
