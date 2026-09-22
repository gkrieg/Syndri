#!/usr/bin/env python3
"""Backfill per-cell count summaries into an existing metrics CSV.

This recreates the raw, Syndri-cleaned, and SPLIT count matrices using the
saved metrics configuration. It does not run normalization, PCA, neighbors,
clustering, or separation metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from metrics_eval import (
    cleaned_counts_for_removed_fraction_maybe_grouped,
    counts_from_layer,
    median_cell_count_summaries,
    parse_float_list,
    parse_str_list,
    sample_adata,
    sanitize_numeric_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Defaults to non_tcell_metrics_config.json beside the CSV.",
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=None,
        help="Override the input H5AD recorded in the saved configuration.",
    )
    parser.add_argument(
        "--label-key",
        default=None,
        help="Override the evaluation label column recorded in the saved configuration.",
    )
    parser.add_argument(
        "--removal-group-key",
        default=None,
        help="Override the cell-type grouping column used for grouped removal.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not preserve the original CSV as <name>.pre_count_summary.csv.",
    )
    return parser.parse_args()


def summarized_matrix(X, labels) -> tuple[float, float]:
    valid = ~pd.Series(labels).isna().to_numpy()
    if not np.any(valid):
        raise ValueError("No cells have a non-null evaluation label.")
    return median_cell_count_summaries(X[valid])


def main() -> None:
    cli = parse_args()
    csv_path = cli.metrics_csv.expanduser().resolve()
    config_path = (
        cli.config_json.expanduser().resolve()
        if cli.config_json is not None
        else csv_path.with_name("non_tcell_metrics_config.json")
    )
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = json.loads(config_path.read_text())
    args = SimpleNamespace(**config)
    if cli.label_key is not None:
        args.label_key = cli.label_key
    if cli.removal_group_key is not None:
        args.removal_group_key = cli.removal_group_key
    if cli.input_h5ad is not None:
        input_path = cli.input_h5ad.expanduser().resolve()
    else:
        inputs = config.get("input_files", [])
        if len(inputs) != 1:
            raise ValueError(
                "Saved config must contain exactly one input file; "
                "otherwise pass --input-h5ad."
            )
        input_path = Path(inputs[0]).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    table = pd.read_csv(csv_path)
    adata = ad.read_h5ad(input_path)
    adata.X = sanitize_numeric_matrix(adata.X)
    adata = sample_adata(adata, args)

    if args.label_key not in adata.obs:
        raise KeyError(f"obs column {args.label_key!r} is missing from {input_path}")
    labels = adata.obs[args.label_key]
    exclude_labels = set(parse_str_list(args.exclude_labels))
    if exclude_labels:
        keep = ~labels.astype(str).isin(exclude_labels)
        adata = adata[keep].copy()
        labels = adata.obs[args.label_key]

    summaries: dict[str, tuple[float, float]] = {
        "admixed": summarized_matrix(adata.X, labels),
    }

    if args.admix_probs_layer not in adata.layers:
        raise KeyError(f"Layer {args.admix_probs_layer!r} is missing from {input_path}")
    admix_probs = adata.layers[args.admix_probs_layer]
    removal_labels = None
    if args.cell_type_specific_removal:
        removal_group_key = args.removal_group_key or args.label_key
        if removal_group_key not in adata.obs:
            raise KeyError(f"obs column {removal_group_key!r} is missing from {input_path}")
        removal_labels = adata.obs[removal_group_key]

    for fraction in parse_float_list(args.target_removed_fractions):
        cleaned, *_ = cleaned_counts_for_removed_fraction_maybe_grouped(
            adata.X,
            admix_probs,
            fraction,
            args,
            labels=removal_labels,
        )
        case = f"cleaned_admixed_removed_frac_{fraction:g}"
        summaries[case] = summarized_matrix(cleaned, labels)

    if args.additional_counts_layer:
        case = args.additional_counts_case_label or args.additional_counts_layer
        additional = counts_from_layer(adata, args.additional_counts_layer)
        summaries[case] = summarized_matrix(additional, labels)

    table["median_genes_per_cell"] = np.nan
    table["median_transcripts_per_cell"] = np.nan
    for case, (median_genes, median_transcripts) in summaries.items():
        matches = table["case"].astype(str).eq(case)
        if not matches.any():
            raise ValueError(f"Metrics CSV has no row for expected case {case!r}")
        table.loc[matches, "median_genes_per_cell"] = median_genes
        table.loc[matches, "median_transcripts_per_cell"] = median_transcripts

    missing = table[["median_genes_per_cell", "median_transcripts_per_cell"]].isna().any(axis=1)
    if missing.any():
        cases = table.loc[missing, "case"].astype(str).tolist()
        raise ValueError(f"No count summary was generated for CSV cases: {cases}")

    if not cli.no_backup:
        backup = csv_path.with_name(f"{csv_path.stem}.pre_count_summary.csv")
        if not backup.exists():
            shutil.copy2(csv_path, backup)
            print(f"Backup: {backup}")

    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(csv_path)
    print(f"Updated {len(table)} rows: {csv_path}")


if __name__ == "__main__":
    main()
