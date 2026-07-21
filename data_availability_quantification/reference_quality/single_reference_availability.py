#!/usr/bin/env python3
"""Quantify single-reference availability for spatial transcriptomics datasets.

This script reads the harmonized metadata produced by HarmonizeDatabases.ipynb,
identifies spatial transcriptomics query datasets, identifies candidate
single-cell/single-nucleus reference datasets, and scores the best single
reference for each spatial dataset. The score intentionally excludes cell type
overlap because most spatial metadata sources do not provide comparable cell
type annotations.
"""

from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("harmonized_metadata.csv")
DEFAULT_TOP_SCORES = Path("single_reference_scores.csv")
DEFAULT_BEST = Path("single_reference_best.csv")
DEFAULT_SUMMARY = Path("single_reference_summary.csv")

LIST_COLUMNS = [
    "assays",
    "tissues",
    "tissue_general",
    "diseases",
    "cell_types",
    "organisms",
    "organisms_harmonized",
    "tissue_general_harmonized",
    "tissues_harmonized",
    "suspension_types",
]

BASE_COLUMNS = [
    "dataset_id",
    "source_database",
    "collection_id",
    "collection_name",
    "dataset_title",
    "dataset_type",
    "assays",
    "tissues",
    "tissue_general",
    "diseases",
    "cell_types",
    "organism",
    "organisms",
    "organisms_harmonized",
    "organisms_harmonized_label",
    "tissue_general_harmonized",
    "tissue_general_harmonized_label",
    "tissues_harmonized",
    "tissues_harmonized_label",
    "suspension_types",
    "n_cells",
    "dataset_total_cell_count",
    "n_donors",
    "hubmap_id",
    "data_access_level",
]

HUBMAP_SPATIAL_DATASET_TYPES = {
    "GeoMx (NGS)",
    "HiFi-Slide",
    "MERFISH",
    "seqFISH",
    "Slideseq",
    "Visium",
    "Visium (no probes)",
    "Visium (with probes)",
    "Xenium",
}

HUBMAP_REFERENCE_DATASET_TYPES = {
    "10X Multiome",
    "RNAseq",
    "RNAseq (with probes)",
    "SNARE-seq2",
}

SPATIAL_KEYWORDS = {
    "visium",
    "xenium",
    "cosmx",
    "geomx",
    "merfish",
    "seqfish",
    "slide-seq",
    "slideseq",
    "stereoseq",
    "stereo-seq",
    "seq-scope",
    "seqscope",
    "hdst",
    "hd-st",
    "dbit-seq",
    "dbitseq",
    "spatial transcriptomics",
}


def parse_list_value(value) -> list[str]:
    """Parse list-like CSV values into clean string lists."""
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, tuple) or isinstance(value, set):
        raw_values = list(value)
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        raw_values = []
    else:
        text = str(value).strip()
        if not text or text.casefold() in {"nan", "none", "null"}:
            raw_values = []
        else:
            try:
                parsed = literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = text
            raw_values = parsed if isinstance(parsed, list) else [parsed]

    out = []
    seen = set()
    for item in raw_values:
        if item is None or (isinstance(item, float) and np.isnan(item)):
            continue
        text = str(item).strip()
        if not text or text.casefold() in {"nan", "none", "null"}:
            continue
        key = normalize_text(text)
        if key not in seen:
            out.append(text)
            seen.add(key)
    return out


def normalize_text(value) -> str:
    text = str(value).strip().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def normalized_set(values) -> set[str]:
    if isinstance(values, (set, frozenset)):
        return set(values)
    return {normalize_text(value) for value in parse_list_value(values)}


def first_nonempty_list(*values) -> list[str]:
    for value in values:
        parsed = parse_list_value(value)
        if parsed:
            return parsed
    return []


def any_overlap(left, right) -> bool:
    return bool(normalized_set(left) & normalized_set(right))


def numeric_value(row: pd.Series, *columns: str) -> float:
    for column in columns:
        if column not in row.index:
            continue
        value = pd.to_numeric(row.get(column), errors="coerce")
        if not pd.isna(value):
            return float(value)
    return np.nan


def load_harmonized_metadata(input_path: Path) -> pd.DataFrame:
    header = pd.read_csv(input_path, nrows=0).columns
    usecols = [column for column in BASE_COLUMNS if column in header]
    df = pd.read_csv(input_path, usecols=usecols, low_memory=False)

    for column in LIST_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(parse_list_value)

    if "organism" not in df.columns:
        df["organism"] = np.nan

    df["organisms_for_scoring"] = df.apply(
        lambda row: first_nonempty_list(
            row.get("organisms_harmonized"),
            row.get("organisms"),
            row.get("organism"),
        ),
        axis=1,
    )
    df["tissue_general_for_scoring"] = df.apply(
        lambda row: first_nonempty_list(
            row.get("tissue_general_harmonized"),
            row.get("tissue_general"),
        ),
        axis=1,
    )
    df["tissues_for_scoring"] = df.apply(
        lambda row: first_nonempty_list(
            row.get("tissues_harmonized"),
            row.get("tissues"),
        ),
        axis=1,
    )
    df["_organism_set"] = df["organisms_for_scoring"].apply(normalized_set)
    df["_tissue_general_set"] = df["tissue_general_for_scoring"].apply(normalized_set)
    df["_tissues_set"] = df["tissues_for_scoring"].apply(normalized_set)
    df["_tissue_broad_set"] = df.apply(
        lambda row: row["_tissue_general_set"] | row["_tissues_set"],
        axis=1,
    )
    df["_disease_set"] = df["diseases"].apply(normalized_set) if "diseases" in df.columns else [set()] * len(df)
    return df


def row_text(row: pd.Series, columns: list[str]) -> str:
    values = []
    for column in columns:
        value = row.get(column, "")
        if isinstance(value, list):
            values.extend(value)
        elif value is not None and not (isinstance(value, float) and np.isnan(value)):
            values.append(str(value))
    return normalize_text(" ".join(values))


def is_spatial_dataset(row: pd.Series) -> bool:
    source = row.get("source_database")
    if source == "cellxgene":
        return False

    if source == "museum_of_spatial_transcriptomics":
        return True

    dataset_type = row.get("dataset_type")
    if source == "hubmap" and dataset_type in HUBMAP_SPATIAL_DATASET_TYPES:
        return True

    text = row_text(row, ["dataset_type", "assays", "dataset_title", "collection_name"])
    return any(keyword in text for keyword in SPATIAL_KEYWORDS)


def is_candidate_reference(row: pd.Series) -> bool:
    source = row.get("source_database")
    dataset_type = row.get("dataset_type")

    if source == "cellxgene":
        return bool(row.get("organisms_for_scoring")) and not is_spatial_dataset(row)

    if source == "hubmap" and dataset_type in HUBMAP_REFERENCE_DATASET_TYPES:
        return bool(row.get("organisms_for_scoring")) and not is_spatial_dataset(row)

    return False


def organism_matches(query_row: pd.Series, ref_row: pd.Series) -> bool:
    return bool(query_row.get("_organism_set", set()) & ref_row.get("_organism_set", set()))


def tissue_score(query_row: pd.Series, ref_row: pd.Series) -> float:
    query_specific = query_row.get("_tissues_set", set())
    ref_specific = ref_row.get("_tissues_set", set())

    if query_specific & ref_specific:
        return 1.0

    if query_row.get("_tissue_broad_set", set()) & ref_row.get("_tissue_broad_set", set()):
        return 0.6

    return 0.0


def tissue_match_flags(query_row: pd.Series, ref_row: pd.Series) -> dict[str, object]:
    same_tissue = bool(query_row.get("_tissues_set", set()) & ref_row.get("_tissues_set", set()))
    any_tissue = bool(query_row.get("_tissue_broad_set", set()) & ref_row.get("_tissue_broad_set", set()))

    if same_tissue:
        match_level = "specific"
    elif any_tissue:
        match_level = "general"
    else:
        match_level = "none"

    return {
        "has_same_tissue_match": same_tissue,
        "has_tissue_match": any_tissue,
        "tissue_match_level": match_level,
        "missing_same_tissue_reference": not same_tissue,
    }


def disease_score(query_row: pd.Series, ref_row: pd.Series) -> float:
    query_disease = query_row.get("_disease_set", set())
    ref_disease = ref_row.get("_disease_set", set())

    if not query_disease or not ref_disease:
        return 0.0
    if query_disease & ref_disease:
        return 1.0
    if "normal" in ref_disease and "normal" not in query_disease:
        return 0.35
    if "normal" in query_disease and "normal" not in ref_disease:
        return 0.25
    return 0.0


def quality_score(ref_row: pd.Series, min_cells=5_000, good_cells=50_000, min_donors=1, good_donors=5) -> float:
    n_cells = numeric_value(ref_row, "n_cells", "dataset_total_cell_count")
    if pd.isna(n_cells) or n_cells <= 0:
        n_cells = 0.0

    n_donors = numeric_value(ref_row, "n_donors")
    if pd.isna(n_donors):
        n_donors = 0.0

    cell_score = np.clip((n_cells - min_cells) / (good_cells - min_cells), 0, 1)
    donor_score = np.clip((n_donors - min_donors) / (good_donors - min_donors), 0, 1)
    return float(0.7 * cell_score + 0.3 * donor_score)


def same_study_score(query_row: pd.Series, ref_row: pd.Series) -> float:
    query_collection = query_row.get("collection_id")
    ref_collection = ref_row.get("collection_id")
    if pd.isna(query_collection) or pd.isna(ref_collection):
        return 0.0
    return float(str(query_collection) == str(ref_collection))


def score_to_tier(score: float) -> str:
    if score >= 0.80:
        return "A: strong matched reference"
    if score >= 0.60:
        return "B: usable reference"
    if score >= 0.35:
        return "C: partial reference"
    if score > 0:
        return "D: weak reference"
    return "E: no metadata-supported reference"


def reference_score_no_celltype(query_row: pd.Series, ref_row: pd.Series) -> tuple[float, dict[str, float]]:
    components = {
        "tissue": tissue_score(query_row, ref_row),
        "disease": disease_score(query_row, ref_row),
        "quality": quality_score(ref_row),
        "same_study": same_study_score(query_row, ref_row),
    }

    if components["tissue"] == 0:
        total = 0.0
    else:
        # Renormalized from reference_score_v2 after removing cell type overlap:
        # tissue: 0.40 / (0.40 + 0.30), disease: 0.30 / (0.40 + 0.30).
        bio_score = (4 / 7) * components["tissue"] + (3 / 7) * components["disease"]
        modifier = 0.85 + 0.10 * components["quality"] + 0.05 * components["same_study"]
        total = bio_score * modifier

    return float(total), components


def score_spatial_dataset(
    query_row: pd.Series,
    ref_rows: dict[int, pd.Series],
    ref_indices_by_organism: dict[str, set[int]],
    all_ref_indices: set[int],
    top_k: int,
    require_same_species: bool,
) -> pd.DataFrame:
    if require_same_species:
        candidate_indices = set()
        for organism in query_row.get("_organism_set", set()):
            candidate_indices.update(ref_indices_by_organism.get(organism, set()))
    else:
        candidate_indices = set(all_ref_indices)

    query_tissues = query_row.get("_tissue_broad_set", set())
    candidate_indices = [
        index
        for index in candidate_indices
        if ref_rows[index].get("dataset_id") != query_row.get("dataset_id")
        and bool(query_tissues & ref_rows[index].get("_tissue_broad_set", set()))
    ]

    records = []

    for index in candidate_indices:
        ref_row = ref_rows[index]
        score, components = reference_score_no_celltype(query_row, ref_row)
        tissue_flags = tissue_match_flags(query_row, ref_row)
        records.append(
            {
                "spatial_dataset_id": query_row.get("dataset_id"),
                "spatial_dataset_title": query_row.get("dataset_title"),
                "spatial_source_database": query_row.get("source_database"),
                "spatial_collection_id": query_row.get("collection_id"),
                "spatial_collection_name": query_row.get("collection_name"),
                "spatial_dataset_type": query_row.get("dataset_type"),
                "spatial_assays": query_row.get("assays"),
                "spatial_organisms": query_row.get("organisms_for_scoring"),
                "spatial_tissue_general": query_row.get("tissue_general_for_scoring"),
                "spatial_tissues": query_row.get("tissues_for_scoring"),
                "spatial_diseases": query_row.get("diseases"),
                "ref_dataset_id": ref_row.get("dataset_id"),
                "ref_dataset_title": ref_row.get("dataset_title"),
                "ref_source_database": ref_row.get("source_database"),
                "ref_collection_id": ref_row.get("collection_id"),
                "ref_collection_name": ref_row.get("collection_name"),
                "ref_dataset_type": ref_row.get("dataset_type"),
                "ref_assays": ref_row.get("assays"),
                "ref_organisms": ref_row.get("organisms_for_scoring"),
                "ref_tissue_general": ref_row.get("tissue_general_for_scoring"),
                "ref_tissues": ref_row.get("tissues_for_scoring"),
                "ref_diseases": ref_row.get("diseases"),
                "score": score,
                "tier": score_to_tier(score),
                **tissue_flags,
                "tissue_score": components["tissue"],
                "disease_score": components["disease"],
                "quality_score": components["quality"],
                "same_study_score": components["same_study"],
                "ref_n_cells": numeric_value(ref_row, "n_cells", "dataset_total_cell_count"),
                "ref_n_donors": numeric_value(ref_row, "n_donors"),
            }
        )

    if not records:
        return pd.DataFrame(
            [
                {
                    "spatial_dataset_id": query_row.get("dataset_id"),
                    "spatial_dataset_title": query_row.get("dataset_title"),
                    "spatial_source_database": query_row.get("source_database"),
                    "spatial_collection_id": query_row.get("collection_id"),
                    "spatial_collection_name": query_row.get("collection_name"),
                    "spatial_dataset_type": query_row.get("dataset_type"),
                    "spatial_assays": query_row.get("assays"),
                    "spatial_organisms": query_row.get("organisms_for_scoring"),
                    "spatial_tissue_general": query_row.get("tissue_general_for_scoring"),
                    "spatial_tissues": query_row.get("tissues_for_scoring"),
                    "spatial_diseases": query_row.get("diseases"),
                    "ref_dataset_id": None,
                    "ref_dataset_title": None,
                    "ref_source_database": None,
                    "ref_collection_id": None,
                    "ref_collection_name": None,
                    "ref_dataset_type": None,
                    "ref_assays": None,
                    "ref_organisms": None,
                    "ref_tissue_general": None,
                    "ref_tissues": None,
                    "ref_diseases": None,
                    "score": 0.0,
                    "tier": score_to_tier(0.0),
                    "has_same_tissue_match": False,
                    "has_tissue_match": False,
                    "tissue_match_level": "none",
                    "missing_same_tissue_reference": True,
                    "tissue_score": 0.0,
                    "disease_score": 0.0,
                    "quality_score": 0.0,
                    "same_study_score": 0.0,
                }
            ]
        )

    return pd.DataFrame(records).sort_values("score", ascending=False).head(top_k)


def build_reference_indices(ref_meta: pd.DataFrame) -> tuple[dict[int, pd.Series], dict[str, set[int]], set[int]]:
    ref_rows = {}
    ref_indices_by_organism: dict[str, set[int]] = {}

    for index, ref_row in ref_meta.iterrows():
        ref_rows[index] = ref_row
        for organism in ref_row.get("_organism_set", set()):
            ref_indices_by_organism.setdefault(organism, set()).add(index)

    return ref_rows, ref_indices_by_organism, set(ref_rows)


def stringify_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].apply(lambda value: isinstance(value, list)).any():
            out[column] = out[column].apply(
                lambda value: "; ".join(map(str, value)) if isinstance(value, list) else value
            )
    return out


def build_summary(spatial_meta: pd.DataFrame, ref_meta: pd.DataFrame, best_refs: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "n_spatial_datasets", "value": len(spatial_meta)},
        {"metric": "n_reference_candidates", "value": len(ref_meta)},
        {"metric": "n_spatial_with_score_gt_0", "value": int((best_refs["score"] > 0).sum())},
        {"metric": "n_spatial_with_same_tissue_reference", "value": int(best_refs["has_same_tissue_match"].sum())},
        {
            "metric": "n_spatial_without_same_tissue_reference",
            "value": int((~best_refs["has_same_tissue_match"]).sum()),
        },
        {
            "metric": "fraction_spatial_without_same_tissue_reference",
            "value": float((~best_refs["has_same_tissue_match"]).mean()),
        },
        {"metric": "fraction_spatial_with_score_gt_0", "value": float((best_refs["score"] > 0).mean())},
        {"metric": "fraction_spatial_with_score_ge_0.35", "value": float((best_refs["score"] >= 0.35).mean())},
        {"metric": "fraction_spatial_with_score_ge_0.60", "value": float((best_refs["score"] >= 0.60).mean())},
        {"metric": "fraction_spatial_with_score_ge_0.80", "value": float((best_refs["score"] >= 0.80).mean())},
        {"metric": "mean_best_score", "value": float(best_refs["score"].mean())},
        {"metric": "median_best_score", "value": float(best_refs["score"].median())},
    ]

    tier_counts = best_refs["tier"].value_counts(dropna=False)
    for tier, count in tier_counts.items():
        rows.append({"metric": f"tier_count__{tier}", "value": int(count)})

    source_counts = spatial_meta["source_database"].value_counts(dropna=False)
    for source, count in source_counts.items():
        rows.append({"metric": f"spatial_source_count__{source}", "value": int(count)})

    ref_source_counts = ref_meta["source_database"].value_counts(dropna=False)
    for source, count in ref_source_counts.items():
        rows.append({"metric": f"reference_source_count__{source}", "value": int(count)})

    return pd.DataFrame(rows)


def run_quantification(
    input_path: Path,
    top_scores_path: Path,
    best_path: Path,
    summary_path: Path,
    top_k: int,
    require_same_species: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = load_harmonized_metadata(input_path)
    spatial_meta = metadata[metadata.apply(is_spatial_dataset, axis=1)].copy()
    ref_meta = metadata[metadata.apply(is_candidate_reference, axis=1)].copy().reset_index(drop=True)
    ref_rows, ref_indices_by_organism, all_ref_indices = build_reference_indices(ref_meta)

    all_scores = []
    for _, query_row in spatial_meta.iterrows():
        all_scores.append(
            score_spatial_dataset(
                query_row=query_row,
                ref_rows=ref_rows,
                ref_indices_by_organism=ref_indices_by_organism,
                all_ref_indices=all_ref_indices,
                top_k=top_k,
                require_same_species=require_same_species,
            )
        )

    reference_scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    best_refs = (
        reference_scores.sort_values(["spatial_dataset_id", "score"], ascending=[True, False])
        .groupby("spatial_dataset_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    summary = build_summary(spatial_meta, ref_meta, best_refs)

    stringify_list_columns(reference_scores).to_csv(top_scores_path, index=False)
    stringify_list_columns(best_refs).to_csv(best_path, index=False)
    summary.to_csv(summary_path, index=False)

    return reference_scores, best_refs, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top-scores-output", type=Path, default=DEFAULT_TOP_SCORES)
    parser.add_argument("--best-output", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--allow-cross-species",
        action="store_true",
        help="Do not require spatial and reference datasets to have overlapping organisms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, best_refs, summary = run_quantification(
        input_path=args.input,
        top_scores_path=args.top_scores_output,
        best_path=args.best_output,
        summary_path=args.summary_output,
        top_k=args.top_k,
        require_same_species=not args.allow_cross_species,
    )
    print(f"Wrote {args.top_scores_output}")
    print(f"Wrote {args.best_output}")
    print(f"Wrote {args.summary_output}")
    print(summary.to_string(index=False))
    print("\nBest-reference tier counts:")
    print(best_refs["tier"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
