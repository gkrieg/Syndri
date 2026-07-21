import warnings
warnings.filterwarnings("ignore")

import hashlib
import re
import urllib.parse

import numpy as np
import pandas as pd
from collections.abc import Iterable


# -----------------------------
# Helpers
# -----------------------------

def as_set(x):
    """
    Convert list-like CELLxGENE metadata entries into a normalized set.
    Handles [], NaN, strings, tuples, lists, and sets.
    """
    if x is None:
        return set()
    if isinstance(x, float) and np.isnan(x):
        return set()
    if isinstance(x, str):
        return {x.lower().strip()} if x.strip() else set()
    if isinstance(x, Iterable):
        return {
            str(v).lower().strip()
            for v in x
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        }
    return {str(x).lower().strip()}


def jaccard(a, b):
    a = as_set(a)
    b = as_set(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return len(a & b) / len(a | b)


def coverage(query_values, ref_values):
    """
    Fraction of query values covered by reference values.
    Useful for: what fraction of ST cell types are present in the scRNA reference?
    """
    query_values = as_set(query_values)
    ref_values = as_set(ref_values)
    if len(query_values) == 0:
        return np.nan
    return len(query_values & ref_values) / len(query_values)


def any_overlap(a, b):
    return float(len(as_set(a) & as_set(b)) > 0)


# -----------------------------
# Component scores
# -----------------------------

def tissue_score(st_row, ref_row):
    """
    Gives full credit for exact tissue overlap,
    partial credit for tissue_general overlap.
    """
    exact = any_overlap(st_row["tissues"], ref_row["tissues"])
    general = any_overlap(st_row["tissue_general"], ref_row["tissue_general"])

    if exact:
        return 1.0
    elif general:
        return 0.6
    else:
        return 0.0


def disease_score(st_row, ref_row):
    """
    Simple disease match score.
    You may want to add custom mappings later:
    e.g. NSCLC ~ lung cancer, normal as partial match for same tissue.
    """
    st_disease = as_set(st_row["diseases"])
    ref_disease = as_set(ref_row["diseases"])

    if len(st_disease) == 0 or len(ref_disease) == 0:
        return 0.0

    if st_disease & ref_disease:
        return 1.0

    # If the spatial dataset is diseased but ref is normal, give weak partial credit.
    if "normal" in ref_disease and "normal" not in st_disease:
        return 0.35

    # If ST is normal but ref is diseased, usually less useful.
    if "normal" in st_disease and "normal" not in ref_disease:
        return 0.25

    return 0.0


def celltype_score(st_row, ref_row):
    """
    Fraction of spatial cell types represented in candidate scRNA-seq reference.
    """
    score = coverage(st_row["cell_types"], ref_row["cell_types"])
    if np.isnan(score):
        return 0.0
    return score


def quality_score(ref_row, min_cells=5_000, good_cells=50_000, min_donors=1, good_donors=5):
    """
    Reference quality based on cell count and donor count.
    Uses saturating scores so huge datasets do not dominate.
    """
    n_cells = ref_row.get("n_cells", np.nan)
    if pd.isna(n_cells) or n_cells <= 0:
        n_cells = ref_row.get("dataset_total_cell_count", 0)

    n_donors = ref_row.get("n_donors", 0)
    if pd.isna(n_donors):
        n_donors = 0

    # cell score: 0 below min_cells, 1 at good_cells or above
    cell_score = (n_cells - min_cells) / (good_cells - min_cells)
    cell_score = np.clip(cell_score, 0, 1)

    # donor score: 0 below min_donors, 1 at good_donors or above
    donor_score = (n_donors - min_donors) / (good_donors - min_donors)
    donor_score = np.clip(donor_score, 0, 1)

    return 0.7 * cell_score + 0.3 * donor_score


def same_study_score(st_row, ref_row):
    """
    Gives credit if spatial and reference datasets are from same CELLxGENE collection.
    This is a reasonable proxy for same publication/study.
    """
    st_collection = st_row.get("collection_id", None)
    ref_collection = ref_row.get("collection_id", None)

    if pd.isna(st_collection) or pd.isna(ref_collection):
        return 0.0

    return float(st_collection == ref_collection)


# -----------------------------
# Main score
# -----------------------------

def reference_score(
    st_row,
    ref_row,
    weights=None,
):
    if weights is None:
        weights = {
            "tissue": 0.30,
            "disease": 0.25,
            "celltype": 0.25,
            "quality": 0.10,
            "same_study": 0.10,
        }

    components = {
        "tissue": tissue_score(st_row, ref_row),
        "disease": disease_score(st_row, ref_row),
        "celltype": celltype_score(st_row, ref_row),
        "quality": quality_score(ref_row),
        "same_study": same_study_score(st_row, ref_row),
    }

    total = sum(weights[k] * components[k] for k in weights)

    return total, components

def reference_score_v2(st_row, ref_row):
    components = {
        "tissue": tissue_score(st_row, ref_row),
        "disease": disease_score(st_row, ref_row),
        "celltype": celltype_score(st_row, ref_row),
        "quality": quality_score(ref_row),
        "same_study": same_study_score(st_row, ref_row),
    }

    bio_score = (
        0.40 * components["tissue"]
        + 0.30 * components["disease"]
        + 0.30 * components["celltype"]
    )

    # Require at least broad tissue match.
    # This prevents same-study/quality from creating false reference matches.
    if components["tissue"] == 0:
        total = 0.0
    else:
        modifier = (
            0.85
            + 0.10 * components["quality"]
            + 0.05 * components["same_study"]
        )
        total = bio_score * modifier

    return total, components

# def score_to_tier(score):
#     if score >= 0.85:
#         return "A: strong matched reference"
#     elif score >= 0.65:
#         return "B: usable reference"
#     elif score >= 0.40:
#         return "C: partial reference"
#     elif score > 0:
#         return "D: weak reference"
#     else:
#         return "E: no usable reference"
    
def score_to_tier(score):
    if score >= 0.80:
        return "A: strong matched reference"
    elif score >= 0.60:
        return "B: usable reference"
    elif score >= 0.35:
        return "C: partial reference"
    elif score > 0:
        return "D: weak reference"
    else:
        return "E: no metadata-supported reference"
    

def is_probably_scrna_reference(row):
    """
    Conservative filter for likely single-cell/nucleus RNA-seq references.
    Adjust depending on what CELLxGENE assays you see.
    """
    assays = as_set(row.get("assays", []))
    suspension_types = as_set(row.get("suspension_types", []))

    has_cells = row.get("n_cells", 0) > 0

    # CELLxGENE suspension_type is often "cell" or "nucleus" for sc/snRNA-seq.
    is_cell_or_nucleus = bool(suspension_types & {"cell", "nucleus"})

    # Exclude obvious spatial assays if present.
    spatial_keywords = {
        "visium",
        "slide-seq",
        "slideseq",
        "merfish",
        "xenium",
        "cosmx",
        "seq-scope",
        "stereoseq",
        "stereo-seq",
    }

    assay_text = " ".join(assays)
    is_spatial = any(k in assay_text for k in spatial_keywords)

    return has_cells and is_cell_or_nucleus and not is_spatial

def get_first_value(x):
    vals = list(as_set(x))
    return vals[0] if vals else None


def as_organism_set(x):
    return {
        value.replace("_", " ").replace("-", " ")
        for value in as_set(x)
    }


def row_organism_set(row):
    organism = as_organism_set(row.get("organism", None))
    if organism:
        return organism
    return as_organism_set(row.get("organisms", []))


def organism_matches(st_row, ref_row):
    st_org = row_organism_set(st_row)
    ref_org = row_organism_set(ref_row)

    if not st_org or not ref_org:
        return False

    return bool(st_org & ref_org)

def find_best_references_for_spatial_dataset(
    st_row,
    ref_meta,
    top_k=5,
    require_same_species=True,
):
    # Start from all candidate sc/snRNA references
    candidates = ref_meta.copy()

    # Species/organism gate only
    if require_same_species:
        candidates = candidates[
            candidates.apply(lambda r: organism_matches(st_row, r), axis=1)
        ].copy()

    # Avoid matching dataset to itself
    candidates = candidates[candidates["dataset_id"] != st_row["dataset_id"]].copy()

    records = []

    for _, ref_row in candidates.iterrows():
        score, components = reference_score_v2(st_row, ref_row)

        records.append({
            "spatial_dataset_id": st_row["dataset_id"],
            "spatial_dataset_title": st_row.get("dataset_title", None),
            "spatial_collection_name": st_row.get("collection_name", None),
            "spatial_source_database": st_row.get("source_database", None),
            "ref_dataset_id": ref_row["dataset_id"],
            "ref_dataset_title": ref_row.get("dataset_title", None),
            "ref_collection_name": ref_row.get("collection_name", None),
            "ref_collection_id": ref_row.get("collection_id", None),
            "ref_source_database": ref_row.get("source_database", None),
            "score": score,
            "tier": score_to_tier(score),
            "tissue_score": components["tissue"],
            "disease_score": components["disease"],
            "celltype_score": components["celltype"],
            "quality_score": components["quality"],
            "same_study_score": components["same_study"],
            "ref_n_cells": ref_row.get("n_cells", np.nan),
            "ref_n_donors": ref_row.get("n_donors", np.nan),
            "ref_assays": ref_row.get("assays", []),
            "ref_tissues": ref_row.get("tissues", []),
            "ref_tissue_general": ref_row.get("tissue_general", []),
            "ref_diseases": ref_row.get("diseases", []),
            "ref_organism": ref_row.get("organism", ref_row.get("organisms", None)),
        })

    out = pd.DataFrame(records)

    if out.empty:
        return pd.DataFrame([{
            "spatial_dataset_id": st_row["dataset_id"],
            "spatial_dataset_title": st_row.get("dataset_title", None),
            "spatial_collection_name": st_row.get("collection_name", None),
            "spatial_source_database": st_row.get("source_database", None),
            "ref_dataset_id": None,
            "ref_dataset_title": None,
            "score": 0.0,
            "tier": "E: no metadata-supported reference",
            "tissue_score": 0.0,
            "disease_score": 0.0,
            "celltype_score": 0.0,
            "quality_score": 0.0,
            "same_study_score": 0.0,
        }])

    return out.sort_values("score", ascending=False).head(top_k)


MUSEUM_ST_SHEET_ID = "1sJDb9B7AtYmfKv4-m8XR7uc3XXw_k4kGSout8cqZ8bY"
MUSEUM_ST_GID = "588531469"
MUSEUM_ST_SHEET_TABS = [
    "ROI Selection",
    "NGS Barcoding",
    "smFISH",
    "ISS",
    "De Novo",
    "Analysis",
    "Multiomics",
    "LCM",
]


def museum_spatial_transcriptomics_csv_url(
    sheet_id=MUSEUM_ST_SHEET_ID,
    gid=MUSEUM_ST_GID,
    sheet_name=None,
):
    if sheet_name is not None:
        return (
            "https://docs.google.com/spreadsheets/d/"
            f"{sheet_id}/gviz/tq?tqx=out:csv&sheet={urllib.parse.quote(str(sheet_name))}"
        )

    return (
        "https://docs.google.com/spreadsheets/d/"
        f"{sheet_id}/export?format=csv&gid={urllib.parse.quote(str(gid))}"
    )


def fetch_museum_spatial_transcriptomics(
    sheet_id=MUSEUM_ST_SHEET_ID,
    gid=MUSEUM_ST_GID,
    sheet_names=MUSEUM_ST_SHEET_TABS,
):
    """
    Read one or more tabs from the Museum of Spatial Transcriptomics public Google Sheet.
    Returns the raw sheet table; use normalize_museum_spatial_transcriptomics()
    to convert it into the scoring schema.
    """
    if sheet_names is None:
        return pd.read_csv(museum_spatial_transcriptomics_csv_url(sheet_id=sheet_id, gid=gid))

    if isinstance(sheet_names, str):
        sheet_names = [sheet_names]

    frames = []
    errors = {}
    for sheet_name in sheet_names:
        try:
            frame = pd.read_csv(
                museum_spatial_transcriptomics_csv_url(
                    sheet_id=sheet_id,
                    sheet_name=sheet_name,
                )
            )
        except Exception as exc:
            errors[sheet_name] = str(exc)
            continue

        frame["museum_sheet_tab"] = sheet_name
        frame["museum_sheet_row"] = np.arange(len(frame))
        frames.append(frame)

    if not frames:
        raise RuntimeError(
            "No Museum of Spatial Transcriptomics sheets could be fetched. "
            f"Fetch errors: {errors}"
        )

    out = pd.concat(frames, ignore_index=True, sort=False)
    out.attrs["fetch_errors"] = errors
    if errors:
        warnings.warn(f"Some Museum of ST tabs could not be fetched: {errors}")
    return out


def _clean_museum_value(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    text = str(x).strip()
    return text if text else None


def _clean_museum_list(*values):
    out = []
    seen = set()
    for value in values:
        text = _clean_museum_value(value)
        if text is None:
            continue
        key = text.lower()
        if key not in seen:
            out.append(text)
            seen.add(key)
    return out


def _parse_museum_number(x):
    text = _clean_museum_value(x)
    if text is None:
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
    return float(match.group()) if match else np.nan


def _normalize_museum_disease(pathological):
    text = _clean_museum_value(pathological)
    if text is None:
        return ["normal"]
    low = text.lower()
    if low in {"false", "no", "none", "normal", "healthy", "0"}:
        return ["normal"]
    if low in {"true", "yes", "1"}:
        return ["pathological"]
    return [text]


def _museum_dataset_id(row, index):
    parts = [
        _clean_museum_value(row.get("accession")),
        _clean_museum_value(row.get("pmid")),
        _clean_museum_value(row.get("title")),
        _clean_museum_value(row.get("method")),
        _clean_museum_value(row.get("library")),
        _clean_museum_value(row.get("tissue")),
        _clean_museum_value(row.get("organ")),
        _clean_museum_value(row.get("species")),
        _clean_museum_value(row.get("developmental_stage")),
        _clean_museum_value(row.get("sex")),
        _clean_museum_value(row.get("data_link")),
    ]
    digest = hashlib.sha1("|".join(p or "" for p in parts).encode("utf-8")).hexdigest()[:12]
    return f"museum_st_{digest}"


def normalize_museum_spatial_transcriptomics(museum_df):
    """
    Normalize Museum of Spatial Transcriptomics rows into the same dataset-level
    metadata schema used for spatial/reference scoring.
    """
    records = []

    for index, row in museum_df.reset_index(drop=True).iterrows():
        method = _clean_museum_value(row.get("method"))
        tissue = _clean_museum_value(row.get("tissue"))
        organ = _clean_museum_value(row.get("organ"))
        species = _clean_museum_value(row.get("species"))
        title = _clean_museum_value(row.get("title"))
        pmid = _clean_museum_value(row.get("pmid"))
        accession = _clean_museum_value(row.get("accession"))
        n_cells = _parse_museum_number(row.get("n_cells/bins/spots"))
        n_genes = _parse_museum_number(row.get("n_genes"))
        n_subjects = _parse_museum_number(row.get("n_subjects"))
        n_samples = _parse_museum_number(row.get("n_samples"))

        tissues = _clean_museum_list(tissue, organ)
        tissue_general = _clean_museum_list(organ, tissue)
        assays = _clean_museum_list(method, row.get("library"))

        records.append({
            "dataset_id": _museum_dataset_id(row, index),
            "source_database": "museum_of_spatial_transcriptomics",
            "collection_id": pmid or accession,
            "collection_name": title,
            "dataset_title": title,
            "dataset_type": method,
            "assays": assays,
            "tissues": tissues,
            "tissue_general": tissue_general,
            "diseases": _normalize_museum_disease(row.get("pathological")),
            "cell_types": [],
            "organism": species,
            "organisms": _clean_museum_list(species),
            "suspension_types": [],
            "donor_id": np.nan,
            "n_donors": n_subjects,
            "n_samples": n_samples,
            "n_cells": n_cells,
            "dataset_total_cell_count": n_cells,
            "median_n_measured_vars": n_genes,
            "museum_sheet_tab": _clean_museum_value(row.get("museum_sheet_tab")),
            "museum_sheet_row": row.get("museum_sheet_row", np.nan),
            "museum_duplicate_count": 1,
            "museum_source_tabs": _clean_museum_list(row.get("museum_sheet_tab")),
            "date_published": _clean_museum_value(row.get("date_published")),
            "pmid": pmid,
            "journal": _clean_museum_value(row.get("journal")),
            "url": _clean_museum_value(row.get("URL")),
            "accession": accession,
            "data_link": _clean_museum_value(row.get("data_link")),
            "developmental_stage": _clean_museum_value(row.get("developmental_stage")),
            "sex": _clean_museum_value(row.get("sex")),
            "n_cells_source": "museum.n_cells/bins/spots" if not pd.isna(n_cells) else None,
            "dataset_total_cell_count_source": "museum.n_cells/bins/spots" if not pd.isna(n_cells) else None,
            "median_n_measured_vars_source": "museum.n_genes" if not pd.isna(n_genes) else None,
        })

    return pd.DataFrame(records)


def deduplicate_museum_spatial_transcriptomics(museum_meta):
    """
    Collapse exact duplicate Museum rows after normalization.

    Rows are considered duplicate when their stable dataset_id matches. The
    retained row carries all source tabs where the duplicate appeared and a
    duplicate count for auditability.
    """
    if museum_meta.empty or "dataset_id" not in museum_meta.columns:
        return museum_meta.copy()

    def first_non_null(values):
        for value in values:
            if value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            return value
        return np.nan

    rows = []
    for _, group in museum_meta.groupby("dataset_id", sort=False, dropna=False):
        row = group.iloc[0].copy()
        tabs = []
        seen = set()
        for value in group.get("museum_sheet_tab", pd.Series(dtype=object)):
            text = _clean_museum_value(value)
            if text is not None and text.lower() not in seen:
                tabs.append(text)
                seen.add(text.lower())
        row["museum_source_tabs"] = tabs
        row["museum_duplicate_count"] = len(group)
        row["museum_sheet_tab"] = first_non_null(group.get("museum_sheet_tab", []))
        row["museum_sheet_row"] = first_non_null(group.get("museum_sheet_row", []))
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)

SPATIAL_TECH_KEYWORDS = [
    # common commercial / platform names
    "visium",
    "xenium",
    "cosmx",
    "merscope",
    "merfish",
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

    # image-based / FISH-style methods
    "seqfish",
    "seqfish+",
    "osmfish",
    "starfish",
    "starmap",
    "iss",
    "in situ sequencing",

    # generic
    # "spatial",
]

STRICT_SPATIAL_TECH_KEYWORDS = [
    "visium",
    "xenium",
    "cosmx",
    "merscope",
    "merfish",
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
    "seqfish",
    "seqfish+",
    "osmfish",
    "starfish",
    "starmap",
    "in situ sequencing",
    "spatial transcriptomics",
]
def normalize_text(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if isinstance(x, list):
        return " ".join(map(str, x)).lower()
    return str(x).lower()


def contains_spatial_keyword(x, keywords=STRICT_SPATIAL_TECH_KEYWORDS):
    text = normalize_text(x)
    return any(k.lower() in text for k in keywords)


def is_spatial_dataset(row, keywords=STRICT_SPATIAL_TECH_KEYWORDS):
    """
    Returns True if a dataset appears to be spatial transcriptomics,
    based on assay metadata and backup text fields.
    """

    fields_to_check = [
        "assays",
        "dataset_title",
        "collection_name",
        "dataset_h5ad_path",
        "citation",
    ]

    combined_text = " ".join(
        normalize_text(row.get(col, ""))
        for col in fields_to_check
    )

    return any(k.lower() in combined_text for k in keywords)
