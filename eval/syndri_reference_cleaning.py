"""AnnData/cache adapter for the reusable per-cell threshold selector."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from syndri_thresholds import clean_cells_by_reference


def load_profiles(adata, location, *, reference_file="", input_layer="X",
                  required_settings=None):
    """Load explicit profile h5ad or uniquely match an existing sparsity cache.

    Directory matching uses the original RCTD reference identity and ordered
    shared panel. Ambiguous caches require an explicit file, never a guess.
    """
    location = Path(location).expanduser()
    if location.is_file():
        path = location
    else:
        import re
        suffix = re.sub(r"[^0-9A-Za-z]+", "_", input_layer).strip("_") or "unnamed"
        for key in [f"rctd_run_metadata_{suffix}", "rctd_run_metadata"]:
            if key == "rctd_run_metadata" and input_layer != "X":
                continue
            metadata = adata.uns.get(key, {})
            if metadata.get("reference_file"):
                reference_file = str(metadata["reference_file"])
                break
        if not reference_file:
            raise ValueError("Cache matching needs RCTD reference_file metadata or --reference-data; "
                             "alternatively pass an explicit --syndri-reference-profiles h5ad")
        reference_path = Path(reference_file).expanduser().resolve()
        with h5py.File(reference_path, "r") as handle:
            ref_genes = ad.io.read_elem(handle["var"]).index
        expected_genes = list(adata.var_names.intersection(ref_genes))
        candidates = []
        for candidate in sorted(location.glob("*.h5ad")):
            if ".tmp-" in candidate.name:
                continue
            with h5py.File(candidate, "r") as handle:
                key = "uns/cache_metadata_json"
                if key not in handle:
                    continue
                metadata = json.loads(handle[key][()])
            if (Path(metadata["reference_file"]).resolve() == reference_path
                    and metadata["genes"] == expected_genes):
                stat = reference_path.stat()
                settings = metadata.get("settings", {})
                settings_match = all(settings.get(key) == value
                                     for key, value in (required_settings or {}).items())
                if (metadata["size"] == stat.st_size and metadata["mtime_ns"] == stat.st_mtime_ns
                        and settings_match):
                    candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError(f"Found {len(candidates)} matching profile caches in {location}. "
                             "Run split_sparsity first to build the cache, or pass one explicit "
                             "--syndri-reference-profiles h5ad to resolve ambiguity.")
        path = candidates[0]
    profiles = ad.read_h5ad(path)
    if not profiles.obs_names.is_unique or not profiles.var_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell-type profile names and gene names must be unique")
    return profiles, path.resolve()


def prepare_reference_cleaned(adata, args, source_file, *, input_layer="X"):
    profiles, profile_path = load_profiles(
        adata, args.syndri_reference_profiles, reference_file=args.reference_data,
        input_layer=input_layer,
        required_settings={
            "min_UMI": getattr(args, "syndri_profile_min_umi", 10),
            "max_UMI": getattr(args, "syndri_profile_max_umi", 2000),
            "cell_min": getattr(args, "syndri_profile_cell_min", 25),
            "n_max_cells": getattr(args, "syndri_profile_max_cells", 10000),
            "cell_type_col": getattr(args, "syndri_profile_cell_type_col", "cell_type"),
        })
    genes = adata.var_names.intersection(profiles.var_names)
    if not len(genes):
        raise ValueError("No shared spatial/profile genes")
    indices = adata.var_names.get_indexer(genes)
    ref = profiles[:, genes].X
    if sparse.issparse(ref):
        ref = ref.toarray()
    frame = pd.DataFrame(ref, index=profiles.obs_names, columns=genes)
    label_key = args.syndri_cell_type_key or args.label_key
    labels = adata.obs[label_key]
    original = sparse.csr_matrix(adata.X)
    probs = adata.layers[args.admix_probs_layer]
    cleaned, diagnostics = clean_cells_by_reference(
        original[:, indices], probs[:, indices], labels, frame, window=args.syndri_cosine_window)
    if len(diagnostics) and diagnostics.status.eq("missing_reference").all():
        raise ValueError(f"No obs[{label_key!r}] labels match the reference profiles; "
                         "set --syndri-cell-type-key to the matching annotation column")
    # Keep genes without a reference entry unchanged.
    outside = np.flatnonzero(~adata.var_names.isin(genes))
    full = sparse.hstack([cleaned, original[:, outside]], format="csr")
    full = full[:, np.argsort(np.r_[indices, outside])].tocsr()
    diagnostics.insert(0, "cell_id", adata.obs_names)
    diagnostics.insert(0, "input_layer", input_layer)
    diagnostics.insert(0, "source_file", source_file)
    diagnostics["reference_profiles"] = str(profile_path)
    diagnostics["window"] = args.syndri_cosine_window
    diagnostics["n_shared_genes"] = len(genes)
    diagnostics["label_key"] = label_key
    directory = Path(args.output_dir) / "syndri_thresholds"
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{source_file}\0{input_layer}".encode()).hexdigest()[:12]
    diagnostics.to_csv(directory / f"{Path(source_file).stem}.{key}.csv", index=False)
    statuses = diagnostics.status.value_counts().to_dict()
    print(f"Syndri reference-cosine selection ({input_layer}): {statuses}; profiles={profile_path}")
    total = float(original.sum())
    achieved = (total - float(full.sum())) / total if total else 0.
    return full, achieved, str(profile_path)
