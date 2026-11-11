"""AnnData bridge to the unmodified, cloned cellAdmix R scoring functions."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from score_admixture_slice import (
    align_and_prepare, infer_reference_path, parse_csv,
    threshold_for_removed_fraction, zero_counts_at_probability,
)


def run(args):
    output = args.output_dir.expanduser().resolve()
    score_path = output / "admixture_scores.csv"
    if score_path.exists() and not args.overwrite:
        raise FileExistsError(f"{score_path} exists; pass --overwrite.")
    if args.n_jobs < 2:
        raise ValueError("The cloned R adjacency function requires --n-jobs >= 2 (its cut(..., 1) fails).")
    if args.min_expr_frac_specific != args.min_expr_frac_contam:
        raise ValueError("The author backend has one min.expr.frac; supply equal expression-fraction settings.")
    fractions = [float(x) for x in parse_csv(args.target_removed_fractions)]
    if any(not 0 < x <= 1 for x in fractions):
        raise ValueError("Removal fractions must be in (0, 1].")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(score_version="3_celladmix_R", adjacency_source="spatial_cell_coordinates",
                  score_weighting="counts", zero_count_policy="NaN",
                  correlation_preprocessing="log1p(counts/full_panel_cell_total*10000), then shared genes",
                  notebook_adapter="Notebook get_dbl_scores parameters; direct matrix scoring preserves fractional SPLIT counts")
    for name, repo in [("celladmix", args.celladmix_repo), ("celladmix_notebooks", args.celladmix_notebooks_repo)]:
        config[name + "_commit"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    spatial = ad.read_h5ad(args.spatial_h5ad)
    ref_path = args.reference_h5ad or infer_reference_path(spatial)
    print(f"Using reference: {ref_path}", flush=True)
    spatial, reference = align_and_prepare(spatial, ad.read_h5ad(ref_path), args)
    config.update(resolved_reference_h5ad=str(ref_path), n_shared_genes=spatial.n_vars,
                  n_scored_cells=spatial.n_obs)
    # Keep one row per retained cell, even when it has no counts or valid coordinates.
    obs = pd.DataFrame({"cell": spatial.obs_names.astype(str),
                        "cell_type": spatial.obs[args.spatial_cell_type_key].to_numpy(),
                        "x": pd.to_numeric(spatial.obs[args.coord_x], errors="coerce").to_numpy(),
                        "y": pd.to_numeric(spatial.obs[args.coord_y], errors="coerce").to_numpy(),
                        "fov": spatial.obs[args.fov_key].to_numpy() if args.fov_key else "slice",
                        "library_size": spatial.obs['_admixture_library_size'].to_numpy()})
    ref_obs = pd.DataFrame({"cell": [f"reference_{i}" for i in range(reference.n_obs)],
                            "cell_type": reference.obs[args.reference_cell_type_key].to_numpy(),
                            "library_size": reference.obs['_admixture_library_size'].to_numpy()})
    probabilities = spatial.layers[args.probability_layer]
    if args.split_layer and args.split_layer not in spatial.layers:
        raise KeyError(args.split_layer)
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_suffix(".R")
    frames = []
    with tempfile.TemporaryDirectory(prefix="celladmix-", dir=os.environ.get("SLURM_TMPDIR") or None) as tmp:
        work = Path(tmp)
        config["output_dir"] = str(output)
        (work / "config.json").write_text(json.dumps(config))
        obs.to_csv(work / "spatial_obs.csv", index=False)
        ref_obs.to_csv(work / "reference_obs.csv", index=False)
        pd.DataFrame({"gene": spatial.var_names}).to_csv(work / "genes.csv", index=False)
        mmwrite(work / "raw.mtx", sparse.coo_matrix(spatial.X.T))
        mmwrite(work / "reference.mtx", sparse.coo_matrix(reference.X.T))
        del reference

        def invoke(mode):
            subprocess.run([args.rscript, str(script), mode, str(work)], check=True)

        invoke("prepare")
        # One condition matrix at a time bounds temporary disk use.
        conditions = [("raw", 0.0), *(([("SPLIT", np.nan)]) if args.split_layer else []),
                      *[(f"syndri_remove_{f:g}", f) for f in fractions]]
        for setting, target in conditions:
            threshold = achieved = np.nan
            if setting == "raw":
                matrix, achieved = spatial.X, 0.0
            elif setting == "SPLIT":
                matrix = spatial.layers[args.split_layer]
            else:
                threshold, achieved = threshold_for_removed_fraction(
                    spatial.X, probabilities, target, args.clean_prob_threshold)
                matrix = zero_counts_at_probability(spatial.X, probabilities, threshold)
            values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid counts in {setting}")
            print(f"Scoring {setting} with cloned cellAdmix R code", flush=True)
            mmwrite(work / "condition.mtx", sparse.coo_matrix(matrix.T))
            invoke("score")
            frame = pd.read_csv(work / "scores.csv", dtype={"cell": str, "cell_type": str})
            frame["setting"] = setting
            frame["removed_frac_target"] = target
            frame["removed_frac"] = achieved
            frame["removed_pct"] = achieved * 100
            frame["threshold"] = threshold
            frames.append(frame)
        scores = pd.concat(frames, ignore_index=True)
        scores.insert(0, "slice", args.spatial_h5ad.stem)
        scores.insert(0, "source_file", str(args.spatial_h5ad.resolve()))
        scores["score_version"] = config["score_version"]
        # Publish the complete CSV atomically; failed runs never leave a partial score table.
        scores.to_csv(output / "admixture_scores.csv.tmp", index=False)
        os.replace(output / "admixture_scores.csv.tmp", score_path)
    config["n_excluded_genes"] = int(pd.read_csv(output / "gene_correlation_preservation.csv")["excluded"].sum())
    (output / "admixture_score_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(scores):,} scores to {score_path}", flush=True)
