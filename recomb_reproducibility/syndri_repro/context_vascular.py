"""Vascular strict-win response and scRNA-seq plausibility from local predictions.

Ported from the standalone vascular panel in ResponsivenessQueryNativeMarkers.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.stats import linregress, rankdata, spearmanr
from . import ROOT


def compute_vascular(run):
    run=Path(run)
    output=run/'vascular_all32_plausibility_vs_response.csv'
    if output.exists():
        return output
    design=json.loads((run/'design.json').read_text())
    VASC_PLOT_SLICES=design['slices']
    VASC_PLOT_GENES=design['marker_genes']
    VASC_PLOT_RUN=run
    VASC_PLOT_TYPE='capillary endothelial cell'
    VASC_PLOT_REFERENCE=ROOT/'assets/data/BX/reference.h5ad'
    VASC_PLOT_QUERY_SLOPES_PATH=run/'vascular_all32_query_slopes.csv.gz'
    VASC_PLOT_REFERENCE_PATH=run/'vascular_reference_plausibility.csv'
    # Compute the original per-query slope from saved context predictions.
    slope_records = []
    for entry in VASC_PLOT_SLICES:
        prediction_path = Path(entry["predictions"])
        prediction = ad.read_h5ad(prediction_path)
        obs = prediction.obs.reset_index(drop=True)
        names = prediction.var.get(
            "feature_name", pd.Series(prediction.var_names, index=prediction.var_names)
        ).fillna("").astype(str).to_numpy()
        names = np.where(names == "", prediction.var_names.astype(str), names)
        gene_idx = {gene: i for i, gene in enumerate(names)}
        marker_cols = {gene: gene_idx[gene] for gene in VASC_PLOT_GENES if gene in gene_idx}
        native_genes = design['query_native_marker_sets'][VASC_PLOT_TYPE]
        native_cols = [gene_idx[gene] for gene in native_genes if gene in gene_idx]
        probs = prediction.layers["admix_probs"]
        if sparse.issparse(probs):
            probs = probs.toarray()
        raw = prediction.X

        def _vascular_dense_row(matrix, row_index):
            row = matrix[row_index]
            return row.toarray().ravel() if sparse.issparse(row) else np.asarray(row).ravel()

        query_mask = obs.query_cell_type.eq(VASC_PLOT_TYPE)
        for query_id, query_rows in obs.loc[query_mask].groupby("focal_cell_id", sort=False).groups.items():
            query_rows = np.asarray(list(query_rows), dtype=int)
            original_rows = query_rows[obs.loc[query_rows, "context_kind"].eq("original").to_numpy()]
            if len(original_rows) != 1:
                continue
            original_row = int(original_rows[0])
            raw_original = _vascular_dense_row(raw, original_row)
            expressed_native = [col for col in native_cols if raw_original[col] > 0]
            if not expressed_native:
                continue
            delta_malignant = (
                obs.loc[query_rows, "context_malignant_token_count"].to_numpy(dtype=float)
                - float(obs.loc[original_row, "query_malignant_token_count"])
            )
            representative = obs.loc[query_rows, "context_kind"].eq("representative").to_numpy()
            response_rows = query_rows[representative]
            delta_malignant = delta_malignant[representative]
            if len(np.unique(delta_malignant)) < 3:
                continue
            for gene, gene_col in marker_cols.items():
                if raw_original[gene_col] <= 0:
                    continue
                native_probability = probs[np.ix_(response_rows, np.asarray(expressed_native))]
                wins = (probs[response_rows, gene_col, None] > native_probability).mean(axis=1)
                slope = linregress(delta_malignant, wins).slope
                slope_records.append({
                    "slice": entry["slice"], "slice_label": entry["label"],
                    "query_cell_id": query_id, "gene": gene,
                    "beta_rank_per_malignant_token": slope,
                    "n_contexts": len(response_rows),
                    "n_delta_levels": len(np.unique(delta_malignant)),
                })
        del prediction, probs, raw

    gene_query_slopes = pd.DataFrame(slope_records)
    gene_query_slopes.to_csv(VASC_PLOT_QUERY_SLOPES_PATH, index=False, compression="gzip")
    rng = np.random.default_rng(741)
    slope_summary_rows = []
    for gene in VASC_PLOT_GENES:
        part = gene_query_slopes.loc[gene_query_slopes.gene.eq(gene)]
        per_slice = [g.beta_rank_per_malignant_token.to_numpy()
                     for _, g in part.groupby("slice") if len(g)]
        if not per_slice:
            slope_summary_rows.append(dict(gene=gene, beta=np.nan, ci_low=np.nan, ci_high=np.nan,
                n_queries=0, n_slices=0, response_class="unassessed"))
            continue
        beta = float(np.median(np.concatenate(per_slice)))
        bootstraps = []
        for _ in range(2000):
            sampled = np.concatenate([values[rng.integers(0, len(values), len(values))]
                                      for values in per_slice])
            bootstraps.append(np.median(sampled))
        ci_low, ci_high = np.quantile(bootstraps, [0.025, 0.975])
        slope_summary_rows.append(dict(gene=gene, beta=beta, ci_low=ci_low, ci_high=ci_high,
            n_queries=sum(map(len, per_slice)), n_slices=len(per_slice),
            response_class="negative" if ci_high < 0 else "positive" if ci_low > 0 else "neutral"))
    slope_summary = pd.DataFrame(slope_summary_rows)

    # Compute reference prevalence for all panel genes directly from the saved reference.
    reference = ad.read_h5ad(VASC_PLOT_REFERENCE)
    reference_label = "cell_type" if "cell_type" in reference.obs else "Harmonised_Level4"
    reference_types = reference.obs[reference_label].astype(str).to_numpy()
    reference_names = reference.var.get(
        "feature_name", pd.Series(reference.var_names, index=reference.var_names)
    ).astype("object").fillna("").astype(str).to_numpy()
    reference_names = np.where(reference_names == "", reference.var_names.astype(str), reference_names)
    reference_gene_idx = {gene: i for i, gene in enumerate(reference_names)}
    reference_matrix = reference.layers["SoupX"] if "SoupX" in reference.layers else reference.X
    reference_rows = []
    for gene in VASC_PLOT_GENES:
        if gene not in reference_gene_idx:
            continue
        column = reference_gene_idx[gene]
        for cell_type in [VASC_PLOT_TYPE, "malignant cell"]:
            rows = np.flatnonzero(reference_types == cell_type)
            values = reference_matrix[rows, column]
            values = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
            reference_rows.append(dict(gene=gene, cell_type=cell_type, n_cells=len(rows),
                n_expressing=int((values > 0).sum()),
                detection_prevalence=float((values > 0).mean()) if len(values) else np.nan,
                mean_expression=float(values.mean()) if len(values) else np.nan))
    reference_plausibility = pd.DataFrame(reference_rows)
    reference_plausibility.to_csv(VASC_PLOT_REFERENCE_PATH, index=False)
    reference_wide = reference_plausibility.pivot(index="gene", columns="cell_type",
        values=["detection_prevalence", "mean_expression", "n_cells", "n_expressing"])
    reference_wide.columns = [f"{metric}_{cell_type}" for metric, cell_type in reference_wide.columns]
    reference_wide = reference_wide.reset_index()
    all32 = pd.DataFrame({"gene": VASC_PLOT_GENES}).merge(slope_summary, on="gene", how="left", validate="one_to_one")
    all32 = all32.merge(reference_wide, on="gene", how="left", validate="one_to_one")

    prev_ec = f"detection_prevalence_{VASC_PLOT_TYPE}"
    prev_mal = "detection_prevalence_malignant cell"
    n_ec = f"n_cells_{VASC_PLOT_TYPE}"
    n_mal = "n_cells_malignant cell"
    all32["endothelial_plausibility_prevalence_difference"] = all32[prev_ec] - all32[prev_mal]
    p_ec = (all32[f"n_expressing_{VASC_PLOT_TYPE}"] + 0.5) / (all32[n_ec] + 1.0)
    p_mal = (all32["n_expressing_malignant cell"] + 0.5) / (all32[n_mal] + 1.0)
    all32["endothelial_plausibility_log2_ratio"] = np.log2(p_ec / p_mal)
    all32["response_status"] = np.where(np.isfinite(all32.beta), "available", "missing slope")

    def _vascular_panel_spearman(x, y, seed=20261022, n_boot=5000, n_perm=10000):
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = np.asarray(x)[keep], np.asarray(y)[keep]
        n = len(x)
        if n < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
            return dict(n_genes=n, rho=np.nan, ci_low=np.nan, ci_high=np.nan, permutation_p=np.nan)
        rho = float(spearmanr(x, y).statistic)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            ix = rng.integers(0, n, n)
            if np.unique(x[ix]).size > 1 and np.unique(y[ix]).size > 1:
                boots.append(spearmanr(x[ix], y[ix]).statistic)
        ci_low, ci_high = np.quantile(boots, [0.025, 0.975]) if boots else (np.nan, np.nan)
        rank_x, rank_y = rankdata(x), rankdata(y)
        rank_x, rank_y = rank_x - rank_x.mean(), rank_y - rank_y.mean()
        denom = np.sqrt(np.dot(rank_x, rank_x) * np.dot(rank_y, rank_y))
        observed = np.dot(rank_x, rank_y) / denom
        null = np.empty(n_perm)
        for index in range(n_perm):
            null[index] = np.dot(rank_x, rng.permutation(rank_y)) / denom
        p_value = (1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1)
        return dict(n_genes=n, rho=rho, ci_low=float(ci_low), ci_high=float(ci_high), permutation_p=float(p_value))

    association_rows = []
    for label, xcol in [
        ("prevalence difference", "endothelial_plausibility_prevalence_difference"),
        ("smoothed log2 prevalence ratio", "endothelial_plausibility_log2_ratio"),
    ]:
        association_rows.append(dict(plausibility_measure=label,
            **_vascular_panel_spearman(all32[xcol].to_numpy(), all32.beta.to_numpy())))
    vascular_all32_association = pd.DataFrame(association_rows)

    all32.to_csv(output,index=False)
    vascular_all32_association.to_csv(run/'vascular_all32_plausibility_response_association.csv',index=False)
    return output
