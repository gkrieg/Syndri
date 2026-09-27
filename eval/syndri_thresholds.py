"""Per-cell Syndri threshold selection, independent of the metric evaluator.

Cosines use linear expression on a fixed gene panel. Removing genes changes
only their dot-product and squared-norm contributions. No log transform or
renormalization of the remaining genes is performed (cosine is scale invariant).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class ThresholdResult:
    threshold: float
    cosine_before: float
    cosine_after: float
    rolling_cosine: float
    n_removed_genes: int
    status: str
    trace: dict | None = None


def _scan(values, probabilities, reference_values, reference_norm, window,
          return_trace, min_probability=0.0, eligibility_probabilities=None):
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    if not np.isfinite(min_probability) or not 0 <= min_probability <= 1:
        raise ValueError("min_probability must be in [0, 1]")
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Counts must be finite/nonnegative and expressed-gene probabilities in [0, 1]")
    if not len(values) or reference_norm == 0:
        return ThresholdResult(np.inf, np.nan, np.nan, np.nan, 0,
                               "zero_cell" if not len(values) else "zero_reference")
    eligibility_probabilities = (
        probabilities if eligibility_probabilities is None
        else np.asarray(eligibility_probabilities, dtype=float)
    )
    if eligibility_probabilities.shape != probabilities.shape:
        raise ValueError("eligibility probabilities must align with ordering probabilities")
    if (not np.isfinite(eligibility_probabilities).all()
            or ((eligibility_probabilities < 0) | (eligibility_probabilities > 1)).any()):
        raise ValueError("Eligibility probabilities must be finite and in [0, 1]")
    # The shared/type-level probabilities define order; the cell's own
    # cell-gene probabilities determine which genes are eligible for removal.
    eligible = eligibility_probabilities >= min_probability
    fixed_x = values[~eligible].astype(np.float64)
    fixed_r = reference_values[~eligible]
    order = np.argsort(-probabilities[eligible], kind="stable")
    p = probabilities[eligible][order]
    x = values[eligible][order].astype(np.float64)
    r = reference_values[eligible][order]
    # Only tied-probability group endpoints define realizable thresholds.
    removed = np.r_[0, np.flatnonzero(np.r_[p[1:] != p[:-1], True]) + 1] if len(p) else np.array([0])
    thresholds = np.r_[np.inf, p[removed[1:] - 1]] if len(p) else np.array([np.inf])
    # Algebraically D_new = D_old - sum(x_g*r_g), S_new = S_old - sum(x_g**2).
    # Reverse cumulative sums avoid cancellation when almost all genes are removed.
    fixed_dot = float(fixed_x @ fixed_r)
    fixed_norm = float(fixed_x @ fixed_x)
    dots = fixed_dot + np.r_[np.cumsum((x * r)[::-1])[::-1], 0.][removed]
    norms = fixed_norm + np.r_[np.cumsum((x * x)[::-1])[::-1], 0.][removed]
    valid = norms > 0
    cosines = np.full(len(removed), np.nan)
    cosines[valid] = np.clip(dots[valid] / (np.sqrt(norms[valid]) * reference_norm), 0, 1)
    # Trailing rolling mean, min_periods=1. All-zero candidates are undefined
    # and cannot win; include the no-removal candidate in the rolling series.
    totals = np.r_[0., np.cumsum(np.where(valid, cosines, 0.))]
    numbers = np.r_[0, np.cumsum(valid)]
    ends = np.arange(1, len(removed) + 1)
    starts = np.maximum(ends - window, 0)
    rolling = np.full(len(removed), np.nan)
    rolling[valid] = ((totals[ends] - totals[starts])[valid]
                      / (numbers[ends] - numbers[starts])[valid])
    best_score = np.nanmax(rolling)
    # Treat floating-point ties as ties and prefer fewer removed genes.
    best = np.flatnonzero(valid & np.isclose(rolling, best_score, rtol=1e-12, atol=1e-14))[0]
    trace = None
    if return_trace:
        trace = dict(threshold=thresholds, cosine=cosines, rolling_cosine=rolling,
                     n_removed_genes=removed)
    return ThresholdResult(float(thresholds[best]), float(cosines[0]),
                           float(cosines[best]), float(rolling[best]),
                           int(removed[best]), "ok", trace)


def select_cell_threshold(counts, probabilities, reference_profile, *, window=5,
                          min_probability=0.0, return_trace=False):
    """Choose a per-cell threshold maximizing trailing rolling-average cosine.

    Vectors must share gene order. Counts may be a vector or a one-row sparse
    matrix; probabilities/reference_profile are dense vectors. Only expressed
    genes with probability >= ``min_probability`` create candidates. Zero genes
    with probability >= result.threshold;
    infinity means leave unchanged. Tied probabilities are removed together.
    ``window=1`` maximizes unsmoothed cosine. Ties prefer less removal. An empty
    cell or zero reference is left unchanged, with NaN cosine and a status.
    ``return_trace=True`` exposes every candidate for plotting in a notebook.
    """
    r = np.asarray(reference_profile, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if r.ndim != 1 or p.shape != r.shape:
        raise ValueError("Reference and probabilities must be aligned 1D vectors")
    if not np.isfinite(r).all() or (r < 0).any():
        raise ValueError("Reference profile must be finite and nonnegative")
    if sparse.issparse(counts):
        row = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
        if row.shape != (1, len(r)):
            raise ValueError("Sparse counts must have shape (1, n_genes)")
        row.sum_duplicates()
        row.eliminate_zeros()
        indices, values = row.indices, row.data
    else:
        x = np.asarray(counts, dtype=np.float64)
        if x.shape != r.shape:
            raise ValueError("Counts and reference must have the same vector shape")
        indices = np.flatnonzero(x)
        values = x[indices]
    return _scan(values, p[indices], r[indices], np.linalg.norm(r), window,
                 return_trace, min_probability=min_probability)


def clean_cells_by_reference(
    X,
    probabilities,
    labels,
    profiles,
    *,
    window=5,
    probability_ordering="cell_type_mean",
    min_removal_probability=0.0,
    one_se_multiplier=1.0,
    return_curves=False,
):
    """Clean cells with a shared cell-type ranking and a one-SE stopping rule.

    For every cell type, gene scores are mean Syndri probabilities among cells
    in which that gene is expressed. Genes are ranked once by this score. For
    every prefix ``k``, the mean improvement in cell-to-reference cosine is
    computed across cells. The selected prefix is the smallest ``k`` whose
    mean is within ``one_se_multiplier`` exact bootstrap standard errors of the
    best mean. This is the exact nonparametric-bootstrap SE of a sample mean,
    so no Monte Carlo resampling is needed.

    ``window``, ``probability_ordering`` and ``min_removal_probability`` remain
    accepted for call compatibility but do not affect this selection rule.
    There is deliberately no fixed probability threshold. If ``return_curves``
    is true, return ``(cleaned, cell_diagnostics, cell_type_curves)``.
    """
    import pandas as pd

    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    if not np.isfinite(one_se_multiplier) or one_se_multiplier < 0:
        raise ValueError("one_se_multiplier must be finite and nonnegative")
    out = sparse.csr_matrix(X, dtype=np.float64, copy=True)
    out.sum_duplicates()
    out.eliminate_zeros()
    if probabilities.shape != out.shape or len(labels) != out.shape[0] or profiles.shape[1] != out.shape[1]:
        raise ValueError("Counts, probabilities, labels and profiles must have aligned dimensions")
    if not profiles.index.is_unique:
        raise ValueError("Profile cell types must be unique")
    ref = profiles.to_numpy(dtype=np.float64)
    if not np.isfinite(ref).all() or (ref < 0).any() or not np.isfinite(out.data).all() or (out.data < 0).any():
        raise ValueError("Counts and reference profiles must be finite/nonnegative")
    norms = np.linalg.norm(ref, axis=1)
    lookup = {str(label): i for i, label in enumerate(profiles.index)}
    probs = sparse.csr_matrix(probabilities) if sparse.issparse(probabilities) else np.asarray(probabilities)
    probability_values = probs.data if sparse.issparse(probs) else probs
    if (not np.isfinite(probability_values).all()
            or ((probability_values < 0) | (probability_values > 1)).any()):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    label_values = np.asarray(labels, dtype=object)
    label_text = np.asarray([None if pd.isna(x) else str(x) for x in label_values], dtype=object)
    rows = [None] * out.shape[0]
    curve_frames = []
    n_genes = out.shape[1]

    for cell_type in pd.unique(label_text[label_text != None]):  # noqa: E711
        cell_idx = np.flatnonzero(label_text == cell_type)
        if cell_type not in lookup:
            for i in cell_idx:
                start, end = out.indptr[i:i + 2]
                values = out.data[start:end]
                rows[i] = dict(cell_type=label_values[i], threshold=np.nan,
                    cosine_before=np.nan, cosine_after=np.nan, rolling_cosine=np.nan,
                    n_removed_genes=0, n_expressed_genes_before=end-start,
                    removed_count=0.0, original_count=float(values.sum()),
                    probability_ordering="cell_type_expressed_mean", min_removal_probability=np.nan,
                    selected_k=0, status="missing_reference")
            continue

        # s_gc = mean probability among cells of type c where g is expressed.
        probability_sum = np.zeros(n_genes, dtype=np.float64)
        expression_n = np.zeros(n_genes, dtype=np.int64)
        for i in cell_idx:
            start, end = out.indptr[i:i + 2]
            genes = out.indices[start:end]
            if sparse.issparse(probs):
                cell_p = probs[i, genes].toarray().ravel()
            else:
                cell_p = np.asarray(probs[i, genes]).ravel()
            probability_sum[genes] += cell_p
            expression_n[genes] += 1
        scores = np.full(n_genes, -np.inf)
        expressed = expression_n > 0
        scores[expressed] = probability_sum[expressed] / expression_n[expressed]
        ranking = np.argsort(-scores, kind="stable")
        rank = np.empty(n_genes, dtype=np.int64)
        rank[ranking] = np.arange(n_genes)

        # Accumulate changes in delta and delta squared at their global prefix
        # positions. Cumulative sums then give the whole mean curve in O(nnz+G).
        delta_events = np.zeros(n_genes + 2, dtype=np.float64)
        square_events = np.zeros(n_genes + 2, dtype=np.float64)
        cell_state = {}
        reference = ref[lookup[cell_type]]
        reference_norm = norms[lookup[cell_type]]
        n_valid = 0
        for i in cell_idx:
            start, end = out.indptr[i:i + 2]
            genes = out.indices[start:end]
            values = out.data[start:end].astype(np.float64, copy=False)
            total = float(values.sum())
            if not len(genes) or reference_norm == 0:
                cell_state[i] = (np.nan, np.nan, total, "zero_cell" if not len(genes) else "zero_reference")
                continue
            dot = float(values @ reference[genes])
            norm_sq = float(values @ values)
            cosine0 = dot / (np.sqrt(norm_sq) * reference_norm)
            old_delta = 0.0
            order = np.argsort(rank[genes], kind="stable")
            for position in order:
                gene = genes[position]
                dot -= float(values[position] * reference[gene])
                norm_sq -= float(values[position] ** 2)
                cosine = dot / (np.sqrt(norm_sq) * reference_norm) if norm_sq > 0 else 0.0
                new_delta = cosine - cosine0
                event = int(rank[gene]) + 1
                delta_events[event] += new_delta - old_delta
                square_events[event] += new_delta * new_delta - old_delta * old_delta
                old_delta = new_delta
            cell_state[i] = (cosine0, rank, total, "ok")
            n_valid += 1

        if n_valid:
            mean_delta = np.cumsum(delta_events[:n_genes + 1]) / n_valid
            second_moment = np.cumsum(square_events[:n_genes + 1]) / n_valid
            # Exact conditional nonparametric-bootstrap SE for the sample mean.
            bootstrap_se = np.sqrt(np.maximum(second_moment - mean_delta ** 2, 0.0) / n_valid)
            k_max = int(np.flatnonzero(np.isclose(mean_delta, np.max(mean_delta),
                rtol=1e-12, atol=1e-14))[0])
            target = mean_delta[k_max] - one_se_multiplier * bootstrap_se[k_max]
            selected_k = int(np.flatnonzero(mean_delta >= target - 1e-14)[0])
        else:
            mean_delta = np.full(n_genes + 1, np.nan)
            bootstrap_se = np.full(n_genes + 1, np.nan)
            k_max = selected_k = 0
            target = np.nan

        removed_gene = np.empty(n_genes + 1, dtype=object)
        removed_gene[0] = None
        removed_gene[1:] = np.asarray(profiles.columns, dtype=object)[ranking]
        ranked_scores = scores[ranking].copy()
        ranked_scores[~np.isfinite(ranked_scores)] = np.nan
        ranked_score = np.r_[np.nan, ranked_scores]
        curve_frames.append(pd.DataFrame(dict(cell_type=cell_type,
            k=np.arange(n_genes + 1), removed_gene=removed_gene,
            removed_gene_score=ranked_score, mean_cosine_improvement=mean_delta,
            bootstrap_se=bootstrap_se, is_k_max=np.arange(n_genes + 1) == k_max,
            is_selected=np.arange(n_genes + 1) == selected_k,
            selected_k=selected_k, k_max=k_max, one_se_target=target,
            n_cells=n_valid, one_se_multiplier=one_se_multiplier)))

        remove_gene = rank < selected_k
        for i in cell_idx:
            start, end = out.indptr[i:i + 2]
            genes = out.indices[start:end]
            values = out.data[start:end]
            cosine0, _, total, status = cell_state[i]
            remove = remove_gene[genes]
            removed_count = float(values[remove].sum())
            values[remove] = 0
            remaining_norm = np.linalg.norm(values)
            cosine_after = (float(values @ reference[genes]) /
                (remaining_norm * reference_norm)) if remaining_norm and reference_norm else np.nan
            rows[i] = dict(cell_type=label_values[i], threshold=np.nan,
                cosine_before=cosine0, cosine_after=cosine_after, rolling_cosine=np.nan,
                n_removed_genes=int(remove.sum()), n_expressed_genes_before=len(genes),
                removed_count=removed_count, original_count=total,
                probability_ordering="cell_type_expressed_mean", min_removal_probability=np.nan,
                selected_k=selected_k, k_max=k_max, one_se_target=target,
                status=status)

    for i, row in enumerate(rows):
        if row is None:
            start, end = out.indptr[i:i + 2]
            rows[i] = dict(cell_type=label_values[i], threshold=np.nan,
                cosine_before=np.nan, cosine_after=np.nan, rolling_cosine=np.nan,
                n_removed_genes=0, n_expressed_genes_before=end-start,
                removed_count=0.0, original_count=float(out.data[start:end].sum()),
                probability_ordering="cell_type_expressed_mean", min_removal_probability=np.nan,
                selected_k=0, status="missing_reference")
    out.eliminate_zeros()
    diagnostics = pd.DataFrame(rows)
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    return (out, diagnostics, curves) if return_curves else (out, diagnostics)
