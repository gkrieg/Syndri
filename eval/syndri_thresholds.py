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


def _scan(values, probabilities, reference_values, reference_norm, window, return_trace):
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Counts must be finite/nonnegative and expressed-gene probabilities in [0, 1]")
    if not len(values) or reference_norm == 0:
        return ThresholdResult(np.inf, np.nan, np.nan, np.nan, 0,
                               "zero_cell" if not len(values) else "zero_reference")
    order = np.argsort(-probabilities, kind="stable")
    p = probabilities[order]
    x = values[order].astype(np.float64)
    r = reference_values[order]
    # Only tied-probability group endpoints define realizable thresholds.
    removed = np.r_[0, np.flatnonzero(np.r_[p[1:] != p[:-1], True]) + 1]
    thresholds = np.r_[np.inf, p[removed[1:] - 1]]
    # Algebraically D_new = D_old - sum(x_g*r_g), S_new = S_old - sum(x_g**2).
    # Reverse cumulative sums avoid cancellation when almost all genes are removed.
    dots = np.r_[np.cumsum((x * r)[::-1])[::-1], 0.][removed]
    norms = np.r_[np.cumsum((x * x)[::-1])[::-1], 0.][removed]
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
                          return_trace=False):
    """Choose a per-cell threshold maximizing trailing rolling-average cosine.

    Vectors must share gene order. Counts may be a vector or a one-row sparse
    matrix; probabilities/reference_profile are dense vectors. Only expressed
    genes create candidates. Zero genes with probability >= result.threshold;
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
    return _scan(values, p[indices], r[indices], np.linalg.norm(r), window, return_trace)


def clean_cells_by_reference(X, probabilities, labels, profiles, *, window=5):
    """Sparse batch wrapper; profiles is a cell-type-by-gene pandas DataFrame.

    X/probabilities must have the same shape and profiles.columns must already
    match their gene order. Missing labels/profiles preserve the cell and are
    recorded in diagnostics. Returns (CSR cleaned counts, per-cell DataFrame).
    No cell-by-gene dense matrix is allocated; type-profile norms are computed once.
    """
    import pandas as pd

    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
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
    rows = []
    for i, label in enumerate(labels):
        start, end = out.indptr[i:i + 2]
        genes = out.indices[start:end]
        values = out.data[start:end]
        total = float(values.sum())
        if pd.isna(label) or str(label) not in lookup:
            result = ThresholdResult(np.inf, np.nan, np.nan, np.nan, 0, "missing_reference")
        else:
            k = lookup[str(label)]
            if sparse.issparse(probs):
                p = probs[i, genes].toarray().ravel()
            else:
                p = np.asarray(probs[i, genes]).ravel()
            result = _scan(values, p, ref[k, genes], norms[k], window, False)
            if np.isfinite(result.threshold):
                values[p >= result.threshold] = 0
        rows.append(dict(cell_type=label, threshold=result.threshold,
                         cosine_before=result.cosine_before, cosine_after=result.cosine_after,
                         rolling_cosine=result.rolling_cosine, n_removed_genes=result.n_removed_genes,
                         removed_count=total - float(values.sum()), original_count=total,
                         status=result.status))
    out.eliminate_zeros()
    return out, pd.DataFrame(rows)
