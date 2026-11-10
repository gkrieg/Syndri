"""Whole-gene count removal, preserving the original figure threshold rule."""
import numpy as np
import pandas as pd
from .io import csr


def correct(counts, probabilities, labels=None, target_fraction=0.5, min_probability=0.5, unit='cell_type'):
    if not 0 <= target_fraction <= 1 or not 0 <= min_probability <= 1:
        raise ValueError('Fractions and probability floors must be in [0, 1]')
    counts, probabilities = csr(counts), csr(probabilities)
    if counts.shape != probabilities.shape:
        raise ValueError('Counts and probabilities must have the same shape and identifier order')
    if unit not in {'cell', 'cell_type'}:
        raise ValueError('unit must be cell or cell_type')
    if unit == 'cell_type' and labels is None:
        raise ValueError('Cell-type threshold selection requires labels')
    groups = np.arange(counts.shape[0]).astype(str) if unit == 'cell' else np.asarray(labels).astype(str)
    result = counts.copy()
    diagnostics = []
    # Work on observed entries only. Ties at the cutoff are removed together,
    # matching the archived notebooks; achieved removal can exceed the target.
    for group in pd.unique(groups):
        rows = np.flatnonzero(groups == group)
        block = counts[rows].tocoo()
        total = float(block.data.sum())
        if not total or target_fraction == 0:
            diagnostics.append((group, np.inf, 0.0)); continue
        p = np.asarray(probabilities[rows][block.row, block.col]).ravel()
        eligible = p >= min_probability
        if not eligible.any():
            diagnostics.append((group, np.inf, 0.0)); continue
        order = np.argsort(p[eligible])[::-1]
        cumulative = np.cumsum(block.data[eligible][order], dtype=np.float64)
        stop = min(np.searchsorted(cumulative, target_fraction * total, side='left'), len(order) - 1)
        threshold = float(p[eligible][order[stop]])
        remove = eligible & (p >= threshold)
        block.data[remove] = 0
        result[rows] = block.tocsr()
        diagnostics.append((group, threshold, float(counts[rows].sum() - block.data.sum()) / total))
    result.eliminate_zeros()
    return result, pd.DataFrame(diagnostics, columns=['group', 'threshold', 'achieved_removed_fraction'])

