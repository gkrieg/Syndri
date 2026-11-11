#!/usr/bin/env python3
"""Add a hard-threshold Syndri corrected-count layer to inference outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import anndata as ad
import numpy as np
from scipy import sparse

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('input', type=Path)
p.add_argument('--probability-layer', default='admix_probs')
p.add_argument('--counts-layer', default='admixed_counts')
p.add_argument('--output-layer', default='syndri_corrected_counts')
p.add_argument('--threshold', type=float, default=0.5)
a = p.parse_args()
if not 0 <= a.threshold <= 1:
    p.error('--threshold must be between 0 and 1')
data = ad.read_h5ad(a.input)
if a.probability_layer not in data.layers or a.counts_layer not in data.layers:
    raise KeyError(f"Need layers {a.probability_layer!r} and {a.counts_layer!r}")
counts = data.layers[a.counts_layer]
probs = data.layers[a.probability_layer]
if sparse.issparse(counts):
    corrected = counts.tocsr(copy=True)
    row, col = corrected.nonzero()
    p_nonzero = np.asarray(probs[row, col]).ravel()
    corrected.data[p_nonzero >= a.threshold] = 0
    corrected.eliminate_zeros()
else:
    p_dense = probs.toarray() if sparse.issparse(probs) else np.asarray(probs)
    corrected = np.where(p_dense >= a.threshold, 0, counts)
data.layers[a.output_layer] = corrected
data.uns['syndri_correction'] = {
    'probability_layer': a.probability_layer,
    'counts_layer': a.counts_layer,
    'probability_threshold': float(a.threshold),
    'rule': 'zero counts in genes whose predicted admixture probability meets threshold',
}
data.write_h5ad(a.input, compression='gzip')
print(f'Added {a.output_layer!r} to {a.input}')
