import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr


def build_knn_edges(coords, k=6):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    _, inds = nn.kneighbors(coords)

    src = np.repeat(np.arange(coords.shape[0]), k)
    dst = inds[:, 1:].reshape(-1)

    return np.column_stack([src, dst])


def count_type_edges_fast(type_codes, edges, n_types):
    src_types = type_codes[edges[:, 0]]
    dst_types = type_codes[edges[:, 1]]

    pair_codes = src_types * n_types + dst_types
    print(pair_codes.min())
    counts = np.bincount(
        pair_codes,
        minlength=n_types * n_types
    ).reshape(n_types, n_types)

    return counts


def adjacency_log_enrichment(type_codes, edges, n_types, pseudocount=1e-8):
    counts = count_type_edges_fast(type_codes, edges, n_types)

    observed = counts / counts.sum()

    type_counts = np.bincount(type_codes, minlength=n_types)
    p = type_counts / type_counts.sum()

    expected = np.outer(p, p)

    enrich = np.log2((observed + pseudocount) / (expected + pseudocount))

    return enrich

def flatten_upper_triangle(mat):
    arr = mat
    idx = np.triu_indices_from(arr)
    return arr[idx]

def flatten_profile(mat):
    return mat.ravel()