import anndata as ad
import numpy as np
import pytest
from scipy.sparse import csr_array

from Heimdall.admixture_sampling import (
    build_neighbor_dict,
    sample_admixture_from_neighbor_dict,
    validate_precomputed_neighbor_celltype_dict,
    validate_precomputed_neighbor_dict,
)


def test_validate_precomputed_neighbor_dict_accepts_query_subset():
    adata = ad.AnnData(X=csr_array(np.ones((3, 2))))
    adata.obs_names = ["query", "neighbor_a", "neighbor_b"]
    adata.uns["neighbor_dict"] = {"query": ["neighbor_a", "neighbor_b"]}

    result = validate_precomputed_neighbor_dict(adata)

    assert result == {"query": ["neighbor_a", "neighbor_b"]}


def test_validate_precomputed_neighbor_dict_rejects_missing_ids():
    adata = ad.AnnData(X=csr_array(np.ones((2, 2))))
    adata.obs_names = ["query", "neighbor"]
    adata.uns["neighbor_dict"] = {"query": ["missing"]}

    with pytest.raises(ValueError, match="missing"):
        validate_precomputed_neighbor_dict(adata)


def test_validate_precomputed_neighbor_dict_requires_nonempty_dict():
    adata = ad.AnnData(X=csr_array(np.ones((2, 2))))
    adata.uns["neighbor_dict"] = {}

    with pytest.raises(ValueError, match="non-empty dictionary"):
        validate_precomputed_neighbor_dict(adata)


def test_validate_precomputed_neighbor_dict_enforces_context_capacity():
    adata = ad.AnnData(X=csr_array(np.ones((3, 2))))
    adata.obs_names = ["query", "neighbor_a", "neighbor_b"]
    adata.uns["neighbor_dict"] = {"query": ["neighbor_a", "neighbor_b"]}

    with pytest.raises(ValueError, match="k_neighbors=1"):
        validate_precomputed_neighbor_dict(adata, max_neighbors=1)


def test_validate_precomputed_neighbor_celltypes_supports_query_only_adata():
    adata = ad.AnnData(X=csr_array(np.ones((2, 2))))
    adata.obs_names = ["query_a", "query_b"]
    adata.uns["cell_type_dict"] = {"T cell": -1, "tumor": -2}
    adata.uns["neighbor_celltype_dict"] = {
        "query_a": ["T cell", "tumor"],
        "query_b": ["tumor", "tumor"],
    }

    result = validate_precomputed_neighbor_celltype_dict(adata, max_neighbors=2)

    assert result["query_a"] == ["T cell", "tumor"]


def test_build_neighbor_dict_can_toggle_same_cell_type_neighbors():
    adata = ad.AnnData(X=csr_array(np.ones((3, 2))))
    adata.obs_names = ["query", "same_type", "other_type"]
    adata.obs["x"] = [0.0, 1.0, 10.0]
    adata.obs["y"] = [0.0, 0.0, 0.0]
    adata.obs["cell_type"] = ["A", "A", "B"]
    adata.obs["tissue"] = ["T", "T", "T"]

    without_same_type = build_neighbor_dict(
        adata,
        k_neighbors=1,
        coord_cols=("x", "y"),
        tissue_col="tissue",
        min_source_counts=0,
        use_same_cell_type_neighbors=False,
    )
    with_same_type = build_neighbor_dict(
        adata,
        k_neighbors=1,
        coord_cols=("x", "y"),
        tissue_col="tissue",
        min_source_counts=0,
        use_same_cell_type_neighbors=True,
    )

    assert without_same_type.uns["neighbor_dict"]["query"] == ["other_type"]
    assert with_same_type.uns["neighbor_dict"]["query"] == ["same_type"]


def test_sample_admixture_can_toggle_query_expressed_gene_filter():
    adata = ad.AnnData(
        X=csr_array(
            np.array(
                [
                    [10, 0],
                    [100, 1],
                ],
            ),
        ),
    )
    adata.obs_names = ["query", "source"]
    adata.uns["neighbor_dict"] = {"query": ["source"], "source": []}

    only_unexpressed, _ = sample_admixture_from_neighbor_dict(
        adata,
        tau_range=(0.99, 0.99),
        min_sampled_transcript_threshold=0,
        only_sample_unexpressed_genes=True,
        rng=np.random.default_rng(0),
    )
    all_source_genes, _ = sample_admixture_from_neighbor_dict(
        adata,
        tau_range=(0.99, 0.99),
        min_sampled_transcript_threshold=0,
        only_sample_unexpressed_genes=False,
        rng=np.random.default_rng(0),
    )

    assert only_unexpressed[0, 0] == 0
    assert only_unexpressed[0, 1] == 1
    assert all_source_genes[0, 0] == 100
    assert all_source_genes[0, 1] == 1
