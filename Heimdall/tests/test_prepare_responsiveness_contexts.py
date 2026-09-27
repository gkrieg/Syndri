import anndata as ad
import numpy as np
from scipy.sparse import csr_array

from scripts.prepare_responsiveness_contexts import prepare_tissue


def test_prepare_tissue_copies_each_focal_across_same_type_contexts(tmp_path):
    adata = ad.AnnData(
        X=csr_array(
            np.array(
                [
                    [1, 0],
                    [2, 0],
                    [0, 3],
                    [0, 4],
                ],
            ),
        ),
    )
    adata.obs_names = ["t1", "t2", "tumor", "myeloid"]
    adata.obs["cell_type"] = ["T cell", "T cell", "tumor", "myeloid"]
    adata.obs["x"] = [0.0, 10.0, 1.0, 9.0]
    adata.obs["y"] = 0.0
    input_path = tmp_path / "tissue.h5ad"
    output_path = tmp_path / "prepared.h5ad"
    adata.write_h5ad(input_path)

    result = prepare_tissue(
        input_path,
        output_path,
        cell_type="T cell",
        cell_type_col="cell_type",
        coord_cols=("x", "y"),
        tissue_col=None,
        k_neighbors=1,
        min_source_counts=0,
        use_same_cell_type_neighbors=False,
        split_col="split",
        split_value="val",
    )

    assert result.n_obs == 4
    assert result.obs["focal_cell_id"].tolist() == ["t1", "t1", "t2", "t2"]
    assert result.obs["context_source_cell_id"].tolist() == ["t1", "t2", "t1", "t2"]
    assert result.obs["is_original"].tolist() == [True, False, False, True]
    assert set(result.obs["cell_type"]) == {"T cell"}
    assert "neighbor_dict" not in result.uns
    assert len(result.uns["neighbor_celltype_dict"]) == 4
    assert output_path.is_file()

    roundtrip = ad.read_h5ad(output_path)
    assert roundtrip.n_obs == 4
    assert roundtrip.obsm["neighborhood_proportions"].shape == (4, 3)


def test_prepare_tissue_samples_queries_contexts_and_repeats(tmp_path):
    adata = ad.AnnData(X=csr_array(np.ones((7, 2))))
    adata.obs_names = ["t1", "t2", "t3", "t4", "a", "b", "c"]
    adata.obs["cell_type"] = ["T cell"] * 4 + ["A", "B", "C"]
    adata.obs["x"] = [0.0, 10.0, 20.0, 30.0, 1.0, 11.0, 21.0]
    adata.obs["y"] = 0.0
    input_path = tmp_path / "sampled_tissue.h5ad"
    output_path = tmp_path / "sampled_prepared.h5ad"
    adata.write_h5ad(input_path)

    result = prepare_tissue(
        input_path,
        output_path,
        cell_type="T cell",
        cell_type_col="cell_type",
        coord_cols=("x", "y"),
        tissue_col=None,
        k_neighbors=1,
        min_source_counts=0,
        use_same_cell_type_neighbors=False,
        split_col="split",
        split_value="val",
        n_contexts=2,
        n_query_cells=3,
        n_repeats=2,
        random_seed=11,
        include_original_context=True,
    )

    representative = result.obs["is_representative_context"]
    assert representative.sum() == 2 * 3 * 2
    assert set(result.obs["sampling_repeat"]) == {0, 1}
    assert result.obs.groupby(
        ["sampling_repeat", "focal_cell_id"],
        observed=True,
    )["is_original"].any().all()
    assert result.n_obs <= 2 * (3 * 2 + 3)
