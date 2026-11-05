from pathlib import Path

import anndata as ad
import numpy as np
from omegaconf import OmegaConf
import pytest
from pytest import fixture
from scipy.sparse import csr_array

from Heimdall.fc import AdmixtureFc


# def test_dummy_getitem(geneformer_fc, scgpt_fc):
#     dummy_index = -1

#     identity_inputs, expression_inputs, padding_mask = geneformer_fc[dummy_index]
#     assert np.all(identity_inputs == geneformer_fc.fg.pad_value)
#     assert np.all(expression_inputs == geneformer_fc.fe.pad_value)
#     assert np.all(padding_mask)

#     identity_inputs, expression_inputs, padding_mask = scgpt_fc[dummy_index]
#     assert np.all(identity_inputs == scgpt_fc.fg.pad_value)
#     assert np.all(expression_inputs == scgpt_fc.fe.pad_value)
#     assert np.all(padding_mask)


# def test_geneformer_fc_preprocess_cells_and_getitem(zero_expression_mock_dataset, geneformer_fc):
#     identity_expected = csr_array(
#         np.array(
#             [
#                 [1, 2, 3],
#                 [2, 3, 0],
#                 [3, 0, 1],
#                 [0, 1, 2],
#             ],
#         ),
#     )

#     _, raw_seq_length = identity_expected.shape

#     for cell_index in range(len(zero_expression_mock_dataset)):
#         identity_inputs, expression_inputs, padding_mask = geneformer_fc[cell_index]
#         assert np.allclose(identity_expected[[cell_index], :].toarray(), identity_inputs[:raw_seq_length])
#         assert len(identity_inputs) == geneformer_fc.max_input_length

#         assert not np.any(padding_mask[:raw_seq_length])
#         assert np.all(padding_mask[raw_seq_length:])


# def test_scgpt_fc_preprocess_cells_and_getitem(zero_expression_mock_dataset, scgpt_fc):
#     identity_expected = csr_array(
#         np.array(
#             [
#                 [1, 2, 3],
#                 [0, 2, 3],
#                 [0, 1, 3],
#                 [0, 1, 2],
#             ],
#         ),
#     )

#     expression_expected = csr_array(
#         np.array(
#             [
#                 [3, 2, 1],
#                 [1, 3, 2],
#                 [2, 1, 3],
#                 [3, 2, 1],
#             ],
#         ),
#     )

#     _, raw_seq_length = identity_expected.shape

#     seed = 0
#     rng = np.random.default_rng(seed)
#     for cell_index in range(len(zero_expression_mock_dataset)):
#         identity_inputs, expression_inputs, padding_mask = scgpt_fc[cell_index]
#         sample_indices = rng.choice(raw_seq_length, raw_seq_length, replace=False)
#         assert np.allclose(identity_expected[[cell_index], sample_indices], identity_inputs)
#         assert np.allclose(expression_expected[[cell_index], sample_indices], expression_inputs)
#         assert len(identity_inputs) == scgpt_fc.max_input_length

#         assert not np.any(padding_mask[: scgpt_fc.max_input_length])


# def test_scBERT_fc_preprocess_cells_and_getitem(zero_expression_mock_dataset, scbert_fc):
#     identity_expected = csr_array(
#         np.array(
#             [
#                 [1, 2, 3],
#                 [0, 2, 3],
#                 [0, 1, 3],
#                 [0, 1, 2],
#             ],
#         ),
#     )

#     expression_expected = csr_array(
#         np.array(
#             [
#                 [2, 2, 1],
#                 [1, 2, 2],
#                 [2, 1, 2],
#                 [2, 2, 1],
#             ],
#         ),
#     )

#     _, raw_seq_length = identity_expected.shape

#     seed = 0
#     rng = np.random.default_rng(seed)
#     for cell_index in range(len(zero_expression_mock_dataset)):
#         identity_inputs, expression_inputs, padding_mask = scbert_fc[cell_index]
#         sample_indices = rng.choice(raw_seq_length, raw_seq_length, replace=False)
#         assert np.allclose(identity_expected[[cell_index], sample_indices], identity_inputs)
#         assert np.allclose(expression_expected[[cell_index], sample_indices], expression_inputs)
#         assert len(identity_inputs) == scbert_fc.max_input_length

#         assert not np.any(padding_mask[: scbert_fc.max_input_length])


# # def test_uce_fc_order(mock_dataset_all_valid_genes, uce_fc):
# #     _, num_genes = weighted_sampling_fe.adata.shape
# #
# #     expected = np.array(
# #         [
# #             [1, 1, 0, 0, 2],
# #             [2, 2, 2, 1, 2],
# #             [2, 0, 2, 0, 2],
# #             [0, 2, 1, 0, 0],
# #         ],
# #     )
# #
# #     for cell_index in range(len(identity_fg.adata)):
# #         cell_identity_inputs, cell_expression_inputs = weighted_sampling_fe[cell_index]
# #         assert np.allclose(expected[cell_index], cell_identity_inputs)
# #
# #     assert weighted_sampling_fe.pad_value == 4
# #     assert weighted_sampling_fe.mask_value == 5


# def test_uce_fc_preprocess_cells_and_getitem(mock_dataset_all_valid_genes, uce_fc):
#     identity_expected = csr_array(
#         np.array(
#             [
#                 [-4, 0, 0],
#                 [-2, 1, -1],
#                 [-2, 1, 1],
#                 [-4, 0, 0],
#             ],
#         ),
#     )

#     expression_expected = csr_array(
#         np.array(
#             [
#                 [-4, 1, 1],
#                 [-2, 1, -1],
#                 [-2, 2, 2],
#                 [-4, 4, 4],
#             ],
#         ),
#     )

#     _, raw_seq_length = identity_expected.shape

#     for cell_index in range(len(mock_dataset_all_valid_genes)):
#         identity_inputs, expression_inputs, padding_mask = uce_fc[cell_index]
#         assert np.allclose(identity_expected[[cell_index], :].toarray(), identity_inputs[:raw_seq_length])
#         assert np.allclose(expression_expected[[cell_index], :].toarray(), expression_inputs[:raw_seq_length])
#         assert len(identity_inputs) == uce_fc.max_input_length

#         assert not np.any(padding_mask[:raw_seq_length])


# def test_geneformer_fc_reduce(geneformer_fc):
#     ...
#     # geneformer_fc.reduce() # TODO: fill out function call

#     # output = mock_dataset.obsm["cell_identity_inputs"]

#     # expected = np.array(
#     #     [
#     #         [1, 2, 3, 0],
#     #         [2, 3, 0, 1],
#     #         [3, 0, 1, 2],
#     #         [0, 1, 2, 3],
#     #     ],
#     # )

#     # assert np.allclose(expected, output)

def test_admixture_fc_extract_surr_cell_types(mock_dataset_admixture, admixture_fc):
    for i in range(len(mock_dataset_admixture)):

        # should update admixture_fc.surr_cell_tokens each generation
        admixture_fc[i]

        # reproduce token indexing in extract_surr_cell_types
        cell_id = mock_dataset_admixture.obs_names[i]
        surr_cell_ids = mock_dataset_admixture.uns["neighbor_dict"][cell_id]
        surr_cell_types = [mock_dataset_admixture.obs.loc[cid,'cell_type'] for cid in surr_cell_ids]
        expected_tokens = [mock_dataset_admixture.uns['cell_type_dict'][t] for t in surr_cell_types]
        assert admixture_fc.surr_cell_tokens == expected_tokens

    

def test_admixture_fc_getitem(mock_dataset_admixture, admixture_fc):
    for i in range(len(mock_dataset_admixture)):
        identity_inputs, expression_inputs, padding = admixture_fc[i]

        print(sum(padding))
        n_prefix = len(admixture_fc.surr_cell_tokens)

        # check that surrounding cell type tokens prepended to identity inputs
        assert all(admixture_fc.surr_cell_tokens == identity_inputs[:n_prefix])

        # check that surrounding cell type tokens prepended to expression inputs
        assert all(admixture_fc.surr_cell_tokens == expression_inputs[:n_prefix])


def test_admixture_fc_getitem_with_query_cell_type_token(mock_dataset_admixture, admixture_fc):
    admixture_fc.query_cell_type_token_mode = "metadata"

    for i in range(len(mock_dataset_admixture)):
        identity_inputs, expression_inputs, _ = admixture_fc[i]
        cell_id = mock_dataset_admixture.obs_names[i]
        query_cell_type = mock_dataset_admixture.obs.loc[cell_id, "cell_type"]
        query_token = mock_dataset_admixture.uns["cell_type_dict"][query_cell_type]
        expected_prefix = admixture_fc.surr_cell_tokens + [query_token]
        n_prefix = len(expected_prefix)

        assert admixture_fc.metadata_prefix_tokens == expected_prefix
        assert admixture_fc.num_prefix_tokens == n_prefix
        assert all(expected_prefix == identity_inputs[:n_prefix])
        assert all(expected_prefix == expression_inputs[:n_prefix])


def test_admixture_fc_predicted_cls_query_cell_type_token_raises(admixture_fc):
    admixture_fc.query_cell_type_token_mode = "predicted_cls"

    with pytest.raises(NotImplementedError):
        admixture_fc[0]


@pytest.mark.parametrize(
    ("include_query", "include_neighbors"),
    ((True, False), (False, True), (False, False)),
)
def test_admixture_fc_cell_type_token_ablations(
    include_query,
    include_neighbors,
):
    adata = ad.AnnData(np.zeros((1, 1)))
    adata.obs_names = ["query"]
    adata.obs["cell_type"] = ["query_type"]
    adata.uns["cell_type_dict"] = {
        "query_type": -1,
        "neighbor_a": -2,
        "neighbor_b": -3,
    }
    adata.uns["neighbor_celltype_dict"] = {
        "query": ["neighbor_a", "neighbor_b"],
    }

    admixture_fc = AdmixtureFc.__new__(AdmixtureFc)
    admixture_fc._adata = adata
    admixture_fc.query_cell_type_token_mode = "metadata"
    admixture_fc.query_cell_type_col = "cell_type"
    admixture_fc.include_query_cell_type_token = include_query
    admixture_fc.include_neighbor_cell_type_tokens = include_neighbors
    admixture_fc.num_neighbor_cell_type_tokens = None
    admixture_fc.admixture_token_offset = 1

    surr_types, surr_tokens, metadata_prefix = admixture_fc.get_metadata_prefix_tokens("query")
    expected_prefix = []
    if include_neighbors:
        expected_prefix.extend([-2, -3])
    if include_query:
        expected_prefix.append(-1)

    assert surr_types == ["neighbor_a", "neighbor_b"]
    assert surr_tokens == [-2, -3]
    assert metadata_prefix == expected_prefix
    assert admixture_fc.num_prefix_tokens == len(expected_prefix)


@pytest.mark.parametrize("visible_neighbors", [10, 8, 6, 4, 2, 1])
def test_admixture_fc_limits_visible_neighbor_types(visible_neighbors):
    adata = ad.AnnData(np.zeros((1, 1)))
    adata.obs_names = ["query"]
    adata.obs["cell_type"] = ["query_type"]
    neighbors = [f"neighbor_{i}" for i in range(10)]
    adata.uns["neighbor_celltype_dict"] = {"query": neighbors}
    adata.uns["cell_type_dict"] = {"query_type": -1, **{
        cell_type: -(i + 2) for i, cell_type in enumerate(neighbors)
    }}

    fc = AdmixtureFc.__new__(AdmixtureFc)
    fc._adata = adata
    fc.query_cell_type_token_mode = "metadata"
    fc.query_cell_type_col = "cell_type"
    fc.include_query_cell_type_token = True
    fc.include_neighbor_cell_type_tokens = True
    fc.num_neighbor_cell_type_tokens = visible_neighbors

    all_types, all_tokens, prefix = fc.get_metadata_prefix_tokens("query")
    assert all_types == neighbors
    assert len(all_tokens) == 10
    assert prefix == all_tokens[:visible_neighbors] + [-1]
    assert fc.num_prefix_tokens == visible_neighbors + 1



def test_admixture_fc_reduce(mock_dataset_admixture, admixture_fc):

    for i in range(len(mock_dataset_admixture)):
        identity_inputs, expression_inputs,_ = admixture_fc[i]

        # check that reduce is working
        admixture_fc.reduce(identity_inputs,expression_inputs)
        pass
