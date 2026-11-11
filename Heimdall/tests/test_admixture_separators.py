from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from scipy.sparse import csr_matrix

from Heimdall.fc import AdmixtureFc
from Heimdall.embedding import FlexibleTypeLinear
from Heimdall.utils import instantiate_from_config
from scripts.run_admixture_inference import build_fast_inference_loader


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("include_query,include_neighbors", [(True, True), (False, True), (True, False), (False, False)])
def test_separator_training_inference_parity(monkeypatch, enabled, include_query, include_neighbors):
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: pd.DataFrame())
    adata = ad.AnnData(csr_matrix([[2., 3.]]))
    adata.obs_names = ["query"]
    adata.obs["cell_type"] = ["Q"]
    adata.var["identity_embedding_index"] = [0, 1]
    adata.uns["cell_type_dict"] = {"Q": -1, "A": -2, "B": -3}
    adata.uns["neighbor_celltype_dict"] = {"query": ["A", "B"]}
    prefix_len = int(include_query) + 2 * int(include_neighbors) + 3 * int(enabled)
    config = OmegaConf.create({
        "tailor_config": {"type": "Heimdall.tailor.AdmixtureTailor"},
        "order_config": {"type": "Heimdall.order.ExpressionOrder"},
        "reduce_config": {"type": "Heimdall.reduce.AdmixtureSumReduce"},
        "embedding_parameters": {"type": "Heimdall.embedding.GaussianInitEmbedding",
                                 "args": {"num_embeddings": 100, "embedding_dim": 4}},
    })
    fg = SimpleNamespace(pad_value=9)
    fe = SimpleNamespace(pad_value=9)
    fc = AdmixtureFc(fg, fe, adata, gene_metadata_filepath="unused.csv",
                    ensembl_dir=".", species="human", max_input_length=prefix_len + 2,
                    query_cell_type_token_mode="metadata",
                    include_query_cell_type_token=include_query,
                    include_neighbor_cell_type_tokens=include_neighbors,
                    include_separator_tokens=enabled, **config)
    expected = ([-101] if enabled else []) + ([-2, -3] if include_neighbors else [])
    expected += ([-102] if enabled else []) + ([-1] if include_query else [])
    expected += [-103] if enabled else []
    assert fc.num_prefix_tokens == prefix_len
    assert config.embedding_parameters.args.num_embeddings == 100
    fc.extract_surr_cell_types("query")
    train_identity, train_expression = fc.tailor(np.array([0, 1]), np.array([2., 3.]), np.array([0, 1]))
    assert train_identity.tolist() == expected + [0, 1]

    cr = SimpleNamespace(adata=adata, fc=fc, fg=fg, fe=fe,
                         splits={"test": [0]}, dataset_task_cfg=SimpleNamespace(context_length=2))
    batch = next(iter(build_fast_inference_loader(cr, batch_size=1)))
    np.testing.assert_array_equal(batch["identity_inputs"][0], train_identity)
    np.testing.assert_array_equal(batch["expression_inputs"][0], train_expression)
    assert batch["gene_order"].tolist() == [[0, 1]]

    embeddings = instantiate_from_config(fc.embedding_parameters)
    reduced = fc.reduce(batch["identity_inputs"].clone(), torch.nn.Embedding(10, 4),
                        batch["expression_inputs"].clone(), FlexibleTypeLinear(1, 4), embeddings)
    assert reduced.shape == (1, prefix_len + 2, 4)
    reduced.sum().backward()
    if enabled:
        assert torch.all(embeddings.weight.grad[100:103] != 0)
        # Marker identities do not change when observed types change.
        before = fc.separator_tokens
        adata.uns["cell_type_dict"] = {"Q": -1}
        assert fc.separator_tokens == before


def test_fast_loader_rejects_incorrect_prefix_length():
    cr = SimpleNamespace(
        splits={"test": [0]}, adata=ad.AnnData(csr_matrix([[1.]])),
        fc=SimpleNamespace(max_input_length=5, num_prefix_tokens=3),
        dataset_task_cfg=SimpleNamespace(context_length=3),
    )
    with pytest.raises(ValueError, match="tokenizer prefix length"):
        build_fast_inference_loader(cr, 1)
