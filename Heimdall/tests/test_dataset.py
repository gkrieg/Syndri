from pathlib import Path

import anndata as ad
import numpy as np
from omegaconf import OmegaConf
from pytest import fixture
from scipy.sparse import csr_array

from Heimdall.datasets import AdmixtureInstanceDataset
from Heimdall.cell_representations import CellRepresentation

model_configs = [
    """
        model:
          type: Heimdall.models.Transformer
          name: transformer
          args:
            d_model: 128
            pos_enc: BERT
            num_encoder_layers: 6
            nhead: 4
            hidden_act: gelu
            hidden_dropout_prob: 0.1
            attention_probs_dropout_prob: 0.1
            use_flash_attn: false
            pooling: cls_pooling # or "mean_pooling"
    """,
    """
        model:
            type: Heimdall.models.ExpressionWeightedSum
            name: ExpressionWeightedSum
            args:
              d_model: 128
              pos_enc: null
              pooling: mean_pooling # or "mean_pooling"
    """,
    """
        model:
            type: Heimdall.models.Average
            name: ExpressionWeightedSum
            args:
              d_model: 128
              pos_enc: null
              pooling: mean_pooling # or "mean_pooling"
    """,
    """
        model:
            type: Heimdall.models.ExpressionOnly
            name: logistic_regression
    """,
]

def test_admixture_instance_dataset():
    # need CellRepresentation
    cfg = OmegaConf.create(model_configs[0])
    data = CellRepresentation(cfg,)
    dataset = AdmixtureInstanceDataset(data)
    for i in range(len(dataset)):

        output = dataset[i]
        print(output["labels"])
    pass