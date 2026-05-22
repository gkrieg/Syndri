"""MLP baseline modules for admixture prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdmixtureMLP(nn.Module):
    """Predict admixed transcript labels from dense expression and cell-type one-hots."""

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        n_neighbor_tokens: int,
        hidden_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_neighbor_tokens = n_neighbor_tokens

        input_dim = n_genes + (1 + n_neighbor_tokens) * n_cell_types
        layers: list[nn.Module] = []
        dim_in = input_dim
        for dim_out in hidden_dims:
            layers.extend(
                [
                    nn.Linear(dim_in, dim_out),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ],
            )
            dim_in = dim_out
        layers.append(nn.Linear(dim_in, n_genes))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _tokens_to_one_hot(
    tokens: torch.Tensor,
    token_to_cell_index: torch.Tensor,
    n_cell_types: int,
) -> torch.Tensor:
    tokens = tokens.long()
    valid = (tokens >= 0) & (tokens < token_to_cell_index.numel())
    safe_tokens = tokens.clamp(min=0, max=token_to_cell_index.numel() - 1)
    cell_indices = token_to_cell_index[safe_tokens]
    valid = valid & (cell_indices >= 0)
    one_hot = F.one_hot(cell_indices.clamp_min(0), num_classes=n_cell_types).float()
    return one_hot * valid.unsqueeze(-1)


def batch_to_mlp_inputs(
    batch: dict[str, torch.Tensor],
    *,
    n_genes: int,
    n_cell_types: int,
    n_neighbor_tokens: int,
    token_to_cell_index: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a Heimdall admixture batch into MLP features and ordered labels."""

    identity_inputs = batch["identity_inputs"].to(device)
    expression_inputs = batch["expression_inputs"].to(device).float()
    gene_order = batch["gene_order"].to(device).long()
    labels = batch["labels"].to(device).float()
    query_tokens = batch["query_cell_type_token"].to(device).long()
    token_to_cell_index = token_to_cell_index.to(device)

    gene_expression = expression_inputs[:, n_neighbor_tokens : n_neighbor_tokens + gene_order.shape[1]]
    if gene_expression.shape[1] != gene_order.shape[1]:
        gene_expression = expression_inputs[:, -gene_order.shape[1] :]

    valid_gene = (gene_order >= 0) & (gene_order < n_genes)
    dense_expression = torch.zeros(
        (gene_order.shape[0], n_genes),
        dtype=gene_expression.dtype,
        device=device,
    )
    dense_expression.scatter_add_(1, gene_order.clamp(min=0, max=n_genes - 1), gene_expression * valid_gene)

    query_one_hot = _tokens_to_one_hot(query_tokens, token_to_cell_index, n_cell_types)
    neighbor_tokens = identity_inputs[:, :n_neighbor_tokens]
    neighbor_one_hot = _tokens_to_one_hot(neighbor_tokens, token_to_cell_index, n_cell_types).flatten(start_dim=1)
    features = torch.cat([dense_expression, query_one_hot, neighbor_one_hot], dim=1)

    labels = labels.clone()
    labels[~valid_gene] = float("nan")
    return features, gene_order, labels


def gather_gene_ordered_logits(
    full_gene_logits: torch.Tensor,
    gene_order: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_gene = (gene_order >= 0) & (gene_order < full_gene_logits.shape[1])
    logits = full_gene_logits.gather(1, gene_order.clamp(min=0, max=full_gene_logits.shape[1] - 1))
    labels = labels.clone()
    labels[~valid_gene] = float("nan")
    return logits, labels
