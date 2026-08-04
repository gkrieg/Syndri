#!/usr/bin/env python
"""Run MLP baseline admixture inference on a spatial AnnData file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.sparse import csr_matrix, issparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
SYNDRI_ROOT = THIS_DIR.parent
HEIMDALL_ROOT = SYNDRI_ROOT / "Heimdall"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(HEIMDALL_ROOT))
sys.path.insert(0, str(HEIMDALL_ROOT / "scripts"))

import run_admixture_inference as heimdall_infer  # noqa: E402
from Heimdall.cell_representations import CellRepresentation  # noqa: E402
from Heimdall.utils import get_dtype  # noqa: E402
from model import AdmixtureMLP, batch_to_mlp_inputs  # noqa: E402


class PrecomputedMLPInferenceDataset(Dataset):
    def __init__(self, batches: dict[str, torch.Tensor]):
        self.batches = batches
        self.length = next(iter(batches.values())).shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.batches.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--st-data", type=Path, required=True, help="Spatial AnnData input.")
    parser.add_argument("--reference-data", type=Path, required=True, help="Reference AnnData for cell_type_dict.")
    parser.add_argument("--gene-list", type=Path, default=None)
    parser.add_argument("--expression-layer", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-name", default="mlp_prepared.h5ad")
    parser.add_argument("--output-name", default="mlp_with_admix_probs.h5ad")
    parser.add_argument("--experiment", default="admixture_prediction")
    parser.add_argument(
        "--config-source",
        choices=("checkpoint", "experiment"),
        default="checkpoint",
        help="Load config from the MLP checkpoint, or compose from --experiment.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="MLP checkpoint written by train.py --save-checkpoint.",
    )
    parser.add_argument("--split-value", default="val")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--preserve-split-column", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--prob-layer", default="mlp_admix_probs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--coord-cols", default="CenterX_global_px,CenterY_global_px")
    parser.add_argument("--tissue-col", default=None)
    parser.add_argument("--apply-admixture", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extra-overrides", default="")
    parser.add_argument("--hidden-dims", default="1024,512")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"MLP checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _runtime_overrides(args: argparse.Namespace, prepared_path: Path) -> list[str]:
    return heimdall_infer._runtime_overrides(args, prepared_path)


def compose_config(args: argparse.Namespace, prepared_path: Path, checkpoint: dict):
    runtime_overrides = _runtime_overrides(args, prepared_path)

    if args.config_source == "checkpoint":
        if "config" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain a saved config. Re-run training with the updated "
                "--save-checkpoint path or use --config-source experiment."
            )
        config = OmegaConf.create(checkpoint["config"])
        OmegaConf.set_struct(config, False)
        heimdall_infer._apply_dotlist_overrides(config, runtime_overrides)
        print(f"> Loaded config from MLP checkpoint: {args.checkpoint_path}")
        print(OmegaConf.to_yaml(config, resolve=True))
        return config

    overrides = [f"+experiments={args.experiment}", *runtime_overrides]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(HEIMDALL_ROOT / "config")):
        config = hydra.compose(config_name="config", overrides=overrides)
    print(f"> Composed config from experiment: {args.experiment}")
    print(OmegaConf.to_yaml(config, resolve=True))
    return config


def build_token_to_cell_index(cell_type_dict: dict) -> torch.Tensor:
    max_token = max(abs(int(token)) for token in cell_type_dict.values())
    token_to_cell_index = torch.full((max_token + 1,), -1, dtype=torch.long)
    for cell_index, token in enumerate(cell_type_dict.values()):
        token_to_cell_index[abs(int(token))] = cell_index
    return token_to_cell_index


def _neighbor_tokens(adata, cell_id: str, cell_type_dict: dict, max_neighbors: int) -> list[int]:
    neighbor_types = [str(cell_type) for cell_type in adata.uns["neighbor_celltype_dict"].get(cell_id, [])]
    missing = [cell_type for cell_type in neighbor_types[:max_neighbors] if cell_type not in cell_type_dict]
    if missing:
        raise KeyError(f"Neighbor cell types are missing from cell_type_dict. First missing values: {missing[:20]}")
    return [cell_type_dict[cell_type] for cell_type in neighbor_types[:max_neighbors]]


def build_mlp_inference_loader(
    cr: CellRepresentation,
    cfg,
    *,
    batch_size: int,
    cell_type_col: str,
    split: str = "test",
    num_workers: int = 0,
) -> DataLoader:
    indices = np.asarray(cr.splits[split], dtype=np.int64)
    adata = cr.adata
    expression = adata.X.tocsr() if issparse(adata.X) else csr_matrix(adata.X)

    seq_len = int(cr.fc.max_input_length)
    gene_len = int(cfg.tasks.args.context_length)
    neighbor_len = seq_len - gene_len
    if neighbor_len <= 0:
        raise ValueError(f"Expected max_input_length > context_length, got {seq_len=} and {gene_len=}.")

    if cell_type_col not in adata.obs:
        raise KeyError(f"{cell_type_col!r} not found in prepared AnnData obs columns.")

    cell_type_dict = {str(cell_type): token for cell_type, token in dict(adata.uns["cell_type_dict"]).items()}
    cell_types = adata.obs[cell_type_col].astype(str)
    missing_query_types = sorted(set(cell_types.iloc[indices]) - set(cell_type_dict))
    if missing_query_types:
        raise KeyError(
            f"{len(missing_query_types)} query cell types are missing from cell_type_dict. "
            f"First 20: {missing_query_types[:20]}"
        )

    n = len(indices)
    identity_inputs = np.full((n, seq_len), cr.fg.pad_value, dtype=np.int64)
    expression_inputs = np.full((n, seq_len), cr.fe.pad_value, dtype=np.float32)
    gene_order = np.full((n, gene_len), cr.fe.pad_value, dtype=np.int64)
    labels = np.full((n, gene_len), np.nan, dtype=np.float32)
    query_cell_type_token = np.empty(n, dtype=np.int64)

    identity_token_order = adata.var["identity_embedding_index"].to_numpy(dtype=np.int64)
    max_token = int(identity_token_order.max())
    token_to_col = np.full(max_token + 1, cr.fe.pad_value, dtype=np.int64)
    token_to_col[identity_token_order] = np.arange(len(identity_token_order), dtype=np.int64)

    print(f"> Precomputing MLP inference tensors for {n:,} {split!r} cells...")
    for out_i, cell_idx in enumerate(tqdm(indices, desc="precompute MLP inference tensors")):
        cell_id = str(adata.obs_names[cell_idx])
        surr_tokens = _neighbor_tokens(adata, cell_id, cell_type_dict, neighbor_len)
        n_surr = len(surr_tokens)
        if n_surr:
            identity_inputs[out_i, :n_surr] = surr_tokens
            expression_inputs[out_i, :n_surr] = surr_tokens

        row_start = expression.indptr[cell_idx]
        row_stop = expression.indptr[cell_idx + 1]
        cols = expression.indices[row_start:row_stop]
        vals = expression.data[row_start:row_stop].astype(np.float32, copy=False)

        valid_expr = ~np.isnan(vals)
        cols = cols[valid_expr][:gene_len]
        vals = vals[valid_expr][:gene_len]
        tokens = identity_token_order[cols]

        gene_stop = neighbor_len + len(tokens)
        identity_inputs[out_i, neighbor_len:gene_stop] = tokens
        expression_inputs[out_i, neighbor_len:gene_stop] = vals

        valid_tokens = (tokens >= 0) & (tokens < len(token_to_col))
        row_gene_order = gene_order[out_i, : len(tokens)]
        row_gene_order[valid_tokens] = token_to_col[tokens[valid_tokens]]
        query_cell_type_token[out_i] = cell_type_dict[str(cell_types.iloc[cell_idx])]

    expression_padding = expression_inputs == cr.fe.pad_value
    tensor_data = {
        "identity_inputs": torch.from_numpy(identity_inputs),
        "expression_inputs": torch.from_numpy(expression_inputs),
        "expression_padding": torch.from_numpy(expression_padding),
        "labels": torch.from_numpy(labels),
        "gene_order": torch.from_numpy(gene_order),
        "query_cell_type_token": torch.from_numpy(query_cell_type_token),
    }
    dataset = PrecomputedMLPInferenceDataset(tensor_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _parse_hidden_dims(value) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(dim) for dim in value.split(",") if dim)
    return tuple(int(dim) for dim in value)


def build_model(args: argparse.Namespace, checkpoint: dict, cfg, cr: CellRepresentation) -> AdmixtureMLP:
    mlp_args = checkpoint.get("mlp_args", {})
    hidden_dims = _parse_hidden_dims(mlp_args.get("hidden_dims", args.hidden_dims))
    dropout = float(mlp_args.get("dropout", args.dropout))

    model = AdmixtureMLP(
        n_genes=cr.adata.n_vars,
        n_cell_types=len(cr.adata.uns["cell_type_dict"]),
        n_neighbor_tokens=int(cfg.fc.args.max_input_length) - int(cfg.tasks.args.context_length),
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    print(f"> Loaded MLP weights from {args.checkpoint_path}")
    print(f"> MLP architecture: hidden_dims={hidden_dims}, dropout={dropout}")
    return model


def run_inference(args: argparse.Namespace, prepared_path: Path, checkpoint: dict) -> Path:
    if not args.fast_inference:
        print("> MLP inference uses the precomputed tensor path; ignoring --no-fast-inference.")
    if args.apply_admixture:
        raise ValueError("MLP inference does not currently support --apply-admixture; use an input expression layer instead.")

    cfg = compose_config(args, prepared_path, checkpoint)
    cr = CellRepresentation(cfg)
    prediction_indices = np.asarray(cr.splits["test"], dtype=np.int64)
    if len(prediction_indices) == 0:
        raise ValueError(f"No cells found for inference split {args.split_col!r} == {args.split_value!r}.")

    batch_size = int(args.batch_size or cfg.tasks.args.batchsize)
    loader = build_mlp_inference_loader(
        cr,
        cfg,
        batch_size=batch_size,
        cell_type_col=args.cell_type_col,
        split="test",
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = build_model(args, checkpoint, cfg, cr)
    model_dtype = get_dtype(cfg.float_dtype)
    model = model.to(device=device, dtype=model_dtype)
    model.eval()

    batch_kwargs = {
        "n_genes": cr.adata.n_vars,
        "n_cell_types": len(cr.adata.uns["cell_type_dict"]),
        "n_neighbor_tokens": int(cfg.fc.args.max_input_length) - int(cfg.tasks.args.context_length),
        "token_to_cell_index": build_token_to_cell_index(cr.adata.uns["cell_type_dict"]),
    }

    probs_chunks = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="MLP inference"):
            features, _, _ = batch_to_mlp_inputs(batch, device=device, **batch_kwargs)
            logits = model(features.to(model_dtype))
            probs_chunks.append(torch.sigmoid(logits).detach().cpu().numpy())

    probs = np.concatenate(probs_chunks, axis=0).astype(np.float32)
    expected_shape = (len(prediction_indices), cr.adata.X.shape[1])
    if probs.shape != expected_shape:
        raise ValueError(f"Prediction shape {probs.shape} does not match selected split shape {expected_shape}")

    model_output_adata = cr.adata[prediction_indices].copy()
    output_adata = heimdall_infer.build_output_adata_with_input_genes(args, model_output_adata)
    output_adata.layers[args.prob_layer] = heimdall_infer._expand_matrix_to_input_genes(
        probs,
        model_output_adata,
        pd.Index(output_adata.var_names.astype(str)),
        args.prob_layer,
    )

    output_path = args.output_dir / args.output_name
    output_adata.write_h5ad(output_path)
    print(f"> Wrote MLP predictions to {output_path} in layer {args.prob_layer!r}.")
    return output_path


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint_path)
    prepared_path = heimdall_infer.prepare_spatial_adata(args)
    run_inference(args, prepared_path, checkpoint)


if __name__ == "__main__":
    main()
