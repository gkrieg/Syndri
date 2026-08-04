#!/usr/bin/env python
"""Run Heimdall admixture inference from a prepared spatial AnnData file."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.sparse import coo_matrix, csr_matrix, issparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Heimdall.cell_representations import CellRepresentation
from Heimdall.models import HeimdallModel
from Heimdall.trainer import HeimdallTrainer
from Heimdall.utils import count_parameters, get_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--st-data",
        type=Path,
        default=Path("/compute/oven-0-13/jwinchel/spatial_adata/cosmx_nsclc_Lung5-Rep1_adata_annotated.h5ad"),
        help="Spatial AnnData input.",
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=Path("/compute/oven-0-13/jwinchel/scRNA-seq/adata/GSE127465_tumor_integrated.h5ad"),
        help="Reference AnnData used for cell_type_dict and gene consistency.",
    )
    parser.add_argument(
        "--gene-list",
        type=Path,
        default=None,
        help="Optional text file containing one gene per line. Defaults to all spatial AnnData genes.",
    )
    parser.add_argument(
        "--expression-layer",
        default=None,
        help="Optional layer to use as model input expression instead of adata.X.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/compute/oven-0-13/jwinchel/spatial_adata/heimdall_inference"),
        help="Directory for prepared input and prediction output.",
    )
    parser.add_argument("--prepared-name", default="cosmx_nsclc_Lung5-Rep1_prepared.h5ad")
    parser.add_argument("--output-name", default="cosmx_nsclc_Lung5-Rep1_with_admix_probs.h5ad")
    parser.add_argument("--experiment", default="admixture_prediction")
    parser.add_argument(
        "--config-source",
        choices=("checkpoint", "experiment"),
        default="checkpoint",
        help="Load model-defining config from checkpoint config.txt, or compose from --experiment.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/home/jwinchel/Heimdall/admixture_prediction_results/posweight_4p0"),
    )
    parser.add_argument("--milestone", type=int, default=99)
    parser.add_argument("--split-value", default="val")
    parser.add_argument("--split-col", default="split")
    parser.add_argument(
        "--preserve-split-column",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the input AnnData split column and infer only rows matching "
            "--split-value. By default, all prepared cells are assigned --split-value."
        ),
    )
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--prob-layer", default="admix_probs")
    parser.add_argument(
        "--admixture-layer",
        default="admixed_counts",
        help="Layer name for exact sampled admixture counts when --apply-admixture is enabled.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--coord-cols",
        default="CenterX_global_px,CenterY_global_px",
        help="Comma-separated coordinate columns used for neighbor-aware inference.",
    )
    parser.add_argument(
        "--tissue-col",
        default=None,
        help=(
            "Optional tissue column for neighbor/admixture sampling. If omitted, "
            "spatial inference uses no tissue column and apply-admixture mode uses the dataset default."
        ),
    )
    parser.add_argument("--apply-admixture", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--extra-overrides",
        default="",
        help="Optional space-separated dotlist overrides applied after standard runtime overrides.",
    )
    return parser.parse_args()


def genes_from_txt(gene_list_path: Path) -> list[str]:
    genes = [line.strip() for line in gene_list_path.read_text().splitlines() if line.strip()]
    if not genes:
        raise ValueError(f"Gene list is empty: {gene_list_path}")
    return genes


def build_cell_type_dict(adata: ad.AnnData, cell_type_col: str) -> dict[str, int]:
    cell_types = sorted(adata.obs[cell_type_col].astype(str).unique())
    return {cell_type: -(idx + 1) for idx, cell_type in enumerate(cell_types)}


def prepare_spatial_adata(args: argparse.Namespace) -> Path:
    st_data = ad.read_h5ad(args.st_data)
    ref_data = ad.read_h5ad(args.reference_data)
    if args.expression_layer is not None:
        if args.expression_layer not in st_data.layers:
            raise KeyError(f"{args.expression_layer!r} not found in spatial adata.layers")
        st_data.X = st_data.layers[args.expression_layer].copy()
        print(f"> Using spatial AnnData layer {args.expression_layer!r} as model input expression.")
    if args.gene_list is None:
        genes = st_data.var_names.astype(str).tolist()
        print(f"> Using all {len(genes):,} spatial AnnData genes.")
    else:
        genes = genes_from_txt(args.gene_list)
        print(f"> Loaded {len(genes):,} genes from {args.gene_list}.")

    if args.cell_type_col not in st_data.obs:
        raise KeyError(f"{args.cell_type_col!r} not found in spatial adata.obs")

    valid_cell_type = st_data.obs[args.cell_type_col].notna()
    valid_cell_type &= st_data.obs[args.cell_type_col].astype(str).str.lower().ne("nan")
    dropped = int((~valid_cell_type).sum())
    st_data = st_data[valid_cell_type].copy()
    print(f"> Dropped {dropped:,} cells with missing/'nan' {args.cell_type_col!r}.")

    missing_genes = [gene for gene in genes if gene not in st_data.var_names]
    if missing_genes:
        preview = missing_genes[:20]
        raise ValueError(f"{len(missing_genes)} gene-list genes are missing from spatial AnnData. First 20: {preview}")
    st_data = st_data[:, genes].copy()
    print(f"> Prepared spatial AnnData with {len(genes):,} genes.")

    if "cell_type_dict" in ref_data.uns:
        st_data.uns["cell_type_dict"] = dict(ref_data.uns["cell_type_dict"])
        print("> Copied cell_type_dict from reference AnnData.")
    elif "cell_type_dict" in st_data.uns:
        st_data.uns["cell_type_dict"] = dict(st_data.uns["cell_type_dict"])
        print("> Using existing cell_type_dict from spatial AnnData.")
    else:
        st_data.uns["cell_type_dict"] = build_cell_type_dict(st_data, args.cell_type_col)
        print("> Built cell_type_dict from spatial AnnData cell types.")

    if args.preserve_split_column:
        if args.split_col not in st_data.obs:
            raise KeyError(f"{args.split_col!r} not found in input adata.obs")
        split_count = int((st_data.obs[args.split_col] == args.split_value).sum())
        if split_count == 0:
            raise ValueError(f"No cells found with {args.split_col!r} == {args.split_value!r}.")
        print(f"> Preserved {args.split_col!r}; {split_count:,} cells match split value {args.split_value!r}.")
    else:
        st_data.obs[args.split_col] = args.split_value
        print(f"> Set all cells to {args.split_col}={args.split_value!r}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = args.output_dir / args.prepared_name
    st_data.write_h5ad(prepared_path)
    print(f"> Wrote prepared AnnData: {prepared_path}")
    return prepared_path


def _parse_override_value(value: str):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [_parse_override_value(item.strip()) for item in value[1:-1].split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _apply_dotlist_overrides(config, overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be in key=value form, got {override!r}")
        key, value = override.split("=", 1)
        key = key.lstrip("+")
        OmegaConf.update(config, key, _parse_override_value(value), merge=True, force_add=True)


def _runtime_overrides(args: argparse.Namespace, prepared_path: Path) -> list[str]:
    coord_cols = [col.strip() for col in args.coord_cols.split(",") if col.strip()]
    if len(coord_cols) != 2:
        raise ValueError(f"--coord-cols must contain exactly two comma-separated column names, got {args.coord_cols!r}")

    overrides = [
        f"dataset.preprocess_args.data_path={prepared_path}",
        f"++dataset.preprocess_args.apply_admixture={str(args.apply_admixture).lower()}",
        f"++dataset.preprocess_args.admixture_args.coord_cols=[{coord_cols[0]},{coord_cols[1]}]",
        f"++dataset.preprocess_args.admixture_args.cell_type_col={args.cell_type_col}",
        f"tasks.args.splits.col={args.split_col}",
        f"tasks.args.splits.keys_.test={args.split_value}",
    ]
    if args.tissue_col is not None:
        overrides.append(f"++dataset.preprocess_args.admixture_args.tissue_col={args.tissue_col}")
    elif not args.apply_admixture:
        overrides.append("++dataset.preprocess_args.admixture_args.tissue_col=null")
    if args.batch_size is not None:
        overrides.append(f"tasks.args.batchsize={args.batch_size}")
    if args.extra_overrides:
        overrides.extend(shlex.split(args.extra_overrides))
    return overrides


def compose_config(args: argparse.Namespace, prepared_path: Path):
    runtime_overrides = _runtime_overrides(args, prepared_path)

    if args.config_source == "checkpoint":
        checkpoint_config_path = args.checkpoint_dir / "config.txt"
        if not checkpoint_config_path.is_file():
            raise FileNotFoundError(f"Checkpoint config not found: {checkpoint_config_path}")
        config = OmegaConf.load(checkpoint_config_path)
        OmegaConf.set_struct(config, False)
        _apply_dotlist_overrides(config, runtime_overrides)
        print(f"> Loaded model config from checkpoint: {checkpoint_config_path}")
        print(OmegaConf.to_yaml(config, resolve=True))
        return config

    overrides = [f"+experiments={args.experiment}", *runtime_overrides]

    with hydra.initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "config")):
        config = hydra.compose(config_name="config", overrides=overrides)
    print(f"> Composed model config from experiment: {args.experiment}")
    print(OmegaConf.to_yaml(config, resolve=True))
    return config


class PrecomputedAdmixtureInferenceDataset(Dataset):
    def __init__(self, batches: dict[str, torch.Tensor]):
        self.batches = batches
        self.length = next(iter(batches.values())).shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.batches.items()}


def _neighbor_tokens(adata: ad.AnnData, cell_id: str, cell_type_dict: dict, max_neighbors: int) -> list[int]:
    neighbor_types = adata.uns["neighbor_celltype_dict"].get(cell_id, [])
    return [cell_type_dict[cell_type] for cell_type in neighbor_types[:max_neighbors]]


def build_fast_inference_loader(cr: CellRepresentation, batch_size: int, split: str = "test") -> DataLoader:
    """Materialize inference tensors once, avoiding slow AnnData work per batch."""

    indices = np.asarray(cr.splits[split], dtype=np.int64)
    adata = cr.adata
    expression = adata.X.tocsr() if issparse(adata.X) else csr_matrix(adata.X)

    seq_len = int(cr.fc.max_input_length)
    gene_len = int(cr.dataset_task_cfg.context_length)
    neighbor_len = seq_len - gene_len
    if neighbor_len <= 0:
        raise ValueError(f"Expected max_input_length > context_length, got seq_len={seq_len} and gene_len={gene_len}.")

    n = len(indices)
    identity_inputs = np.full((n, seq_len), cr.fg.pad_value, dtype=np.int64)
    expression_inputs = np.full((n, seq_len), cr.fe.pad_value, dtype=np.float32)
    gene_order = np.full((n, gene_len), cr.fe.pad_value, dtype=np.int64)
    labels = np.full((n, gene_len), np.nan, dtype=np.float32)

    identity_token_order = adata.var["identity_embedding_index"].to_numpy(dtype=np.int64)
    max_token = int(identity_token_order.max())
    token_to_col = np.full(max_token + 1, cr.fe.pad_value, dtype=np.int64)
    token_to_col[identity_token_order] = np.arange(len(identity_token_order), dtype=np.int64)
    cell_type_dict = dict(adata.uns["cell_type_dict"])

    print(f"> Precomputing fast inference tensors for {n:,} {split!r} cells...")
    for out_i, cell_idx in enumerate(tqdm(indices, desc="precompute inference tensors")):
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

    expression_padding = expression_inputs == cr.fe.pad_value
    tensor_data = {
        "identity_inputs": torch.from_numpy(identity_inputs),
        "expression_inputs": torch.from_numpy(expression_inputs),
        "expression_padding": torch.from_numpy(expression_padding),
        "labels": torch.from_numpy(labels),
        "gene_order": torch.from_numpy(gene_order),
    }
    dataset = PrecomputedAdmixtureInferenceDataset(tensor_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _prediction_var_names(model_adata: ad.AnnData) -> pd.Index:
    return pd.Index(model_adata.var_names.astype(str), name=model_adata.var_names.name)


def _expand_matrix_to_input_genes(
    matrix,
    model_adata: ad.AnnData,
    input_var_names: pd.Index,
    layer_name: str,
) -> csr_matrix:
    matrix = matrix.tocsr() if issparse(matrix) else csr_matrix(matrix)
    prediction_var_names = _prediction_var_names(model_adata)
    input_index = pd.Index(input_var_names.astype(str))
    output_cols = input_index.get_indexer(prediction_var_names)
    valid = output_cols >= 0

    if not valid.all():
        missing = prediction_var_names[~valid].tolist()
        preview = missing[:20]
        raise ValueError(
            f"{len(missing)} {layer_name!r} prediction genes are missing from the original input AnnData. "
            f"First 20: {preview}"
        )

    matrix = matrix[:, valid]
    matrix = matrix.tocoo()
    expanded = coo_matrix(
        (matrix.data, (matrix.row, output_cols[valid][matrix.col])),
        shape=(matrix.shape[0], len(input_index)),
        dtype=matrix.dtype,
    ).tocsr()
    print(
        f"> Expanded {layer_name!r} from {matrix.shape[1]:,} model genes "
        f"to {len(input_index):,} input genes; unmodeled genes are zero."
    )
    return expanded


def build_output_adata_with_input_genes(args: argparse.Namespace, model_adata: ad.AnnData) -> ad.AnnData:
    input_adata = ad.read_h5ad(args.st_data)
    output_obs_names = pd.Index(model_adata.obs_names.astype(str))
    missing_obs = output_obs_names.difference(pd.Index(input_adata.obs_names.astype(str)))
    if len(missing_obs):
        preview = missing_obs[:20].tolist()
        raise ValueError(f"{len(missing_obs)} prediction cells are missing from input AnnData. First 20: {preview}")

    output_adata = input_adata[output_obs_names, :].copy()
    for column in model_adata.obs.columns:
        output_adata.obs[column] = model_adata.obs[column].reindex(output_adata.obs_names).values
    output_adata.uns.update(model_adata.uns)
    return output_adata


def run_inference(args: argparse.Namespace, prepared_path: Path) -> Path:
    if args.apply_admixture and args.fast_inference:
        print("> Disabling fast inference because apply_admixture=true requires the standard dataset path.")
        args.fast_inference = False

    config = compose_config(args, prepared_path)
    cr = CellRepresentation(config)

    float_dtype = get_dtype(config.float_dtype)
    model = HeimdallModel(data=cr, model_config=config.model, task_config=config.tasks.args).to(float_dtype)
    print(f"> Model parameters: {count_parameters(model):,}")

    trainer = HeimdallTrainer(cfg=config, model=model, data=cr, run_wandb=False)
    trainer.initialize_checkpointing(args.checkpoint_dir)
    trainer.load_checkpoint(specific_milestone=args.milestone)

    prediction_indices = np.asarray(cr.splits["test"], dtype=np.int64)
    if len(prediction_indices) == 0:
        raise ValueError(f"No cells found for inference split {args.split_col!r} == {args.split_value!r}.")

    prediction_loader = trainer.dataloader_test
    if args.fast_inference:
        prediction_loader = build_fast_inference_loader(cr, config.tasks.args.batchsize, split="test")
        prediction_loader = trainer.accelerator.prepare(prediction_loader)

    probs = np.concatenate(trainer.get_predictions(prediction_loader), axis=0).astype(np.float32)
    expected_shape = (len(prediction_indices), trainer.data.adata.X.shape[1])
    if probs.shape != expected_shape:
        raise ValueError(f"Prediction shape {probs.shape} does not match selected split shape {expected_shape}")

    model_output_adata = trainer.data.adata[prediction_indices].copy()
    output_adata = build_output_adata_with_input_genes(args, model_output_adata)
    output_adata.layers[args.prob_layer] = _expand_matrix_to_input_genes(
        probs,
        model_output_adata,
        output_adata.var_names,
        args.prob_layer,
    )
    if args.apply_admixture:
        admixture_matrix = trainer.data.fe.current_admixture_matrix
        if admixture_matrix is None:
            raise RuntimeError(
                "apply_admixture=True, but no current admixture matrix was materialized for inference."
            )
        output_adata.layers[args.admixture_layer] = _expand_matrix_to_input_genes(
            admixture_matrix[prediction_indices],
            model_output_adata,
            output_adata.var_names,
            args.admixture_layer,
        )
        print(f"> Added exact sampled admixture counts to layer {args.admixture_layer!r}.")

    output_path = args.output_dir / args.output_name
    output_adata.write_h5ad(output_path)
    print(f"> Wrote predictions to {output_path} in layer {args.prob_layer!r}.")
    return output_path


def main() -> None:
    args = parse_args()
    prepared_path = prepare_spatial_adata(args)
    run_inference(args, prepared_path)


if __name__ == "__main__":
    main()
