"""Run RCTD annotation on AnnData spatial files and save results back into .h5ad files.

Designed for use inside the `rctd` conda environment:
    conda run -n rctd python Syndri/code/rctd_annotation.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Pattern, Sequence

import anndata as ad
import numpy as np
import rctd
from rctd import RCTDConfig, Reference, run_rctd
ad.settings.allow_write_nullable_strings = True

def sanitize_key(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value))
    return re.sub(r"__+", "_", text).strip("_") or "unnamed"


def find_matching_layers(adata: ad.AnnData, layer_pattern: str) -> List[str]:
    regex = re.compile(layer_pattern, flags=re.IGNORECASE)
    candidates = ["X", *list(adata.layers.keys())]
    matches = [layer for layer in candidates if regex.search(layer)]
    return sorted(dict.fromkeys(matches), key=str)


def ensure_common_genes(spatial: ad.AnnData, reference: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData]:
    common = spatial.var_names.intersection(reference.var_names)
    if len(common) == 0:
        raise ValueError(
            f"No shared genes between spatial ({spatial.shape[1]}) and reference ({reference.shape[1]}) AnnData objects."
        )
    spatial_common = spatial[:, common].copy()
    reference_common = reference[:, common].copy()
    return spatial_common, reference_common


def prepare_spatial_for_layer(spatial: ad.AnnData, layer_key: str) -> ad.AnnData:
    if layer_key == "X":
        return spatial.copy()
    if layer_key not in spatial.layers:
        raise KeyError(f"Layer {layer_key!r} not found in AnnData.layers")

    spatial_layer = spatial.copy()
    spatial_layer.X = spatial_layer.layers[layer_key].copy()
    return spatial_layer


def build_reference(
    reference_adata: ad.AnnData,
    cell_type_col: str = "cell_type",
    cell_min: int = 25,
    n_max_cells: int = 10000,
    min_UMI: int = 10,
    max_UMI: int = 2000,
) -> Reference:
    if min_UMI < 1 or max_UMI < min_UMI:
        raise ValueError("Reference UMI bounds must satisfy 1 <= min_UMI <= max_UMI")
    # Reference does not implement an upper bound. Apply both bounds before
    # its cell-type minimum and subsampling, using the already shared gene panel.
    totals = np.asarray(reference_adata.X.sum(axis=1)).ravel()
    keep = np.isfinite(totals) & (totals >= min_UMI) & (totals <= max_UMI)
    if not keep.any():
        raise ValueError(f"No reference cells have {min_UMI}–{max_UMI} shared-panel UMIs")
    return Reference(
        reference_adata[keep].copy(),
        cell_type_col=cell_type_col,
        cell_min=cell_min,
        n_max_cells=n_max_cells,
        min_UMI=min_UMI,
    )


def build_rctd_config(
    *,
    class_df: Optional[Dict[str, str]] = None,
    UMI_min: int = 10,
    counts_MIN: int = 10,
    UMI_min_sigma: int = 100,
    device: str = "auto",
    compile: bool = True,
    **extra,
) -> RCTDConfig:
    config = RCTDConfig(
        UMI_min=UMI_min,
        counts_MIN=counts_MIN,
        UMI_min_sigma=UMI_min_sigma,
        device=device,
        compile=compile,
        class_df=class_df,
    )
    for key, value in extra.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    return config


def prepare_obsm_keys(spatial_path: Path, layer_key: str) -> dict[str, str]:
    sample_name = sanitize_key(spatial_path.stem)
    layer_suffix = sanitize_key(layer_key)
    return {
        "harmony": f"X_pca_harmony_{sample_name}_{layer_suffix}",
        "weights": f"rctd_weights_{layer_suffix}",
        "weights_doublet": f"rctd_weights_doublet_{layer_suffix}",
    }


def pad_result_array(values: np.ndarray, pixel_mask: Optional[np.ndarray], n_obs: int, fill_value: float = 0.0) -> np.ndarray:
    if pixel_mask is None:
        if values.shape[0] == n_obs:
            return values
        raise ValueError(f"Cannot pad result with shape {values.shape} to {n_obs} obs without a pixel mask.")

    if values.shape[0] != int(np.count_nonzero(pixel_mask)):
        raise ValueError(
            f"Result has {values.shape[0]} rows but pixel_mask selects {int(np.count_nonzero(pixel_mask))} observations."
        )

    padded = np.full((n_obs, values.shape[1]), fill_value, dtype=values.dtype)
    kept = np.flatnonzero(pixel_mask)
    padded[kept, :] = values
    return padded


def prepare_uns_keys(layer_key: str) -> dict[str, str]:
    suffix = sanitize_key(layer_key)
    return {
        "cell_type_names": f"rctd_cell_type_names_{suffix}",
        "class_df": f"rctd_class_df_{suffix}",
        "postprocessing": f"rctd_postprocessing_{suffix}",
        "run_metadata": f"rctd_run_metadata_{suffix}",
        "simulated_dropout": f"simulated_dropout_{suffix}",
        "source_h5_files": f"source_h5_files_{suffix}",
    }


def create_postprocessing_payload(result: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "cell_type_names": list(result.cell_type_names),
    }
    for key in [
        "spot_class",
        "first_type",
        "second_type",
        "first_class",
        "second_class",
        "first_class_name",
        "second_class_name",
        "min_score",
        "singlet_score",
        "converged",
    ]:
        if hasattr(result, key):
            payload[key] = getattr(result, key)
    return payload


def create_run_metadata(
    spatial_path: Path,
    reference_path: Path,
    layer_key: str,
    mode: str,
    config: RCTDConfig,
    source_h5_files: Sequence[str],
) -> Dict[str, Any]:
    metadata = {
        "spatial_file": str(spatial_path),
        "reference_file": str(reference_path),
        "count_layer": layer_key,
        "mode": mode,
        "source_h5_files": list(source_h5_files),
    }
    config_data = getattr(config, "__dict__", None)
    if config_data is None:
        try:
            config_data = asdict(config)
        except Exception:
            config_data = {}  # type: ignore[assignment]
    metadata["config"] = config_data
    return metadata


def annotate_single_layer(
    adata: ad.AnnData,
    spatial_path: Path,
    reference_path: Path,
    layer_key: str,
    cell_type_col: str,
    output_path: Path,
    mode: str = "doublet",
    class_df: Optional[Dict[str, str]] = None,
    simulated_dropout: Optional[Dict[str, Any]] = None,
    force: bool = False,
    batch_size: str | int = "auto",
    sigma_override: Optional[int] = None,
    **config_kwargs,
) -> None:
    output_keys = prepare_obsm_keys(spatial_path, layer_key)
    uns_keys = prepare_uns_keys(layer_key)
    if not force:
        if output_keys["weights"] in adata.obsm or output_keys["weights_doublet"] in adata.obsm:
            raise FileExistsError(
                f"RCTD weights already present for layer {layer_key} in {spatial_path}. Use --force to overwrite."
            )

    reference = ad.read_h5ad(reference_path)
    spatial_work, reference_work = ensure_common_genes(
        prepare_spatial_for_layer(adata, layer_key), reference
    )
    reference_obj = build_reference(
        reference_work,
        cell_type_col=cell_type_col,
        cell_min=config_kwargs.get("cell_min", 25),
        n_max_cells=config_kwargs.get("n_max_cells", 10000),
        min_UMI=config_kwargs.get("reference_min_UMI", 10),
        max_UMI=config_kwargs.get("reference_max_UMI", 2000),
    )
    config = build_rctd_config(
        class_df=class_df,
        UMI_min=config_kwargs.get("UMI_min", 10),
        counts_MIN=config_kwargs.get("counts_MIN", 10),
        UMI_min_sigma=config_kwargs.get("UMI_min_sigma", 100),
        device=config_kwargs.get("device", "auto"),
        compile=config_kwargs.get("compile", True),
        **{
            k: v
            for k, v in config_kwargs.items()
            if k not in {"cell_min", "n_max_cells", "reference_min_UMI", "reference_max_UMI", "UMI_min", "counts_MIN", "UMI_min_sigma", "device", "compile"}
        },
    )

    result = run_rctd(
        spatial_work,
        reference_obj,
        mode=mode,
        config=config,
        batch_size=batch_size,
        sigma_override=sigma_override,
    )

    weights = pad_result_array(result.weights, getattr(result, "pixel_mask", None), adata.n_obs)
    weights_doublet = getattr(result, "weights_doublet", None)
    if weights_doublet is not None:
        weights_doublet = pad_result_array(weights_doublet, getattr(result, "pixel_mask", None), adata.n_obs)

    uns_metadata = create_run_metadata(
        spatial_path,
        reference_path,
        layer_key,
        mode,
        config,
        source_h5_files=[str(spatial_path), str(reference_path)],
    )
    uns_metadata["reference_filtering"] = {
        "min_UMI": config_kwargs.get("reference_min_UMI", 10),
        "max_UMI": config_kwargs.get("reference_max_UMI", 2000),
        "cell_min": config_kwargs.get("cell_min", 25),
        "n_max_cells": config_kwargs.get("n_max_cells", 10000),
        "umi_gene_scope": "shared_spatial_reference_genes",
    }

    harmony_key = output_keys["harmony"]
    if "X_pca_harmony" in adata.obsm:
        adata.obsm[harmony_key] = adata.obsm["X_pca_harmony"].copy()

    adata.obsm[output_keys["weights"]] = weights
    if weights_doublet is not None:
        adata.obsm[output_keys["weights_doublet"]] = weights_doublet

    adata.uns[uns_keys["cell_type_names"]] = list(result.cell_type_names)
    adata.uns[uns_keys["class_df"]] = class_df
    adata.uns[uns_keys["postprocessing"]] = create_postprocessing_payload(result)
    adata.uns[uns_keys["run_metadata"]] = uns_metadata
    adata.uns[uns_keys["simulated_dropout"]] = simulated_dropout
    adata.uns[uns_keys["source_h5_files"]] = [str(spatial_path), str(reference_path)]


def annotate_one_layer(
    spatial_path: Path,
    reference_path: Path,
    layer_key: str,
    cell_type_col: str,
    output_path: Path,
    mode: str = "doublet",
    class_df: Optional[Dict[str, str]] = None,
    simulated_dropout: Optional[Dict[str, Any]] = None,
    force: bool = False,
    batch_size: str | int = "auto",
    sigma_override: Optional[int] = None,
    **config_kwargs,
) -> None:
    print(f"Processing {spatial_path} layer={layer_key}")
    adata = ad.read_h5ad(spatial_path)
    layer_keys = find_matching_layers(adata, layer_key)
    if not layer_keys:
        raise ValueError(
            f"Layer pattern {layer_key!r} did not match any layers in {spatial_path}."
        )

    for resolved_layer in layer_keys:
        print(f"  -> annotating layer {resolved_layer}")
        annotate_single_layer(
            adata,
            spatial_path,
            reference_path,
            resolved_layer,
            cell_type_col,
            output_path,
            mode=mode,
            class_df=class_df,
            simulated_dropout=simulated_dropout,
            force=force,
            batch_size=batch_size,
            sigma_override=sigma_override,
            **config_kwargs,
        )

    if output_path == spatial_path:
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        adata.write_h5ad(tmp_path)
        os.replace(tmp_path, output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path)

    print(f"Saved annotated AnnData to {output_path}")


def parse_class_df(value: Optional[str]) -> Optional[Dict[str, str]]:
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--class-df must be a JSON object or a path to a JSON file"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rctd on one or more spatial AnnData files and store results in .h5ad."
    )
    parser.add_argument("--reference", required=True, help="Reference AnnData .h5ad file")
    parser.add_argument(
        "--spatial",
        required=True,
        nargs="+",
        help="Spatial AnnData .h5ad file(s) to annotate.",
    )
    parser.add_argument(
        "--layer-pattern",
        default=r"counts?$|raw|X",
        help="Regex pattern to select count layers in the spatial AnnData.",
    )
    parser.add_argument(
        "--cell-type-col",
        default="cell_type",
        help="Reference obs column containing cell type labels.",
    )
    parser.add_argument(
        "--mode",
        choices=["doublet", "full", "multi"],
        default="doublet",
        help="RCTD mode to run.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory to write annotated .h5ad files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing RCTD outputs if they exist in the AnnData file.",
    )
    parser.add_argument(
        "--class-df",
        default=None,
        help="JSON string or JSON file path describing class_df for RCTD.",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size passed to rctd.run_rctd; use 'auto' or integer.",
    )
    parser.add_argument(
        "--sigma-override",
        type=int,
        default=None,
        help="Optional sigma_override passed to rctd.run_rctd.",
    )
    parser.add_argument(
        "--simulated-dropout",
        default=None,
        help="Optional JSON string or file describing simulated dropout settings.",
    )
    parser.add_argument(
        "--reference-min-umi",
        type=int,
        default=10,
        help="Inclusive minimum shared-panel UMIs for reference cells (independent of sigma estimation).",
    )
    parser.add_argument("--reference-max-umi", type=int, default=2000,
                        help="Inclusive maximum shared-panel UMIs for reference cells.")
    parser.add_argument("--umi-min-sigma", type=int, default=100,
                        help="Spatial UMI threshold for sigma estimation; does not filter reference cells.")
    parser.add_argument(
        "--cell-min",
        type=int,
        default=25,
        help="Minimum cells per reference cell type.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reference_min_umi < 1 or args.reference_max_umi < args.reference_min_umi:
        parser.error("Reference UMI bounds must satisfy 1 <= minimum <= maximum")
    args.class_df = parse_class_df(args.class_df)
    args.simulated_dropout = parse_class_df(args.simulated_dropout)
    if args.batch_size != "auto":
        args.batch_size = int(args.batch_size)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else None
    for spatial_path in args.spatial:
        spatial_path = Path(spatial_path)
        if not spatial_path.exists():
            raise FileNotFoundError(f"Spatial AnnData file not found: {spatial_path}")
        out_path = spatial_path
        if output_dir is not None:
            out_path = output_dir / spatial_path.name
        annotate_one_layer(
            spatial_path=spatial_path,
            reference_path=Path(args.reference),
            layer_key=args.layer_pattern,
            cell_type_col=args.cell_type_col,
            output_path=out_path,
            mode=args.mode,
            class_df=args.class_df,
            simulated_dropout=args.simulated_dropout,
            force=args.force,
            batch_size=args.batch_size,
            sigma_override=args.sigma_override,
            cell_min=args.cell_min,
            reference_min_UMI=args.reference_min_umi,
            reference_max_UMI=args.reference_max_umi,
            UMI_min_sigma=args.umi_min_sigma,
        )


if __name__ == "__main__":
    main()
