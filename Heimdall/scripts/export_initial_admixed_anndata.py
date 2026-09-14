#!/usr/bin/env python
"""Export the initial sampled admixture AnnData without training a model."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Heimdall.cell_representations import CellRepresentation


def _get_export_cfg(config):
    return config.get("export_admixed", {})


def _get_layer_name(config) -> str:
    export_cfg = _get_export_cfg(config)
    layer_name = export_cfg.get("layer_name", None)
    if layer_name is not None:
        return str(layer_name)

    configured_layer = config.dataset.preprocess_args.get("admix_layer", None)
    if configured_layer is not None:
        return str(configured_layer)
    return "admixed_counts"


def _get_output_path(config, epoch: int) -> Path:
    output_path = _get_export_cfg(config).get("output_path", None)
    if output_path is not None:
        return Path(output_path)

    return Path(config.work_dir) / f"initial_admixed_epoch_{epoch}.h5ad"


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))

    epoch = int(_get_export_cfg(config).get("epoch", 0))
    layer_name = _get_layer_name(config)
    output_path = _get_output_path(config, epoch)

    cr = CellRepresentation(config)
    if cr.admixture_epoch_store is None:
        raise ValueError(
            "No AdmixtureEpochStore was created. Ensure dataset.preprocess_args.admixture_args "
            "exists and dataset.preprocess_args.apply_admixture is true.",
        )

    if "neighbor_dict" not in cr.adata.uns:
        raise KeyError("Expected 'neighbor_dict' in adata.uns after preprocessing.")
    if "neighbor_celltype_dict" not in cr.adata.uns:
        raise KeyError("Expected 'neighbor_celltype_dict' in adata.uns after preprocessing.")

    admixed_counts = cr.admixture_epoch_store.get(
        epoch,
        obs_names=cr.adata.obs_names,
        var_names=cr.adata.var_names,
    )
    cr.adata.layers[layer_name] = admixed_counts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cr.adata.write_h5ad(output_path)

    print(f"> Wrote initial sampled admixture AnnData: {output_path}")
    print(f"  layer: {layer_name}")
    print(f"  epoch: {epoch}")
    print(f"  shape: {cr.adata.shape}")
    print("  uns keys: neighbor_dict, neighbor_celltype_dict")


if __name__ == "__main__":
    main()



# python scripts/export_initial_admixed_anndata.py \
#   +experiments=admixture_prediction \
#   ++export_admixed.output_path=/work/magroup/skrieger/Syndri/data/trainsets/parameter_tests/xenium_moreneighbors_test.h5ad \
#   ++export_admixed.layer_name=admixed_counts \
#   ++export_admixed.epoch=0