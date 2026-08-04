"""Copy SPLIT/RCTD AnnData entries from merged h5ad files into matching h5ad files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import pandas as pd


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def matches_prefix(key: str, prefixes: Sequence[str]) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def matching_keys(keys: Iterable[str], prefixes: Sequence[str]) -> list[str]:
    return sorted(key for key in keys if matches_prefix(str(key), prefixes))


def check_axes(source: ad.AnnData, target: ad.AnnData, source_path: Path, target_path: Path) -> None:
    if not source.obs_names.equals(target.obs_names):
        raise ValueError(f"obs_names differ between {source_path} and {target_path}")
    if not source.var_names.equals(target.var_names):
        raise ValueError(f"var_names differ between {source_path} and {target_path}")


def copy_obs_columns(source: ad.AnnData, target: ad.AnnData, prefixes: Sequence[str], overwrite: bool) -> list[str]:
    copied: list[str] = []
    for column in matching_keys(source.obs.columns, prefixes):
        if column in target.obs.columns and not overwrite:
            continue
        target.obs[column] = source.obs[column].copy()
        copied.append(column)
    return copied


def copy_layers(source: ad.AnnData, target: ad.AnnData, prefixes: Sequence[str], overwrite: bool) -> list[str]:
    copied: list[str] = []
    for key in matching_keys(source.layers.keys(), prefixes):
        if key in target.layers and not overwrite:
            continue
        target.layers[key] = source.layers[key].copy()
        copied.append(key)
    return copied


def copy_obsm(source: ad.AnnData, target: ad.AnnData, prefixes: Sequence[str], overwrite: bool) -> list[str]:
    copied: list[str] = []
    for key in matching_keys(source.obsm.keys(), prefixes):
        if key in target.obsm and not overwrite:
            continue
        target.obsm[key] = source.obsm[key].copy()
        copied.append(key)
    return copied


def copy_uns(source: ad.AnnData, target: ad.AnnData, prefixes: Sequence[str], overwrite: bool) -> list[str]:
    copied: list[str] = []
    for key in matching_keys(source.uns.keys(), prefixes):
        if key in target.uns and not overwrite:
            continue
        target.uns[key] = source.uns[key]
        copied.append(key)
    return copied


def copy_one(
    source_path: Path,
    target_path: Path,
    *,
    output_path: Path,
    layer_prefixes: Sequence[str],
    obs_prefixes: Sequence[str],
    obsm_prefixes: Sequence[str],
    uns_prefixes: Sequence[str],
    overwrite: bool,
    dry_run: bool,
) -> dict[str, object]:
    if dry_run:
        source = ad.read_h5ad(source_path, backed="r")
        target = ad.read_h5ad(target_path, backed="r")
        try:
            check_axes(source, target, source_path, target_path)

            copied_layers = matching_keys(source.layers.keys(), layer_prefixes)
            copied_obs = matching_keys(source.obs.columns, obs_prefixes)
            copied_obsm = matching_keys(source.obsm.keys(), obsm_prefixes)
            copied_uns = matching_keys(source.uns.keys(), uns_prefixes)

            if not overwrite:
                copied_layers = [key for key in copied_layers if key not in target.layers]
                copied_obs = [key for key in copied_obs if key not in target.obs.columns]
                copied_obsm = [key for key in copied_obsm if key not in target.obsm]
                copied_uns = [key for key in copied_uns if key not in target.uns]
        finally:
            source.file.close()
            target.file.close()
    else:
        source = ad.read_h5ad(source_path)
        target = ad.read_h5ad(target_path)
        check_axes(source, target, source_path, target_path)

        copied_layers = matching_keys(source.layers.keys(), layer_prefixes)
        copied_obs = matching_keys(source.obs.columns, obs_prefixes)
        copied_obsm = matching_keys(source.obsm.keys(), obsm_prefixes)
        copied_uns = matching_keys(source.uns.keys(), uns_prefixes)

        if not overwrite:
            copied_layers = [key for key in copied_layers if key not in target.layers]
            copied_obs = [key for key in copied_obs if key not in target.obs.columns]
            copied_obsm = [key for key in copied_obsm if key not in target.obsm]
            copied_uns = [key for key in copied_uns if key not in target.uns]

        for key in copied_layers:
            target.layers[key] = source.layers[key].copy()
        for key in copied_obs:
            target.obs[key] = source.obs[key].copy()
        for key in copied_obsm:
            target.obsm[key] = source.obsm[key].copy()
        for key in copied_uns:
            target.uns[key] = source.uns[key]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path == target_path:
            tmp_path = target_path.with_suffix(target_path.suffix + f".tmp-{os.getpid()}")
            target.write_h5ad(tmp_path)
            os.replace(tmp_path, target_path)
        else:
            target.write_h5ad(output_path)

    return {
        "source": str(source_path),
        "target": str(target_path),
        "output": str(output_path),
        "n_layers": len(copied_layers),
        "n_obs": len(copied_obs),
        "n_obsm": len(copied_obsm),
        "n_uns": len(copied_uns),
        "layers": ";".join(copied_layers),
        "obs": ";".join(copied_obs),
        "obsm": ";".join(copied_obsm),
        "uns": ";".join(copied_uns),
        "status": "dry_run" if dry_run else "completed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="/work/magroup/skrieger/Syndri/data/xenium_noise_experiment/split_h5ad_layers/merged_h5ad",
        help="Directory containing merged h5ad files with SPLIT/RCTD entries.",
    )
    parser.add_argument(
        "--target-dir",
        default="/work/magroup/skrieger/Syndri/data/xenium_noise_experiment",
        help="Directory containing matching destination h5ad files.",
    )
    parser.add_argument("--glob", default="*.h5ad")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. If unset, destination files are updated in place.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing matching keys in targets.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be copied without writing files.")
    parser.add_argument("--layer-prefixes", default="split_purified_counts")
    parser.add_argument("--obs-prefixes", default="split_,rctd_")
    parser.add_argument("--obsm-prefixes", default="rctd_,X_pca_harmony_")
    parser.add_argument("--uns-prefixes", default="split_,rctd_,source_h5_files,simulated_dropout")
    parser.add_argument(
        "--manifest",
        default=None,
        help="CSV manifest path. Defaults to copy_split_rctd_h5ad_info_manifest.csv in target/output dir.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    manifest_path = Path(args.manifest) if args.manifest else (output_dir or target_dir) / "copy_split_rctd_h5ad_info_manifest.csv"

    layer_prefixes = parse_csv(args.layer_prefixes)
    obs_prefixes = parse_csv(args.obs_prefixes)
    obsm_prefixes = parse_csv(args.obsm_prefixes)
    uns_prefixes = parse_csv(args.uns_prefixes)

    rows: list[dict[str, object]] = []
    source_paths = sorted(source_dir.glob(args.glob))
    if not source_paths:
        raise FileNotFoundError(f"No files matching {args.glob!r} found in {source_dir}")

    for source_path in source_paths:
        target_path = target_dir / source_path.name
        output_path = (output_dir / source_path.name) if output_dir else target_path
        if not target_path.exists():
            rows.append({
                "source": str(source_path),
                "target": str(target_path),
                "output": str(output_path),
                "status": "missing_target",
            })
            print(f"[missing] {source_path.name}")
            continue

        print(f"[copy] {source_path.name}")
        rows.append(
            copy_one(
                source_path,
                target_path,
                output_path=output_path,
                layer_prefixes=layer_prefixes,
                obs_prefixes=obs_prefixes,
                obsm_prefixes=obsm_prefixes,
                uns_prefixes=uns_prefixes,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        )

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
