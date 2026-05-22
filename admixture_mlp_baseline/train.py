"""Train a simple MLP baseline for Heimdall admixture prediction."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset, default_collate
from torchmetrics.functional.classification import binary_auroc, binary_average_precision
from tqdm.auto import tqdm
from transformers import get_scheduler

THIS_DIR = Path(__file__).resolve().parent
SYNDRI_ROOT = THIS_DIR.parent
HEIMDALL_ROOT = SYNDRI_ROOT / "Heimdall"
sys.path.insert(0, str(HEIMDALL_ROOT))

import Heimdall.losses  # noqa: E402
from Heimdall.cell_representations import CellRepresentation  # noqa: E402
from Heimdall.utils import get_dtype  # noqa: E402

from model import AdmixtureMLP, batch_to_mlp_inputs, gather_gene_ordered_logits


class QueryCellTypeDataset(Dataset):
    """Wrap a Heimdall dataset and add the query cell-type token."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.full_dataset = self._unwrap_dataset(dataset)
        adata = self.full_dataset.data.adata
        self.adata = adata
        self.cell_type_dict = dict(adata.uns["cell_type_dict"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = dict(self.dataset[idx])
        full_idx = self._full_index(self.dataset, idx)
        cell_type = self.adata.obs.iloc[full_idx]["cell_type"]
        sample["query_cell_type_token"] = self.cell_type_dict[cell_type]
        return sample

    def _unwrap_dataset(self, dataset):
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        return dataset

    def _full_index(self, dataset, idx):
        if isinstance(dataset, Subset):
            return self._full_index(dataset.dataset, int(dataset.indices[idx]))
        return int(idx)


class MeanCellwiseBinaryRankingMetric:
    """Average a binary ranking metric across cells, skipping undefined rows."""

    def __init__(self, metric_fn):
        self.metric_fn = metric_fn
        self.metric_sum = 0.0
        self.metric_count = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.detach().cpu()
        labels = labels.detach().cpu()
        valid = torch.isfinite(labels)
        for row_logits, row_labels, row_valid in zip(logits, labels, valid):
            row_logits = row_logits[row_valid]
            row_labels = row_labels[row_valid].to(torch.int)
            if row_labels.numel() == 0 or row_labels.unique().numel() < 2:
                continue
            score = self.metric_fn(row_logits, row_labels)
            if torch.isfinite(score):
                self.metric_sum += float(score)
                self.metric_count += 1

    def compute(self):
        if self.metric_count == 0:
            return float("nan")
        return self.metric_sum / self.metric_count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="admixture_prediction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dims", default="1024,512")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_known_args()


def compose_config(experiment: str, overrides: list[str]):
    os.chdir(HEIMDALL_ROOT)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(HEIMDALL_ROOT / "config")):
        return hydra.compose(config_name="config", overrides=[f"+experiments={experiment}", *overrides])


def make_dataloaders(cr: CellRepresentation, cfg, num_workers: int):
    return {
        split: DataLoader(
            QueryCellTypeDataset(dataset),
            batch_size=cfg.tasks.args.batchsize,
            shuffle=cfg.tasks.args.shuffle if split == "train" and len(dataset) > 0 else False,
            collate_fn=default_collate,
            num_workers=num_workers,
        )
        for split, dataset in cr.datasets.items()
        if split in {"train", "val", "test"}
    }


def build_token_to_cell_index(cell_type_dict: dict) -> torch.Tensor:
    max_token = max(abs(int(token)) for token in cell_type_dict.values())
    token_to_cell_index = torch.full((max_token + 1,), -1, dtype=torch.long)
    for cell_index, token in enumerate(cell_type_dict.values()):
        token_to_cell_index[abs(int(token))] = cell_index
    return token_to_cell_index


def make_loss(cfg):
    loss_args = OmegaConf.to_container(cfg.loss.get("args", {}), resolve=True)
    if cfg.loss.name == "MaskedBCEWithLogitsLoss":
        return Heimdall.losses.MaskedBCEWithLogitsLoss(**loss_args)
    if cfg.loss.name == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss(**loss_args)
    raise ValueError(f"Unsupported MLP baseline loss: {cfg.loss.name}")


def set_dataset_epoch(cr: CellRepresentation, epoch: int):
    full_dataset = cr.datasets.get("full")
    if hasattr(full_dataset, "set_epoch"):
        full_dataset.set_epoch(epoch)


def evaluate(model, dataloader, loss_fn, batch_kwargs, device, fastdev=False):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    auroc = MeanCellwiseBinaryRankingMetric(binary_auroc)
    auprc = MeanCellwiseBinaryRankingMetric(binary_average_precision)
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval", leave=False):
            features, gene_order, labels = batch_to_mlp_inputs(batch, device=device, **batch_kwargs)
            logits, ordered_labels = gather_gene_ordered_logits(model(features.to(model_dtype)), gene_order, labels)
            loss = loss_fn(logits, ordered_labels.clone())
            total_loss += float(loss)
            total_batches += 1
            auroc.update(logits, ordered_labels)
            auprc.update(logits, ordered_labels)
            if fastdev:
                break
    return {
        "loss": total_loss / max(total_batches, 1),
        "AUROC": auroc.compute(),
        "AUPRC": auprc.compute(),
    }


def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, batch_kwargs, device, grad_norm_clip, fastdev=False):
    model.train()
    total_loss = 0.0
    total_batches = 0
    auroc = MeanCellwiseBinaryRankingMetric(binary_auroc)
    auprc = MeanCellwiseBinaryRankingMetric(binary_average_precision)
    model_dtype = next(model.parameters()).dtype
    for batch in tqdm(dataloader, desc="train", leave=False):
        features, gene_order, labels = batch_to_mlp_inputs(batch, device=device, **batch_kwargs)
        logits, ordered_labels = gather_gene_ordered_logits(model(features.to(model_dtype)), gene_order, labels)
        loss = loss_fn(logits, ordered_labels.clone())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.detach())
        total_batches += 1
        auroc.update(logits, ordered_labels)
        auprc.update(logits, ordered_labels)
        if fastdev:
            break
    return {
        "loss": total_loss / max(total_batches, 1),
        "AUROC": auroc.compute(),
        "AUPRC": auprc.compute(),
    }


def prefixed_metrics(prefix: str, metrics: dict) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def finite_metric_payload(payload: dict) -> dict:
    finite_payload = {}
    skipped = []
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            if math.isfinite(value):
                finite_payload[key] = float(value)
            else:
                skipped.append(key)
        else:
            finite_payload[key] = value
    if skipped:
        print(f"Skipping non-finite W&B metrics: {', '.join(skipped)}")
    return finite_payload


def log_epoch_metrics(epoch: int, wandb_run=None, **metric_groups):
    payload = {"epoch": epoch}
    for prefix, metrics in metric_groups.items():
        payload.update(prefixed_metrics(prefix, metrics))
    print(payload)
    if wandb_run is not None:
        wandb_run.log(finite_metric_payload(payload), step=epoch)


def finite_or_neginf(value):
    return value if value is not None and math.isfinite(value) else float("-inf")


def main():
    args, overrides = parse_args()
    cfg = compose_config(args.experiment, overrides)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(args.device)
    float_dtype = get_dtype(cfg.float_dtype)

    cr = CellRepresentation(cfg)
    dataloaders = make_dataloaders(cr, cfg, args.num_workers)

    n_genes = cr.adata.n_vars
    n_cell_types = len(cr.adata.uns["cell_type_dict"])
    n_neighbor_tokens = cfg.fc.args.max_input_length - cfg.tasks.args.context_length
    hidden_dims = tuple(int(dim) for dim in args.hidden_dims.split(",") if dim)

    model = AdmixtureMLP(
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        n_neighbor_tokens=n_neighbor_tokens,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(device=device, dtype=float_dtype)
    optimizer = getattr(torch.optim, cfg.optimizer.name)(model.parameters(), **OmegaConf.to_container(cfg.optimizer.args))
    total_steps = max(len(dataloaders["train"]) * cfg.tasks.args.epochs, 1)
    warmup_steps = int(cfg.scheduler.warmup_ratio * total_steps)
    scheduler = get_scheduler(
        name=cfg.scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = make_loss(cfg).to(device)

    batch_kwargs = {
        "n_genes": n_genes,
        "n_cell_types": n_cell_types,
        "n_neighbor_tokens": n_neighbor_tokens,
        "token_to_cell_index": build_token_to_cell_index(cr.adata.uns["cell_type_dict"]),
    }

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project or cfg.project_name,
            name=f"{cfg.run_name}_mlp",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        wandb_mode = getattr(wandb_run.settings, "mode", "unknown")
        print(
            "W&B logging enabled: "
            f"project={wandb_run.project}, name={wandb_run.name}, mode={wandb_mode}, url={wandb_run.url}"
        )
    else:
        print("W&B logging disabled by --no-wandb")

    best_valid_auprc = float("-inf")
    work_dir = Path(cfg.work_dir) / "mlp_baseline"
    work_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.tasks.args.epochs):
        set_dataset_epoch(cr, epoch)
        train_metrics = train_epoch(
            model,
            dataloaders["train"],
            loss_fn,
            optimizer,
            scheduler,
            batch_kwargs,
            device,
            cfg.trainer.grad_norm_clip,
            fastdev=cfg.trainer.fastdev,
        )
        valid_metrics = evaluate(model, dataloaders["val"], loss_fn, batch_kwargs, device, fastdev=cfg.trainer.fastdev)
        test_metrics = evaluate(model, dataloaders["test"], loss_fn, batch_kwargs, device, fastdev=cfg.trainer.fastdev)

        log_epoch_metrics(
            epoch,
            wandb_run,
            train=train_metrics,
            valid=valid_metrics,
            test=test_metrics,
        )

        if args.save_checkpoint and finite_or_neginf(valid_metrics["AUPRC"]) > best_valid_auprc:
            best_valid_auprc = finite_or_neginf(valid_metrics["AUPRC"])
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                work_dir / "best_model.pt",
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
