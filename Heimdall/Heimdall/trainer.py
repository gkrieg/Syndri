"""Heimdall trainer."""

import importlib.util
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import copy
import scanpy as sc
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from anndata import AnnData
from omegaconf import OmegaConf
from torchmetrics import Metric
from torchmetrics.classification import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    MatthewsCorrCoef,
    Precision,
    Recall,
)
from torchmetrics.functional.classification import binary_auroc, binary_average_precision
from torchmetrics.regression import MeanSquaredError, R2Score
from tqdm import tqdm
from transformers import get_scheduler
from torch.utils.data import DataLoader, Subset
from scipy.sparse import csr_matrix,issparse
from sklearn.metrics import silhouette_score

import Heimdall.datasets
import Heimdall.losses
from Heimdall.admixture_sampling import sample_admixture_from_neighbor_dict
from Heimdall.utils import heimdall_collate_fn

import wandb
torch.autograd.set_detect_anomaly(True)


class MeanCellwiseBinaryRankingMetric(Metric):
    """Average a binary ranking metric across cells, skipping undefined rows."""

    def __init__(self, metric_fn):
        super().__init__()
        self.metric_fn = metric_fn
        self.add_state("metric_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("metric_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits, labels):
        logits = logits.detach()
        labels = labels.detach()
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
            labels = labels.unsqueeze(0)
        elif logits.ndim > 2:
            logits = logits.flatten(start_dim=1)
            labels = labels.flatten(start_dim=1)

        labels = labels.to(logits.device)
        valid = torch.isfinite(labels)
        for row_logits, row_labels, row_valid in zip(logits, labels, valid):
            row_logits = row_logits[row_valid]
            row_labels = row_labels[row_valid].to(torch.int)
            if row_labels.numel() == 0 or row_labels.unique().numel() < 2:
                continue
            score = self.metric_fn(row_logits, row_labels)
            if torch.isfinite(score):
                self.metric_sum += score
                self.metric_count += 1

    def compute(self):
        if self.metric_count == 0:
            return torch.tensor(float("nan"), device=self.metric_sum.device)
        return self.metric_sum / self.metric_count

class HeimdallTrainer:
    def __init__(
        self,
        cfg,
        model,
        data,
        run_wandb=False,
        custom_loss_func=None,
        custom_metrics=None,
    ):
        self.cfg = cfg
        self.model = model
        self.data = data

        # cell type label
        # label_key = self.cfg.tasks.args.label_col_name
        # if not pd.api.types.is_categorical_dtype(self.data.adata.obs[label_key]):
        #    self.data.adata.obs[label_key] = self.data.adata.obs[label_key].astype("category")

        # class_names will now align with integer labels returned by .codes
        # self.class_names = self.data.adata.obs[label_key].cat.categories.tolist()

        # assert len(self.class_names) == self.num_labels, "Mismatch between classes and label indices"

        args = self.cfg.tasks.args

        # TODO: since we use the label_key in the CellRepresentation setup, we shouldn't need it here.
        # It should all be accessible in the data.labels... Delete the block below if possible...?

        # Unified label key handling: support .obs or .obsm
        label_key = getattr(args, "label_col_name", None)
        label_obsm_key = getattr(args, "label_obsm_name", None)
        print(label_obsm_key)

        if label_key is not None:
            # Single-label classification using .obs[label_key]
            if not pd.api.types.is_categorical_dtype(self.data.adata.obs[label_key]):
                self.data.adata.obs[label_key] = self.data.adata.obs[label_key].astype("category")
            self.class_names = self.data.adata.obs[label_key].cat.categories.tolist()
            self.num_labels = len(self.class_names)

        elif label_obsm_key is not None:
            if args.task_type in ["binary", "admixture_binary"]:
                self.class_names = ["0", "1"]
                self.num_labels = data.num_tasks
            else:
                # Multi-label classification using .obsm[label_obsm_key]
                label_obsm = self.data.adata.obsm[label_obsm_key]
                if hasattr(label_obsm, "columns"):
                    self.class_names = label_obsm.columns.tolist()
                else:
                    self.class_names = [str(i) for i in range(label_obsm.shape[1])]
                self.num_labels = len(self.class_names)

        else:
            # Auto infering
            self.class_names = data.adata.uns["task_order"]  # NOTE: first entry might be NULL
            self.num_labels = data.num_tasks

        # else:
        #    raise ValueError("Must specify either `label_col_name` or `label_obsm_name` in the config.")

        self.run_wandb = run_wandb
        self.process = psutil.Process()
        self.custom_loss_func = custom_loss_func
        self.custom_metrics = custom_metrics or {}

        accelerator_log_kwargs = {}
        if run_wandb:
            accelerator_log_kwargs["log_with"] = "wandb"
            accelerator_log_kwargs["project_dir"] = cfg.work_dir
        set_seed(cfg.seed)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.trainer.accumulate_grad_batches,
            step_scheduler_with_optimizer=False,
            **accelerator_log_kwargs,
        )

        if hasattr(model.encoder, "use_flash_attn") and model.encoder.use_flash_attn:
            assert self.accelerator.mixed_precision == "bf16", "If using Flash Attention, mixed precision must be bf16"

        self.optimizer = self._initialize_optimizer()
        self.loss_fn = self._get_loss_function()

        self.accelerator.wait_for_everyone()
        self.print_r0(f"> Using Device: {self.accelerator.device}")
        self.print_r0(f"> Number of Devices: {self.accelerator.num_processes}")

        self._initialize_wandb()
        self._initialize_lr_scheduler()
        self.step = 0

        # after accelerator.prepare(...)
        model_unwrapped = self.accelerator.unwrap_model(self.model)

        with torch.no_grad():
            for name, p in model_unwrapped.named_parameters():
                if torch.isnan(p).any():
                    print("PARAM INIT NaN:", name)


        self.dataloader_train_ref = copy.copy(self.dataloader_train)
        self.dataloader_val_ref = copy.copy(self.dataloader_val)
        self.dataloader_test_ref = copy.copy(self.dataloader_test)
        (
            self.model,
            self.optimizer,
            self.dataloader_train,
            self.dataloader_val,
            self.dataloader_test,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.dataloader_train,
            self.dataloader_val,
            self.dataloader_test,
            self.lr_scheduler,
        )

        self.external_eval = self._initialize_external_eval()

        if self.accelerator.is_main_process:
            print("> Finished Wrapping the model, optimizer, and dataloaders in accelerate")
            print("> run HeimdallTrainer.train() to begin training")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        for split in ["train", "val", "test"]:
            setattr(self, f"dataloader_{split}", data.dataloaders[split])

    def print_r0(self, payload):
        if self.accelerator.is_main_process:
            print(f"{payload}")

    def _initialize_optimizer(self):
        optimizer_class = getattr(torch.optim, self.cfg.optimizer.name)
        return optimizer_class(self.model.parameters(), **OmegaConf.to_container(self.cfg.optimizer.args))

    def _get_loss_function(self):
        loss_args = OmegaConf.to_container(self.cfg.loss.get("args", {}), resolve=True)
        if self.custom_loss_func:
            self.print_r0(f"> Using Custom Loss Function: {self.custom_loss_func.__name__}")
            return self.custom_loss_func
        elif self.cfg.loss.name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss(**loss_args)
        elif self.cfg.loss.name == "BCEWithLogitsLoss":
            return torch.nn.BCEWithLogitsLoss(**loss_args)
        elif self.cfg.loss.name == "MaskedBCEWithLogitsLoss":
            return Heimdall.losses.MaskedBCEWithLogitsLoss(**loss_args)
        elif self.cfg.loss.name == "CrossEntropyFocalLoss":
            return Heimdall.losses.CrossEntropyFocalLoss(**loss_args)
        elif self.cfg.loss.name == "MSELoss":
            return nn.MSELoss(**loss_args)
        else:
            raise ValueError(f"Unsupported loss function: {self.cfg.loss.name}")

    def _initialize_wandb(self):
        if self.run_wandb and self.accelerator.is_main_process:
            print("==> Starting a new WANDB run")
            new_tags = (self.cfg.dataset.dataset_name, self.cfg.fg.type, self.cfg.fe.type, self.cfg.fc.type)
            wandb_config = {
                "wandb": {
                    "tags": new_tags,
                    "name": self.cfg.run_name,
                    "entity": self.cfg.entity,
                },
            }
            self.accelerator.init_trackers(
                project_name=self.cfg.project_name,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                init_kwargs=wandb_config,
            )
            print("==> Initialized Run")

    def _initialize_lr_scheduler(self):
        dataset_config = self.cfg.tasks.args
        global_batch_size = dataset_config.batchsize
        train_samples = len(self.dataloader_train.dataset)
        total_steps = train_samples // global_batch_size * dataset_config.epochs
        inference_mode = not self.cfg.dataset.preprocess_args.get("apply_admixture", True)
        if train_samples == 0:
            if inference_mode:
                self.print_r0("> No train samples found; initializing scheduler with one placeholder step for inference.")
                total_steps = 1
            else:
                raise ValueError("No train samples found in the split column.")
        warmup_ratio = self.cfg.scheduler.warmup_ratio
        warmup_step = int(warmup_ratio * total_steps)

        self.lr_scheduler = get_scheduler(
            name=self.cfg.scheduler.name,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=total_steps,
        )
        self.print_r0("!!! Remember that config batchsize here is GLOBAL Batchsize !!!")
        self.print_r0(f"> global batchsize: {global_batch_size}")
        self.print_r0(f"> total_samples: {train_samples}")
        self.print_r0(f"> Warm Up Steps: {warmup_step}")
        self.print_r0(f"> Total Steps: {total_steps}")
        self.print_r0(f"> per_device_batch_size: {global_batch_size // self.accelerator.num_processes}")

    def _initialize_metrics(self):
        """Initializing the metrics based on the hydra config."""
        metrics = {}
        task_type = self.cfg.tasks.args.task_type

        # First, add custom metrics if provided, TODO this is not implemented yet
        assert self.custom_metrics == {}, "Custom Metrics Not Implemented Yet"
        metrics.update(self.custom_metrics)

        # Then, add built-in metrics if not overridden by custom metrics
        if task_type == "multiclass":
            num_classes = self.num_labels
            for metric_name in self.cfg.tasks.args.metrics:
                if metric_name not in metrics:
                    if metric_name == "Accuracy":
                        metrics[metric_name] = Accuracy(task="multiclass", num_classes=num_classes)
                    elif metric_name == "Precision":
                        metrics[metric_name] = Precision(task="multiclass", num_classes=num_classes, average="macro")
                    elif metric_name == "Recall":
                        metrics[metric_name] = Recall(task="multiclass", num_classes=num_classes, average="macro")
                    elif metric_name == "F1Score":
                        metrics[metric_name] = F1Score(task="multiclass", num_classes=num_classes, average="macro")
                    elif metric_name == "MatthewsCorrCoef":
                        metrics[metric_name] = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
                    elif metric_name == "ConfusionMatrix":
                        metrics[metric_name] = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        elif task_type == "regression":
            for metric_name in self.cfg.tasks.args.metrics:
                if metric_name not in metrics:
                    if metric_name == "R2Score":
                        metrics[metric_name] = R2Score()
                    elif metric_name == "MSE":
                        metrics[metric_name] = MeanSquaredError()
        elif task_type == "binary" or task_type == "admixture_binary":
            # num_labels = self.num_labels
            num_labels = 2
            metrics["AUROC"] = MeanCellwiseBinaryRankingMetric(binary_auroc)
            metrics["AUPRC"] = MeanCellwiseBinaryRankingMetric(binary_average_precision)
            for metric_name in self.cfg.tasks.args.metrics:
                if metric_name not in metrics:
                    if metric_name == "Accuracy":
                        metrics[metric_name] = Accuracy(task="binary", num_labels=num_labels)
                    elif metric_name == "Precision":
                        metrics[metric_name] = Precision(task="binary", num_labels=num_labels, average="macro")
                    elif metric_name == "Recall":
                        metrics[metric_name] = Recall(task="binary", num_labels=num_labels, average="macro")
                    elif metric_name == "F1Score":
                        metrics[metric_name] = F1Score(task="binary", num_labels=num_labels, average="macro")
                    elif metric_name == "MatthewsCorrCoef":
                        metrics[metric_name] = MatthewsCorrCoef(task="binary", num_labels=num_labels)
                    elif metric_name == "AUROC":
                        metrics[metric_name] = MeanCellwiseBinaryRankingMetric(binary_auroc)
                    elif metric_name == "AUPRC":
                        metrics[metric_name] = MeanCellwiseBinaryRankingMetric(binary_average_precision)

        return {k: v.to(self.accelerator.device) if hasattr(v, "to") else v for k, v in metrics.items()}

    def _prepare_logits_and_labels_for_metrics(self, logits, labels):
        task_type = self.cfg.tasks.args.task_type
        if task_type in ["multiclass"]:
            labels = labels.to(torch.int)
        elif task_type in ["binary", "admixture_binary"]:
            flattened_labels = labels.flatten()
            flattened_logits = logits.flatten()
            mask = ~torch.isnan(flattened_labels)

            labels = flattened_labels[mask].to(torch.int)
            logits = flattened_logits[mask]

        return logits, labels

    def _prepare_cellwise_logits_and_labels_for_metrics(self, logits, labels, masks=None):
        task_type = self.cfg.tasks.args.task_type
        if task_type not in ["binary", "admixture_binary"]:
            return logits, labels
        if masks is None:
            return logits, labels

        if masks.shape == labels.shape:
            masked_labels = labels.float().masked_fill(~masks.bool(), float("nan"))
            return logits, masked_labels

        return logits[masks], labels[masks]

    def _get_adata_from_dataloader(self, dl):
        ds = dl.dataset
        while isinstance(ds, Subset):
            ds = ds.dataset
        return ds.data.adata

    def _set_dataset_epoch(self, epoch: int):
        full_dataset = self.data.datasets.get("full")
        if isinstance(full_dataset, Heimdall.datasets.AdmixtureInstanceDataset):
            full_dataset.set_epoch(epoch)

    def _initialize_external_eval(self):
        eval_cfg = self.cfg.tasks.args.get("external_eval", {})
        if not eval_cfg or not eval_cfg.get("enabled", False):
            return None

        data_path = eval_cfg.get("data_path", None)
        if not data_path:
            raise ValueError("tasks.args.external_eval.data_path is required when external_eval.enabled=true.")

        every_n_epochs = int(eval_cfg.get("every_n_epochs", 1))
        if every_n_epochs < 1:
            raise ValueError("tasks.args.external_eval.every_n_epochs must be >= 1.")

        from Heimdall.cell_representations import CellRepresentation

        eval_config = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        data_path = str(data_path)
        dataset_name = eval_cfg.get("dataset_name", None)
        if dataset_name is None:
            dataset_name = f"{self.cfg.dataset.dataset_name}_external_eval_{Path(data_path).stem}"

        eval_config.dataset.dataset_name = str(dataset_name)
        eval_config.dataset.preprocess_args.data_path = data_path
        eval_config.tasks.args.shuffle = False
        eval_config.work_dir = str(Path(self.cfg.work_dir) / "external_eval" / str(dataset_name))

        overrides = eval_cfg.get("overrides", None)
        if overrides:
            eval_config = OmegaConf.merge(eval_config, overrides)

        self.print_r0(
            "> Initializing external evaluation dataset "
            f"{eval_config.dataset.dataset_name!r} from {data_path}",
        )
        external_data = CellRepresentation(eval_config)
        if eval_cfg.get("require_matching_var_names", True):
            self._validate_external_eval_gene_alignment(external_data)

        split = eval_cfg.get("split", "full")
        if split not in external_data.dataloaders:
            valid_splits = ", ".join(sorted(external_data.dataloaders))
            raise ValueError(
                f"External eval split {split!r} is not available. "
                f"Available splits: {valid_splits}",
            )

        raw_external_dataloader = external_data.dataloaders[split]
        external_dataloader = self.accelerator.prepare(raw_external_dataloader)
        fixed_epoch = eval_cfg.get("fixed_epoch", None)
        if fixed_epoch is not None:
            fixed_epoch = int(fixed_epoch)

        return {
            "data": external_data,
            "dataloader": external_dataloader,
            "raw_dataloader": raw_external_dataloader,
            "row_indices": self._dataset_row_indices(external_data.datasets[split]),
            "every_n_epochs": every_n_epochs,
            "fixed_epoch": fixed_epoch,
            "metrics_eval": eval_cfg.get("metrics_eval", {}),
            "prefix": str(eval_cfg.get("prefix", "external_eval")),
            "run_model_metrics": bool(eval_cfg.get("run_model_metrics", True)),
            "split": str(split),
            "source_file": data_path,
        }

    def _validate_external_eval_gene_alignment(self, external_data):
        if not external_data.adata.var_names.equals(self.data.adata.var_names):
            raise ValueError(
                "External eval AnnData var_names do not match the training AnnData var_names. "
                "Reorder/subset the external h5ad to the training genes before evaluation, "
                "or set tasks.args.external_eval.require_matching_var_names=false if you have "
                "verified that the token mapping is compatible.",
            )

        train_tokens = self.data.adata.var.get("identity_embedding_index", None)
        external_tokens = external_data.adata.var.get("identity_embedding_index", None)
        if train_tokens is None or external_tokens is None:
            return

        if not np.array_equal(train_tokens.to_numpy(), external_tokens.to_numpy()):
            raise ValueError(
                "External eval identity_embedding_index does not match the training token order. "
                "This model cannot safely evaluate the external dataset with the current tokenization.",
            )

    def _set_external_eval_epoch(self, epoch: int):
        if self.external_eval is None:
            return

        eval_epoch = self.external_eval["fixed_epoch"]
        if eval_epoch is None:
            eval_epoch = epoch

        full_dataset = self.external_eval["data"].datasets.get("full")
        if isinstance(full_dataset, Heimdall.datasets.AdmixtureInstanceDataset):
            full_dataset.set_epoch(eval_epoch)

    def evaluate_external_dataset(self, epoch: int):
        if self.external_eval is None:
            return None

        completed_epochs = epoch + 1
        every_n_epochs = self.external_eval["every_n_epochs"]
        if completed_epochs % every_n_epochs != 0:
            return None

        eval_epoch = self.external_eval["fixed_epoch"]
        if eval_epoch is None:
            eval_epoch = epoch
        self._set_external_eval_epoch(epoch)

        prefix = self.external_eval["prefix"]
        self.print_r0(
            f"> Running external evaluation {prefix!r} after {completed_epochs} completed epoch(s) "
            f"using eval epoch {eval_epoch}.",
        )
        result = None
        if self.external_eval["run_model_metrics"]:
            result = self.validate_model(
                self.external_eval["dataloader"],
                dataset_type=prefix,
                extra_log={
                    f"{prefix}_completed_epochs": completed_epochs,
                    f"{prefix}_eval_epoch": eval_epoch,
                },
            )
        self.evaluate_external_metrics_eval(epoch, completed_epochs, eval_epoch)
        return result

    def _load_metrics_eval_module(self):
        eval_cfg = self.external_eval["metrics_eval"]
        script_path = eval_cfg.get("script_path", None)
        if script_path is None:
            script_path = Path(__file__).resolve().parents[2] / "eval" / "metrics_eval.py"
        script_path = Path(script_path).expanduser().resolve()
        spec = importlib.util.spec_from_file_location("syndri_metrics_eval", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load metrics_eval.py from {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _set_namespace_values(namespace, values):
        if not values:
            return namespace
        for key, value in values.items():
            setattr(namespace, str(key).replace("-", "_"), value)
        return namespace

    @staticmethod
    def _safe_metric_key(value):
        value = str(value)
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
        return value or "all"

    def _metrics_eval_scalar_log(self, frames: dict[str, pd.DataFrame], prefix: str) -> dict:
        log = {}
        for frame_name, frame in frames.items():
            if frame is None or frame.empty:
                continue

            metric_rows = frame
            if "error" in metric_rows.columns:
                metric_rows = metric_rows[metric_rows["error"].fillna("").astype(str) == ""]
            if metric_rows.empty:
                continue

            numeric_cols = metric_rows.select_dtypes(include=[np.number]).columns
            numeric_cols = [
                col
                for col in numeric_cols
                if col not in {"dropout_fraction"}
            ]
            if not numeric_cols:
                continue

            group_col = "case" if "case" in metric_rows.columns else "condition" if "condition" in metric_rows.columns else None
            if group_col is None:
                means = metric_rows[numeric_cols].mean(numeric_only=True)
                for col, value in means.items():
                    if np.isfinite(value):
                        log[f"{prefix}/metrics_eval/{frame_name}/{col}_mean"] = float(value)
                continue

            for group_value, group in metric_rows.groupby(group_col, dropna=False):
                group_key = self._safe_metric_key(group_value)
                means = group[numeric_cols].mean(numeric_only=True)
                for col, value in means.items():
                    if np.isfinite(value):
                        log[f"{prefix}/metrics_eval/{frame_name}/{group_key}/{col}_mean"] = float(value)
        return log

    def _write_metrics_eval_outputs(
        self,
        metrics_module,
        args,
        frames: dict[str, pd.DataFrame],
        output_dir: Path,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        output_map = {
            "cell_type": args.cell_type_output,
            "tcell": args.tcell_output,
            "tcell_effect": args.tcell_effect_output,
            "tcell_lr_gene": args.tcell_lr_gene_output,
            "cell_type_profile": args.cell_type_profile_output,
            "tcell_cosine": args.tcell_cosine_output,
        }
        for name, file_name in output_map.items():
            frame = frames.get(name)
            if frame is None or frame.empty:
                continue
            metrics_module.write_csv(frame, output_dir / file_name)

    def evaluate_external_metrics_eval(self, epoch: int, completed_epochs: int, eval_epoch: int):
        metrics_cfg = self.external_eval["metrics_eval"]
        if not metrics_cfg or not metrics_cfg.get("enabled", False):
            return None

        prob_batches = self.get_predictions(
            self.external_eval["raw_dataloader"],
            data=self.external_eval["data"],
        )
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return None

        probs = np.vstack(prob_batches)
        row_indices = self.external_eval["row_indices"]
        if probs.shape[0] != len(row_indices):
            raise ValueError(
                "External metrics_eval prediction rows do not match selected external rows "
                f"({probs.shape[0]} vs {len(row_indices)}).",
            )

        metrics_module = self._load_metrics_eval_module()
        args = metrics_module.build_parser().parse_args([])
        metrics_args = OmegaConf.to_container(metrics_cfg.get("args", {}), resolve=True)
        self._set_namespace_values(args, metrics_args)

        source_file = self.external_eval["source_file"]
        source_path = Path(source_file)
        tissue = metrics_cfg.get("tissue", None) or metrics_module.tissue_name_from_path(
            source_path,
            args.tissue_name_regex,
        )
        sample_replicate = metrics_cfg.get("sample_replicate", None) or metrics_module.sample_replicate_from_path(
            source_path,
            args.sample_replicate_regex,
        )
        dropout_label, dropout_fraction = metrics_module.dropout_from_path(source_path)

        probability_layer = str(metrics_cfg.get("probability_layer", "_training_syndri_probs"))
        metric_adata = self.external_eval["data"].adata[row_indices].copy()
        metric_adata.layers[probability_layer] = csr_matrix(probs)
        metric_adata.obs[args.source_file_key] = source_file
        metric_adata.obs["_metrics_tissue"] = tissue
        metric_adata.obs["_metrics_sample_replicate"] = sample_replicate
        metric_adata.obs["_metrics_dropout_label"] = dropout_label
        metric_adata.obs["_metrics_dropout_fraction"] = dropout_fraction
        metric_adata.X = metrics_module.sanitize_numeric_matrix(metric_adata.X)
        metric_adata = metrics_module.sample_adata(metric_adata, args)

        args.admix_probs_layer = probability_layer
        reference_adata = None
        reference_profile_cache: dict = {}
        if not args.skip_tcell_contamination and not args.skip_tcell_cosine_similarity and args.reference_data:
            reference_adata = metrics_module.ad.read_h5ad(Path(args.reference_data).expanduser().resolve(), backed="r")

        frames = {
            "cell_type": pd.DataFrame(),
            "tcell": pd.DataFrame(),
            "tcell_effect": pd.DataFrame(),
            "tcell_lr_gene": pd.DataFrame(),
            "cell_type_profile": pd.DataFrame(),
            "tcell_cosine": pd.DataFrame(),
        }
        if not args.skip_cell_type_separation:
            rows = metrics_module.run_cell_type_separation(metric_adata, tissue, source_file, args)
            frames["cell_type"] = pd.DataFrame(rows)

        if not args.skip_tcell_contamination:
            tcell_df, coefficient_df, profile_df, cosine_df = metrics_module.run_tcell_contamination(
                metric_adata.copy(),
                tissue,
                source_file,
                args,
                reference_adata=reference_adata,
                reference_profile_cache=reference_profile_cache,
            )
            frames["tcell"] = tcell_df
            frames["tcell_effect"] = metrics_module.summarize_tcell_effects(tcell_df, args)
            frames["tcell_lr_gene"] = coefficient_df
            frames["cell_type_profile"] = profile_df
            frames["tcell_cosine"] = cosine_df

        if reference_adata is not None and getattr(reference_adata, "file", None) is not None:
            reference_adata.file.close()

        prefix = self.external_eval["prefix"]
        scalar_log = self._metrics_eval_scalar_log(frames, prefix)
        scalar_log.update(
            {
                f"{prefix}/metrics_eval/completed_epochs": completed_epochs,
                f"{prefix}/metrics_eval/eval_epoch": eval_epoch,
                "epoch": epoch,
            }
        )

        if self.run_wandb:
            self.accelerator.log(scalar_log, step=self.step)
        else:
            print(scalar_log)

        if metrics_cfg.get("write_outputs", True):
            output_dir = metrics_cfg.get("output_dir", None)
            if output_dir is None:
                output_dir = (
                    Path(self.cfg.work_dir)
                    / "external_metrics_eval"
                    / prefix
                    / f"epoch_{completed_epochs:04d}"
                )
            self._write_metrics_eval_outputs(metrics_module, args, frames, Path(output_dir))

        return scalar_log

    @staticmethod
    def _dataset_row_indices(dataset):
        if isinstance(dataset, Subset):
            parent_indices = HeimdallTrainer._dataset_row_indices(dataset.dataset)
            return parent_indices[np.asarray(dataset.indices, dtype=np.int64)]
        if hasattr(dataset, "idx"):
            return np.asarray(dataset.idx, dtype=np.int64)
        return np.arange(len(dataset), dtype=np.int64)

    @staticmethod
    def _as_csr(matrix):
        if issparse(matrix):
            return matrix.tocsr().astype(np.float32)
        return csr_matrix(np.asarray(matrix, dtype=np.float32))

    def _get_epoch_admixture_matrix(self, epoch: int):
        if self.data.admixture_epoch_store is not None:
            return self.data.admixture_epoch_store.get(
                epoch,
                obs_names=self.data.adata.obs_names,
                var_names=self.data.adata.var_names,
            )

        configured_layer = self.data.dataset_preproc_cfg.get("admix_layer", None)
        candidate_layers = []
        if configured_layer is not None:
            layer_parts = configured_layer.rsplit("_", 1)
            layer_prefix = (
                layer_parts[0]
                if len(layer_parts) == 2 and layer_parts[1].isdigit()
                else configured_layer
            )
            candidate_layers.extend([f"{layer_prefix}_{epoch}", configured_layer])
        candidate_layers.extend(["admixed_counts", "admixed"])

        for layer in candidate_layers:
            if layer in self.data.adata.layers:
                return self.data.adata.layers[layer]
        return None

    def _preprocess_for_silhouette(self, X, metric_cfg):
        metric_adata = AnnData(X=X.copy())
        normalize_total = metric_cfg.get("normalize_total", 10000)
        if normalize_total is not None:
            sc.pp.normalize_total(metric_adata, target_sum=float(normalize_total))
        if metric_cfg.get("log1p", True):
            sc.pp.log1p(metric_adata)

        n_pcs = int(metric_cfg.get("n_pcs", 50))
        n_comps = min(n_pcs, metric_adata.n_obs - 1, metric_adata.n_vars - 1)
        if n_comps < 1:
            return None
        sc.pp.pca(
            metric_adata,
            n_comps=n_comps,
            random_state=int(metric_cfg.get("random_state", self.cfg.seed)),
        )
        return metric_adata.obsm["X_pca"]

    def compute_silhouette_delta_metric(self, epoch: int):
        if not self.accelerator.is_main_process:
            return None

        metric_cfg = self.cfg.tasks.args.get("silhouette_delta", {})
        if not metric_cfg.get("enabled", False):
            return None
        if self.cfg.tasks.args.task_type != "admixture_binary":
            self.print_r0("> Skipping silhouette delta metric; task_type is not 'admixture_binary'.")
            return None

        split = metric_cfg.get("split", "val")
        if split not in self.data.datasets:
            self.print_r0(f"> Skipping silhouette delta metric; split {split!r} is not available.")
            return None

        label_col = metric_cfg.get("label_col_name", "cell_type")
        if label_col not in self.data.adata.obs:
            self.print_r0(f"> Skipping silhouette delta metric; obs column {label_col!r} is not available.")
            return None

        admixture_matrix = self._get_epoch_admixture_matrix(epoch)
        if admixture_matrix is None:
            self.print_r0("> Skipping silhouette delta metric; no admixture matrix/layer is available.")
            return None

        dataloader = DataLoader(
            self.data.datasets[split],
            batch_size=self.cfg.tasks.args.batchsize,
            shuffle=False,
            collate_fn=heimdall_collate_fn,
            num_workers=0,
        )

        row_indices = self._dataset_row_indices(self.data.datasets[split])
        labels = pd.Series(self.data.adata.obs[label_col].iloc[row_indices].to_numpy()).astype("category")
        valid = ~labels.isna().to_numpy()
        row_indices = row_indices[valid]
        labels = labels[valid].cat.remove_unused_categories().to_numpy()
        if len(row_indices) < 3 or len(np.unique(labels)) < 2:
            self.print_r0("> Skipping silhouette delta metric; not enough labeled cells/classes.")
            return None

        probs = np.vstack(self.get_predictions(dataloader))
        probs = probs[valid]
        if probs.shape[0] != len(row_indices):
            self.print_r0(
                "> Skipping silhouette delta metric; prediction rows do not match split rows "
                f"({probs.shape[0]} vs {len(row_indices)}).",
            )
            return None

        max_cells = metric_cfg.get("max_cells", None)
        if max_cells is not None and len(row_indices) > int(max_cells):
            rng = np.random.default_rng(int(metric_cfg.get("random_state", self.cfg.seed)))
            keep = np.sort(rng.choice(len(row_indices), size=int(max_cells), replace=False))
            row_indices = row_indices[keep]
            labels = labels[keep]
            probs = probs[keep]
            if len(np.unique(labels)) < 2:
                self.print_r0("> Skipping silhouette delta metric; subsample has fewer than two classes.")
                return None

        original = self._as_csr(self.data.adata.X[row_indices])
        admixed = original + self._as_csr(admixture_matrix[row_indices])
        cleaned = admixed.tocoo(copy=True)
        remove_mask = probs[cleaned.row, cleaned.col] >= float(metric_cfg.get("prediction_threshold", 0.5))
        cleaned.data[remove_mask] = 0
        cleaned = cleaned.tocsr()
        cleaned.eliminate_zeros()

        original_embedding = self._preprocess_for_silhouette(original, metric_cfg)
        cleaned_embedding = self._preprocess_for_silhouette(cleaned, metric_cfg)
        if original_embedding is None or cleaned_embedding is None:
            self.print_r0("> Skipping silhouette delta metric; PCA preprocessing produced no components.")
            return None

        silhouette_metric = metric_cfg.get("metric", "cosine")
        original_score = silhouette_score(original_embedding, labels, metric=silhouette_metric)
        cleaned_score = silhouette_score(cleaned_embedding, labels, metric=silhouette_metric)
        prefix = f"{split}_silhouette_delta"
        log = {
            f"{prefix}_original": float(original_score),
            f"{prefix}_cleaned": float(cleaned_score),
            f"{prefix}_cleaned_minus_original": float(cleaned_score - original_score),
            f"{prefix}_n_cells": int(len(row_indices)),
            f"{prefix}_prediction_threshold": float(metric_cfg.get("prediction_threshold", 0.5)),
            "epoch": epoch,
        }

        if self.run_wandb and self.accelerator.is_main_process:
            self.accelerator.log(log, step=self.step)
        if not self.run_wandb and self.accelerator.is_main_process:
            print(log)
        return log
    
    def fit(self, resume_from_checkpoint=True, checkpoint_every_n_epochs=1):
        """Train the model with automatic checkpointing and resumption."""
        # Initialize checkpointing
        self.initialize_checkpointing()

        # Try to resume from checkpoint if requested
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint()

        # If the tracked parameter is specified
        track_metric = None
        if self.cfg.tasks.args.get("track_metric", False):
            track_metric = self.cfg.tasks.args.track_metric
            best_metric = {
                f"best_val_{track_metric}": float("-inf"),
                f"reported_test_{track_metric}": float("-inf"),
            }
            assert (
                track_metric in self.cfg.tasks.args.metrics
            ), "The tracking metric is not in the list of metrics, please check your configuration task file"

        # Initialize early stopping parameters
        early_stopping = self.cfg.tasks.args.get("early_stopping", False)
        early_stopping_patience = self.cfg.tasks.args.get("early_stopping_patience", 5)
        patience_counter = 0

        best_val_embed = None
        best_test_embed = None
        best_epoch = 0

        # only add admixture once to validation and testing
        # add admixture from neighbors:
        # adm_config = self.cfg.dataset.preprocess_args.admixture_args
        # dl_list = [self.dataloader_val_ref,self.dataloader_test_ref]
        # for dl in dl_list:
        #     if self.accelerator.is_main_process:
        #         adata = self._get_adata_from_dataloader(dl)
        #         X_admixed, _ = sample_admixture_from_neighbor_dict(adata,
        #                                                             frac=adm_config['frac_transcript_per_cell'],
        #                                                             min_sampled_transcript_threshold=adm_config['min_sampled_transcript_threshold'],
        #                                                             mask_out_random_genes=adm_config['mask_out_random_genes'],
        #                                                               mask_out_frac=adm_config['mask_out_frac'])
        #         adata.layers['admixed'] = X_admixed
        #         adata.layers['is_transcript_added'] = (X_admixed > 0).astype(int)
        #     self.accelerator.wait_for_everyone()


        for epoch in range(start_epoch, self.cfg.tasks.args.epochs):
            self._set_dataset_epoch(epoch)

            
            # if self.accelerator.is_main_process:
            #     adata =  self._get_adata_from_dataloader(self.dataloader_train_ref)
            #     X_admixed, _ = sample_admixture_from_neighbor_dict(adata,
            #                                                         frac=adm_config['frac_transcript_per_cell'],
            #                                                         min_sampled_transcript_threshold=adm_config['min_sampled_transcript_threshold'],
            #                                                         mask_out_random_genes=adm_config['mask_out_random_genes'],
            #                                                         mask_out_frac=adm_config['mask_out_frac'])
                
            #     adata.layers['admixed'] = X_admixed
            #     adata.layers['is_transcript_added'] = (X_admixed > 0).astype(int)
            # self.accelerator.wait_for_everyone()
                
            # Validation and test evaluation
            self.validate_model(self.dataloader_train, dataset_type="train_eval")
            valid_log, val_embed = self.validate_model(self.dataloader_val, dataset_type="valid")
            test_log, test_embed = self.validate_model(self.dataloader_test, dataset_type="test")


            # Track the best metric if specified
            if track_metric:
                val_metric = valid_log.get(f"valid_{track_metric}", float("-inf"))
                print(val_metric)
                if val_metric > best_metric[f"best_val_{track_metric}"]:

                    best_val_embed = val_embed
                    best_test_embed = test_embed
                    best_epoch = epoch

                    best_metric[f"best_val_{track_metric}"] = val_metric
                    self.print_r0(f"New best validation {track_metric}: {val_metric}")
                    best_metric["reported_epoch"] = epoch  # log the epoch for convenience
                    for metric in self.cfg.tasks.args.metrics:
                        best_metric[f"reported_test_{metric}"] = test_log.get(f"test_{metric}", float("-inf"))
                    patience_counter = 0  # Reset patience counter since we have a new best
                    
                    
                    # Save checkpoint for best model
                    self.save_checkpoint(epoch)
                    self.print_r0(f"> Saved best model checkpoint at epoch {epoch}")
                else:
                    patience_counter += 1
                    if early_stopping:
                        self.print_r0(
                            f"No improvement in validation {track_metric}. "
                            f"Patience counter: {patience_counter}/{early_stopping_patience}",
                        )

            # Check early stopping condition
            if early_stopping and patience_counter >= early_stopping_patience:
                self.print_r0(
                    f"Early stopping triggered. No improvement in {track_metric} for {early_stopping_patience} epochs.",
                )
                break

            # Train for one epoch
            self.train_epoch(epoch)
            self.evaluate_external_dataset(epoch)
            self.compute_silhouette_delta_metric(epoch)

            # Save checkpoint at regular intervals if requested
            if (epoch + 1) % checkpoint_every_n_epochs == 0:
                self.save_checkpoint(epoch)
                self.print_r0(f"> Saved regular checkpoint at epoch {epoch}")

        # # Save final checkpoint ## no need to save the final checkpoint
        # self.save_checkpoint(epoch)
        # self.print_r0(f"> Saved final checkpoint at epoch {epoch}")

        if self.run_wandb and self.accelerator.is_main_process:
            if track_metric:  # logging the best val score and the tracked test scores
                self.accelerator.log(best_metric, step=self.step)
            self.accelerator.end_training()

        if (
            self.accelerator.is_main_process
            and self.cfg.model.name != "logistic_regression"
            and not isinstance(self.data.datasets["full"], Heimdall.datasets.PairedInstanceDataset)
        ):
            print(best_test_embed,best_val_embed)
            self.save_adata_umap(best_test_embed, best_val_embed)
            self.print_r0(f"> Saved best UMAP checkpoint at epoch {best_epoch}")

        if self.accelerator.is_main_process:
            self.print_r0("> Model has finished Training")

    def get_loss(self, logits, labels, masks=None):
        if masks is not None:
            logits, labels = logits[masks], labels[masks]

        if self.custom_loss_func:
            loss = self.loss_fn(logits, labels)
        elif self.cfg.loss.name.endswith("BCEWithLogitsLoss"):
            loss = self.loss_fn(logits, labels)
        elif self.cfg.loss.name.endswith("CrossEntropyFocalLoss"):
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.cfg.loss.name == "CrossEntropyLoss":
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.cfg.loss.name == "MSELoss":
            loss = self.loss_fn(logits, labels)
        else:
            raise NotImplementedError("Only custom, CrossEntropyLoss, and MSELoss are supported right now")

        return loss

    def train_epoch(self, epoch):
        self.model.train()
        step = len(self.dataloader_train) * epoch
        log_every = self.cfg.trainer.get("log_every", 50)

        with tqdm(self.dataloader_train, disable=not self.accelerator.is_main_process) as t:
            for batch in t:
                step += 1
                is_logging = step % log_every == 0

                lr = self.lr_scheduler.get_last_lr()[0]
                with self.accelerator.accumulate(self.model):

                    inputs = (batch["identity_inputs"], batch["expression_inputs"])

                    # id_inp, expr_inp = inputs
                    # pad = batch.get("expression_padding")

                    # def stats(name, x):
                    #     print(
                    #         name,
                    #         "shape:", tuple(x.shape),
                    #         "dtype:", x.dtype,
                    #         "nan:", torch.isnan(x).any().item(),
                    #         "min:", float(x.min()) if x.numel() > 0 else None,
                    #         "max:", float(x.max()) if x.numel() > 0 else None,
                    #     )

                    # stats("identity_inputs", id_inp)
                    # stats("expression_inputs", expr_inp)
                    # if pad is not None:
                    #     print("expression_padding shape:", pad.shape, "dtype:", pad.dtype, "unique:", torch.unique(pad).cpu())

                    outputs = self.model(
                        inputs=inputs,
                        attention_mask=batch.get("expression_padding"),
                    )

                    # REORDER BASED ON TAILORED ORDER
                    labels = batch["labels"].to(outputs.device)
                    if (masks := batch.get("masks")) is not None:
                        masks = masks.to(outputs.device)
                    

                    loss = self.get_loss(outputs.logits, labels, masks=masks)
                    
                    # if torch.isnan(loss):
                    #     print("NaN loss at step", self.step)
                    #     print("labels shape:", labels.shape, "logits shape:", outputs.logits.shape)
                    #     if masks is not None:
                    #         print("mask sum:", masks.sum().item())
                    #     print("labels NaNs:", torch.isnan(labels).any().item())
                    #     print("logits NaNs:", torch.isnan(outputs.logits).any().item())
                    #     # maybe break or raise here to inspect
                    #     raise RuntimeError("NaN loss detected")
                    self.accelerator.backward(loss)

                    # # 🔍 DEBUG: check gradients before clipping / optimizer
                    # has_nan_grad = False
                    # for name, p in self.model.named_parameters():
                    #     if p.grad is not None and torch.isnan(p.grad).any():
                    #         print(f"NaN grad in param: {name}, shape={tuple(p.grad.shape)}")
                    #         has_nan_grad = True
                    #         break
                    # print("Any NaN grad before clipping?", has_nan_grad)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.cfg.trainer.grad_norm_clip,
                        )
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        self.step += 1
                grad_norm =0
                if is_logging:
                    t.set_description(
                        f"Epoch: {epoch} "
                        f"Step {self.step} "
                        f"Loss: {loss.item():.4f} "
                        f"LR: {lr:.1e} "
                        f"grad_norm: {grad_norm:.4f} ",
                    )

                if is_logging:
                    log = {
                        "train_loss": loss.item(),
                        "global_step": self.step,
                        "learning_rate": lr,
                        "epoch": epoch,
                        "grad_norm": grad_norm,
                    }
                    if self.run_wandb and self.accelerator.is_main_process:
                        self.accelerator.log(log, step=self.step)

                if self.cfg.trainer.fastdev:
                    break

    # Add these methods to the HeimdallTrainer class
    def save_adata_umap(self, best_test_embed, best_val_embed):
        # Case 1: predefined splits
        if hasattr(self.cfg.tasks.args, "splits"):
            test_adata = self.data.adata[
                self.data.adata.obs[self.cfg.tasks.args.splits.col] == self.cfg.tasks.args.splits.keys_.test
            ].copy()
            val_adata = self.data.adata[
                self.data.adata.obs[self.cfg.tasks.args.splits.col] == self.cfg.tasks.args.splits.keys_.val
            ].copy()

        # Case 2: random splits
        elif hasattr(self.data, "splits"):
            # breakpoint()
            test_adata = self.data.adata[self.splits["test"]].copy()
            val_adata = self.data.adata[self.splits["val"]].copy()

        else:
            raise ValueError("No split information found in config")

        test_adata.obsm["heimdall_latents"] = best_test_embed
        val_adata.obsm["heimdall_latents"] = best_val_embed

        sc.pp.neighbors(test_adata, use_rep="heimdall_latents")
        sc.tl.leiden(test_adata)
        sc.tl.umap(test_adata)

        sc.pp.neighbors(val_adata, use_rep="heimdall_latents")
        sc.tl.leiden(val_adata)
        sc.tl.umap(val_adata)

        AnnData.write(test_adata, self.results_folder / "test_adata.h5ad")
        AnnData.write(val_adata, self.results_folder / "val_adata.h5ad")

    def initialize_checkpointing(self, results_folder_path=None):
        """Initialize checkpoint directory."""
        if results_folder_path is None:
            self.results_folder = Path(self.cfg.work_dir)
        else:
            self.results_folder = Path(results_folder_path)

        # Create directory if it doesn't exist
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(parents=True, exist_ok=True)
            self.print_r0(f"> Checkpoint directory initialized at {self.results_folder}")

    def save_checkpoint(self, epoch):
        """Save model checkpoint at the given epoch."""
        # Only save on the main process
        if not self.accelerator.is_main_process:
            return

        # Ensure results folder exists
        if not hasattr(self, "results_folder"):
            self.initialize_checkpointing()

        # Calculate current step based on epoch
        # step = len(self.dataloader_train) * epoch

        # Prepare the data to save
        data = {
            "epoch": epoch,
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if (self.accelerator.scaler is not None) else None,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "version": 1.0,
        }

        # Save checkpoint
        checkpoint_path = self.results_folder / f"model-{epoch}.pt"
        torch.save(data, str(checkpoint_path))
        self.print_r0(f"> Saved checkpoint to {checkpoint_path}")

        # Overwrite 'milestone.txt' with the new milestone
        milestone_file = self.results_folder / "milestone.txt"
        with open(milestone_file, "w") as f:
            f.write(str(epoch))
        self.print_r0(f"> Updated milestone.txt to milestone {epoch}")

        config_path = self.results_folder / "config.txt"
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

    def load_checkpoint(self, specific_milestone=None):
        """Load a checkpoint based on milestone.txt or a specific milestone
        number."""
        # Ensure results folder is initialized
        if not hasattr(self, "results_folder"):
            self.initialize_checkpointing()

        if not self.results_folder.exists():
            self.print_r0(f"> Results folder {self.results_folder} does not exist. Starting from scratch.")
            return 0

        # Determine which milestone to load
        if specific_milestone is not None:
            milestone = specific_milestone
        else:
            milestone_file = self.results_folder / "milestone.txt"
            if not milestone_file.exists():
                self.print_r0("> No milestone.txt found. Starting from scratch.")
                return 0

            # Read the milestone number
            with open(milestone_file) as f:
                milestone_str = f.read().strip()
                if not milestone_str.isdigit():
                    self.print_r0("milestone.txt is invalid. Starting from scratch.")
                    return 0
                milestone = int(milestone_str)

        # Load the checkpoint
        load_path = self.results_folder / f"model-{milestone}.pt"
        if not load_path.exists():
            self.print_r0(f"> Checkpoint file {load_path} does not exist. Starting from scratch.")
            return 0

        self.print_r0(f"> Loading checkpoint from {load_path}")

        # Load the data
        device = self.accelerator.device
        data = torch.load(str(load_path), map_location=device, weights_only=False)

        # Unwrap model and restore parameters
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        # Restore optimizer and scheduler states
        self.optimizer.load_state_dict(data["optimizer"])
        if (data["scaler"] is not None) and (self.accelerator.scaler is not None):
            self.accelerator.scaler.load_state_dict(data["scaler"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])

        # Restore random states
        random.setstate(data["python_rng_state"])
        np.random.set_state(data["numpy_rng_state"])

        # Handle torch RNG state
        torch_rng_state = data["torch_rng_state"]
        if isinstance(torch_rng_state, torch.Tensor) and torch_rng_state.device.type != "cpu":
            torch_rng_state = torch_rng_state.cpu()
        torch.random.set_rng_state(torch_rng_state)

        # Handle CUDA RNG states
        if data["cuda_rng_state_all"] is not None and torch.cuda.is_available():
            num_visible_devices = torch.cuda.device_count()
            if len(data["cuda_rng_state_all"]) != num_visible_devices:
                self.print_r0(
                    "Warning: Number of visible CUDA devices does not match the number of saved CUDA RNG states. "
                    "Skipping CUDA RNG state restoration.",
                )
            else:
                new_cuda_states = []
                for state in data["cuda_rng_state_all"]:
                    if isinstance(state, torch.Tensor) and state.device.type != "cpu":
                        state = state.cpu()
                    new_cuda_states.append(state)
                torch.cuda.set_rng_state_all(new_cuda_states)

        epoch = data["epoch"]
        self.step = data["step"]
        # step = data.get("step", epoch * len(self.dataloader_train))

        if "version" in data:
            self.print_r0(f"> Checkpoint version: {data['version']}")
        self.print_r0(f"> Resumed from epoch {epoch}, step {self.step}")

        return epoch + 1  # Return the next epoch to start from

    def validate_model(self, dataloader, dataset_type, extra_log=None):
        self.model.eval()
        metrics = self._initialize_metrics()
        # print(metrics)
        loss = 0
        encoded_list = []

        y_true_batches, preds_batches = [], []

        with torch.no_grad():
            for b,batch in tqdm(enumerate(dataloader), disable=not self.accelerator.is_main_process):
                inputs = (batch["identity_inputs"], batch["expression_inputs"])
                
                outputs = self.model(
                    inputs=inputs,
                    attention_mask=batch.get("expression_padding"),
                )
                



                logits = outputs.logits
                # print(outputs,flush=True)

                # identity_token_order = self.data.fc.adata.var["identity_embedding_index"]
                # logits = []

                labels = batch["labels"].to(outputs.device)

                if self.cfg.tasks.args.task_type == "multiclass":
                    preds = logits.argmax(dim=1)
                elif self.cfg.tasks.args.task_type == "binary" or self.cfg.tasks.args.task_type == "admixture_binary":
                    # multi-label binary classification → use sigmoid + threshold
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()


                elif self.cfg.tasks.args.task_type == "regression":
                    preds = logits


                else:
                    raise ValueError(f"Unsupported task_type: {self.cfg.tasks.args.task_type}")

                # if b == 0:
                #     print('printing syndri outputs:', labels[:,:20],labels[:,-20:])
                #     print('output shape:', probs.shape,labels.shape)
                y_true_batches.append(labels.cpu())
                preds_batches.append(preds.cpu())

                if self.cfg.model.name != "logistic_regression":
                    encoded_list.append(outputs.cls_embeddings.detach().cpu().numpy())

                if (masks := batch.get("masks")) is not None:
                    masks = masks.to(outputs.device)
                cellwise_metric_logits, cellwise_metric_labels = self._prepare_cellwise_logits_and_labels_for_metrics(
                    logits,
                    labels,
                    masks=masks,
                )
                if masks is not None:
                    logits, labels = logits[masks], labels[masks]

                # perform a .clone() so that the labels are not updated in-place
                # print(len(logits[0]),len(labels[0]))
                if b == 0 and self.cfg.tasks.args.task_type in ["binary", "admixture_binary"]:
                    flat = labels.flatten()
                    mask = ~torch.isnan(flat)
                    if mask.any():
                        print("positive rate:", flat[mask].mean().item(), "valid labels:", mask.sum().item())
                    else:
                        print("positive rate: no valid labels")

                loss += self.get_loss(logits, labels.clone()).item()

                # predictions = outputs["logits"] if isinstance(outputs, dict) else outputs
                # labels = batch['labels']

                # print(metrics)
                # print("---")
                for metric_name, metric in metrics.items():  # noqa: B007
                    # Built-in metric
                    # print(metric)
                    # print(metric_name)
                    if metric_name in ["AUROC", "AUPRC"]:
                        metric.update(cellwise_metric_logits, cellwise_metric_labels)
                    else:
                        metric_logits, metric_labels = self._prepare_logits_and_labels_for_metrics(logits, labels)
                        metric.update(metric_logits, metric_labels)
                if self.cfg.trainer.fastdev:
                    break

        all_encoded = None
        if self.cfg.model.name != "logistic_regression":
            all_encoded = np.concatenate(encoded_list, axis=0)

        loss = loss / len(dataloader)

        # concatenate & gather once per epoch
        y_true_all = torch.cat(y_true_batches, 0)
        preds_all = torch.cat(preds_batches, 0)

        if self.accelerator.num_processes > 1:
            loss = self.accelerator.gather(torch.tensor(loss)).mean().item()

        log = {f"{dataset_type}_loss": loss}
        if extra_log:
            log.update(extra_log)
        for metric_name, metric in metrics.items():
            if metric_name != "ConfusionMatrix":
                # Built-in metric
                log[f"{dataset_type}_{metric_name}"] = metric.compute().item()
                if metric_name in ["Accuracy", "Precision", "Recall", "F1Score", "MathewsCorrCoef"]:
                    log[f"{dataset_type}_{metric_name}"] *= 100  # Convert to percentage for these metrics

        if "ConfusionMatrix" in metrics:
            # 1. Gather counts from all processes and sum
            cm_local = metrics["ConfusionMatrix"].compute()  # (C, C) tensor
            cm_counts = self.accelerator.reduce(cm_local, reduction="sum")  # global counts

            # 3) If binary and flat, reshape to (2, 2)
            if cm_counts.dim() == 1:
                c = int(cm_counts.numel() ** 0.5)  # should be 2
                cm_counts = cm_counts.view(c, c)

            # 2. Row-wise normalisation → per-class accuracy matrix
            cm_norm = cm_counts.float()
            cm_norm = cm_norm / (cm_norm.sum(dim=1, keepdim=True) + 1e-8)

            # 3. Per-class accuracy vector (for dashboard scalars)
            per_class_acc = cm_norm.diag().cpu().numpy() * 100
            log[f"{dataset_type}_per_class_accuracy"] = {
                name: float(acc) for name, acc in zip(self.class_names, per_class_acc)
            }

            # 4. Log interactive confusion matrix to WandB (main process only)
            if self.run_wandb and self.accelerator.is_main_process:
                wandb_cm = wandb.plot.confusion_matrix(
                    y_true=y_true_all.numpy().tolist(),
                    preds=preds_all.numpy().tolist(),
                    class_names=self.class_names,  # same order as metric
                )
                self.accelerator.log(
                    {f"{dataset_type}_confusion_matrix": wandb_cm},
                    step=self.step,
                )

        rss = self.process.memory_info().rss / (1024**3)
        log["Process_mem_rss"] = rss

        if self.run_wandb and self.accelerator.is_main_process:
            self.accelerator.log(log, step=self.step)

        if not self.run_wandb and self.accelerator.is_main_process:
            print(log)


        return log, all_encoded

    def get_predictions(self, dataloader, data=None):
        self.model.eval()
        all_probs_batches = []
        device = self.accelerator.device
        data = data or self.data

        def move_to_device(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, tuple):
                return tuple(move_to_device(v) for v in value)
            if isinstance(value, list):
                return [move_to_device(v) for v in value]
            return value

        with torch.no_grad():
            for batch in tqdm(dataloader, disable=not self.accelerator.is_main_process):
                inputs = (
                    move_to_device(batch["identity_inputs"]),
                    move_to_device(batch["expression_inputs"]),
                )

                gene_reorderings = move_to_device(batch['gene_order'])
                
                outputs = self.model(
                    inputs=inputs,
                    attention_mask=move_to_device(batch.get("expression_padding")),
                )

                logits = outputs.logits

                # identity_token_order = self.data.fc.adata.var["identity_embedding_index"]
                # logits = []

                labels = batch["labels"].to(outputs.device)

                if self.cfg.tasks.args.task_type == "multiclass":
                    preds = logits.argmax(dim=1)
                elif self.cfg.tasks.args.task_type == "binary" or self.cfg.tasks.args.task_type == "admixture_binary":
                    # multi-label binary classification → use sigmoid + threshold
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()

                elif self.cfg.tasks.args.task_type == "regression":
                    preds = logits


                else:
                    raise ValueError(f"Unsupported task_type: {self.cfg.tasks.args.task_type}")

                # Reorder token-level probabilities back into adata.var order.
                probs_np = probs.detach().cpu().numpy()
                gene_reorderings_np = gene_reorderings.detach().cpu().numpy()
                new_probs = np.zeros((len(probs_np), data.adata.X.shape[1]), dtype=probs_np.dtype)

                valid = (
                    (gene_reorderings_np != data.fe.pad_value)
                    & (gene_reorderings_np != data.fe.mask_value)
                    & (gene_reorderings_np >= 0)
                    & (gene_reorderings_np < data.adata.X.shape[1])
                )
                row_idx, token_idx = np.nonzero(valid)
                new_probs[row_idx, gene_reorderings_np[row_idx, token_idx]] = probs_np[row_idx, token_idx]
                all_probs_batches.append(new_probs)
        return all_probs_batches

                
