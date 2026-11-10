"""Training and inference using the bundled, checkpoint-compatible runtime."""
from argparse import Namespace
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from . import ROOT
from .io import config, dataset, data_dir, spatial_files, read, write
from .correction import correct
from .parameters import configuration_path


def runtime_config(name, output=None, training_path=None):
    cfg = OmegaConf.load(configuration_path('models', name))
    OmegaConf.set_struct(cfg, False)
    cfg.data_path = str(ROOT / 'assets/metadata')
    cfg.ensembl_dir = cfg.data_path
    cfg.fc.args.ensembl_dir = cfg.data_path
    cfg.fc.args.gene_metadata_filepath = str(ROOT / 'assets/metadata/gene_metadata/species_chrom.csv')
    cfg.dataset.preprocess_args.data_path = str(training_path or data_dir(name) / 'training.h5ad')
    cfg.work_dir = str(output or ROOT / f'outputs/models/{name}')
    cfg.cache_preprocessed_dataset_dir = None
    if 'external_eval' in cfg.tasks.args:
        cfg.tasks.args.external_eval.enabled = False
    return cfg


def train(name, epochs=None, resume=False, rebuilt=False, batch_size=None):
    from accelerate.utils import set_seed
    from Heimdall.cell_representations import CellRepresentation
    from Heimdall.models import HeimdallModel
    from Heimdall.trainer import HeimdallTrainer
    from Heimdall.utils import get_dtype
    output = ROOT / f'outputs/models/{name}'
    if output.exists() and any(output.glob('model-*.pt')) and not resume:
        raise FileExistsError(f'{output} already has checkpoints; use --resume to continue')
    source = ROOT / f'outputs/preprocessed/{name}/training.h5ad' if rebuilt else None
    cfg = runtime_config(name, output, source)
    if epochs is not None:
        cfg.tasks.args.epochs = epochs
    if batch_size is not None:
        cfg.tasks.args.batchsize = batch_size
    set_seed(cfg.seed)
    data = CellRepresentation(cfg)
    import anndata as ad
    from .io import csr
    model_reference = ad.AnnData(csr(data.adata.X[:1]), obs=data.adata.obs.iloc[:1].copy(), var=data.adata.var.copy())
    model_reference.uns['cell_type_dict'] = dict(data.adata.uns['cell_type_dict'])
    write(model_reference, output/'reference.h5ad')
    model = HeimdallModel(data=data, model_config=cfg.model, task_config=cfg.tasks.args).to(get_dtype(cfg.float_dtype))
    trainer = HeimdallTrainer(cfg=cfg, model=model, data=data, run_wandb=False)
    trainer.fit(resume_from_checkpoint=resume)


def infer(name, slice_name=None, batch_size=16, retrained=False, milestone=None, unit=None, max_cells=None, generated_test=False):
    from ._inference import prepare_spatial_adata, run_inference
    checkpoint = ROOT / ('outputs/models' if retrained else 'assets/models') / name
    if milestone is None:
        milestone = (int((checkpoint / 'milestone.txt').read_text()) if retrained
                     else dataset(name)['milestone'])
    if not (checkpoint / f'model-{milestone}.pt').exists():
        raise FileNotFoundError(checkpoint / f'model-{milestone}.pt')
    sources = sorted((ROOT/'outputs/simulated_test').glob('*.h5ad')) if generated_test else spatial_files(name)
    if generated_test and name != 'SIM':
        raise ValueError('--generated-test requires --dataset SIM')
    if slice_name:
        sources = [p for p in sources if p.stem == slice_name]
    if not sources:
        raise ValueError(f'No matching input slices for {name}: {slice_name}')
    for source in sources:
        output = ROOT / ('outputs/.checks/predictions' if max_cells else 'outputs/predictions') / name
        output.mkdir(parents=True, exist_ok=True)
        cfg = runtime_config(name, output)
        if retrained:
            saved = OmegaConf.load(checkpoint / 'config.txt')
            # Keep architecture and training settings from the new checkpoint,
            # but relocate all metadata and output paths before use.
            saved.data_path = cfg.data_path; saved.ensembl_dir = cfg.ensembl_dir
            saved.fc.args.ensembl_dir = cfg.fc.args.ensembl_dir
            saved.fc.args.gene_metadata_filepath = cfg.fc.args.gene_metadata_filepath
            saved.work_dir = str(output); saved.cache_preprocessed_dataset_dir = None
            cfg = saved
        st_path = source
        if max_cells is not None:
            # Explicit development option; never enabled by the paper commands.
            obj = read(source)
            obj = obj[:max_cells].copy()
            st_path = output / f'{source.stem}_subset_input.h5ad'; write(obj, st_path)
        reference_path = checkpoint/'reference.h5ad' if retrained else data_dir(name)/'training.h5ad'
        args = Namespace(st_data=st_path, reference_data=reference_path,
            expression_layer=None, gene_list=None, cell_type_col=dataset(name)['label'],
            preserve_split_column=False, split_col='split', split_value='val', output_dir=output,
            prepared_name=f'{source.stem}_prepared.h5ad', output_name=source.name,
            config=cfg, checkpoint_dir=checkpoint, milestone=milestone,
            coord_cols=','.join(dataset(name)['coords']), tissue_col=None,
            batch_size=batch_size, fast_inference=True, dry_run_tokenization=False,
            apply_admixture=False, prob_layer='admix_probs')
        prepared = prepare_spatial_adata(args)
        result_path = run_inference(args, prepared)
        obj = read(result_path, layers=['admix_probs', 'original_counts', 'synthetic_admixture_counts'])
        settings = dict(config()['correction']); settings['unit'] = unit or settings['unit']
        cleaned, diagnostics = correct(obj.X, obj.layers['admix_probs'], obj.obs[args.cell_type_col], **settings)
        obj.layers['syndri_corrected_counts'] = cleaned
        obj.uns['syndri_correction'] = settings
        write(obj, result_path)
        diagnostics.to_csv(output / f'{source.stem}_thresholds.csv', index=False)

