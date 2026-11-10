"""Fixed-query Xenium context interventions, inference, and figure regeneration.

The archived summary tables support plotting without running inference. Explicit
prepare/infer/analyze stages regenerate the experiment using only local assets.
"""
import json
from argparse import Namespace
from pathlib import Path
from . import ROOT
from .parameters import configuration_path

ARCHIVED = ROOT / 'assets/context'
DEFAULT_RUN = ROOT / 'outputs/context'


def configuration():
    return json.loads(configuration_path('context').read_text())


def initialize(run=DEFAULT_RUN, resample=False):
    run = Path(run).resolve()
    design = configuration()
    design['replay_sampling'] = not resample
    design['sampling_dir'] = str(ARCHIVED / 'sampling')
    design['reference'] = str(ROOT / 'assets/data/BX/training.h5ad')
    design['checkpoint'] = str(ROOT / 'assets/models/BX')
    for entry in design['slices']:
        entry['input'] = str(ROOT / entry['input'])
        entry['contexts'] = str(run / 'contexts' / (entry['slice'] + '.h5ad'))
        entry['predictions'] = str(run / 'predictions' / (entry['slice'] + '.h5ad'))
    path = run / 'design.json'
    if path.exists() and json.loads(path.read_text()) != design:
        raise ValueError('Existing context design differs; select a fresh --output directory.')
    run.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(design, indent=2) + '\n')
    for directory in ['contexts', 'predictions', 'burden_stratified_context_auroc_response']:
        (run / directory).mkdir(exist_ok=True)
    import pandas as pd
    pd.DataFrame([(t, f) for t, f in design['families'].items() if f != 'Malignant'],
                 columns=['query_cell_type', 'query_family']).to_csv(
        run / 'burden_stratified_context_auroc_response/cell_family_mapping.csv', index=False)
    return design


def infer_section(run, index, batch_size=16):
    from accelerate.utils import set_seed
    from .model import runtime_config
    from ._inference import prepare_spatial_adata, run_inference
    run = Path(run)
    design = json.loads((run / 'design.json').read_text())
    entry = design['slices'][index]
    input_path, output_path = Path(entry['contexts']), Path(entry['predictions'])
    if output_path.exists():
        raise FileExistsError(f'{output_path}: use a fresh --output directory to regenerate predictions')
    if not input_path.exists():
        raise FileNotFoundError(f'Prepare contexts first: {input_path}')
    cfg = runtime_config('BX', run / 'predictions')
    set_seed(cfg.seed)
    args = Namespace(st_data=input_path, reference_data=Path(design['reference']),
        expression_layer=None, gene_list=None, cell_type_col='harmony_cell_type',
        preserve_split_column=True, split_col='split', split_value='val',
        output_dir=run / 'predictions', prepared_name=entry['slice'] + '_prepared.h5ad',
        output_name=output_path.name, config=cfg, checkpoint_dir=Path(design['checkpoint']),
        milestone=design['milestone'], coord_cols='x_centroid,y_centroid', tissue_col=None,
        batch_size=batch_size, fast_inference=True, dry_run_tokenization=False,
        use_precomputed_neighbor=True, apply_admixture=False, prob_layer='admix_probs')
    # The supplied neighbor token sequences must survive preprocessing. Rebuilding
    # physical neighbors here would destroy the fixed-query intervention.
    prepared = prepare_spatial_adata(args)
    return run_inference(args, prepared)


def aggregate_curves(run):
    """Equal query, type, and original-context-stratum aggregation from the notebook."""
    import numpy as np
    import pandas as pd
    run = Path(run)
    out = run / 'burden_stratified_context_auroc_response'
    rows = pd.read_csv(run / 'marker_auroc_context_responses/marker_auroc_context_responses.csv.gz',
                       dtype={'query_neighbor_level': str})
    rows = rows.loc[rows.context_selected_for_shared_panel.eq(True)
        & rows.context_kind.ne('unchanged_control')
        & rows.query_burden_quantile.isin(['low', 'medium', 'high'])
        & np.isfinite(rows.delta_marker_auroc)].copy()
    rows['delta_malignant_neighbors'] = (rows.context_malignant_token_count
                                       - rows.query_malignant_token_count).astype(int)
    keys = ['slice', 'query_cell_type', 'query_burden_quantile',
            'query_neighbor_level', 'delta_malignant_neighbors']
    queries = rows.groupby(keys + ['query_cell_id'], observed=True).delta_marker_auroc.mean().reset_index()
    types = queries.groupby(keys, observed=True).agg(
        delta_marker_auroc=('delta_marker_auroc', 'median'),
        n_queries=('query_cell_id', 'nunique')).reset_index()
    types['query_family'] = types.query_cell_type.map(configuration()['families'])
    if types.query_family.isna().any():
        raise ValueError('Unmapped query types')
    strata = ['query_family', 'query_burden_quantile', 'query_neighbor_level', 'delta_malignant_neighbors']
    sections = types.loc[types.n_queries.ge(5)].groupby(['slice'] + strata, observed=True).agg(
        delta_marker_auroc=('delta_marker_auroc', 'mean'),
        n_cell_types=('query_cell_type', 'nunique'), n_queries=('n_queries', 'sum')).reset_index()
    supported = sections.groupby(strata, observed=True).agg(
        n_sections=('slice', 'nunique')).reset_index()
    supported = supported.loc[supported.n_sections.ge(3)]
    selected = sections.merge(supported[strata], on=strata, how='inner', validate='many_to_one')
    keys = ['query_family', 'query_burden_quantile', 'delta_malignant_neighbors']
    section_curves = selected.groupby(['slice'] + keys, observed=True).agg(
        delta_marker_auroc=('delta_marker_auroc', 'mean'),
        n_original_context_strata=('query_neighbor_level', 'nunique')).reset_index()
    curves = section_curves.groupby(keys, observed=True).agg(
        delta_marker_auroc=('delta_marker_auroc', 'median'),
        q25=('delta_marker_auroc', lambda s: s.quantile(.25)),
        q75=('delta_marker_auroc', lambda s: s.quantile(.75)),
        n_sections=('slice', 'nunique')).reset_index()
    types.to_csv(out / 'cell_family_contributing_type_responses.csv.gz', index=False)
    sections.to_csv(out / 'cell_family_grid_section_responses.csv.gz', index=False)
    section_curves.to_csv(out / 'combined_family_auroc_section_curves.csv', index=False)
    curves.to_csv(out / 'combined_family_auroc_curves.csv', index=False)


def analyze(run=DEFAULT_RUN):
    from .context_native import compute_native_marker_analysis
    from .context_within_type import compute_control_native_within_type
    from .context_vascular import compute_vascular
    run = Path(run)
    design = json.loads((run / 'design.json').read_text())
    missing = [e['predictions'] for e in design['slices'] if not Path(e['predictions']).exists()]
    if missing:
        raise FileNotFoundError('Complete context inference for every section first:\n' + '\n'.join(missing))
    compute_native_marker_analysis(run, min_queries=10, min_delta_levels=3)
    aggregate_curves(run)
    compute_control_native_within_type(run)
    compute_vascular(run)


def execute(stage='plot', run=DEFAULT_RUN, slice_name=None, batch_size=16,
            resample=False, generated=False):
    if stage in ['prepare', 'infer', 'all']:
        design = initialize(run, resample)
        entries = [(i, e) for i, e in enumerate(design['slices'])
                   if slice_name is None or slice_name in [e['slice'], e['label']]]
        if not entries:
            raise ValueError(f'Unknown section: {slice_name}')
        for index, entry in entries:
            if stage in ['prepare', 'all']:
                from .context_design import prepare
                prepare(run, index)
            if stage in ['infer', 'all']:
                infer_section(run, index, batch_size)
    if stage in ['analyze', 'all']:
        analyze(run)
    if stage in ['plot', 'all']:
        from .context_figure import plot_context
        return plot_context(Path(run) if generated or stage == 'all' else ARCHIVED)
