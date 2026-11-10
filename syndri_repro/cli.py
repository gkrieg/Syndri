"""Single command interface; heavyweight dependencies are imported per task."""
import argparse
import os
from pathlib import Path
from . import ROOT


def main(argv=None):
    parser=argparse.ArgumentParser(description='Syndri training, inference, and paper figure reproduction')
    parser.add_argument('--params',type=Path,help='Setup YAML/JSON file; defaults to configs/reproduction.yaml')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('setup',help='Prepare required data and trained checkpoints from included files, local copies, or downloads')
    p.add_argument('--verify',action='store_true',default=None)
    p=sub.add_parser('download',help='Use the bundled assets or download missing files from the release manifest')
    p.add_argument('--verify',action='store_true',default=None);p.add_argument('--base-url')
    p.add_argument('--sources',action='store_true',help='List the original public study portals and accessions')
    p=sub.add_parser('preprocess',help='Audit archived inputs or rebuild pseudo-spatial mappings')
    p.add_argument('--dataset',choices=['BM','NC','BX','SIM','all'],default='all')
    p.add_argument('--rebuild',action='store_true')
    p.add_argument('--select-representative',action='store_true')
    p=sub.add_parser('train',help='Train Syndri from scratch using the archived dataset configuration')
    p.add_argument('--dataset',choices=['BM','NC','BX','SIM'],required=True)
    p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int)
    p.add_argument('--resume',action='store_true');p.add_argument('--rebuilt',action='store_true')
    p=sub.add_parser('infer',help='Infer probabilities and create corrected counts per slice')
    p.add_argument('--dataset',choices=['BM','NC','BX','SIM'],required=True)
    p.add_argument('--slice');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--retrained',action='store_true');p.add_argument('--milestone',type=int)
    p.add_argument('--unit',choices=['cell','cell_type']);p.add_argument('--max-cells',type=int)
    p.add_argument('--generated-test',action='store_true')
    p=sub.add_parser('evaluate',help='Calculate Raw/Syndri separation, reference, and variance metrics')
    p.add_argument('--dataset',choices=['BM','NC','BX','all'],default='all')
    p.add_argument('--recompute',action='store_true');p.add_argument('--max-cells',type=int)
    p.add_argument('--unit',choices=['cell','cell_type'])
    p=sub.add_parser('simulate',help='Regenerate test admixture with K=9 and tau=[0,0.95]')
    p.add_argument('--rebuilt',action='store_true');p.add_argument('--seed',type=int,default=0)
    p=sub.add_parser('figures',help='Export Syndri-only figure panels and numerical source tables')
    p.add_argument('--panel',choices=['figure1','figure2','figure3','figure4','context','all'],default='all')
    p.add_argument('--recompute',action='store_true');p.add_argument('--skip-metrics',action='store_true')
    p=sub.add_parser('context',help='Reproduce the fixed-query Xenium context-perturbation figure')
    p.add_argument('--stage',choices=['plot','prepare','infer','analyze','all'],default='plot')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/context')
    p.add_argument('--slice',help='Optional section label, e.g. B1_1, for prepare/infer')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--resample',action='store_true',help='Resample instead of replaying the frozen queries/donors; use a fresh output directory')
    p.add_argument('--generated',action='store_true',help='Plot newly computed summaries instead of the bundled tables')
    p=sub.add_parser('reproduce',help='Verify inputs, audit them, evaluate, and export all panels using bundled predictions')
    p.add_argument('--recompute',action='store_true')
    for command_parser in sub.choices.values():
        command_parser.add_argument('--params',type=Path,default=argparse.SUPPRESS,
                                    help='Setup YAML/JSON file (also accepted before the command)')
    args=parser.parse_args(argv)
    from .parameters import configure
    configure(args.params)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.matplotlib'))
    os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'outputs/.numba'))
    os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
    os.environ.setdefault('WANDB_MODE','disabled')
    if args.command not in ['setup','download']:
        from .assets import download
        download(quiet=True)
    if args.command=='setup':
        from .assets import download
        download(args.verify)
    elif args.command=='download':
        from .assets import download
        if args.sources:
            from .assets import show_sources
            show_sources()
        else:
            download(args.verify,args.base_url)
    elif args.command=='preprocess':
        from .preprocess import validate,rebuild,select_representative
        for name in ['BM','NC','BX','SIM'] if args.dataset=='all' else [args.dataset]:
            validate(name)
            selected=select_representative(name) if args.select_representative and name!='SIM' else None
            if args.rebuild:rebuild(name, selected)
    elif args.command=='train':
        from .model import train
        train(args.dataset,args.epochs,args.resume,args.rebuilt,args.batch_size)
    elif args.command=='infer':
        from .model import infer
        infer(args.dataset,args.slice,args.batch_size,args.retrained,args.milestone,args.unit,args.max_cells,args.generated_test)
    elif args.command=='evaluate':
        from .metrics import evaluate
        for name in ['BM','NC','BX'] if args.dataset=='all' else [args.dataset]:
            evaluate(name,args.recompute,args.max_cells,args.unit)
    elif args.command=='simulate':
        from .simulation import generate_test
        generate_test(args.rebuilt,args.seed)
    elif args.command=='figures':
        from . import figures
        for panel in ['figure1','figure2','figure3','figure4','context'] if args.panel=='all' else [args.panel]:
            if panel=='figure2':figures.figure2(args.recompute,not args.skip_metrics)
            elif panel=='context':
                from .context_figure import plot_context
                plot_context()
            else:getattr(figures,panel)(args.recompute)
    elif args.command=='context':
        from .context import execute
        execute(args.stage,args.output,args.slice,args.batch_size,args.resample,args.generated)
    elif args.command=='reproduce':
        from .preprocess import validate
        from .metrics import evaluate
        from . import figures
        for name in ['BM','NC','BX','SIM']:validate(name)
        for name in ['BM','NC','BX']:evaluate(name,args.recompute)
        for panel in ['figure1','figure2','figure3','figure4']:getattr(figures,panel)(args.recompute)
        from .context_figure import plot_context
        plot_context()

