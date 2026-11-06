"""Run native CosMx DE using an existing slice-specific ResolVI checkpoint."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import scvi


def main():
    p=argparse.ArgumentParser();p.add_argument('request',type=Path)
    request=p.parse_args().request
    cfg=json.loads(request.read_text());out=request.parent
    if (out/'de.csv').exists():raise FileExistsError(out/'de.csv')
    if scvi.__version__!='1.3.3':raise RuntimeError('Requires scvi-tools 1.3.3')
    model_dir=Path(cfg['model_dir'])
    if not (model_dir/'model.pt').is_file() or not (model_dir/'adata.h5ad').is_file():
        raise FileNotFoundError(f'Missing checkpoint or registered AnnData: {model_dir}')
    scvi.settings.seed=0
    model=scvi.external.RESOLVI.load(str(model_dir))
    work=model.adata
    cohort=pd.read_csv(cfg['cohort'],dtype={'cell_id':str})
    if cohort.cell_id.duplicated().any() or not cohort.cell_id.isin(work.obs_names).all():
        raise ValueError('Cohort IDs missing from saved model or duplicated')
    if not cohort.region.isin(['tumor','stroma']).all():raise ValueError('Unexpected cohort region')
    if not work.obs.loc[cohort.cell_id,'cell_type_harmonized'].eq('Fibroblast').all():
        raise ValueError('Cohort includes non-fibroblasts')
    expected=work.obs.loc[cohort.cell_id,'niche'].astype('string').map({'tumor interior':'tumor','stroma':'stroma'})
    if not np.array_equal(expected.to_numpy(),cohort.region.to_numpy()):
        raise ValueError('Saved model region assignments differ from comparison cohort')
    masks=[work.obs_names.isin(cohort.loc[cohort.region.eq(g),'cell_id']) for g in ['tumor','stroma']]
    if min(map(np.sum,masks))<cfg['min_group_size']:raise ValueError('Insufficient tumor/stroma fibroblasts')
    print('Loaded saved model:',model_dir,'; tumor/stroma:',list(map(np.sum,masks)),flush=True)
    result=model.differential_expression(idx1=masks[0],idx2=masks[1],mode='change',
        delta=cfg['delta'],fdr_target=cfg['fdr'],batch_correction=False,
        weights=cfg['weights'],pseudocounts=cfg['pseudocounts'],
        filter_outlier_cells=cfg['filter_outlier_cells'],test_mode=cfg['test_mode'])
    required={'lfc_mean','proba_de',f"is_de_fdr_{cfg['fdr']}"}
    if not required<=set(result.columns) or result.empty:raise ValueError('Incomplete native DE output')
    if not np.isfinite(result.proba_de).all() or not result.proba_de.between(0,1).all():
        raise ValueError('Invalid native DE probabilities')
    result.index.name='feature_id';result.to_csv(out/'de.csv',mode='x')
    (out/'complete.json').write_text(json.dumps({'refit':False,'scvi_version':scvi.__version__,
        'model_dir':str(model_dir),'n_tumor':int(masks[0].sum()),'n_stroma':int(masks[1].sum()),
        'n_genes':len(result)},indent=2))
    print('Saved native DE:',out/'de.csv',flush=True)

if __name__=='__main__':main()
