"""Attach within-query neighborhood-composition changes to saved response tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from all_types_panel import FAMILIES


def original_family_proportions(run):
    import anndata as ad
    records=[]
    for entry in json.loads((Path(run)/'design.json').read_text())['slices']:
        a=ad.read_h5ad(entry['contexts'],backed='r')
        try:
            obs=a.obs
            keep=obs['context_kind'].eq('original').to_numpy()
            baseline=obs.loc[keep,['focal_cell_id']].copy()
            if baseline.focal_cell_id.duplicated().any():
                raise ValueError(f'Multiple baselines in {entry["slice"]}')
            comp=a.obsm['neighborhood_proportions'].iloc[np.flatnonzero(keep)]
            for family,types in FAMILIES.items():
                part=pd.DataFrame(dict(slice=entry['slice'],query_cell_id=baseline.focal_cell_id.astype(str).to_numpy(),
                                       neighbor_family=family,original_neighbor_proportion=comp.reindex(columns=types,fill_value=0).sum(axis=1).to_numpy()))
                records.append(part)
        finally:
            a.file.close()
    result=pd.concat(records,ignore_index=True)
    if result.duplicated(['slice','query_cell_id','neighbor_family']).any():raise ValueError('Duplicate baseline keys')
    return result


def attach_delta_neighbor_proportion(examples,baselines):
    result=examples.merge(baselines,on=['slice','query_cell_id','neighbor_family'],how='left',validate='many_to_one')
    if result.original_neighbor_proportion.isna().any():raise ValueError('Missing original context for example query')
    result['delta_neighbor_proportion']=result.neighbor_proportion-result.original_neighbor_proportion
    if not result.delta_neighbor_proportion.between(-1-1e-6,1+1e-6).all():raise ValueError('Invalid composition change')
    return result

if __name__=='__main__':
    import sys
    run=Path(sys.argv[1]);destination=run/'summaries_delta_context';destination.mkdir(exist_ok=False)
    baselines=original_family_proportions(run)
    baselines.to_csv(destination/'original_family_proportions.csv',index=False)
    for directory,filename,output in [
        ('summaries_logits','example_context_responses.csv.gz','logit_examples.csv.gz'),
        ('summaries_ranks','example_marker_rank_changes.csv.gz','rank_examples.csv.gz')]:
        examples=pd.read_csv(run/directory/filename)
        result=attach_delta_neighbor_proportion(examples,baselines)
        if len(result)!=len(examples):raise ValueError('Example row count changed')
        result.to_csv(destination/output,index=False,compression='gzip')
        print(output,len(result),'rows; x range',result.delta_neighbor_proportion.min(),result.delta_neighbor_proportion.max())
    (destination/'complete.json').write_text(json.dumps(dict(x_axis='substituted minus original family proportion',n_baselines=len(baselines)),indent=2)+'\n')
