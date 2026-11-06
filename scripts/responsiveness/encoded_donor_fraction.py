"""Attach malignant composition of the actual encoded five-neighbor contexts."""
from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from encoded_context_rank import _column

def encoded_donor_fraction(run, source_contexts, output_dir):
    run,out=Path(run),Path(output_dir)
    out.mkdir(exist_ok=True)
    table_path=out/'example_donor_fraction_k5_different_type_responses.csv.gz'
    correlation_path=out/'query_context_donor_fraction_k5_correlations.csv'
    complete=out/'encoded_donor_fraction_k5_complete.json'
    if complete.exists():
        return pd.read_csv(table_path),pd.read_csv(correlation_path)
    frames=[]
    for entry in json.loads((run/'design.json').read_text())['slices']:
        section=entry['slice']
        source=source_contexts.loc[source_contexts['slice'].eq(section)].copy()
        with h5py.File(entry['predictions']) as f:
            obs=pd.DataFrame({k:_column(f['obs/'+k]) for k in ['focal_cell_id','context_source_cell_id','context_kind']})
            fraction=f['obsm/neighborhood_proportions/malignant cell'][:]
            count=np.rint(fraction*5).astype(int)
            if not np.allclose(fraction,count/5) or not np.all((count>=0)&(count<=5)):
                raise ValueError('Encoded composition is not a five-token fraction')
            original=obs.loc[obs.context_kind.eq('original')].copy()
            if original.focal_cell_id.duplicated().any():raise ValueError('Duplicate original contexts')
            original['malignant_count']=count[original.index.to_numpy()]
            lookup=original.set_index('focal_cell_id').malignant_count
            query_count=obs.focal_cell_id.map(lookup).to_numpy()
            if pd.isna(query_count).any():raise ValueError('Missing true query context')
            obs['donor_malignant_fraction_k5']=count/5
            obs['query_malignant_fraction_k5']=query_count/5
            # Compute the integer-count difference first so mathematically equal deltas stay tied.
            obs['delta_donor_malignant_fraction_k5']=(count-query_count)/5
        obs=obs.rename(columns={'focal_cell_id':'query_cell_id'})
        merged=source.merge(obs,on=['query_cell_id','context_source_cell_id','context_kind'],how='left',validate='one_to_one')
        if merged.donor_malignant_fraction_k5.isna().any():raise ValueError('Unmatched encoded context composition')
        frames.append(merged)
    result=pd.concat(frames,ignore_index=True)
    correlations=[]
    for (section,q),group in result.loc[result.context_kind.eq('representative')].groupby(['slice','query_cell_id']):
        for metric in ['delta_marker_logit','delta_marker_rank_percentile']:
            values=group[['delta_donor_malignant_fraction_k5',metric]].dropna()
            supported=(len(values)>=10 and values.delta_donor_malignant_fraction_k5.nunique()>=3 and values[metric].nunique()>1)
            rho=float(spearmanr(values.delta_donor_malignant_fraction_k5,values[metric]).statistic) if supported else np.nan
            correlations.append(dict(slice=section,query_cell_id=q,query_cell_type=group.query_cell_type.iloc[0],metric=metric,
                spearman_rho=rho,n_contexts=len(values),status='ok' if supported else 'insufficient_variation'))
    controls=result.loc[result.context_kind.eq('unchanged_control')]
    if not np.allclose(controls.delta_donor_malignant_fraction_k5,0):raise ValueError('Controls change composition')
    result.to_csv(table_path,index=False,compression='gzip')
    corr=pd.DataFrame(correlations);corr.to_csv(correlation_path,index=False)
    complete.write_text(json.dumps(dict(neighbors=5,exclude_self=True,exclude_same_focal_cell_type=True,
        source='saved encoded neighborhood_proportions and original-context baseline',
        source_count_rule='existing preparation: eligible source cells have >10 counts',
        x='(donor malignant count - original query malignant count) / 5',
        marker_genes=json.loads((run/'requested_marker_genes_20261019.json').read_text())),indent=2)+'\n')
    return result,corr
