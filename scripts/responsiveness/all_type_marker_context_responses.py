"""Marker-score and marker-rank changes for all saved query/context pairs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
from scipy.stats import rankdata
from encoded_context_rank import _column,_matrix

def all_type_marker_context_responses(run,markers,output_dir):
    run,out=Path(run),Path(output_dir);out.mkdir(exist_ok=True)
    path=out/'all_type_marker_context_responses.csv.gz';complete=out/'marker_context_complete.json'
    config=dict(marker_genes=list(markers),logit_clip=1e-6,rank='percentiles among expressed genes from raw probabilities',min_rank_genes=10)
    if complete.exists():
        if json.loads(complete.read_text())!=config:raise ValueError('Marker context cache differs')
        return pd.read_csv(path)
    frames=[]
    for entry in json.loads((run/'design.json').read_text())['slices']:
        with h5py.File(entry['predictions']) as f:
            obs=pd.DataFrame({k:_column(f['obs/'+k]) for k in ['focal_cell_id','context_source_cell_id','context_kind','query_cell_type']})
            names=_column(f['var/feature_name']);marker_mask=np.isin(names,markers)
            raw=_matrix(f['X']);p=_matrix(f['layers/admix_probs']);p=p.toarray() if hasattr(p,'toarray') else np.asarray(p)
            pm=np.clip(p[:,marker_mask].astype(float),1e-6,1-1e-6)
            logits=np.log(pm)-np.log1p(-pm)
            rm=raw[:,marker_mask];rm=rm.toarray() if hasattr(rm,'toarray') else np.asarray(rm)
            expressed=rm>0;number=expressed.sum(axis=1)
            scores=np.divide((logits*expressed).sum(axis=1),number,out=np.full(len(obs),np.nan),where=number>0)
            section_rows=[]
            for q,group in obs.groupby('focal_cell_id',sort=False,observed=True):
                ix=group.index.to_numpy();base=group.index[group.context_kind.eq('original')].to_numpy()
                if len(base)!=1:raise ValueError('Expected one original context')
                b=int(base[0]);rr=raw[b];genes=np.flatnonzero((rr.toarray().ravel() if hasattr(rr,'toarray') else np.asarray(rr).ravel())>0)
                rank_score=np.full(len(ix),np.nan)
                marker_columns=np.isin(names[genes],markers)
                if len(genes)>=10 and marker_columns.any():
                    ranks=rankdata(p[ix][:,genes],axis=1,method='average')
                    rank_score=((ranks[:,marker_columns]-1)/(len(genes)-1)).mean(axis=1)
                local_base=int(np.flatnonzero(ix==b)[0])
                section_rows.append(pd.DataFrame(dict(slice=entry['slice'],query_cell_id=str(q),
                    query_cell_type=str(obs.loc[b,'query_cell_type']),context_source_cell_id=group.context_source_cell_id.to_numpy(),
                    context_kind=group.context_kind.to_numpy(),n_expressed_markers=int(number[b]),n_expressed_genes=len(genes),
                    delta_marker_logit=scores[ix]-scores[b],delta_marker_rank_percentile=rank_score-rank_score[local_base])))
        frame=pd.concat(section_rows,ignore_index=True);frames.append(frame)
        print('All-type marker-context responses:',entry['label'],len(frame),'rows',flush=True)
    result=pd.concat(frames,ignore_index=True)
    controls=result.loc[result.context_kind.eq('unchanged_control')]
    for col in ['delta_marker_logit','delta_marker_rank_percentile']:
        if not np.allclose(controls[col].dropna(),0,atol=1e-12):raise ValueError('Unchanged controls differ')
    result.to_csv(path,index=False,compression='gzip');complete.write_text(json.dumps(config,indent=2)+'\n')
    return result
