"""Evaluate unchanged method matrices on shared Syndri-defined cell cohorts."""
from __future__ import annotations
import argparse,copy,gc,hashlib,json,os,sys
from pathlib import Path
import h5py
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.insert(0,'/home/jwinchel/Syndri/eval')
import metrics_eval as ev

FRACTIONS=[.01,.02,.05,.1,.2,.3,.4,.5]

def metadata(path):
    with h5py.File(path) as f:
        return ad.io.read_elem(f['obs']),ad.io.read_elem(f['var'])

def matrix(path,layer=None):
    with h5py.File(path) as f:
        x=ad.io.read_elem(f['X'] if layer is None else f['layers'][layer])
    x=sparse.csr_matrix(x)
    if not np.isfinite(x.data).all() or (x.data<0).any():raise ValueError(f'Invalid counts: {path}:{layer}')
    return x

def signature(path):
    s=Path(path).stat()
    return dict(path=str(path),size=s.st_size,mtime_ns=s.st_mtime_ns)

def configured(config):
    parser=ev.build_parser()
    known={a.dest for a in parser._actions}
    parser.set_defaults(**{k:v for k,v in config.items() if k in known})
    args=parser.parse_args([])
    args.n_jobs=int(os.environ.get('SLURM_CPUS_PER_TASK',4))
    args.separation_min_counts=0;args.separation_min_genes=0
    return args

def cohort_mask(raw,prob,obs,args,qc_fraction):
    labels=obs[args.label_key]
    base=labels.notna().to_numpy() & ~labels.astype(str).isin(ev.parse_str_list(args.exclude_labels)).to_numpy()
    ids=np.flatnonzero(base)
    if qc_fraction is None:return ids,dict(n_annotated=len(ids),n_pass=len(ids))
    removal_labels=obs.iloc[ids][args.removal_group_key or args.label_key]
    cleaned=ev.cleaned_counts_for_removed_fraction_maybe_grouped(raw[ids],prob[ids],qc_fraction,args,labels=removal_labels)[0]
    cleaned=sparse.csr_matrix(cleaned)
    keep=(np.asarray(cleaned.sum(1)).ravel()>=10)&(np.asarray((cleaned>0).sum(1)).ravel()>=5)
    return ids[keep],dict(n_annotated=len(ids),n_pass=int(keep.sum()),n_fail=int((~keep).sum()))

def run(record,qc_fraction,family,prepare_only=False):
    source=Path(record['input']);obs,var=metadata(source)
    if not obs.index.is_unique or not var.index.is_unique:raise ValueError('Nonunique source IDs')
    config=json.loads(Path(record['configs']['non_tcell']).read_text())
    args=configured(config)
    # Match the standard Syndri cleaning sweep, without recalibrating after cohort selection.
    args.cleaning_mode='probability_scale';args.probability_transform='power';args.probability_power=2.
    args.clean_prob_threshold=.5;args.cell_type_specific_removal=True
    args.removal_group_key=args.removal_group_key or args.label_key
    raw=matrix(source);prob=matrix(source,'syndri_admix_probs_X')
    idx,audit=cohort_mask(raw,prob,obs,args,qc_fraction)
    cohort='all_cells' if qc_fraction is None else f'syndri_qc_frac_{qc_fraction:g}'
    out=Path(record['output'])/family/cohort/source.stem
    out.mkdir(parents=True,exist_ok=False)
    ids=obs.index[idx]
    cohort_sha=hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest()
    pd.DataFrame({'cell_id':ids}).to_csv(out/'cells.csv.gz',index=False)
    common=dict(dataset=record['dataset'],slice=source.stem,cohort=cohort,
                qc_target_fraction=qc_fraction,cohort_sha256=cohort_sha,n_cohort_cells=len(idx))
    provenance=dict(**common,**audit,settings=vars(args),source=signature(source),
                    min_counts=10 if qc_fraction is not None else 0,
                    min_genes=5 if qc_fraction is not None else 0,
                    cleaning_calibration='full annotated slice, before QC selection',
                    matrices='each method original matrix; only subset rows',family=family)
    if family=='tcell':
        targs=configured(json.loads(Path(record['configs']['tcell']).read_text()))
        targs.only_additional_counts=True;targs.additional_counts_layer='_shared_method_counts'
        ref=ad.read_h5ad(targs.reference_data,backed='r') if targs.reference_data else None
        reference_cache={}
        provenance['tcell_settings']=vars(targs)
        check=ad.AnnData(X=raw[idx].copy(),obs=obs.iloc[idx].copy(),var=var.copy())
        ev.ensure_tcell_inputs(check,source.stem,targs)
        if targs.filter_tcell_invalid_cells:
            valid_check,dropped=ev.filtered_tcell_adata(check,targs)
        else:
            valid_check,dropped=check,{}
        pd.DataFrame({'cell_id':valid_check.obs_names}).to_csv(out/'tcell_valid_cells.csv.gz',index=False)
        provenance['tcell_invalid_dropped']=dropped
        provenance['n_tcell_valid_cells']=valid_check.n_obs
        del check,valid_check
    results=[];status=[];tframes=[[],[],[],[]]
    def evaluate(name,counts,fraction=None,origin=None,n_unscored=0):
        if counts.shape!=(len(idx),len(var)):raise ValueError(f'{name}: incorrect aligned shape {counts.shape}')
        info=dict(**common,method=name,target_removed_fraction=fraction,matrix_source=str(origin or source),
                  n_resolvi_unscored=n_unscored)
        if prepare_only:
            row=dict(**info,total_counts=float(counts.sum()),n_cells=counts.shape[0],error='')
            results.append(row)
        elif family=='non_tcell':
            case='admixed' if name=='Raw' else (f'cleaned_admixed_removed_frac_{fraction:g}' if name=='Syndri' else name)
            row=ev.score_case(source.stem,str(source),case,counts,obs.iloc[idx][args.label_key],args,
                              target_removed_fraction=fraction if fraction is not None else np.nan)
            row.update(info);results.append(row)
            if row.get('error'):status.append(dict(**info,status='error',error=row['error']))
        else:
            local=copy.copy(targs)
            label='raw_X' if name=='Raw' else (f'cleaned_admixed_removed_frac_{fraction:g}' if name=='Syndri' else name)
            local.additional_counts_case_label=label;local.purified_condition_label=label
            a=ad.AnnData(X=raw[idx].copy(),obs=obs.iloc[idx].copy(),var=var.copy())
            a.layers[local.admix_probs_layer]=prob[idx].copy()
            a.layers['_shared_method_counts']=counts
            frames=ev.run_tcell_contamination(a,source.stem,str(source),local,
                    reference_adata=ref,reference_profile_cache=reference_cache)
            for collection,frame in zip(tframes,frames):
                if len(frame):
                    for key,value in info.items():frame[key]=value
                    collection.append(frame)
                    if 'error' in frame and frame.error.fillna('').astype(str).ne('').any():
                        status.append(dict(**info,status='error',error='; '.join(frame.error.dropna().astype(str).unique())))
            del a
        print(cohort,family,name,fraction,'cells',len(idx),flush=True)
    evaluate('Raw',raw[idx])
    evaluate('SPLIT',matrix(source,'split_purified_counts_X')[idx])
    base=obs[args.label_key].notna().to_numpy() & ~obs[args.label_key].astype(str).isin(ev.parse_str_list(args.exclude_labels)).to_numpy()
    baseidx=np.flatnonzero(base)
    positions=pd.Index(baseidx).get_indexer(idx)
    for fraction in (FRACTIONS if qc_fraction is None else [qc_fraction]):
        cleaned=ev.cleaned_counts_for_removed_fraction_maybe_grouped(raw[baseidx],prob[baseidx],fraction,args,
                  labels=obs.iloc[baseidx][args.removal_group_key])[0]
        evaluate('Syndri',sparse.csr_matrix(cleaned)[positions],fraction)
        del cleaned;gc.collect()
    for method,external in record['external'].items():
        path=Path(external['path'])
        if not path.exists():
            status.append(dict(**common,method=method,status='unavailable',error='Missing corrected matrix',path=str(path)))
            continue
        before=signature(path)
        eo,evr=metadata(path)
        if not eo.index.is_unique or not evr.index.is_unique:raise ValueError(f'Duplicate IDs in {path}')
        ri=eo.index.get_indexer(ids);ci=evr.index.get_indexer(var.index)
        if (ri<0).any() or (ci<0).any():
            status.append(dict(**common,method=method,status='unavailable',error='Missing cohort cells or panel genes',
                               n_missing_cells=int((ri<0).sum()),n_missing_genes=int((ci<0).sum()),path=str(path)))
            continue
        counts=matrix(path,external['layer'])[ri][:,ci]
        if before!=signature(path):raise RuntimeError(f'Input replaced during read: {path}')
        unscored=int((~eo.iloc[ri].resolvi_scored.astype(bool)).sum()) if 'resolvi_scored' in eo and method=='ResolVI' else 0
        # Preserve the exported matrix, including any documented raw fallback rows.
        evaluate(method,counts,origin=path,n_unscored=unscored)
        provenance.setdefault('external_signatures',{})[method]=before
        del counts;gc.collect()
    if prepare_only or family=='non_tcell':pd.DataFrame(results).to_csv(out/'non_tcell_metrics.csv',index=False)
    else:
        for name,frames in zip(['tcell_contamination_metrics.csv','tcell_lr_top_genes.csv',
                               'tissue_cell_type_profiles.csv','tcell_cosine_similarity.csv'],tframes):
            frame=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
            frame.to_csv(out/name,index=False)
            if name=='tcell_contamination_metrics.csv':
                ev.summarize_tcell_effects(frame,targs).to_csv(out/'tcell_contamination_effects.csv',index=False)
        if ref is not None:ref.file.close()
    pd.DataFrame(status,columns=list(dict.fromkeys([*common,'method','status','error','path','n_missing_cells','n_missing_genes','target_removed_fraction']))).to_csv(out/'status.csv',index=False)
    (out/'config.json').write_text(json.dumps(provenance,indent=2,default=str))
    if any(s['status']=='error' for s in status):raise RuntimeError(f'Metric errors recorded in {out}')
    (out/'complete.json').write_text(json.dumps(dict(**common,prepare_only=prepare_only),indent=2))
    print('Complete:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('task',type=int)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    task=json.loads(a.manifest.read_text())[a.task]
    run(task['record'],task['qc_fraction'],task['family'],a.prepare_only)
