"""Standalone Raw/Syndri figure panels, with data tables saved beside plots."""
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from . import ROOT
from .io import config, data_dir, dataset, read, predicted_file, spatial_files, qc, valid_labels, symbols, csr
from .metrics import matrices, normalize_log, adjacency

COLORS = {'Raw': '#4c72b0', 'Syndri': '#dd8452', 'Syndri (binarized)': '#f2c5a5'}
PANEL_DIR = ROOT/'outputs/panels'
TABLE_DIR = ROOT/'outputs/tables'


def style():
    PANEL_DIR.mkdir(parents=True, exist_ok=True); TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='white', context='paper')
    plt.rcParams.update({'font.size':8, 'axes.labelsize':8, 'xtick.labelsize':8,
        'ytick.labelsize':8, 'axes.titlesize':10, 'legend.fontsize':8,
        'svg.fonttype':'none', 'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False})


def save(fig, stem):
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(PANEL_DIR/f'{stem}.{suffix}', dpi=600, bbox_inches='tight')
    plt.close(fig)


def figure1(recompute=False):
    style()
    composition = pd.read_csv(ROOT/'assets/localization/cell_transcript_composition.csv')
    distances = pd.read_parquet(ROOT/'assets/localization/transcript_boundary_distances.parquet')
    inside = distances[distances['inside_cell_polygon']]
    summary = inside.groupby(['slice_id','cell_id','gene_annot'], observed=True).agg(
        mean_normalized_distance=('normalized_distance','mean'),
        n_transcripts=('normalized_distance','size')).reset_index()
    summary['Transcript Type'] = summary.gene_annot.map({'t-cell marker':'T-cell','tumor marker':'Tumor'})
    summary.to_csv(TABLE_DIR/'figure1_localization.csv', index=False)
    tumor = composition[composition.gene_annot == 'tumor marker']
    fig, ax = plt.subplots(figsize=(2.3,2.0))
    sns.histplot(tumor, x='transcript_proportion', bins=np.linspace(0,1,51), stat='probability',
                 color='#999999', ax=ax)
    ax.set(xlabel='Fraction of tumor transcripts per T cell', ylabel='Fraction of T cells', title='Transcript prevalence')
    save(fig, 'figure1b_prevalence')
    fig, ax = plt.subplots(figsize=(2.3,2.0))
    sns.histplot(summary, x='mean_normalized_distance', hue='Transcript Type', hue_order=['T-cell','Tumor'],
        palette=['#999999','#e41a1c'], bins=np.linspace(0,1,51), stat='probability',
        common_norm=False, element='step', fill=False, ax=ax)
    ax.set(xlabel='Mean distance to boundary / cell radius', ylabel='Fraction of T cells', title='Transcript localization')
    save(fig, 'figure1b_localization')


def embedding(template, counts):
    obj = ad.AnnData(csr(counts), obs=template.obs.copy(), var=template.var.copy())
    sc.pp.normalize_total(obj, target_sum=10000); sc.pp.log1p(obj)
    n_pcs = min(50, min(obj.shape)-1)
    sc.pp.pca(obj, n_comps=n_pcs, random_state=42)
    sc.pp.neighbors(obj, n_neighbors=min(15, obj.n_obs-1), n_pcs=n_pcs, random_state=42)
    sc.tl.umap(obj, min_dist=.4, random_state=42)
    return obj.obsm['X_umap']


def figure2(recompute=False, include_metrics=True):
    style(); name='BX'; cfg=dataset(name)
    reference=read(data_dir(name)/'reference.h5ad')
    source=data_dir(name)/'spatial'/f'{cfg["representative"]}.h5ad'
    obj=read(predicted_file(name, source), layers=['admix_probs'])
    shared=obj.var_names.intersection(reference.var_names, sort=False)
    obj=obj[:,shared].copy();reference=reference[:,shared].copy()
    raw_clean=matrices(obj,name)
    keep=valid_labels(obj.obs[cfg['label']]) & qc(raw_clean['Raw']) & qc(raw_clean['Syndri'])
    panels=[('Reference',reference,reference.X,cfg['reference_label'])]
    panels.extend((method,obj[keep].copy(),counts[keep],cfg['label']) for method,counts in raw_clean.items())
    labels=sorted(set().union(*(set(item[1].obs[item[3]].astype(str)) for item in panels)))
    palette=dict(zip(labels,sns.color_palette('tab20',len(labels))))
    fig,axes=plt.subplots(1,3,figsize=(7.35,2.2))
    for ax,(method,template,counts,key) in zip(axes,panels):
        cache=TABLE_DIR/f'figure2_umap_{method}.csv'
        if cache.exists() and not recompute:
            table=pd.read_csv(cache,index_col=0)
            if not table.index.equals(template.obs_names):
                raise ValueError('UMAP cache cell IDs changed; rerun with --recompute')
            xy=table[['UMAP1','UMAP2']].to_numpy()
        else:
            xy=embedding(template,counts)
            pd.DataFrame({'UMAP1':xy[:,0],'UMAP2':xy[:,1],'cell_type':template.obs[key].astype(str).to_numpy()},
                         index=template.obs_names).to_csv(cache)
        colors=[palette[label] for label in template.obs[key].astype(str)]
        ax.scatter(xy[:,0],xy[:,1],c=colors,s=1,linewidths=0,rasterized=True)
        ax.set_title(method);ax.set_axis_off()
        single,one=plt.subplots(figsize=(2.25,2.25))
        one.scatter(xy[:,0],xy[:,1],c=colors,s=1,linewidths=0,rasterized=True)
        one.set_axis_off();single.subplots_adjust(0,0,1,1)
        single.savefig(PANEL_DIR/f'figure2a_{method}_scatter.png',dpi=900,transparent=True)
        plt.close(single)
    fig.tight_layout();save(fig,'figure2a_umaps')
    fig,ax=plt.subplots(figsize=(3, max(1.,len(labels)*.15)));ax.set_axis_off()
    handles=[Line2D([],[],ls='',marker='o',ms=4,color=palette[label],label=label) for label in labels]
    ax.legend(handles=handles,loc='center',frameon=False,ncol=1);save(fig,'figure2a_legend')
    if not include_metrics:return
    files=[p for name in ['BM','NC','BX'] for p in (ROOT/'outputs/metrics'/name).glob('*.csv') if not p.stem.endswith('_celltypes')]
    if not files:
        raise FileNotFoundError('Run evaluate for BM, NC and BX first, or use --skip-metrics for UMAP-only export')
    table=pd.concat([pd.read_csv(path) for path in files],ignore_index=True)
    table.to_csv(TABLE_DIR/'figure2_metrics.csv',index=False)
    metrics=['ARI','NMI','Silhouette','reference_cosine','variance_spearman']
    fig,axes=plt.subplots(1,len(metrics),figsize=(7.35,2.0),sharex=True)
    for ax,key in zip(axes,metrics):
        sns.barplot(table,x='dataset',y=key,hue='method',hue_order=['Raw','Syndri'],palette=COLORS,errorbar='se',ax=ax)
        ax.set(xlabel='',ylabel='',title={'reference_cosine':'Reference cosine','variance_spearman':'Variance rho'}.get(key,key))
        if ax.get_legend():ax.get_legend().remove()
    fig.legend(handles=[plt.Rectangle((0,0),1,1,color=COLORS[x],label=x) for x in ['Raw','Syndri']],
               loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.1,1,1]);save(fig,'figure2b_raw_syndri_metrics')
    fig,axes=plt.subplots(2,1,figsize=(2.4,3.2),sharex=True)
    for ax,key,title in zip(axes,['median_genes','median_transcripts'],['Detected genes per cell','Transcripts per cell']):
        sns.barplot(table,x='dataset',y=key,hue='method',palette=COLORS,errorbar='se',ax=ax)
        ax.set(xlabel='',ylabel='Median',title=title)
        if ax.get_legend():ax.get_legend().remove()
    fig.tight_layout();save(fig,'figure2c_counts')


def _ordered_types(sets):
    common=set.intersection(*sets)
    order=[ct for family in config()['families'].values() for ct in family if ct in common]
    return order + sorted(common-set(order))


def _matrix_plot(matrix, labels, title, stem, cmap='RdBu_r', limit=4):
    fig,ax=plt.subplots(figsize=(3.5,3.0))
    image=ax.imshow(matrix,cmap=cmap,vmin=-limit,vmax=limit,interpolation='nearest')
    ax.set_xticks(range(len(labels)),labels,rotation=90)
    ax.set_yticks(range(len(labels)),labels)
    ax.set(title=title,xlabel='Neighbor cell type',ylabel='Query cell type')
    fig.colorbar(image,ax=ax,label='log2 observed / expected',shrink=.65)
    fig.tight_layout();save(fig,stem)


def figure3(recompute=False):
    from .simulation import mmd_table, pr_table
    style();table=mmd_table(recompute)
    long=table.melt(id_vars=['slice','cell_type','seed','n_query'],value_vars=['mmd2_before','mmd2_after'],
                    var_name='condition',value_name='MMD2')
    long['condition']=long.condition.map({'mmd2_before':'Before','mmd2_after':'After'})
    # Match notebook boxplots: average over simulation seeds within each type/slice.
    box=long.groupby(['slice','cell_type','condition'],as_index=False).MMD2.mean()
    fig,ax=plt.subplots(figsize=(2.5,2.2))
    sns.boxplot(box,x='condition',y='MMD2',hue='slice',order=['Before','After'],width=.8,gap=.05,fliersize=1,ax=ax)
    ax.set(xlabel='',ylabel='RBF MMD²');ax.legend(fontsize=7,frameon=False)
    fig.tight_layout();save(fig,'figure3a_mmd')
    sl=config()['simulation']['slice']
    mapped=read(data_dir('SIM')/'training.h5ad')
    mapped=mapped[mapped.obs['spatial_slice'].astype(str)==sl].copy()
    real=read(data_dir('NC')/'spatial'/f'{sl}.h5ad')
    reference=read(data_dir('SIM')/'reference.h5ad')
    order=_ordered_types([set(mapped.obs.cell_type.astype(str)),set(real.obs.cell_type_harmonized.astype(str)),
                          set(reference.obs.cell_type.astype(str))])
    real_adj=adjacency(real,'cell_type_harmonized',dataset('NC')['coords'],order)
    sim_adj=adjacency(mapped,'cell_type',dataset('SIM')['coords'],order)
    for label,values in [('Real',real_adj),('Simulated',sim_adj)]:
        pd.DataFrame(values,index=order,columns=order).to_csv(TABLE_DIR/f'figure3_adjacency_{label}.csv')
        _matrix_plot(values,order,label,f'figure3b_adjacency_{label}')
    # Pair the same directed statistic in the two triangles. For i<j, the
    # forward view shows sim[i,j] at (i,j) and real[i,j] at (j,i).
    for reverse in [False,True]:
        a,b=(real_adj.T,sim_adj.T) if reverse else (real_adj,sim_adj)
        values=np.zeros_like(a)
        i,j=np.triu_indices(len(order),1);values[i,j]=b[i,j];values[j,i]=a[i,j]
        np.fill_diagonal(values,(np.diag(a)+np.diag(b))/2)
        suffix='reverse' if reverse else 'forward'
        pd.DataFrame(values,index=order,columns=order).to_csv(TABLE_DIR/f'figure3_adjacency_{suffix}.csv')
        _matrix_plot(values,order,f'{suffix.capitalize()} direction',f'figure3b_adjacency_{suffix}')
    shared=set(reference.var_names)&set(mapped.var_names)&set(real.var_names)
    markers=pd.read_csv(ROOT/'assets/markers/cosmx_deseq2.csv').dropna(subset=['cell_type','gene','padj'])
    markers=markers.sort_values(['cell_type','padj','log2FoldChange'],ascending=[True,True,False],kind='stable')
    selected=[];centers=[];gene_records=[]
    alias=dict(zip(reference.var_names,symbols(reference)))
    for ct in order:
        genes=list(dict.fromkeys(g for g in markers.loc[markers.cell_type.astype(str)==ct,'gene'].astype(str)
                                  if g in shared and alias.get(g,'') not in {'','nan'}))[:3]
        if len(genes)!=3:raise ValueError(f'{ct}: three shared marker genes required')
        centers.append(len(selected)+1);selected.extend(genes)
        gene_records.extend(dict(cell_type=ct,gene=g,symbol=alias[g]) for g in genes)
    pd.DataFrame(gene_records).to_csv(TABLE_DIR/'figure3_marker_genes.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(7.35,2.9))
    for ax,obj,title in zip(axes,[reference,mapped,real],['scRNA-seq reference','Simulated ST','Real ST']):
        # All cells contribute to every gene pair, regardless of their labels.
        detection=(obj[:,selected].X>0).astype(np.float64)
        values=(detection.T@detection).toarray()/obj.n_obs
        pd.DataFrame(values,index=selected,columns=selected).to_csv(TABLE_DIR/f'figure3_codetection_{title.replace(" ","_")}.csv')
        image=ax.imshow(values,cmap='Blues',vmin=0,vmax=.1)
        ax.set_xticks(centers,order,rotation=90);ax.set_yticks(centers,order if ax is axes[0] else [])
        ax.set_title(title)
    fig.colorbar(image,ax=axes.tolist(),label='Co-detection fraction',shrink=.55)
    fig.subplots_adjust(left=.14,right=.86,bottom=.4,wspace=.12);save(fig,'figure3c_codetection')
    curves,summary=pr_table(recompute)
    fig,ax=plt.subplots(figsize=(2.5,2.2))
    for method,group in curves.groupby('method',sort=False):
        ax.plot(group.recall,group.precision,label=method,color=COLORS[method],lw=1)
        if method=='Syndri':
            best=summary[summary.method==method].iloc[0]
            ax.scatter(best.best_recall,best.best_precision,s=12,color=COLORS[method])
            ax.annotate(f'F1 = {best.best_f1:.2f}',(best.best_recall,best.best_precision),xytext=(4,4),textcoords='offset points',fontsize=8)
    baseline=float(summary.positive_prevalence.iloc[0])
    ax.axhline(baseline,color='gray',ls='--',lw=.7,label=f'Positive prevalence ({baseline:.2f})')
    ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));ax.legend(frameon=False,fontsize=7)
    fig.tight_layout();save(fig,'figure3d_pr')
    fig,ax=plt.subplots(figsize=(2.5,1.4))
    sorted_summary=summary.sort_values('average_precision')
    ax.barh(sorted_summary.method,sorted_summary.average_precision,color=[COLORS[x] for x in sorted_summary.method])
    ax.set(xlabel='Average precision (AUPRC)',xlim=(0,1))
    fig.tight_layout();save(fig,'figure3d_auprc')
    removed=pd.read_csv(TABLE_DIR/'figure3_removed_counts.csv')
    removed['count_class']=removed.count_class.map({'original_counts':'Native counts','synthetic_admixture_counts':'Known admixture'})
    fig,ax=plt.subplots(figsize=(2.5,2.2))
    sns.barplot(removed,x='count_class',y='fraction_removed',errorbar='se',color=COLORS['Syndri'],ax=ax)
    ax.set(xlabel='',ylabel='Fraction removed',ylim=(0,1));ax.tick_params(axis='x',rotation=15)
    fig.tight_layout();save(fig,'figure3_removed_counts')


def figure4(recompute=False):
    style();cfg=config()['fibroblasts'];sources=[]
    for sl in cfg['slices']:
        source=data_dir('NC')/'spatial'/f'{sl}.h5ad'
        sources.append(read(predicted_file('NC',source),layers=['admix_probs']))
    shared=sources[0].var_names
    for obj in sources[1:]:shared=shared.intersection(obj.var_names,sort=False)
    raw_blocks=[];clean_blocks=[];regions=[];members=[]
    for sl,obj in zip(cfg['slices'],sources):
        counts=matrices(obj,'NC')
        region=obj.obs[cfg['region_column']].map(cfg['regions'])
        keep=(obj.obs.cell_type_harmonized==cfg['label']).to_numpy() & region.notna().to_numpy()
        keep &= qc(counts['Raw']) & qc(counts['Syndri'])
        cols=obj.var_names.get_indexer(shared)
        raw_blocks.append(counts['Raw'][keep][:,cols]);clean_blocks.append(counts['Syndri'][keep][:,cols])
        regions.extend(region[keep]);members.extend(dict(slice=sl,cell_id=cell) for cell in obj.obs_names[keep])
    raw=sparse.vstack(raw_blocks,format='csr');clean=sparse.vstack(clean_blocks,format='csr')
    regions=np.asarray(regions);gene_symbols=symbols(sources[0])[sources[0].var_names.get_indexer(shared)]
    if pd.Index(gene_symbols).has_duplicates:raise ValueError('Gene aliases must be unique for DE')
    if min((regions=='tumor').sum(),(regions=='stroma').sum())<30:
        raise ValueError('At least 30 tumor and stromal fibroblasts are required')
    pd.DataFrame(members).assign(region=regions).to_csv(TABLE_DIR/'figure4_cells.csv',index=False)
    removal=[]
    for category in ['tumor','macrophage','native']:
        cols=np.isin(gene_symbols,cfg[category]);total=float(raw[:,cols].sum())
        removal.append(dict(category=category,raw_count=total,corrected_count=float(clean[:,cols].sum()),
                            fraction_removed=1-float(clean[:,cols].sum())/total if total else np.nan))
    table=pd.DataFrame(removal);table.to_csv(TABLE_DIR/'figure4_marker_removal.csv',index=False)
    fig,ax=plt.subplots(figsize=(2,1.7));sns.heatmap(table.set_index('category')[['fraction_removed']],
        annot=True,fmt='.2f',cmap='RdBu_r',center=0,vmin=-1,vmax=1,ax=ax,
        cbar_kws={'label':'Fraction of Raw counts removed'})
    ax.set_xticklabels(['Syndri']);ax.set(xlabel='',ylabel='Marker category')
    fig.tight_layout();save(fig,'figure4a_removal')
    results=[]
    for method,counts in [('Raw',raw),('Syndri',clean)]:
        dest=TABLE_DIR/f'figure4_de_{method}.csv'
        if dest.exists() and not recompute:
            de=pd.read_csv(dest)
        else:
            obj=ad.AnnData(counts.copy(),obs=pd.DataFrame({'region':pd.Categorical(regions)}),
                           var=pd.DataFrame(index=gene_symbols))
            sc.pp.normalize_total(obj,target_sum=10000);sc.pp.log1p(obj)
            sc.tl.rank_genes_groups(obj,groupby='region',groups=['tumor'],reference='stroma',method='wilcoxon',
                                   tie_correct=True,corr_method='benjamini-hochberg')
            de=sc.get.rank_genes_groups_df(obj,group='tumor');de.to_csv(dest,index=False)
        de['method']=method;results.append(de)
    fig,axes=plt.subplots(1,2,figsize=(4.8,2.2),sharey=True)
    marker_colors={'tumor':'#8b0000','macrophage':'#2676b8','other':'#bdbdbd'}
    for ax,de,method in zip(axes,results,['Raw','Syndri']):
        category=np.where(de.names.isin(cfg['tumor']),'tumor',np.where(de.names.isin(cfg['macrophage']),'macrophage','other'))
        x=np.clip(de.logfoldchanges.to_numpy(),-5,5);y=-np.log10(np.clip(de.pvals_adj,1e-30,1))
        ax.scatter(x,y,c=[marker_colors[c] for c in category],s=3,alpha=.7,rasterized=True)
        for cutoff in [-np.log2(1.5),np.log2(1.5)]:ax.axvline(cutoff,c='red',ls='--',lw=.5)
        ax.axhline(-np.log10(.05),c='red',ls='--',lw=.5)
        ax.set(xlabel='LFC tumor vs stroma',title=method)
        if method=='Raw':ax.set_ylabel('−log10 BH-adjusted P')
        if method=='Syndri':_annotate(ax,de,cfg,x,y)
    fig.tight_layout();save(fig,'figure4b_volcano')
    joined=results[0].set_index('names')[['scores']].join(results[1].set_index('names')[['scores']],lsuffix='_raw',rsuffix='_clean')
    joined.to_csv(TABLE_DIR/'figure4_z.csv')
    fig,ax=plt.subplots(figsize=(2.4,2.2))
    cat=np.where(joined.index.isin(cfg['tumor']),'tumor',np.where(joined.index.isin(cfg['macrophage']),'macrophage','other'))
    ax.scatter(joined.scores_raw,joined.scores_clean,c=[marker_colors[c] for c in cat],s=4,alpha=.7,rasterized=True)
    bound=max(np.abs(joined.to_numpy()).max(),1);ax.plot([-bound,bound],[-bound,bound],color='gray',ls='--',lw=.5)
    ax.set(xlabel='Z (original)',ylabel='Z (cleaned)',title='Syndri')
    de=results[1].copy();coordinates=joined.reindex(de.names)
    _annotate(ax,de,cfg,coordinates.scores_raw.to_numpy(),coordinates.scores_clean.to_numpy())
    fig.tight_layout();save(fig,'figure4c_z')


def _annotate(ax,de,cfg,x,y):
    for side,key,color in [('left','stromal_region','#287a69'),('right','tumor_region','#b23a48')]:
        for rank,gene in enumerate(cfg[key]):
            matches=np.flatnonzero(de.names.to_numpy()==gene)
            if len(matches):
                i=matches[0]
                ax.annotate(gene,(x[i],y[i]),xytext=(.02 if side=='left' else .98,.96-rank*.075),
                    textcoords='axes fraction',ha=side,va='top',fontsize=7,color=color,
                    arrowprops={'arrowstyle':'-','color':color,'lw':.4})

