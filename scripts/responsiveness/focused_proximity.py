"""Focused distance/sensitivity panels with all sections and explicit support."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def focused_panel(query_type,query_data,correlations,slice_order,slice_labels,palette,
                  edges=(0,10,25,50,100,200,np.inf),minimum_bin_queries=3, marker_label="Epithelial-marker"):
    metric='malignant_marker_score_sd'
    cells=query_data.loc[query_data.query_cell_type.eq(query_type)].dropna(subset=['query_distance_um',metric]).copy()
    if cells.empty:return None, cells, pd.DataFrame(), pd.DataFrame(), {'status':'no_expressed_markers'}
    cells['distance_bin']=pd.cut(cells.query_distance_um,edges,right=False,include_lowest=True)
    bins=cells.groupby(['slice','distance_bin'],observed=True).agg(mean_distance_um=('query_distance_um','mean'),
           median_responsiveness=(metric,'median'),n_queries=('query_cell_id','nunique')).reset_index()
    bins=bins.loc[bins.n_queries>=minimum_bin_queries]
    rho=correlations.loc[correlations.query_cell_type.eq(query_type)&correlations.metric.eq(metric)].set_index('slice').reindex(slice_order).reset_index()
    supported=rho.loc[rho.status.eq('ok')];median=supported.spearman_rho.median();negative=int(supported.spearman_rho.lt(0).sum())
    fig,(left,right)=plt.subplots(1,2,figsize=(10.5,4.3),gridspec_kw={'width_ratios':[1.8,1]})
    for sid in slice_order:
        cell=cells[cells['slice'].eq(sid)];line=bins[bins['slice'].eq(sid)].sort_values('mean_distance_um')
        left.scatter(cell.query_distance_um,cell[metric],color=palette[sid],s=17,alpha=.3,edgecolors='none')
        if not line.empty:left.plot(line.mean_distance_um,line.median_responsiveness,color=palette[sid],marker='o',markersize=4,linewidth=1.4)
    left.set_xscale('symlog',linthresh=10);left.set_xlim(left=0);left.set_ylim(bottom=0)
    left.set(xlabel='Query distance to nearest malignant cell (µm)',ylabel=f'{marker_label} context sensitivity\n(logit SD across matched contexts)',title='A  Actual query proximity')
    left.text(.98,.98,f'{len(cells)} marker-supported queries; {cells["slice"].nunique()} sections',ha='right',va='top',transform=left.transAxes,fontsize=8)
    b1=[i for i,sid in enumerate(slice_order) if slice_labels[sid].startswith('B1_')]
    if b1:right.axhspan(min(b1)-.45,max(b1)+.45,color='0.93',zorder=0)
    right.axvline(0,color='.4',linestyle='--',linewidth=.9)
    if pd.notna(median):right.axvline(median,color='.65',linestyle=':',linewidth=1)
    labels=[]
    for i,sid in enumerate(slice_order):
        row=rho.loc[rho['slice'].eq(sid)].iloc[0];n=int(row.n_queries) if pd.notna(row.n_queries) else 0
        labels.append(f'{slice_labels[sid]} (n={n})')
        if row.status=='ok' and pd.notna(row.spearman_rho):right.scatter(row.spearman_rho,i,color=palette[sid],s=42,marker='o' if i in b1 else 's',zorder=3)
        else:right.text(.02,i,'Insufficient support',ha='left',va='center',fontsize=7)
    right.set_yticks(range(len(slice_order)),labels);right.set_ylim(len(slice_order)-.5,-.5);right.set_xlim(-1,1)
    right.set(xlabel='Within-section Spearman ρ',title='B  Consistency across sections')
    right.text(.02,-.21,f'{negative}/{len(supported)} supported sections: ρ < 0\nMedian ρ = {median:.2f}; shaded rows = B1 replicates',transform=right.transAxes,ha='left',va='top',fontsize=8)
    fig.suptitle(f'{query_type.capitalize()}: proximity and {marker_label.lower()} context sensitivity',fontsize=11,y=1.01)
    handles=[plt.Line2D([0],[0],marker='o',linestyle='',markersize=5,color=palette[sid],label=slice_labels[sid]) for sid in slice_order if sid in set(cells['slice'])]
    fig.legend(handles=handles,title='Section',loc='lower center',ncol=8,bbox_to_anchor=(.36,-.12),frameon=False)
    fig.tight_layout()
    return fig,cells,bins,rho,dict(status='ok',n_marker_supported_queries=len(cells),n_supported_sections=len(supported),n_negative_sections=negative,median_rho=median)


def export_focused(fig,cells,bins,rho,output,query_type):
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    tag=query_type.replace(' ','_');index=1
    suffixes=['.pdf','.png','_cells.csv','_bins.csv','_correlations.csv']
    while any((output/f'focused_{tag}_distance_{index}{suffix}').exists() for suffix in suffixes):index+=1
    stem=output/f'focused_{tag}_distance_{index}'
    fig.savefig(str(stem)+'.pdf',bbox_inches='tight');fig.savefig(str(stem)+'.png',dpi=300,bbox_inches='tight')
    cells.to_csv(str(stem)+'_cells.csv',index=False);bins.to_csv(str(stem)+'_bins.csv',index=False);rho.to_csv(str(stem)+'_correlations.csv',index=False)
    return stem
