"""Recreate the supplied five-panel context figure from numerical source tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from scipy.cluster.hierarchy import linkage, leaves_list
from . import ROOT
from .context import ARCHIVED, configuration

FAMILIES = ['Lymphoid', 'Myeloid', 'Stromal', 'Vascular']
COLORS = dict(zip(FAMILIES, ['#539ecd', '#e32f27', '#56b567', '#bce4b5']))
BURDENS = ['low', 'medium', 'high']
STYLE = {'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
         'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.titlesize': 10,
         'svg.fonttype': 'none', 'pdf.fonttype': 42}


def _format(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='both', labelsize=8, width=.6, length=2.5)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)


def _legend(fig, axes, anchor):
    handles = [Line2D([], [], color=COLORS[f], marker='o', ms=3, lw=1, label=f) for f in FAMILIES]
    # An explicit figure-coordinate anchor keeps legends outside the data panels.
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=anchor,
               ncol=4, frameon=False, columnspacing=.9, handlelength=1.2, handletextpad=.4)


def schematic(ax):
    """Editable vector illustration: fixed gray query, altered context tokens."""
    rng = np.random.default_rng(39)
    for x, colors in [(0, ['#ed7378'] * 5), (2.7, ['#ed7378', '#539ecd', '#ae6b9c', '#ed7378', '#ae6b9c'])]:
        for k, color in enumerate(colors):
            theta = .45 + k * 2 * np.pi / 5
            center = np.array([x + .76 * np.cos(theta), .73 * np.sin(theta)])
            angles = np.linspace(0, 2 * np.pi, 26, endpoint=False)
            radius = .33 * (1 + .12 * rng.normal(size=len(angles)))
            vertices = center + np.column_stack([radius * np.cos(angles), .78 * radius * np.sin(angles)])
            ax.add_patch(Polygon(vertices, facecolor=color, edgecolor='white', lw=1.3, alpha=.92))
            ax.add_patch(Circle(center, .09, facecolor=color, edgecolor='.45', lw=.5))
        ax.add_patch(Circle((x, 0), .35, facecolor='.66', edgecolor='.25', ls='--', lw=.8, zorder=4))
        ax.add_patch(Circle((x, 0), .10, facecolor='.55', edgecolor='.35', lw=.5, zorder=5))
    ax.add_patch(FancyArrowPatch((.1, .16), (2.55, .16), connectionstyle='arc3,rad=-.65',
                                arrowstyle='-|>', mutation_scale=12, lw=1, color='.15', zorder=6))
    ax.text(1.35, 1.30, 'Map to new context', ha='center', va='center', fontsize=10)
    ax.text(1.35, -1.28, 'Context perturbation with query cell held fixed',
            ha='center', va='center', fontsize=8)
    ax.set(xlim=(-1.25, 3.95), ylim=(-1.6, 1.7), aspect='equal')
    ax.axis('off')


def response_curves(axes, source):
    curves = pd.read_csv(Path(source) / 'burden_stratified_context_auroc_response/combined_family_auroc_curves.csv')
    for index, (ax, burden) in enumerate(zip(axes, BURDENS)):
        for family in FAMILIES:
            values = curves.loc[curves.query_family.eq(family) & curves.query_burden_quantile.eq(burden)]
            values = values.set_index('delta_malignant_neighbors').reindex(range(-5, 6))
            values.loc[values.n_sections.fillna(0).lt(3), ['delta_marker_auroc', 'q25', 'q75']] = np.nan
            ax.fill_between(values.index.to_numpy(), values.q25.to_numpy(float), values.q75.to_numpy(float),
                            color=COLORS[family], alpha=.1, linewidth=0)
            ax.plot(values.index, values.delta_marker_auroc, color=COLORS[family], marker='o', ms=2.5, lw=1)
        ax.axhline(0, color='.65', lw=.6, zorder=0)
        ax.axvline(0, color='.65', lw=.6, ls=':', zorder=0)
        ax.set_title(burden.capitalize(), pad=3)
        ax.set_xticks([-4, -2, 0, 2, 4])
        # The parent figure supplies one shared x label for these three axes.
        ax.set_xlabel('')
        if index == 0:
            ax.set_ylabel('Δ malignant marker\nremoval priority', labelpad=3)
        else:
            ax.tick_params(labelleft=False)
        _format(ax)


def marker_heatmap(ax, source, annotate=True, colorbar=True):
    table = pd.read_csv(Path(source) / 'gene_native_win_stratified_context_correlations/gene_win_correlation_summary.csv')
    table = table.loc[table.group_kind.eq('family')]
    values = table.pivot(index='gene', columns='query_group', values='median_rho').reindex(
        index=configuration()['marker_genes'], columns=FAMILIES)
    support = table.pivot(index='gene', columns='query_group', values='n_sections').reindex_like(values)
    agreement = table.pivot(index='gene', columns='query_group', values='n_same_sign').reindex_like(values)
    values = values.mask(support.lt(3))
    supported = values.index[values.notna().any(axis=1)].tolist()
    missing = values.index[values.isna().all(axis=1)].tolist()
    imputed = values.loc[supported].copy()
    imputed = imputed.T.fillna(imputed.mean(axis=1)).T
    if len(supported) > 1:
        tree = linkage(imputed.to_numpy(), method='average', metric='euclidean', optimal_ordering=True)
        supported = imputed.index[leaves_list(tree)].tolist()
        if values.loc[supported[0]].mean() < values.loc[supported[-1]].mean():
            supported.reverse()
    values = values.reindex(supported + missing)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#d9d9d9')
    image = ax.imshow(values.T.to_numpy(), aspect='auto', cmap=cmap, vmin=-1, vmax=1,
                      interpolation='none')
    ax.set_xticks(range(len(values)), values.index, rotation=90)
    ax.set_yticks(range(len(FAMILIES)), FAMILIES)
    ax.set_xlabel('32-gene breast cancer panel', labelpad=4)
    ax.set_ylabel('Query superclass')
    ax.set_title('Malignant-gene response relative to query-type native markers', fontsize=10, pad=6)
    for row, family in enumerate(FAMILIES):
        for column, gene in enumerate(values.index):
            value = values.loc[gene, family]
            if annotate and np.isfinite(value):
                ax.text(column, row, f'{int(agreement.loc[gene, family])}/{int(support.loc[gene, family])}',
                        ha='center', va='center', fontsize=5.5, color='white' if abs(value) > .65 else '.2')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', labelsize=8, length=2)
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, fraction=.022, pad=.02, shrink=.85, ticks=[-1, 0, 1])
        bar.set_label('Median section–stratum\nSpearman ρ', fontsize=8)
        bar.ax.tick_params(labelsize=8)
    return values


def vascular(ax, source):
    table = pd.read_csv(Path(source) / 'vascular_all32_plausibility_vs_response.csv')
    x = table.endothelial_plausibility_log2_ratio.to_numpy(float)
    y = table.beta.to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    ax.scatter(x, y, s=15, color='#527e95', alpha=.8)
    if len(x) > 1 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.array([x.min(), x.max()])
        ax.plot(xx, slope * xx + intercept, color='#ad563e', lw=1)
    ax.axhline(0, color='.2', lw=.6)
    ax.axvline(0, color='.7', lw=.6)
    ax.set_xlabel('Log₂ vascular / malignant prevalence')
    ax.set_ylabel('Vascular β')
    ax.set_title('Vascular plausibility vs\nmalignant-context response', fontsize=10)
    _format(ax)


def slopes(axes, source):
    path = Path(source) / 'auroc_control_native_within_type'
    sections = pd.read_csv(path / 'section_control_native_slopes.csv')
    summary = pd.read_csv(path / 'control_native_summary.csv')
    valid = sections.loc[sections.status.eq('valid') & np.isfinite(sections.beta_ctrl)
                         & np.isfinite(sections.beta_native)]
    for index, (ax, burden) in enumerate(zip(axes, BURDENS)):
        ax.axhline(0, color='.7', lw=.6, zorder=0)
        ax.axvline(0, color='.7', lw=.6, zorder=0)
        ax.plot([-.025, .025], [-.025, .025], color='.65', ls='--', lw=.6, zorder=0)
        for family in FAMILIES:
            part = valid.loc[valid.query_family.eq(family) & valid.query_burden_quantile.eq(burden)]
            ax.scatter(part.beta_ctrl, part.beta_native, color=COLORS[family], s=10, alpha=.35, edgecolor='none')
            rows = summary.loc[summary.query_family.eq(family) & summary.query_burden_quantile.eq(burden)
                               & summary.n_sections.ge(3)]
            for row in rows.itertuples():
                ax.errorbar(row.beta_ctrl, row.beta_native,
                    xerr=[[row.beta_ctrl-row.beta_ctrl_q25], [row.beta_ctrl_q75-row.beta_ctrl]],
                    yerr=[[row.beta_native-row.beta_native_q25], [row.beta_native_q75-row.beta_native]],
                    fmt='o', ms=4, color=COLORS[family], markeredgecolor='.35', markeredgewidth=.4,
                    elinewidth=.9, capsize=1.8)
        ax.set_title(burden.capitalize(), pad=3)
        ax.set_xlim(-.025, .025)
        ax.set_ylim(-.025, .025)
        ax.set_xticks([-.02, 0, .02])
        ax.set_yticks([-.02, 0, .02])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('')
        if index == 0:
            ax.set_ylabel('β_native: within-type\noriginal-context slope', labelpad=3)
        else:
            ax.tick_params(labelleft=False)
        _format(ax)


def _save(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf', 'svg']:
        # Fixed canvas preserves the requested 7.5-inch figure width.
        fig.savefig(stem.with_suffix('.' + suffix), dpi=600, facecolor='white')
    plt.close(fig)
    print('Saved:', stem)


def plot_panel(panel, source=ARCHIVED, output=None, annotate=True):
    output = Path(output or ROOT / 'outputs/panels/context')
    with plt.rc_context(STYLE):
        if panel == 'a':
            fig, ax = plt.subplots(figsize=(3.2, 2.1))
            schematic(ax)
            fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)
        elif panel in ['b', 'e']:
            fig, axes = plt.subplots(1, 3, figsize=(5.2, 2.6), sharey=True)
            (response_curves if panel == 'b' else slopes)(axes, source)
            fig.subplots_adjust(left=.14, right=.99, bottom=.36, top=.83, wspace=.18)
            if panel == 'b':
                fig.suptitle('Query malignant-marker burden', fontsize=10, y=.99)
            fig.text(.56, .22, 'Δ malignant neighbors given as context' if panel == 'b'
                     else 'β_ctrl: within-query AUROC slope', ha='center', fontsize=8)
            _legend(fig, axes, (.54, .11))
        elif panel == 'c':
            fig, ax = plt.subplots(figsize=(7.5, 2.4))
            marker_heatmap(ax, source, annotate)
            fig.subplots_adjust(left=.13, right=.92, bottom=.40, top=.85)
        elif panel == 'd':
            fig, ax = plt.subplots(figsize=(2.7, 2.7))
            vascular(ax, source)
            fig.subplots_adjust(left=.27, right=.98, bottom=.24, top=.78)
        else:
            raise ValueError(panel)
        _save(fig, output / f'context_{panel}')


def plot_context(source=ARCHIVED, output=None, annotate=True):
    source = Path(source)
    output = Path(output or ROOT / 'outputs/panels/context')
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.5, 7.1))
        grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.85], height_ratios=[1.15, .85, 1.1],
                               left=.12, right=.98, bottom=.12, top=.95, hspace=.92, wspace=.55)
        a = fig.add_subplot(grid[0, 0])
        schematic(a)
        bgrid = grid[0, 1].subgridspec(1, 3, wspace=.17)
        b = [fig.add_subplot(bgrid[0, k]) for k in range(3)]
        response_curves(b, source)
        limits = [ax.get_ylim() for ax in b]
        for ax in b:
            ax.set_ylim(min(lo for lo, _ in limits), max(hi for _, hi in limits))
        c = fig.add_subplot(grid[1, :])
        values = marker_heatmap(c, source, annotate, colorbar=False)
        cpos = c.get_position()
        c.set_position([.15, cpos.y0, .72, cpos.height])
        color_ax = fig.add_axes([.89, cpos.y0 + .015, .009, cpos.height - .03])
        bar = fig.colorbar(c.images[0], cax=color_ax, ticks=[-1, 0, 1])
        bar.set_label('Median section–stratum\nSpearman ρ', fontsize=8, labelpad=4)
        bar.ax.tick_params(labelsize=8, length=2)
        d = fig.add_subplot(grid[2, 0])
        vascular(d, source)
        egrid = grid[2, 1].subgridspec(1, 3, wspace=.17)
        e = [fig.add_subplot(egrid[0, k]) for k in range(3)]
        slopes(e, source)
        for label, ax in [('a', a), ('b', b[0]), ('c', c), ('d', d), ('e', e[0])]:
            pos = ax.get_position()
            fig.text(.025 if label in 'acd' else pos.x0 - .025, min(pos.y1 + .02, .98),
                     label, fontsize=12, weight='bold', va='top')
        fig.text(.75, .995, 'Query malignant-marker burden', ha='center', va='top', fontsize=10)
        fig.text(.75, b[0].get_position().y0 - .05, 'Δ malignant neighbors given as context',
                 ha='center', fontsize=8)
        fig.text(.75, e[0].get_position().y0 - .047, 'β_ctrl: within-query AUROC slope',
                 ha='center', fontsize=8)
        _legend(fig, b, (.69, .695))
        _legend(fig, e, (.69, .06))
        _save(fig, output / 'context_responsiveness')
        for panel in 'abcde':
            plot_panel(panel, source, output, annotate)
    tables = ROOT / 'outputs/tables/context'
    tables.mkdir(parents=True, exist_ok=True)
    values.to_csv(tables / 'heatmap_displayed_values.csv')
    (tables / 'plot_provenance.json').write_text(__import__('json').dumps({
        'source': str(source.resolve()), 'archived': source.resolve() == ARCHIVED.resolve(),
        'support_annotation': 'sections whose median stratum rho agrees in sign / supported sections; not significance',
        'model': 'BX milestone 50', 'section_count': 8}, indent=2) + '\n')
    return output
