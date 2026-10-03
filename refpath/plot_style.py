"""Unified figure style of the REF-Path figures.
Bars / lines: our method in blue, baselines in greys of decreasing darkness;
accuracy / confusion heat-maps: 'Blues' with in-cell annotations; signed maps: 'RdBu_r'; serif fonts (Times New Roman when
available), no top/right spines, light dotted horizontal grid, values printed above bars, 'higher/lower is better' notes."""
import matplotlib; import matplotlib.pyplot as plt; import numpy as np
BLUE = '#1f6fff'; BLUE_LIGHT = '#7fb0ff'; GREYS = ['#3c3c3c', '#6d6d6d', '#9a9a9a', '#c4c4c4', '#e2e2e2']
CLASS_COLORS = ['#1f6fff', '#e69f00', '#009e73', '#d55e00', '#cc79a7', '#56b4e9', '#f0e442', '#000000', '#999999', '#8c564b']   # Okabe-Ito based (classes / machine types)
CMAP_DIV = 'RdBu_r'; CMAP_MAP = 'Blues'
from matplotlib.colors import LinearSegmentedColormap as _LSC
CMAP_ACC = _LSC.from_list('Blues_light', plt.get_cmap('Blues')(np.linspace(0.02, 0.78, 256)))   # lighter Blues (row-normalized confusion matrices)
CLASS_BLUES = ['#0a2f6b', '#1f6fff', '#7fb0ff', '#9a9a9a']    # class colours inside blue/grey figures: navy / blue / light blue / grey (grey = the failure case)
TSNE_COLORS = ['#4a7fd6', '#f2a541', '#d94a4a', '#5aa469', '#8e6bbf', '#4cc3d0', '#8c8c8c', '#c9a227', '#e377c2', '#7f7f7f']            # soft blue / orange / red / green ... (t-SNE panels)


def apply(size=9):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': size,
                         'axes.titlesize': size, 'axes.labelsize': size, 'xtick.labelsize': size - 1, 'ytick.labelsize': size - 1, 'legend.fontsize': size - 1, 'legend.frameon': False,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'axes.axisbelow': True, 'savefig.dpi': 200, 'pdf.fonttype': 42})


def grid(ax, axis='y'):
    ax.grid(True, axis=axis, color='#d0d0d0', lw=0.5, ls=':')


def annotate_bars(ax, bars, fmt='{:.2f}', dy=0.01, size=None, color='k'):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h): ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha='center', va='bottom', fontsize=size or plt.rcParams['font.size'] - 2, color=color)


def note(ax, text='higher is better ↑', loc='upper right'):
    x, ha = (0.98, 'right') if 'right' in loc else (0.02, 'left'); y, va = (0.97, 'top') if 'upper' in loc else (0.03, 'bottom')
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=plt.rcParams['font.size'] - 1, color='#666', style='italic')


def confusion(ax, M, labels, cmap=None, size=None, bold_diag=True):
    """row-normalized confusion matrix in the light-Blues style: white cell separators, bold diagonal, black/white text"""
    cmap = cmap or CMAP_ACC; im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1); n = M.shape[0]
    for i in range(n):
        for j in range(n): ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=size or plt.rcParams['font.size'] - 1, color='w' if M[i, j] > 0.62 else 'k', fontweight='bold' if (bold_diag and i == j) else 'normal')
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True); ax.grid(which='minor', color='w', lw=1.8); ax.tick_params(which='minor', length=0); ax.tick_params(which='major', length=2)
    for sp in ax.spines.values(): sp.set_visible(False)
    return im


def heat(ax, M, vmin=0.4, vmax=1.0, fmt='{:.2f}', cmap=None, text_thresh=None, size=None):
    cmap = cmap or CMAP_ACC
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto'); thr = text_thresh if text_thresh is not None else vmin + 0.55 * (vmax - vmin)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]): ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center', fontsize=size or plt.rcParams['font.size'] - 2, color='w' if M[i, j] > thr else 'k')
    return im
