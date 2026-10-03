"""Figure: 27-cell (microphone x speed x load) per-cell accuracy of the three fused branches and of the deployable
REF-Path fusion (three-branch temperature-calibrated product of seed ensembles, temperatures by nested leave-one-task-out)."""
import os, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.insert(0, ROOT); from refpath import plot_style as PS; PS.apply(8)
d = json.load(open(os.path.join(ROOT, 'outputs', 'fm_stats_mic_x_speed_x_load.json'))); cells = d['cells']; S = d['summary']
E = json.load(open(os.path.join(ROOT, 'outputs', 'ensemble_temperature_mic_x_speed_x_load.json'))); assert E['cells'] == cells
R = E['results']   # the three branches exactly as they enter the fusion (same values as Table 3)
mats = {'(a) Physics branch': np.array(R['physics, seed-averaged']['per_cell']), '(b) BEATs branch': np.array(R['BEATs, seed-averaged']['per_cell']),
        '(c) Dasheng branch, frozen probe': np.array(R['Dasheng frozen probe']['per_cell']), '(d) REF-Path fusion (deployable)': np.array(E['results']['T-product E_phys x E_beats x Dasheng-frozen']['nested_per_cell'])}
mics = ['CRY333', 'BK4966_2', 'BK4966_3']; MICN = ['CRY333', 'B&K #2', 'B&K #3']; rpms = [800, 1000, 1200]; loads = [0, 15, 30]
def grid(v):
    G = np.full((3, 9), np.nan)
    for c, x in zip(cells, v):
        m, r, l = c.split(':')[1].split('@'); G[mics.index(m), rpms.index(int(r)) * 3 + loads.index(int(l))] = x
    return G
fig, axes = plt.subplots(2, 2, figsize=(9.5, 3.9)); axes = axes.ravel()
for k, (ax, (name, v)) in enumerate(zip(axes, mats.items())):
    G = grid(v); im = ax.imshow(G, vmin=0.4, vmax=1.0, cmap=PS.CMAP_ACC, aspect='auto')
    for i in range(3):
        for j in range(9): ax.text(j, i, f'{G[i, j]:.2f}', ha='center', va='center', fontsize=7, color='w' if G[i, j] > 0.8 else 'k')
    ax.set_xticks(range(9)); ax.set_xticklabels([f'{r}\n{l}%' for r in rpms for l in loads] if k >= 2 else [''] * 9, fontsize=7); ax.set_yticks(range(3)); ax.set_yticklabels(MICN if k % 2 == 0 else [''] * 3, fontsize=8)
    ax.set_xticks(np.arange(-0.5, 9, 1), minor=True); ax.set_yticks(np.arange(-0.5, 3, 1), minor=True); ax.grid(which='minor', color='w', lw=1.2); ax.tick_params(which='minor', length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(f'{name}: mean {np.nanmean(G):.3f}, worst cell {np.nanmin(G):.2f}', fontsize=8.5, loc='left')
    for x in (2.5, 5.5): ax.axvline(x, color='w', lw=3)
    if k >= 2: ax.set_xlabel('held-out speed (rpm) / load')
fig.subplots_adjust(left=0.07, right=0.91, top=0.93, bottom=0.17, wspace=0.05, hspace=0.3)
cax = fig.add_axes([0.925, 0.2, 0.012, 0.65]); fig.colorbar(im, cax=cax, label='accuracy')
out = os.path.join(ROOT, 'figures', 'fig_27cell_branches.png'); fig.savefig(out, dpi=200); fig.savefig(out.replace('.png', '.pdf')); print('saved', out)
