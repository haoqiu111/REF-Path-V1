"""Confusion matrices of two complementary nine-cell tasks: CRY333 at 1200 rpm (the physics branch fails, the BEATs branch
does not) and B&K #3 at 1000 rpm (the reverse), each for the physics branch, the BEATs branch and their geometric fusion."""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'outputs'); FIG = os.path.join(ROOT, 'figures')
sys.path.insert(0, ROOT); from refpath import plot_style as PS; PS.apply(8)
CL = ['cage', 'healthy', 'in+out', 'inner', 'outer']
A = np.load(os.path.join(OUT, 'guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_probs.npz')); B = np.load(os.path.join(OUT, 'guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm_refsplit_refgain_s0_probs.npz'))
cells = [('mxs:CRY333@1200', 'CRY333, 1200 rpm'), ('mxs:BK4966_3@1000', 'B&K #3, 1000 rpm')]
fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.3)); LET = 'abcdef'
for r, (t, tn) in enumerate(cells):
    pa, tea = A[f'{t}/prob'], A[f'{t}/te']; pb, yb, teb = B[f'{t}/prob'], B[f'{t}/y'], B[f'{t}/te']
    fiw = (teb // 60) * 119 + 2 * (teb % 60)                     # wave window j of a 60-s recording == feature window 2j (hop 0.5 s)
    pos = {ix: i for i, ix in enumerate(tea)}; pa = pa[np.array([pos[ix] for ix in fiw])]; geo = np.log(pa + 1e-9) + np.log(pb + 1e-9)
    for c, (name, p) in enumerate([('physics branch', pa.argmax(1)), ('BEATs branch', pb.argmax(1)), ('fusion', geo.argmax(1))]):
        cm = np.zeros((5, 5), int)
        for a_, b_ in zip(yb, p): cm[a_, b_] += 1
        ax = axes[r, c]; im = PS.confusion(ax, cm / cm.sum(1, keepdims=True), CL, size=6.5)
        ax.set_xticklabels(CL if r == 1 else [''] * 5, rotation=35, ha='right', fontsize=6.5); ax.set_yticklabels(CL if c == 0 else [''] * 5, fontsize=6.5)
        ax.set_title(f'({LET[3 * r + c]}) {tn}: {name}, acc. {(p == yb).mean():.2f}', fontsize=7.5, loc='left')
        if r == 1: ax.set_xlabel('predicted class')
        if c == 0: ax.set_ylabel('true class')
fig.subplots_adjust(left=0.08, right=0.9, top=0.95, bottom=0.12, wspace=0.08, hspace=0.18)
cax = fig.add_axes([0.915, 0.2, 0.012, 0.6]); cb = fig.colorbar(im, cax=cax); cb.set_label('row-normalized proportion'); cb.outline.set_visible(False)
fig.savefig(os.path.join(FIG, 'fig_confusion_hard_cells.png'), dpi=200); fig.savefig(os.path.join(FIG, 'fig_confusion_hard_cells.pdf')); print('saved fig_confusion_hard_cells (seed-0 models)')
