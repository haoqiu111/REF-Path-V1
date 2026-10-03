"""Open-set figure on the 27-cell protocol (three seeds): (a) AUROC of the softmax score, the feature-space k-NN distance
and their rank average per held-out fault class (mean over cells, error bars = s.d. over seeds); (b) per-cell AUROC of the
rank average against the softmax score for the compound fault and cage fracture (27 cells each); (c) rank-average AUROC by
held-out microphone. Writes figures/fig_openset27.pdf/png and outputs/openset27_stats.json."""
import os, json, glob, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'outputs')
import sys; sys.path.insert(0, ROOT); from refpath import plot_style as PS; PS.apply(8)
C = [('inner_pitting', 'inner'), ('outer_pitting', 'outer'), ('inner_outer_pitting', 'inner+outer'), ('cage_fracture', 'cage')]; MICS = ['CRY333', 'BK4966_2', 'BK4966_3']; K = ['auroc_softmax', 'auroc_knn', 'auroc_rank']
D = {}
for c, _ in C:
    per_seed = {k: [] for k in K}; cells = None
    for fn in sorted(glob.glob(os.path.join(OUT, f'full_mic_x_speed_x_load_mlp_holdout_{c}_s?.json'))):
        r = json.load(open(fn))['results']; cells = sorted(r)
        for k in K: per_seed[k].append([r[t][k] for t in cells])
    D[c] = dict(cells=cells, **{k: np.array(v) for k, v in per_seed.items()})       # (seeds, cells)
stats = {}
for c, _ in C:
    stats[c] = {k: dict(mean=float(D[c][k].mean()), sd_seeds=float(D[c][k].mean(1).std()), min_cell=float(D[c][k].mean(0).min())) for k in K}
    stats[c]['rank_minus_softmax'] = float((D[c]['auroc_rank'] - D[c]['auroc_softmax']).mean()); stats[c]['rank_wins'] = int(((D[c]['auroc_rank'] - D[c]['auroc_softmax']).mean(0) > 0).sum())
    stats[c]['by_mic'] = {m: {k: float(D[c][k].mean(0)[[i for i, t in enumerate(D[c]['cells']) if m in t]].mean()) for k in K} for m in MICS}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
ax = axes[0]; x = np.arange(len(C)); w = 0.2; lab = {'auroc_softmax': '1 − max posterior', 'auroc_knn': 'k-NN distance', 'auroc_rank': 'rank average (transductive)', 'auroc_pct_ref': 'reference z-score sum (deployable)'}; col = {'auroc_softmax': PS.GREYS[1], 'auroc_knn': PS.GREYS[3], 'auroc_rank': PS.BLUE_LIGHT, 'auroc_pct_ref': PS.BLUE}
sc = os.path.join(OUT, 'openset27_scores.json'); KK = list(K)
if os.path.exists(sc):
    R2 = json.load(open(sc))
    if all('zsum_ref' in next(iter(R2[c].values())) for c, _ in C):
        for c, _ in C: stats[c]['auroc_pct_ref'] = dict(mean=float(np.mean([R2[c][t]['zsum_ref']['auroc'] for t in R2[c]])), sd_seeds=0.0)
        KK = K + ['auroc_pct_ref']
for i, k in enumerate(KK):
    ax.bar(x + (i - (len(KK) - 1) / 2) * w, [stats[c][k]['mean'] for c, _ in C], w, yerr=[stats[c][k]['sd_seeds'] for c, _ in C], color=col[k], label=lab[k], capsize=2)
ax.axhline(0.5, color='k', lw=0.6, ls=':'); ax.set_xticks(x); ax.set_xticklabels([n for _, n in C]); ax.set_ylabel('unknown-vs-known AUROC (27 cells, 3 seeds)'); ax.set_ylim(0.2, 1.02); ax.legend(fontsize=7, loc='lower left'); ax.set_title('(a) held-out fault class'); PS.grid(ax); PS.note(ax)
ax = axes[1]
for c, n, mk in [('inner_outer_pitting', 'inner+outer', 'o'), ('cage_fracture', 'cage', 's'), ('inner_pitting', 'inner', '^'), ('outer_pitting', 'outer', 'v')]:
    ax.scatter(D[c]['auroc_softmax'].mean(0), D[c]['auroc_rank'].mean(0), s=16, marker=mk, label=n, alpha=0.9, color=PS.CLASS_BLUES[[x[0] for x in C].index(c)], edgecolors='none')
ax.plot([0, 1], [0, 1], color='#888', ls=':', lw=0.8); ax.set_xlabel('AUROC, 1 − max posterior'); ax.set_ylabel('AUROC, rank average'); ax.set_xlim(0.1, 1.02); ax.set_ylim(0.1, 1.02); ax.legend(fontsize=7, loc='lower right'); ax.set_title('(b) per cell'); PS.grid(ax, 'both'); PS.note(ax, 'above the diagonal: combination helps', 'upper left')
ax = axes[2]; xm = np.arange(3)
for i, (c, n) in enumerate(C):
    bars = ax.bar(xm + (i - 1.5) * 0.2, [stats[c]['by_mic'][m]['auroc_rank'] for m in MICS], 0.2, label=n, color=PS.CLASS_BLUES[i]); PS.annotate_bars(ax, bars, '{:.2f}', dy=0.008, size=5.5)
ax.axhline(0.5, color='k', lw=0.6, ls=':'); ax.set_xticks(xm); ax.set_xticklabels(['CRY333', 'B&K #2', 'B&K #3']); ax.set_ylabel('rank-average AUROC'); ax.set_ylim(0.2, 1.02); ax.legend(fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.12), handlelength=1.2, columnspacing=1.0); ax.set_title('(c) by held-out microphone'); PS.grid(ax); PS.note(ax)
fig.tight_layout(); out = os.path.join(ROOT, 'figures', 'fig_openset27'); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
json.dump(stats, open(os.path.join(OUT, 'openset27_stats.json'), 'w'), indent=1); print('saved', out)
for c, n in C: print(n, {k: f"{stats[c][k]['mean']:.3f}" for k in K}, 'rank-softmax %+.3f wins %d/27' % (stats[c]['rank_minus_softmax'], stats[c]['rank_wins']))
