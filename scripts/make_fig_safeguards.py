"""Conformal figure (two panels): (a) empirical false-alarm rate on healthy test windows against
the nominal level for the commissioning reference (segment 1) and a recent reference (segment 3), band = min / max over the
nine tasks; (b) false-alarm rate at alpha = 0.05 and 0.10 for the commissioning reference, the label-free self-refreshed
reference and the ideal re-recorded reference (three seeds, outputs/conformal_refresh_summary.json), with the BH false-discovery
rate of the alarms. Writes figures/fig_conformal_far.png/pdf."""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'outputs'); FIG = os.path.join(ROOT, 'figures')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})


def far_curve(fn):
    Z = np.load(fn); H = 1; alphas = np.linspace(0.005, 0.2, 40); F = []
    for t in sorted({k.split('/')[0] for k in Z.files}):
        s_cal = 1 - Z[f'{t}/ref_prob'][:, H]; s_te = 1 - Z[f'{t}/prob'][:, H]; y = Z[f'{t}/y']; n = len(s_cal)
        pv = np.array([(np.sum(s_cal >= s) + 1) / (n + 1) for s in s_te]); F.append([(pv[y == H] <= a).mean() for a in alphas])
    return alphas, np.array(F)


fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axes[0]
for fn, lab, c in [('guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_probs.npz', 'reference = segment 1 (commissioning)', '#6d6d6d'), ('guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_refseg3_probs.npz', 'reference = segment 3 (recent)', '#1f6fff')]:
    al, F = far_curve(os.path.join(OUT, fn)); ax.plot(al, F.mean(0), color=c, label=lab); ax.fill_between(al, F.min(0), F.max(0), color=c, alpha=0.13)
ax.plot(al, al, 'k--', lw=0.8, label='nominal level'); ax.set_xlabel('nominal false-alarm level α'); ax.set_ylabel('false-alarm rate, healthy test windows'); ax.legend(fontsize=7, loc='upper left'); ax.set_xlim(0, 0.2); ax.set_ylim(0, 0.6)
ax.set_title('(a) validity against reference age', fontsize=8.5, loc='left')
V = json.load(open(os.path.join(OUT, 'conformal_refresh_summary.json')))['0.01']['variants']; ax = axes[1]
names = [('OLD', 'commissioning\nreference', '#6d6d6d'), ('SELF-H', 'label-free\nself-refresh', '#1f6fff'), ('IDEAL', 're-recorded\nreference', '#7fb0ff')]; x = np.arange(2); w = 0.26
for i, (k, lab, c) in enumerate(names):
    m = [V[k]['far@0.05'], V[k]['far@0.1']]; s = [np.std(V[k]['far@0.05_seeds']), np.std(V[k]['far@0.1_seeds'])]
    bars = ax.bar(x + (i - 1) * w, m, w, yerr=s, color=c, label=lab.replace('\n', ' '), capsize=2, error_kw=dict(lw=0.7))
    for b, v, e in zip(bars, m, s): ax.text(b.get_x() + b.get_width() / 2, v + e + 0.012, f'{v:.3f}', ha='center', fontsize=7)
for xi, a_ in zip(x, (0.05, 0.10)): ax.plot([xi - 0.42, xi + 0.42], [a_, a_], 'k--', lw=0.8)
ax.text(1.44, 0.105, 'nominal', fontsize=6.5, ha='right', va='bottom')
ax.set_xticks(x); ax.set_xticklabels(['α = 0.05', 'α = 0.10']); ax.set_ylabel('false-alarm rate (three seeds)'); ax.set_ylim(0, 0.5); ax.legend(fontsize=7, loc='upper left')
ax.set_title('(b) label-free refresh of the reference', fontsize=8.5, loc='left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_conformal_far.png'), dpi=200); fig.savefig(os.path.join(FIG, 'fig_conformal_far.pdf')); print('saved fig_conformal_far')
