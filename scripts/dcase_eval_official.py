"""DCASE 2026 dev: (1) official aggregate = harmonic mean over all machines x {AUC_source, AUC_target, pAUC};
(2) machine-ID-agnostic evaluation: one kNN scorer over the pooled train-normal clips of ALL machines (no machine
identity used at test time), same metrics; (3) the official baseline numbers (AE on near channel) for comparison.
Uses cached embeddings: dcase_beats.npz (frozen), dcase_beats_refnorm.npz, dcase_beats_finetuned.npz."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from refpath.data.dcase import dcase_index, MACHINES
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import hmean
idx = dcase_index(); mach = np.array([r['machine'] for r in idx]); split = np.array([r['split'] for r in idx]); dom = np.array([r['domain'] for r in idx]); y = np.array([r['label'] for r in idx])
official = {'ToyCarEmu': (69.62, 61.2, 55.89), 'ToyCar': (75.62, 37.87, 54.03), 'bearingEmu': (62.34, 59.56, 59.85), 'fan': (61.45, 46.94, 53.33), 'gearboxEmu': (68.23, 49.78, 52.94), 'sliderEmu': (67.25, 45.05, 50.38), 'valveEmu': (67.74, 68.78, 55.08)}
print('Official AE baseline (near channel): harmonic mean over machines x metrics = %.3f' % (hmean([v / 100 for m in MACHINES for v in official[m]])))


def metrics(scores_fn):
    vals = []
    for m in MACHINES:
        te = (mach == m) & (split == 'test'); s = scores_fn(m, te); yt, dt = y[te], dom[te]
        vals += [roc_auc_score(yt[dt == 'source'], s[dt == 'source']), roc_auc_score(yt[dt == 'target'], s[dt == 'target']), roc_auc_score(yt, s, max_fpr=0.1)]
    return hmean(vals), np.array(vals).reshape(7, 3)


for name in ['dcase_beats', 'dcase_beats_refnorm', 'dcase_beats_finetuned']:
    fn = os.path.join(ROOT, 'cache', name + '.npz')
    if not os.path.exists(fn): continue
    emb = np.load(fn)['emb']
    for feat, Fm in [('near', emb[:, 0]), ('concat', np.concatenate([emb[:, 0], emb[:, 1]], 1))]:
        # machine-aware: kNN to the machine's own train normals (standardized per machine)
        def s_aware(m, te):
            tr = (mach == m) & (split == 'train'); mu, sd = Fm[tr].mean(0), Fm[tr].std(0) + 1e-6; G = (Fm - mu) / sd
            return NearestNeighbors(n_neighbors=5).fit(G[tr]).kneighbors(G[te])[0].mean(1)
        # machine-agnostic: kNN to ALL train normals, global standardization, no machine identity
        tr_all = split == 'train'; mu, sd = Fm[tr_all].mean(0), Fm[tr_all].std(0) + 1e-6; G = (Fm - mu) / sd; nn_all = NearestNeighbors(n_neighbors=5).fit(G[tr_all])
        def s_agn(m, te): return nn_all.kneighbors(G[te])[0].mean(1)
        h1, v1 = metrics(s_aware); h2, v2 = metrics(s_agn)
        print(f'{name:22s} {feat:7s} official-hmean machine-aware {h1:.3f} | machine-agnostic {h2:.3f}')
