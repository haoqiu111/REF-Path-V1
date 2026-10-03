"""(a) BH-FDR control of fault alarms from conformal p-values (Bates et al. 2023 style);
(b) non-exchangeable weighted conformal (Barber et al. 2023): the calibration set is enlarged with the healthy
    windows of the SOURCE domains, each weighted by the similarity of its domain's reference statistics to the target's
    (path/speed similarity), so that the target's small reference is complemented by similar domains.
Uses the physics-MLP probability files (split reference); healthy class index 1.

python scripts/conformal_weighted.py outputs/<probs>.npz
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.eval.protocol import make_tasks, domain_key
Z = np.load(sys.argv[1]); H = 1
z = np.load(os.path.join(ROOT, 'cache', 'guilin_feats.npz'), allow_pickle=True); meta = z['meta']; psd = z['psd']
keys = domain_key(meta); healthy = meta['state'] == 'healthy'
tasks = {t.name: t for t in make_tasks(meta, 'mic_x_speed')}
alphas = [0.01, 0.05, 0.10]


def pvals(s_cal, s_te, w_cal=None):
    if w_cal is None: w_cal = np.ones(len(s_cal))
    W = w_cal.sum() + 1.0
    return np.array([(np.sum(w_cal * (s_cal >= s)) + 1.0) / W for s in s_te])


def bh(p, q):
    n = len(p); o = np.argsort(p); ps = p[o]; k = np.where(ps <= q * np.arange(1, n + 1) / n)[0]
    rej = np.zeros(n, bool)
    if len(k): rej[o[:k.max() + 1]] = True
    return rej


print('=== (a) BH-FDR on conformal p-values (split reference), physics MLP ===')
print(f'{"task":22s} ' + ' '.join(f'| q={q:.2f}: FDR  power' for q in alphas))
agg = {q: [] for q in alphas}
for t in sorted({k.split('/')[0] for k in Z.files}):
    pc = Z[f'{t}/ref_prob']; pt = Z[f'{t}/prob']; y = Z[f'{t}/y']; s_cal = 1 - pc[:, H]; s_te = 1 - pt[:, H]; pv = pvals(s_cal, s_te)
    line = f'{t:22s} '
    for q in alphas:
        rej = bh(pv, q); fdr = (rej & (y == H)).sum() / max(rej.sum(), 1); power = rej[y != H].mean(); agg[q].append((fdr, power)); line += f'| {fdr:.3f} {power:.3f}     '
    print(line)
for q in alphas:
    r = np.array(agg[q]); print(f'q={q:.2f}: mean FDR {r[:,0].mean():.3f} (max {r[:,0].max():.3f})  mean power {r[:,1].mean():.3f}')

print('\n=== (b) weighted non-exchangeable conformal: target reference + source healthy windows weighted by reference similarity ===')
print(f'{"task":22s} n_eff ' + ' '.join(f'| a={a:.2f}: FAR   DET ' for a in alphas))
aggw = {a: [] for a in alphas}
for t in sorted({k.split('/')[0] for k in Z.files}):
    task = tasks[t]; pc = Z[f'{t}/ref_prob']; pt = Z[f'{t}/prob']; y = Z[f'{t}/y']; te = Z[f'{t}/te']; ref_idx = Z[f'{t}/ref_idx']
    s_cal = 1 - pc[:, H]; s_te = 1 - pt[:, H]
    # source healthy windows: we need their scores under the SAME model -> not stored; approximate with the model's
    # calibration behaviour by using the target's own reference only when source scores are unavailable.
    # Here we implement the weighting over the target reference windows using a recency/segment weight = 1 (all same age),
    # plus source-domain healthy windows if their probabilities are present in the file.
    if f'{t}/src_healthy_prob' in Z.files:
        ps = Z[f'{t}/src_healthy_prob']; src_idx = Z[f'{t}/src_healthy_idx']; s_src = 1 - ps[:, H]
        tk = keys[te[0]]; d_t = psd[ref_idx].mean(0)
        w = []
        for k in np.unique(keys[src_idx]):
            d_k = psd[src_idx[keys[src_idx] == k]].mean(0); w.append((k, np.exp(-np.linalg.norm(d_t - d_k) ** 2 / (2 * 4.0 ** 2))))
        wmap = dict(w); w_src = np.array([wmap[keys[i]] for i in src_idx]) * (len(s_cal) / max(len(s_src), 1))
        s_all = np.concatenate([s_cal, s_src]); w_all = np.concatenate([np.ones(len(s_cal)), w_src]); n_eff = w_all.sum()
    else:
        s_all, w_all, n_eff = s_cal, np.ones(len(s_cal)), float(len(s_cal))
    pv = pvals(s_all, s_te, w_all); line = f'{t:22s} {n_eff:5.0f} '
    for a in alphas:
        alarm = pv <= a; far = alarm[y == H].mean(); det = alarm[y != H].mean(); aggw[a].append((far, det)); line += f'| {far:.3f} {det:.3f}      '
    print(line)
for a in alphas:
    r = np.array(aggw[a]); print(f'alpha={a:.2f}: mean FAR {r[:,0].mean():.3f} (max {r[:,0].max():.3f}) mean DET {r[:,1].mean():.3f}')
print('(source-healthy scores are only available when the trainer stores src_healthy_prob; see train_mlp --save_src_healthy)')
