"""(1) Combined open-set score = rank-average of the softmax score (1 - max prob) and the feature-space kNN distance,
    evaluated on the leave-one-fault-class-out runs (needs the *_holdout_* probs with knn distances -> recomputed here).
(2) Drift-aware conformal: split the healthy reference chronologically (first half -> standardization, second half ->
    calibration); estimate the label-free score drift between the two halves and inflate the calibration quantile by
    it; report false-alarm / detection vs. plain split conformal.

python scripts/openset_conformal.py
"""
import os, sys, json, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.eval.protocol import make_tasks, normalize, Task
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import rankdata
dev = 'cuda'
z = np.load(os.path.join(ROOT, 'cache', 'guilin_feats.npz'), allow_pickle=True); meta = z['meta']
F = np.hstack([z['psd'], z['cyclo'].mean(3).reshape(len(meta), -1)]).astype(np.float32)
classes = sorted(np.unique(meta['state']).tolist()); y = np.array([classes.index(s) for s in meta['state']]); H = classes.index('healthy')


def train_mlp(G, tr, n_cls, seed=0, epochs=40, mixup=0.4):
    torch.manual_seed(seed); X = torch.tensor(G, dtype=torch.float32).to(dev); Y = torch.tensor(y).to(dev); trn = torch.tensor(tr).to(dev)
    m = nn.Sequential(nn.Dropout(0.2), nn.Linear(G.shape[1], 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, n_cls)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(perm), 256):
            idx = trn[perm[i:i + 256]]; xb = X[idx]; yb = F_.one_hot(Y[idx], n_cls).float()
            j = torch.randperm(len(idx), device=dev); lam = torch.distributions.Beta(mixup, mixup).sample((len(idx), 1)).to(dev); xb = lam * xb + (1 - lam) * xb[j]; yb = lam * yb + (1 - lam) * yb[j]
            loss = -(yb * F_.log_softmax(m(xb), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return lambda idx: F_.softmax(m(X[torch.tensor(idx).to(dev)]), 1).detach().cpu().numpy()


def split_task(task, chrono=False):
    rid = np.where(task.ref)[0]
    if chrono: a, b = rid[:len(rid) // 2], rid[len(rid) // 2:]
    else: a, b = rid[::2], rid[1::2]
    rn = np.zeros_like(task.ref); rn[a] = True; rc = np.zeros_like(task.ref); rc[b] = True
    return Task(task.name, task.train, task.test, rn), rc


tasks = make_tasks(meta, 'mic_x_speed')
print('=== (1) combined open-set score, leave-one-fault-class-out ===')
for hc_name in [c for c in classes if c != 'healthy']:
    hc = classes.index(hc_name); res = []
    for t in tasks:
        tn, rc = split_task(t); G = normalize(F, meta, tn, 'healthy'); tr = np.where(t.train)[0]; tr = tr[y[tr] != hc]; te = np.where(t.test)[0]
        mu, sd = G[tr].mean(0), G[tr].std(0) + 1e-6; G = (G - mu) / sd
        pred = train_mlp(G, tr, len(classes)); p = pred(te); known = y[te] != hc
        s_soft = 1 - p.max(1); s_knn = NearestNeighbors(n_neighbors=5).fit(G[tr]).kneighbors(G[te])[0].mean(1)
        s_comb = rankdata(s_soft) + rankdata(s_knn)
        res.append([roc_auc_score(~known, s) for s in (s_soft, s_knn, s_comb)] + [(p[known].argmax(1) == y[te][known]).mean()])
    r = np.array(res); print(f'{hc_name:22s} AUROC softmax {r[:,0].mean():.3f}  kNN {r[:,1].mean():.3f}  combined {r[:,2].mean():.3f}  (min combined {r[:,2].min():.3f}) | known acc {r[:,3].mean():.3f}')

print('\n=== (2) drift-aware conformal (chronological reference split; physics MLP) ===')
alphas = [0.01, 0.05, 0.10]; plain = {a: [] for a in alphas}; drift = {a: [] for a in alphas}
for t in tasks:
    tn, rc = split_task(t, chrono=True); G = normalize(F, meta, tn, 'healthy'); tr = np.where(t.train)[0]; te = np.where(t.test)[0]
    mu, sd = G[tr].mean(0), G[tr].std(0) + 1e-6; G = (G - mu) / sd
    pred = train_mlp(G, tr, len(classes)); p_te = pred(te); p_cal = pred(np.where(rc)[0]); p_early = pred(np.where(tn.ref)[0])
    s_te, s_cal, s_early = 1 - p_te[:, H], 1 - p_cal[:, H], 1 - p_early[:, H]
    # label-free drift estimate: how much the healthy score moved between the two reference halves (shift in the 90% quantile)
    shift = max(0.0, np.quantile(s_cal, 0.9) - np.quantile(s_early, 0.9)); n = len(s_cal)
    for a in alphas:
        q = np.quantile(s_cal, min(1, np.ceil((n + 1) * (1 - a)) / n)); alarm = s_te > q
        plain[a].append((alarm[y[te] == H].mean(), alarm[y[te] != H].mean()))
        alarm2 = s_te > q + shift * 2
        drift[a].append((alarm2[y[te] == H].mean(), alarm2[y[te] != H].mean()))
    print(f'{t.name:22s} drift shift {shift:.3f}')
for a in alphas:
    P, D = np.array(plain[a]), np.array(drift[a])
    print(f'alpha={a:.2f}: plain FAR {P[:,0].mean():.3f} (max {P[:,0].max():.3f}) DET {P[:,1].mean():.3f} | drift-aware FAR {D[:,0].mean():.3f} (max {D[:,0].max():.3f}) DET {D[:,1].mean():.3f}')
