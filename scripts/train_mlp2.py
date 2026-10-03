"""Physics branch and linear brackets on the FULL Guilin cache (cache/guilin_full_feats.npz).

Modes (--model):
  linear   : LR on PCA-60 of standardized features (raw / healthy / oracle)       -> deployable & oracle brackets
  mlp      : reference-standardized MLP + cross-domain mixup (ours)               [+ --multilabel, --coral LAM]
Options:
  --protocol mic_x_speed | mic_x_speed_x_load | lomo | loso | lolo
  --ref_seconds 60      : healthy reference restricted to the first 60 s of segment 1 (deployable short reference)
  --smooth K            : decision-level smoothing: log-posteriors averaged over K consecutive windows of the same
                          recording/channel (K=1 -> none). Reports both window-level and smoothed accuracy.
  --holdout_class C     : open-set: exclude C from training; unknown scores = softmax, kNN distance, and their
                          reference-calibrated p-value combinations (Fisher, min-p) -> AUROC
Outputs: outputs/full_<protocol>_<model>[_tags]_s<seed>.json (+ _probs.npz for mlp)
"""
import os, sys, time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.eval.protocol import make_tasks, normalize, domain_key, Task
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import combine_pvalues, rankdata
dev = 'cuda'
PRIM = {'healthy': [], 'inner_pitting': ['inner'], 'outer_pitting': ['outer'], 'inner_outer_pitting': ['inner', 'outer'], 'cage_fracture': ['cage']}
PRIMS = ['inner', 'outer', 'cage']


def rec_key(meta):
    return np.array([f"{m['path']}|{m['rpm']}|{m['load']}|{m['state']}|{m['seg']}" for m in meta])


def smooth_logp(logp, meta, idx, K):
    """Average log-posteriors over a centred window of K consecutive windows (by start time) within each recording/channel."""
    if K <= 1: return logp
    keys = rec_key(meta[idx]); starts = meta['start'][idx]; out = logp.copy(); h = K // 2
    for k in np.unique(keys):
        sel = np.where(keys == k)[0]; order = sel[np.argsort(starts[sel])]; L = logp[order]
        cs = np.cumsum(np.vstack([np.zeros((1, L.shape[1])), L]), 0)
        for j in range(len(order)):
            a, b = max(0, j - h), min(len(order), j + h + 1); out[order[j]] = (cs[b] - cs[a]) / (b - a)
    return out


def split_ref(task):
    rid = np.where(task.ref)[0]; rn = np.zeros_like(task.ref); rn[rid[::2]] = True; rc = np.zeros_like(task.ref); rc[rid[1::2]] = True
    return Task(task.name, task.train, task.test, rn), rc


class Net(nn.Module):
    def __init__(self, d, n_cls, h=256, n_prim=0):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, h), nn.GELU(), nn.Dropout(0.5), nn.Linear(h, h), nn.GELU(), nn.Dropout(0.5))
        self.head = nn.Linear(h, n_cls); self.prim = nn.Linear(h, n_prim) if n_prim else None
    def forward(self, x):
        z = self.enc(x); return self.head(z), (self.prim(z) if self.prim is not None else None), z


def coral_loss(z, d):
    doms = torch.unique(d); loss = 0; n = 0
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            a, b = z[d == doms[i]], z[d == doms[j]]
            if len(a) > 2 and len(b) > 2: loss = loss + ((torch.cov(a.T) - torch.cov(b.T)) ** 2).sum() / (4 * z.shape[1] ** 2); n += 1
    return loss / max(n, 1)


def run_mlp(F, meta, task, classes, a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); tn, rc = split_ref(task)
    G = normalize(F, meta, tn, 'healthy'); y = np.array([classes.index(s) for s in meta['state']])
    tr, te = np.where(task.train)[0], np.where(task.test)[0]
    hc = classes.index(a.holdout_class) if a.holdout_class else -1
    if hc >= 0: tr = tr[y[tr] != hc]
    mu, sd = G[tr].mean(0), G[tr].std(0) + 1e-6
    rci = np.where(rc)[0]; keep = np.concatenate([tr, te, rci]); pos = -np.ones(len(meta), int); pos[keep] = np.arange(len(keep))
    G = ((G[keep] - mu) / sd).astype(np.float32)                       # only the windows this task touches
    tr_l, te_l, rci_l = pos[tr], pos[te], pos[rci]
    X = torch.tensor(G).to(dev); Y = torch.tensor(y[keep]).to(dev); trn = torch.tensor(tr_l).to(dev)
    P = torch.tensor(np.array([[1.0 if p in PRIM[s] else 0.0 for p in PRIMS] for s in meta['state'][keep]], np.float32)).to(dev)
    dom = torch.tensor(np.unique(domain_key(meta)[tr], return_inverse=True)[1]).to(dev)
    net = Net(G.shape[1], len(classes), n_prim=len(PRIMS) if a.multilabel else 0).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    steps = 0
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(perm), a.bs):
            sel = perm[i:i + a.bs]; idx = trn[sel]; xb = X[idx]; yb = F_.one_hot(Y[idx], len(classes)).float(); pb = P[idx]; db = dom[sel]
            if a.mixup > 0:
                j = torch.randperm(len(idx), device=dev); lam = torch.distributions.Beta(a.mixup, a.mixup).sample((len(idx), 1)).to(dev)
                xb = lam * xb + (1 - lam) * xb[j]; yb = lam * yb + (1 - lam) * yb[j]; pb = lam * pb + (1 - lam) * pb[j]
            logits, prim, z = net(xb); loss = -(yb * F_.log_softmax(logits, 1)).sum(1).mean()
            if a.multilabel: loss = loss + 0.5 * F_.binary_cross_entropy_with_logits(prim, pb)
            if a.coral > 0 and a.mixup == 0: loss = loss + a.coral * coral_loss(z, db)
            elif a.coral > 0:   # CORAL on un-mixed features of the same batch
                _, _, z0 = net(X[idx]); loss = loss + a.coral * coral_loss(z0, db)
            opt.zero_grad(); loss.backward(); opt.step(); steps += 1
            if a.max_steps and steps >= a.max_steps: break
        if a.max_steps and steps >= a.max_steps: break
    net.eval()
    with torch.no_grad():
        logp_te = F_.log_softmax(net(X[torch.tensor(te_l).to(dev)])[0], 1).cpu().numpy()
        logp_rc = F_.log_softmax(net(X[torch.tensor(rci_l).to(dev)])[0], 1).cpu().numpy()
    res = {}
    p = logp_te.argmax(1); res['acc'] = accuracy_score(y[te], p); res['f1'] = f1_score(y[te], p, average='macro')
    if a.smooth > 1:
        ps = smooth_logp(logp_te, meta, te, a.smooth).argmax(1); res['acc_smooth'] = accuracy_score(y[te], ps); res['f1_smooth'] = f1_score(y[te], ps, average='macro')
    if hc >= 0:
        known = y[te] != hc; H = classes.index('healthy')
        nn_ = NearestNeighbors(n_neighbors=5).fit(G[tr_l])
        s_soft = 1 - np.exp(logp_te).max(1); s_knn = nn_.kneighbors(G[te_l])[0].mean(1)
        # reference-calibrated p-values (calibration = odd reference windows, healthy): p = P(cal score >= s)
        c_soft = 1 - np.exp(logp_rc).max(1); c_knn = nn_.kneighbors(G[rci_l])[0].mean(1)
        def pv(cal, s):
            cs = np.sort(cal); return (len(cs) - np.searchsorted(cs, s, side='left') + 1) / (len(cs) + 1)
        p1, p2 = pv(c_soft, s_soft), pv(c_knn, s_knn)
        fisher = -2 * (np.log(p1) + np.log(p2)); minp = -np.minimum(p1, p2); rank = rankdata(s_soft) + rankdata(s_knn)
        for nm, sc in [('softmax', s_soft), ('knn', s_knn), ('fisher', fisher), ('minp', minp), ('rank', rank)]:
            res[f'auroc_{nm}'] = roc_auc_score(~known, sc)
        res['acc_known'] = (p[known] == y[te][known]).mean()
    return res, dict(prob=np.exp(logp_te).astype(np.float32), y=y[te], te=te, ref_prob=np.exp(logp_rc).astype(np.float32), ref_idx=rci)


def run_linear(F, meta, task, classes, mode):
    tn, _ = split_ref(task); G = normalize(F, meta, tn if mode == 'healthy' else task, mode); y = meta['state']
    tr, te = task.train, task.test
    c = make_pipeline(StandardScaler(), PCA(n_components=60), LogisticRegression(max_iter=3000, C=0.3)); c.fit(G[tr], y[tr]); p = c.predict(G[te])
    return dict(acc=accuracy_score(y[te], p), f1=f1_score(y[te], p, average='macro'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='mlp'); ap.add_argument('--protocol', default='mic_x_speed_x_load'); ap.add_argument('--ref_seconds', type=float, default=60)
    ap.add_argument('--feats', default='psd,cyclo'); ap.add_argument('--mixup', type=float, default=0.4); ap.add_argument('--multilabel', action='store_true'); ap.add_argument('--coral', type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument('--bs', type=int, default=512); ap.add_argument('--max_steps', type=int, default=0); ap.add_argument('--smooth', type=int, default=9)
    ap.add_argument('--holdout_class', default=''); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--norm', default='healthy', help='linear only: raw|healthy|oracle'); ap.add_argument('--tasks', default=''); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    z = np.load(os.path.join(ROOT, 'cache', 'guilin_full_feats.npz'), allow_pickle=True); meta = z['meta']
    parts = []
    if 'psd' in a.feats: parts.append(z['psd'].astype(np.float32))
    if 'cyclo' in a.feats: parts.append(z['cyclo'].astype(np.float32).reshape(len(meta), -1))
    F = np.hstack(parts); del parts; classes = sorted(np.unique(meta['state']).tolist())
    tasks = make_tasks(meta, a.protocol, ref_seconds=a.ref_seconds)
    if a.tasks: tasks = [t for t in tasks if t.name in a.tasks.split(',')]
    print(f'{a.model} {a.protocol}: {len(tasks)} tasks, {len(meta)} windows, feature dim {F.shape[1]}', flush=True)
    res = {}; probs = {}; t0 = time.time()
    for t in tasks:
        if a.model == 'linear': r = run_linear(F, meta, t, classes, a.norm)
        else: r, pr = run_mlp(F, meta, t, classes, a); probs[t.name] = pr
        res[t.name] = r; print(f'{t.name:28s} ' + ' '.join(f'{k}={v:.3f}' for k, v in r.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)
    keys = [k for k in next(iter(res.values())) if k.startswith('acc') or k.startswith('auroc')]
    print('MEAN ' + ' '.join(f'{k}={np.mean([r[k] for r in res.values()]):.3f}(min {np.min([r[k] for r in res.values()]):.3f})' for k in keys))
    tag = f'{a.model}{"_" + a.norm if a.model == "linear" else ""}{"_" + a.feats.replace(",", "+") if a.feats != "psd,cyclo" else ""}{"_ml" if a.multilabel else ""}{"_coral" + str(a.coral) if a.coral else ""}{"_mix" + str(a.mixup) if a.model == "mlp" and a.mixup != 0.4 else ""}{"_holdout_" + a.holdout_class if a.holdout_class else ""}{a.tag}'
    name = f'full_{a.protocol}_{tag}_s{a.seed}.json'; os.makedirs(os.path.join(ROOT, 'outputs'), exist_ok=True)
    json.dump(dict(args=vars(a), classes=classes, results=res), open(os.path.join(ROOT, 'outputs', name), 'w'), indent=1); print('saved outputs/' + name)
    if probs: np.savez(os.path.join(ROOT, 'outputs', name.replace('.json', '_probs.npz')), **{k + '/' + kk: v for k, d in probs.items() for kk, v in d.items()})
