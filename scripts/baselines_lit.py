"""Literature baselines under the identical Reference-Only mic x speed protocol (Guilin), all on the same cached
features (log-PSD 128 + order-normalized cyclostationary map 32x100) unless stated. Every baseline is trained on the
source domains only; what differs is how (or whether) the target domain's data is used at test time.

  src_only      : MLP with BatchNorm on RAW features (no reference standardization)                [lower bound]
  adabn_ref     : same model; at test time BN statistics re-estimated on the target HEALTHY REFERENCE only   [Reference-Only]
  adabn_test    : BN statistics re-estimated on the unlabeled target TEST batch (AdaBN, Li et al. 2016)      [transductive]
  tent          : TENT (Wang et al. ICLR 2021): BN affine params updated by entropy minimization on test batch [transductive]
  dann          : domain-adversarial training over the 8 source domains (Ganin et al. 2016) + reference standardization
  coral         : Deep CORAL alignment between source domains (Sun & Saenko 2016) + reference standardization
  cscoh_cnn     : 2D CNN on the cyclic-spectral map with FIXED cyclic-frequency axis, raw (Chen et al. MSSP 2020 style)
  cscoh_cnn_ord : same CNN on the ORDER-normalized map, raw  (isolates the effect of the classifier vs. our normalization)
  cscoh_cnn_ref : same CNN on the order-normalized map with healthy-reference standardization

python scripts/baselines_lit.py [--which src_only,adabn_ref,...] [--seed 0]
"""
import os, sys, time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.eval.protocol import make_tasks, normalize, domain_key, Task
from sklearn.metrics import accuracy_score, f1_score
dev = 'cuda'


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None


def mlp_bn(d_in, n_cls, hidden=256):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.5),
                         nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.5), nn.Linear(hidden, n_cls))


class CSCohCNN(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.h = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, n_cls))
    def forward(self, x): return self.h(self.f(x).flatten(1))


def split_ref(task):
    rid = np.where(task.ref)[0]; rn = np.zeros_like(task.ref); rn[rid[::2]] = True
    return Task(task.name, task.train, task.test, rn)


def fit(model, X, y, tr, epochs=40, bs=256, lr=1e-3, dom=None, mode='ce', lam=0.1, n_dom=0, seed=0):
    torch.manual_seed(seed); model = model.to(dev); opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-2)
    Xt = torch.tensor(X, dtype=torch.float32).to(dev); Y = torch.tensor(y).to(dev); trn = torch.tensor(tr).to(dev)
    D = torch.tensor(dom).to(dev) if dom is not None else None
    dom_head = nn.Linear(256, n_dom).to(dev) if mode == 'dann' else None
    if dom_head is not None: opt.add_param_group({'params': dom_head.parameters()})
    feat = lambda xb: model[:-1](xb) if isinstance(model, nn.Sequential) else model.f(xb).flatten(1)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(perm), bs):
            idx = trn[perm[i:i + bs]]; xb, yb = Xt[idx], Y[idx]
            if mode == 'dann':
                z = feat(xb); logits = model[-1](z); loss = F_.cross_entropy(logits, yb)
                p = min(1.0, (ep * len(perm) + i) / (epochs * len(perm))); l = 2 / (1 + np.exp(-10 * p)) - 1
                loss = loss + lam * F_.cross_entropy(dom_head(GRL.apply(z, l)), D[idx])
            elif mode == 'coral':
                z = feat(xb); logits = model[-1](z); loss = F_.cross_entropy(logits, yb); d = D[idx]; cl = 0; n = 0
                doms = torch.unique(d)
                for a in range(len(doms)):
                    for b in range(a + 1, len(doms)):
                        za, zb = z[d == doms[a]], z[d == doms[b]]
                        if len(za) > 2 and len(zb) > 2:
                            ca = torch.cov(za.T); cb = torch.cov(zb.T); cl = cl + ((ca - cb) ** 2).sum() / (4 * z.shape[1] ** 2); n += 1
                loss = loss + lam * (cl / max(n, 1))
            else:
                loss = F_.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def predict(model, X, idx, bs=1024):
    model.eval(); out = []
    with torch.no_grad():
        Xt = torch.tensor(X[idx], dtype=torch.float32).to(dev)
        for i in range(0, len(idx), bs): out.append(F_.softmax(model(Xt[i:i + bs]), 1).cpu().numpy())
    return np.concatenate(out)


def adabn(model, X, idx):
    """Re-estimate BatchNorm running statistics on X[idx] (no labels)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)): m.reset_running_stats(); m.momentum = None
    model.train()
    with torch.no_grad():
        Xt = torch.tensor(X[idx], dtype=torch.float32).to(dev)
        for i in range(0, len(idx), 256): model(Xt[i:i + 256])
    return model


def tent(model, X, idx, steps=1, lr=1e-3):
    """TENT: adapt BN affine parameters by entropy minimization on the unlabeled test batch (online, one pass)."""
    adabn(model, X, idx); model.train()
    params = [p for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for p in m.parameters()]
    for p in model.parameters(): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr); Xt = torch.tensor(X[idx], dtype=torch.float32).to(dev)
    for _ in range(steps):
        for i in range(0, len(idx), 256):
            logits = model(Xt[i:i + 256]); loss = -(F_.softmax(logits, 1) * F_.log_softmax(logits, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    for p in model.parameters(): p.requires_grad_(True)
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--which', default='src_only,adabn_ref,adabn_test,tent,dann,coral,cscoh_cnn,cscoh_cnn_ord,cscoh_cnn_ref'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=40); a = ap.parse_args()
    z = np.load(os.path.join(ROOT, 'cache', 'guilin_feats.npz'), allow_pickle=True); meta = z['meta']
    psd = z['psd'].astype(np.float32); cyc = z['cyclo'].mean(3).astype(np.float32); cyc_fixed = np.load(os.path.join(ROOT, 'cache', 'guilin_cyclo_fixedalpha.npy')).astype(np.float32)
    F = np.hstack([psd, cyc.reshape(len(meta), -1)]); classes = sorted(np.unique(meta['state']).tolist()); y = np.array([classes.index(s) for s in meta['state']])
    keys = domain_key(meta); dom_id = np.unique(keys, return_inverse=True)[1]
    tasks = make_tasks(meta, 'mic_x_speed'); res = {w: {} for w in a.which.split(',')}; t0 = time.time()
    for t in tasks:
        tr, te = np.where(t.train)[0], np.where(t.test)[0]; tn = split_ref(t); ref = np.where(tn.ref)[0]
        dtr = np.unique(dom_id[tr], return_inverse=True)[1]; dfull = np.zeros(len(meta), int); dfull[tr] = dtr
        # raw features, standardized with SOURCE statistics only (no reference)
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6; Fraw = (F - mu) / sd
        Fref = normalize(F, meta, tn, 'healthy'); mu2, sd2 = Fref[tr].mean(0), Fref[tr].std(0) + 1e-6; Fref = (Fref - mu2) / sd2
        for w in res:
            if w in ('src_only', 'adabn_ref', 'adabn_test', 'tent'):
                m = fit(mlp_bn(F.shape[1], len(classes)), Fraw, y, tr, a.epochs, seed=a.seed)
                if w == 'adabn_ref': adabn(m, Fraw, ref)
                if w == 'adabn_test': adabn(m, Fraw, te)
                if w == 'tent': tent(m, Fraw, te)
                p = predict(m, Fraw, te)
            elif w in ('dann', 'coral'):
                m = fit(mlp_bn(F.shape[1], len(classes)), Fref, y, tr, a.epochs, dom=dfull, mode=w, n_dom=int(dtr.max()) + 1, seed=a.seed); p = predict(m, Fref, te)
            else:
                M = cyc_fixed if w == 'cscoh_cnn' else cyc
                if w == 'cscoh_cnn_ref':
                    Mf = normalize(M.reshape(len(meta), -1), meta, tn, 'healthy').reshape(len(meta), 32, 100)
                else:
                    Mf = M
                mm, ss = Mf[tr].mean(), Mf[tr].std() + 1e-6; X4 = ((Mf - mm) / ss)[:, None]
                m = fit(CSCohCNN(len(classes)), X4, y, tr, a.epochs, bs=128, seed=a.seed); p = predict(m, X4, te)
            acc = accuracy_score(y[te], p.argmax(1)); f1 = f1_score(y[te], p.argmax(1), average='macro'); res[w][t.name] = dict(acc=acc, f1=f1)
        print(f'{t.name:22s} ' + ' '.join(f'{w}={res[w][t.name]["acc"]:.3f}' for w in res) + f'  ({time.time()-t0:.0f}s)', flush=True)
    print('\nMEAN / MIN over 9 tasks')
    for w in res:
        accs = [v['acc'] for v in res[w].values()]; f1s = [v['f1'] for v in res[w].values()]; print(f'{w:14s} acc {np.mean(accs):.3f} (min {np.min(accs):.3f})  macroF1 {np.mean(f1s):.3f}')
    json.dump(dict(args=vars(a), classes=classes, results=res), open(os.path.join(ROOT, 'outputs', f'guilin_mic_x_speed_baselines_lit_s{a.seed}.json'), 'w'), indent=1)
