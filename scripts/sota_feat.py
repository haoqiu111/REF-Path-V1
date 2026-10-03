"""Same-protocol re-implementations of domain-shift methods on the cached physics features (log-PSD 128 + order-normalized
cyclostationary map 32 x 100), EVERY method in two versions: 'raw' (features z-scored with source statistics only, no target
information) and 'ref' (healthy-reference standardization of every domain first, the operation proposed in the paper).

Protocols: mic_x_speed (9 cells, 60-s clips, 40 epochs, bs 256) and mic_x_speed_x_load (27 cells, full recordings, 4 epochs, bs 512).

Methods (name_raw / name_ref in the output):
  erm        BatchNorm MLP, cross-entropy on the source domains                                        [no target use beyond norm]
  adabnref   erm + AdaBN re-estimated on the target HEALTHY REFERENCE (Li et al. 2018)                 [Reference-Only]
  adabntest  erm + AdaBN on the unlabeled test batch (shuffled, i.i.d. stream)                         [transductive]
  imb        adabntest / tent / sar adapted on a test stream with healthy share h in {0.5, 0.8, 0.95, 1.0}, keys <method>_<norm>_h<h>;
             all healthy test windows plus a class-uniform random subset of the faulty ones, evaluated on the full test set
  tent       erm + TENT entropy minimization of the BN affine parameters (Wang et al. ICLR 2021)       [transductive]
  sar        erm + SAR: sharpness-aware, reliable entropy minimization (Niu et al. ICLR 2023)          [transductive]
  dann       domain-adversarial training over the source domains (Ganin et al. 2016)
  coral      Deep CORAL between source domains (Sun & Saenko 2016)
  sagm       sharpness-aware gradient matching for domain generalization (Wang et al. CVPR 2023)
  ddg        dual disentanglement with mixed fusion of the two views (log-PSD view, cyclostationary view): invariant /
             specific codes per view, domain heads, orthogonality, cross-domain mixing of the specific codes
             (multi-modal dual-disentanglement paradigm, Xia et al. MSSP 2026)
  fgdg       frequency-guided multi-source domain generalization: carrier-spectrum (amplitude) mixing between source
             domains with prediction consistency (frequency-guidance paradigm, Wang et al. Sensors 2026)
  cscoh      2-D CNN on the order-normalized cyclic-spectral map (Chen et al. MSSP 2020 recipe)

python scripts/sota_feat.py --protocol mic_x_speed --seed 0 [--which erm,dann,...] [--epochs N] [--tasks ...]
-> outputs/sota_feat_<protocol>_s<seed>.json   {results: {method_norm: {task: {acc, f1}}}}
"""
import os, sys, time, argparse, json, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.eval.protocol import make_tasks, domain_key
from sklearn.metrics import accuracy_score, f1_score
dev = 'cuda'; NPSD, NF, NO = 128, 32, 100


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None


class BNMLP(nn.Module):
    def __init__(self, d_in, n_cls, h=256):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.5), nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.5))
        self.head = nn.Linear(h, n_cls)
    def forward(self, x): return self.head(self.feat(x))


class CSCohCNN(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.h = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, n_cls))
    def forward(self, x): return self.h(self.f(x.view(-1, 1, NF, NO)).flatten(1))


class DDG(nn.Module):
    """Dual disentanglement of two views (log-PSD, cyclostationary map) into domain-invariant and domain-specific codes."""
    def __init__(self, n_cls, n_dom, d=64):
        super().__init__()
        self.e_psd = nn.Sequential(nn.Linear(NPSD, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3))
        self.e_cyc = nn.Sequential(nn.Linear(NF * NO, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.5))
        self.inv_p, self.spec_p = nn.Linear(128, d), nn.Linear(128, d); self.inv_c, self.spec_c = nn.Linear(256, d), nn.Linear(256, d)
        self.cls = nn.Sequential(nn.GELU(), nn.Dropout(0.3), nn.Linear(2 * d, n_cls))
        self.cls_mix = nn.Sequential(nn.GELU(), nn.Linear(4 * d, n_cls))        # mixed fusion: invariant codes + (swapped) specific codes
        self.dom_spec = nn.Linear(2 * d, n_dom); self.dom_adv = nn.Linear(2 * d, n_dom)
    def codes(self, x):
        hp, hc = self.e_psd(x[:, :NPSD]), self.e_cyc(x[:, NPSD:])
        return torch.cat([self.inv_p(hp), self.inv_c(hc)], 1), torch.cat([self.spec_p(hp), self.spec_c(hc)], 1)
    def forward(self, x): return self.cls(self.codes(x)[0])


def ce_soft(logits, y1h): return -(y1h * F_.log_softmax(logits, 1)).sum(1).mean()


def train(model, X, Y, D, ntr, n_dom, mode, epochs, bs, seed, n_cls):
    torch.manual_seed(seed); model = model.to(dev)
    extra = []
    if mode == 'dann': dom_head = nn.Linear(256, n_dom).to(dev); extra = list(dom_head.parameters())
    opt = torch.optim.AdamW(list(model.parameters()) + extra, 1e-3, weight_decay=1e-2); steps_total = epochs * ((ntr + bs - 1) // bs); step = 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(ntr, device=dev)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            if len(idx) < 4: continue
            xb, yb, db = X[idx], Y[idx], D[idx]; step += 1
            if mode == 'ce':
                loss = F_.cross_entropy(model(xb), yb)
            elif mode == 'dann':
                z = model.feat(xb); l = 2 / (1 + np.exp(-10 * step / steps_total)) - 1
                loss = F_.cross_entropy(model.head(z), yb) + 0.1 * F_.cross_entropy(dom_head(GRL.apply(z, l)), db)
            elif mode == 'coral':
                z = model.feat(xb); loss = F_.cross_entropy(model.head(z), yb); cl = 0; n = 0; doms = torch.unique(db)
                covs = [torch.cov(z[db == d_].T) for d_ in doms if (db == d_).sum() > 2]
                for a_ in range(len(covs)):
                    for b_ in range(a_ + 1, len(covs)): cl = cl + ((covs[a_] - covs[b_]) ** 2).sum() / (4 * z.shape[1] ** 2); n += 1
                loss = loss + 0.1 * (cl / max(n, 1))
            elif mode == 'sagm':   # g + grad at (theta + rho g/|g| - alpha g)
                rho, alpha = 0.05, 0.001; params = [p for p in model.parameters() if p.requires_grad]
                loss = F_.cross_entropy(model(xb), yb); g1 = torch.autograd.grad(loss, params)
                gn = torch.sqrt(sum((g ** 2).sum() for g in g1)) + 1e-12; eps = [rho * g / gn - alpha * g for g in g1]
                with torch.no_grad():
                    for p, e in zip(params, eps): p.add_(e)
                loss2 = F_.cross_entropy(model(xb), yb); g2 = torch.autograd.grad(loss2, params)
                with torch.no_grad():
                    for p, e in zip(params, eps): p.sub_(e)
                opt.zero_grad()
                for p, a_, b_ in zip(params, g1, g2): p.grad = a_ + b_
                opt.step(); continue
            elif mode == 'fgdg':   # carrier-spectrum (amplitude) mixing between samples of other source domains + consistency
                j = torch.randperm(len(idx), device=dev); lam = torch.rand(len(idx), 1, device=dev)
                psd, mp = xb[:, :NPSD], xb[:, NPSD:].view(-1, NF, NO); prof = mp.mean(2)
                psd_a = psd + lam * (psd[j] - psd); mp_a = mp + (lam * (prof[j] - prof)).unsqueeze(2)
                xa = torch.cat([psd_a, mp_a.reshape(len(idx), -1)], 1)
                lo, la = model(xb), model(xa)
                loss = F_.cross_entropy(lo, yb) + F_.cross_entropy(la, yb) + 2.0 * F_.kl_div(F_.log_softmax(la, 1), F_.softmax(lo.detach(), 1), reduction='batchmean')
            elif mode == 'ddg':
                inv, spec = model.codes(xb); l = 2 / (1 + np.exp(-10 * step / steps_total)) - 1; j = torch.randperm(len(idx), device=dev)
                loss = F_.cross_entropy(model.cls(inv), yb) + 0.5 * F_.cross_entropy(model.cls_mix(torch.cat([inv, spec[j]], 1)), yb) \
                    + 0.1 * F_.cross_entropy(model.dom_spec(spec), db) + 0.1 * F_.cross_entropy(model.dom_adv(GRL.apply(inv, l)), db) \
                    + 0.1 * (F_.normalize(inv, dim=1) * F_.normalize(spec, dim=1)).sum(1).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def predict(model, X, sl, bs=4096):
    model.eval(); return torch.cat([model(X[sl][i:i + bs]).argmax(1) for i in range(0, sl.stop - sl.start, bs)]).cpu().numpy()


def bn_layers(model): return [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]


@torch.no_grad()
def adabn(model, X, sl):
    for m in bn_layers(model): m.reset_running_stats(); m.momentum = None
    model.train()
    for i in range(sl.start, sl.stop, 256): model(X[i:min(i + 256, sl.stop)])
    return model


def tent(model, X, sl, lr=1e-3):
    adabn(model, X, sl); model.train(); params = [p for m in bn_layers(model) for p in m.parameters()]
    for p in model.parameters(): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr)
    for i in range(sl.start, sl.stop, 256):
        xb = X[i:min(i + 256, sl.stop)]
        if len(xb) < 4: continue
        lg = model(xb); loss = -(F_.softmax(lg, 1) * F_.log_softmax(lg, 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return model


def sar(model, X, sl, n_cls, lr=1e-3, rho=0.05):
    """SAR (Niu et al., ICLR 2023): entropy minimization on reliable samples (entropy < 0.4 ln C) with a sharpness-aware step,
    norm-layer affine parameters only, one online pass over the test stream."""
    adabn(model, X, sl); model.train(); params = [p for m in bn_layers(model) for p in m.parameters()]
    for p in model.parameters(): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    opt = torch.optim.SGD(params, lr, momentum=0.9); e0 = 0.4 * np.log(n_cls)
    ent = lambda lg: -(F_.softmax(lg, 1) * F_.log_softmax(lg, 1)).sum(1)
    for i in range(sl.start, sl.stop, 256):
        xb = X[i:min(i + 256, sl.stop)]
        if len(xb) < 4: continue
        e = ent(model(xb)); keep = e < e0
        if keep.sum() < 2: continue
        g1 = torch.autograd.grad(e[keep].mean(), params); gn = torch.sqrt(sum((g ** 2).sum() for g in g1)) + 1e-12; eps = [rho * g / gn for g in g1]
        with torch.no_grad():
            for p, d_ in zip(params, eps): p.add_(d_)
        e2 = ent(model(xb)); keep2 = keep & (e2 < e0)
        if keep2.sum() < 2:
            with torch.no_grad():
                for p, d_ in zip(params, eps): p.sub_(d_)
            continue
        g2 = torch.autograd.grad(e2[keep2].mean(), params)
        with torch.no_grad():
            for p, d_ in zip(params, eps): p.sub_(d_)
        opt.zero_grad()
        for p, g in zip(params, g2): p.grad = g
        opt.step()
    for p in model.parameters(): p.requires_grad_(True)
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--protocol', default='mic_x_speed'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--which', default='erm,adabnref,adabntest,tent,sar,imb,dann,coral,sagm,ddg,fgdg,cscoh'); ap.add_argument('--epochs', type=int, default=0); ap.add_argument('--tasks', default=''); ap.add_argument('--tag', default='')
    a = ap.parse_args(); which = a.which.split(','); torch.manual_seed(a.seed); rs = np.random.RandomState(a.seed); HS = [0.5, 0.8, 0.95, 1.0]
    if a.protocol == 'mic_x_speed':
        z = np.load(os.path.join(ROOT, 'cache', 'guilin_feats.npz'), allow_pickle=True); meta = z['meta']
        F = np.hstack([z['psd'].astype(np.float32), z['cyclo'].mean(3).astype(np.float32).reshape(len(meta), -1)]); tasks = make_tasks(meta, a.protocol); epochs, bs = a.epochs or 40, 256
    else:
        z = np.load(os.path.join(ROOT, 'cache', 'guilin_full_feats.npz'), allow_pickle=True); meta = z['meta']
        F = np.hstack([z['psd'].astype(np.float32), z['cyclo'].astype(np.float32).reshape(len(meta), -1)]); tasks = make_tasks(meta, a.protocol, ref_seconds=60); epochs, bs = a.epochs or 4, 512
    if a.tasks: tasks = [t for t in tasks if t.name in a.tasks.split(',')]
    classes = sorted(np.unique(meta['state']).tolist()); y = np.array([classes.index(s) for s in meta['state']]); NC = len(classes); H = classes.index('healthy')
    keys = domain_key(meta); healthy = meta['state'] == 'healthy'
    print(f'{a.protocol}: {len(tasks)} tasks, {len(meta)} windows, dim {F.shape[1]}, epochs {epochs}, bs {bs}', flush=True)
    res = {}; t0 = time.time()
    for t in tasks:
        tr, te = np.where(t.train)[0], np.where(t.test)[0]; rf = np.where(t.ref)[0][::2]     # even reference windows = standardization subset
        idx = np.concatenate([tr, te, rf]); ntr, nte = len(tr), len(te); Fs = F[idx]; ks = keys[idx]; hs = healthy[idx]
        pos = np.arange(len(idx)); is_tr, is_te, is_rf = pos < ntr, (pos >= ntr) & (pos < ntr + nte), pos >= ntr + nte
        G = Fs.copy()
        for k in np.unique(ks):
            d = ks == k; r = d & is_rf if (d & is_te).any() else d & is_tr & hs
            if r.sum() < 2: continue
            G[d] = (Fs[d] - Fs[r].mean(0)) / (Fs[r].std(0) + 1e-6)
        dom = np.unique(ks[:ntr], return_inverse=True)[1]; n_dom = int(dom.max()) + 1
        Y = torch.tensor(y[idx]).to(dev); D = torch.zeros(len(idx), dtype=torch.long, device=dev); D[:ntr] = torch.tensor(dom).to(dev)
        s_tr, s_te, s_rf = slice(0, ntr), slice(ntr, ntr + nte), slice(ntr + nte, len(idx)); yt = y[te]
        def rec(name, p):
            res.setdefault(name, {})[t.name] = dict(acc=float(accuracy_score(yt, p)), f1=float(f1_score(yt, p, average='macro')))
        for norm, A in (('raw', Fs), ('ref', G)):
            mu, sd = A[:ntr].mean(0), A[:ntr].std(0) + 1e-6; X = torch.tensor((A - mu) / sd).to(dev)
            if any(w in which for w in ('erm', 'adabnref', 'adabntest', 'tent', 'sar', 'imb')):
                m = train(BNMLP(F.shape[1], NC), X, Y, D, ntr, n_dom, 'ce', epochs, bs, a.seed, NC); state = copy.deepcopy(m.state_dict())
                if 'erm' in which: rec(f'erm_{norm}', predict(m, X, s_te))
                # the transductive methods adapt on an i.i.d. (shuffled) stream of the unlabeled test windows, the standard test-time-adaptation
                # assumption; in recording order the class-balanced test set is sorted by class and every batch holds a single class
                Xs = X[ntr + torch.tensor(rs.permutation(nte)).to(dev)]; s_all = slice(0, nte)
                for w, fn in (('adabnref', lambda mm: adabn(mm, X, s_rf)), ('adabntest', lambda mm: adabn(mm, Xs, s_all)), ('tent', lambda mm: tent(mm, Xs, s_all)), ('sar', lambda mm: sar(mm, Xs, s_all, NC))):
                    if w in which:
                        m.load_state_dict(state)
                        for bnl in bn_layers(m): bnl.momentum = 0.1
                        rec(f'{w}_{norm}', predict(fn(m), X, s_te))
                if 'imb' in which:   # dependence on the composition of the test stream: healthy share h (0.2 = the class-balanced test set above)
                    hi = np.where(yt == H)[0]; fi = [np.where(yt == c)[0] for c in range(NC) if c != H]
                    for h in HS:
                        n_f = int(round(len(hi) * (1 - h) / h / len(fi))) if h < 1 else 0
                        st = rs.permutation(np.concatenate([hi] + [rs.choice(f, min(n_f, len(f)), replace=False) for f in fi if n_f > 0]))
                        Xh = X[ntr + torch.tensor(st).to(dev)]; s_h = slice(0, len(st))
                        for w, fn in (('adabntest', lambda mm: adabn(mm, Xh, s_h)), ('tent', lambda mm: tent(mm, Xh, s_h)), ('sar', lambda mm: sar(mm, Xh, s_h, NC))):
                            m.load_state_dict(state)
                            for bnl in bn_layers(m): bnl.momentum = 0.1
                            rec(f'{w}_{norm}_h{h}', predict(fn(m), X, s_te))
                del m, Xs
            for w in ('dann', 'coral', 'sagm', 'fgdg'):
                if w in which: rec(f'{w}_{norm}', predict(train(BNMLP(F.shape[1], NC), X, Y, D, ntr, n_dom, w, epochs, bs, a.seed, NC), X, s_te))
            if 'ddg' in which: rec(f'ddg_{norm}', predict(train(DDG(NC, n_dom), X, Y, D, ntr, n_dom, 'ddg', epochs, bs, a.seed, NC), X, s_te))
            if 'cscoh' in which:   # CNN on the map with a global (scalar) scaling, as in the CSCoh-CNN recipe
                Mp = A[:, NPSD:]; mm_, ss_ = Mp[:ntr].mean(), Mp[:ntr].std() + 1e-6; Xm = torch.tensor((Mp - mm_) / ss_).to(dev)
                rec(f'cscoh_{norm}', predict(train(CSCohCNN(NC), Xm, Y, D, ntr, n_dom, 'ce', epochs, min(bs, 256), a.seed, NC), Xm, s_te)); del Xm
            del X; torch.cuda.empty_cache()
        print(f'{t.name:26s} ' + ' '.join(f'{k}={v[t.name]["acc"]:.3f}' for k, v in res.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)
    print('\nMEAN / MIN over tasks')
    for k, v in res.items():
        accs = [x['acc'] for x in v.values()]; print(f'{k:16s} acc {np.mean(accs):.3f} (min {np.min(accs):.3f})')
    name = f'sota_feat_{a.protocol}{a.tag}_s{a.seed}.json'; json.dump(dict(args=vars(a), classes=classes, results=res), open(os.path.join(ROOT, 'outputs', name), 'w'), indent=1); print('saved outputs/' + name)
