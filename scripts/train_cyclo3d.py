"""Train/evaluate Cyclo3DNet on cached cyclo volumes under the Reference-Only protocols.

python scripts/train_cyclo3d.py guilin --protocol mic_x_speed --epochs 30 [--no_ref] [--tasks mxs:CRY333@1200,...]

For each task: training windows from the source domains; every window is paired with the mean healthy-reference
volume of ITS OWN domain (train domains: healthy windows of train segments; test domains: healthy windows of the
reference segment). Reports accuracy / macro-F1 per task and the mean.
"""
import os, sys, time, argparse, json, numpy as np, torch, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.eval.protocol import make_tasks, domain_key
from refpath.models.cyclo3d import Cyclo3DNet
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

PRIM = {'healthy': [], 'inner_pitting': ['inner'], 'outer_pitting': ['outer'], 'inner_outer_pitting': ['inner', 'outer'],
        'cage_fracture': ['cage'], 'ball_pitting': ['ball'], 'ball_crack': ['ball'], 'ball_wear': ['ball'],
        'inner_crack': ['inner'], 'inner_wear': ['inner'], 'outer_crack': ['outer'], 'outer_wear': ['outer']}
PRIMS = ['inner', 'outer', 'ball', 'cage']


def reference_stats(X, meta, task):
    """Per-domain healthy-reference mean and std, broadcast to every window: returns (M, S) with X's shape."""
    keys = domain_key(meta); healthy = meta['state'] == 'healthy'; M = np.zeros_like(X); S = np.ones_like(X)
    for k in np.unique(keys):
        d = keys == k
        ref = d & task.ref if (d & task.test).any() else d & task.train & healthy
        if ref.sum() == 0:
            ref = d & healthy
        M[d] = X[ref].mean(0); S[d] = X[ref].std(0)
    return M, S


def run_task(cyc, meta, task, classes, args, dev='cuda', psd=None):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    R, RS = reference_stats(cyc, meta, task)
    if psd is not None:
        PM, PS = reference_stats(psd, meta, task); psd_z = ((psd - PM) / (PS + 1e-3)).astype(np.float32)
    y = np.array([classes.index(s) for s in meta['state']])
    P = np.array([[1.0 if p in PRIM[s] else 0.0 for p in PRIMS] for s in meta['state']], np.float32)
    tr, te = np.where(task.train)[0], np.where(task.test)[0]
    Xtr = torch.tensor(cyc[tr]).unsqueeze(1).to(dev); Rtr = torch.tensor(R[tr]).unsqueeze(1).to(dev)
    ytr = torch.tensor(y[tr]).to(dev); Ptr = torch.tensor(P[tr]).to(dev)
    Xte = torch.tensor(cyc[te]).unsqueeze(1).to(dev); Rte = torch.tensor(R[te]).unsqueeze(1).to(dev)
    RStr = torch.tensor(RS[tr]).unsqueeze(1).to(dev) if args.ref_std else None; RSte = torch.tensor(RS[te]).unsqueeze(1).to(dev) if args.ref_std else None
    Qtr = torch.tensor(psd_z[tr]).to(dev) if psd is not None else None; Qte = torch.tensor(psd_z[te]).to(dev) if psd is not None else None
    model = Cyclo3DNet(n_f=cyc.shape[1], n_o=cyc.shape[2], n_t=cyc.shape[3], n_classes=len(classes), width=args.width,
                       use_ref=not args.no_ref, multilabel=len(PRIMS) if args.multilabel else None, n_psd=psd.shape[1] if psd is not None else 0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * ((len(tr) + args.bs - 1) // args.bs))
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]; xb, rb, yb, pb = Xtr[idx], Rtr[idx], ytr[idx], Ptr[idx]
            rsb = RStr[idx] if RStr is not None else None; qb = Qtr[idx] if Qtr is not None else None
            if args.aug:   # cheap augmentation: random gain on F axis (path-like) and small order jitter (speed-like)
                xb = xb + 0.3 * torch.randn(xb.shape[0], 1, xb.shape[2], 1, 1, device=dev)
                sh = int(np.random.randint(-2, 3)); xb = torch.roll(xb, sh, dims=3)
            out = model(xb, rb, rsb, qb); loss = F_.cross_entropy(out['logits'], yb, label_smoothing=0.05)
            if args.multilabel:
                loss = loss + 0.5 * F_.binary_cross_entropy_with_logits(out['prim_logits'], pb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(te), 512):
            preds.append(model(Xte[i:i + 512], Rte[i:i + 512], RSte[i:i + 512] if RSte is not None else None, Qte[i:i + 512] if Qte is not None else None)['logits'].argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    return accuracy_score(y[te], p), f1_score(y[te], p, average='macro'), confusion_matrix(y[te], p, labels=range(len(classes)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset'); ap.add_argument('--protocol', default='mic_x_speed'); ap.add_argument('--tasks', default='')
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--width', type=int, default=32); ap.add_argument('--no_ref', action='store_true'); ap.add_argument('--multilabel', action='store_true')
    ap.add_argument('--aug', action='store_true'); ap.add_argument('--ref_std', action='store_true'); ap.add_argument('--psd', action='store_true'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    z = np.load(os.path.join(ROOT, 'cache', f'{a.dataset}_feats.npz'), allow_pickle=True); meta, cyc = z['meta'], z['cyclo'].astype(np.float32)
    psd = z['psd'].astype(np.float32) if a.psd else None
    classes = sorted(np.unique(meta['state']).tolist())
    tasks = make_tasks(meta, a.protocol)
    if a.tasks: tasks = [t for t in tasks if t.name in a.tasks.split(',')]
    res = {}; t0 = time.time()
    for t in tasks:
        acc, f1, cm = run_task(cyc, meta, t, classes, a, psd=psd)
        res[t.name] = dict(acc=acc, f1=f1, cm=cm.tolist()); print(f'{t.name:22s} acc {acc:.3f} f1 {f1:.3f}  ({time.time()-t0:.0f}s)', flush=True)
    accs = [r['acc'] for r in res.values()]; print(f'MEAN acc {np.mean(accs):.3f}  min {np.min(accs):.3f}')
    os.makedirs(os.path.join(ROOT, 'outputs'), exist_ok=True)
    name = f'{a.dataset}_{a.protocol}_cyclo3d{"_noref" if a.no_ref else ""}{"_ml" if a.multilabel else ""}{"_aug" if a.aug else ""}{"_refstd" if a.ref_std else ""}{"_psd" if a.psd else ""}{a.tag}.json'
    json.dump(dict(args=vars(a), classes=classes, results=res), open(os.path.join(ROOT, 'outputs', name), 'w'), indent=1)
    print('saved outputs/' + name)
