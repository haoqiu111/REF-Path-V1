"""Low-capacity learned model on healthy-reference-standardized features (PSD + pooled cyclo volume) with
cross-domain mixup, to test whether a learned model can beat the linear Reference-Only baseline.

python scripts/train_mlp.py guilin --protocol mic_x_speed [--pca 60] [--mixup 0.4] [--epochs 40]
"""
import os, sys, time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.eval.protocol import make_tasks, normalize, domain_key
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score


def run_task(F, meta, task, classes, a, dev='cuda'):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    task_n = task
    if a.ref_split:   # split the reference: even windows -> standardization, odd windows -> reliability / conformal calibration
        from refpath.eval.protocol import Task
        rid = np.where(task.ref)[0]; ref_norm = np.zeros_like(task.ref); ref_norm[rid[::2]] = True; ref_cal = np.zeros_like(task.ref); ref_cal[rid[1::2]] = True
        task_n = Task(task.name, task.train, task.test, ref_norm)
    G = normalize(F, meta, task_n, 'healthy'); y = np.array([classes.index(s) for s in meta['state']])
    tr, te = np.where(task.train)[0], np.where(task.test)[0]
    if a.holdout_class:
        hc = classes.index(a.holdout_class); tr = tr[y[tr] != hc]
    if a.pca:
        p = PCA(n_components=a.pca).fit(G[tr]); G = p.transform(G).astype(np.float32)
    mu, sd = G[tr].mean(0), G[tr].std(0) + 1e-6; G = (G - mu) / sd
    X = torch.tensor(G, dtype=torch.float32).to(dev); Y = torch.tensor(y).to(dev)
    dom = domain_key(meta)[tr]; dom_id = np.unique(dom, return_inverse=True)[1]
    model = nn.Sequential(nn.Dropout(a.in_drop), nn.Linear(G.shape[1], a.hidden), nn.GELU(), nn.Dropout(a.drop), nn.Linear(a.hidden, a.hidden), nn.GELU(), nn.Dropout(a.drop), nn.Linear(a.hidden, len(classes))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); trn = torch.tensor(tr).to(dev)
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(perm), a.bs):
            idx = trn[perm[i:i + a.bs]]; xb, yb = X[idx], F_.one_hot(Y[idx], len(classes)).float()
            if a.mixup > 0:   # mix each sample with a sample from a DIFFERENT domain
                j = torch.randperm(len(idx), device=dev); lam = torch.distributions.Beta(a.mixup, a.mixup).sample((len(idx), 1)).to(dev)
                xb = lam * xb + (1 - lam) * xb[j]; yb = lam * yb + (1 - lam) * yb[j]
            loss = -(yb * F_.log_softmax(model(xb), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    rf = np.where(ref_cal if a.ref_split else task.ref)[0]
    with torch.no_grad():
        prob = F_.softmax(model(X[torch.tensor(te).to(dev)]), 1).cpu().numpy(); rprob = F_.softmax(model(X[torch.tensor(rf).to(dev)]), 1).cpu().numpy()
    p = prob.argmax(1); run_task.last = dict(prob=prob, y=y[te], te=te, ref_prob=rprob, ref_idx=rf)
    if a.save_src_healthy:   # scores of SOURCE-domain healthy training windows (for weighted non-exchangeable conformal)
        sh = np.where(task.train & (meta['state'] == 'healthy'))[0]
        with torch.no_grad(): run_task.last.update(src_healthy_prob=F_.softmax(model(X[torch.tensor(sh).to(dev)]), 1).cpu().numpy(), src_healthy_idx=sh)
    if a.holdout_class:
        from sklearn.metrics import roc_auc_score
        hc = classes.index(a.holdout_class); known = y[te] != hc
        unk_score = -prob.max(1)                                   # low max-prob = unknown
        auroc = roc_auc_score(~known, unk_score); acc_known = (p[known] == y[te][known]).mean()
        ent = -(prob * np.log(prob + 1e-9)).sum(1); auroc_ent = roc_auc_score(~known, ent)
        from sklearn.neighbors import NearestNeighbors                       # feature-space kNN distance to the training set
        nn_ = NearestNeighbors(n_neighbors=5).fit(G[tr]); dist = nn_.kneighbors(G[te])[0].mean(1); auroc_knn = roc_auc_score(~known, dist)
        run_task.last.update(auroc=auroc, auroc_ent=auroc_ent, auroc_knn=auroc_knn, acc_known=acc_known)
        print(f'      [{task.name}] AUROC maxprob {auroc:.3f} entropy {auroc_ent:.3f} kNN-dist {auroc_knn:.3f}', flush=True)
        return acc_known, auroc
    return accuracy_score(y[te], p), f1_score(y[te], p, average='macro')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('--protocol', default='mic_x_speed')
    ap.add_argument('--pca', type=int, default=0); ap.add_argument('--hidden', type=int, default=256); ap.add_argument('--drop', type=float, default=0.5); ap.add_argument('--in_drop', type=float, default=0.2)
    ap.add_argument('--mixup', type=float, default=0.4); ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--feats', default='psd,cyclo'); ap.add_argument('--ref_split', action='store_true'); ap.add_argument('--ref_seg', type=int, default=0, help='override reference segment (drift analysis)'); ap.add_argument('--holdout_class', default='', help='unknown-fault rejection: exclude this class from training'); ap.add_argument('--full3d', action='store_true', help='keep the tau axis of the cyclo volume'); ap.add_argument('--save_src_healthy', action='store_true'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    z = np.load(os.path.join(ROOT, 'cache', f'{a.dataset}_feats.npz'), allow_pickle=True); meta = z['meta']
    parts = []
    if 'psd' in a.feats: parts.append(z['psd'])
    if 'cyclo' in a.feats: parts.append(z['cyclo'].reshape(len(meta), -1) if a.full3d else z['cyclo'].mean(3).reshape(len(meta), -1))
    if 'beats' in a.feats:
        zb = np.load(os.path.join(ROOT, 'cache', f'{a.dataset}_beats.npz'), allow_pickle=True); assert len(zb['meta']) == len(meta); parts.append(zb['emb'])
    F = np.hstack(parts).astype(np.float32); classes = sorted(np.unique(meta['state']).tolist())
    if a.ref_seg:
        from refpath.eval import protocol as P; P.SPLIT[a.dataset] = dict(P.SPLIT[a.dataset], ref=(a.ref_seg,))
    tasks = make_tasks(meta, a.protocol); res = {}; probs = {}; t0 = time.time()
    for t in tasks:
        acc, f1 = run_task(F, meta, t, classes, a); res[t.name] = dict(acc=acc, f1=f1); probs[t.name] = run_task.last; print(f'{t.name:22s} acc {acc:.3f} f1 {f1:.3f} ({time.time()-t0:.0f}s)', flush=True)
    accs = [r['acc'] for r in res.values()]; print(f'MEAN acc {np.mean(accs):.3f} min {np.min(accs):.3f}')
    name = f'{a.dataset}_{a.protocol}_mlp_{a.feats.replace(",", "+")}_pca{a.pca}_mix{a.mixup}{'_refsplit' if a.ref_split else ''}{'_refseg' + str(a.ref_seg) if a.ref_seg else ''}{'_holdout_' + a.holdout_class if a.holdout_class else ''}{'_full3d' if a.full3d else ''}{a.tag}.json'
    json.dump(dict(args=vars(a), classes=classes, results=res), open(os.path.join(ROOT, 'outputs', name), 'w'), indent=1); print('saved outputs/' + name)
    np.savez(os.path.join(ROOT, 'outputs', name.replace('.json', '_probs.npz')), **{k + '/' + kk: v for k, d in probs.items() for kk, v in d.items()})
