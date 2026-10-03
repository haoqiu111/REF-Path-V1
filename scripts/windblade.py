"""Wind-turbine-blade acoustic dataset (24062 x 3 ch x 5120, 5 classes, -10 dB SNR, 10 federated clients).
Reference-Only cross-site protocol: client = sensing site (path). Leave-one-client-out; the held-out client exposes
only its class-0 (normal) windows of the TRAIN split as healthy reference; evaluation on its official TEST windows.
Features: per-channel log band energies (3 x 128) + envelope spectrum (no shaft speed -> fixed alpha axis).
Compares raw / healthy-reference / oracle normalization with LR and the mixup MLP.

python scripts/windblade.py [--split iid|dirichlet_eps0.2]
"""
import os, sys, time, argparse, numpy as np, scipy.signal as sg, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath import paths
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
WB = paths.WINDBLADE_ROOT                    # folder with iid.zip / dirichlet_eps0.2.zip


def load_split(name):
    import zipfile, io
    d = os.path.join(ROOT, 'cache', f'wb_{name}')
    if not os.path.exists(os.path.join(d, 'X.npy')):
        os.makedirs(d, exist_ok=True)
        with zipfile.ZipFile(os.path.join(WB, f'{name}.zip')) as z:
            for n in z.namelist():
                if n.endswith(('.npy', '.npz')): open(os.path.join(d, os.path.basename(n)), 'wb').write(z.read(n))
    X = np.load(os.path.join(d, 'X.npy'), mmap_mode='r'); y = np.load(os.path.join(d, 'y.npy'))
    sp = np.load(os.path.join(d, 'splits.npz'), allow_pickle=True) if os.path.exists(os.path.join(d, 'splits.npz')) else np.load(os.path.join(ROOT, 'cache', 'wb_iid', 'splits.npz'), allow_pickle=True)
    return X, y, sp


def feats(X, idx):
    out = os.path.join(ROOT, 'cache', 'wb_feats.npy')
    if os.path.exists(out): return np.load(out)[idx]
    F = np.zeros((X.shape[0], 3 * 128 + 3 * 64), np.float32); t0 = time.time()
    for i in range(X.shape[0]):
        x = np.asarray(X[i], np.float64); P = np.abs(np.fft.rfft(x, axis=1))[:, 1:2561] ** 2
        psd = np.log10(P.reshape(3, 128, 20).mean(2) + 1e-12)
        env = np.abs(sg.hilbert(x, axis=1)); E = np.abs(np.fft.rfft(env - env.mean(1, keepdims=True), axis=1))[:, 1:641]
        esp = np.log10(E.reshape(3, 64, 10).mean(2) + 1e-9); F[i] = np.concatenate([psd.ravel(), esp.ravel()])
        if i % 5000 == 0: print(f'feats {i}/{X.shape[0]} {time.time()-t0:.0f}s', flush=True)
    np.save(out, F); return F[idx]


def mlp(Xtr, ytr, Xte, n_cls, epochs=40, mixup=0.4, seed=0, dev='cuda'):
    torch.manual_seed(seed); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    X = torch.tensor(Xtr, dtype=torch.float32).to(dev); Y = torch.tensor(ytr).to(dev)
    m = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, n_cls)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]; xb = X[idx]; yb = F_.one_hot(Y[idx], n_cls).float()
            if mixup > 0:
                j = torch.randperm(len(idx), device=dev); lam = torch.distributions.Beta(mixup, mixup).sample((len(idx), 1)).to(dev); xb = lam * xb + (1 - lam) * xb[j]; yb = lam * yb + (1 - lam) * yb[j]
            loss = -(yb * F_.log_softmax(m(xb), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(torch.tensor(Xte, dtype=torch.float32).to(dev)).argmax(1).cpu().numpy()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--split', default='iid'); a = ap.parse_args()
    X, y, sp = load_split(a.split); n = len(y); F = feats(X, np.arange(n)); print('features', F.shape, 'classes', np.bincount(y))
    tr_c = [np.asarray(c, int) for c in sp['dict_users_train']]; te_c = [np.asarray(c, int) for c in sp['dict_users_test']]
    client = -np.ones(n, int)
    for c in range(10): client[tr_c[c]] = c; client[te_c[c]] = c
    print('unassigned windows (val):', (client < 0).sum())
    is_train = np.zeros(n, bool); is_train[np.concatenate(tr_c)] = True; is_test = np.zeros(n, bool); is_test[np.concatenate(te_c)] = True
    NORMAL = 0
    def normalize(mode):
        G = F.copy()
        for c in range(10):
            d = client == c; ref = d & is_train & (y == NORMAL) if mode == 'healthy' else d & (is_train | is_test)
            if mode == 'raw' or ref.sum() < 2: continue
            G[d] = (F[d] - F[ref].mean(0)) / (F[ref].std(0) + 1e-6)
        return G
    print('\n=== centralized (official train/test, all clients) ===')
    for mode in ['raw', 'healthy']:
        G = normalize(mode); tr, te = is_train, is_test
        c = make_pipeline(StandardScaler(), PCA(n_components=100), LogisticRegression(max_iter=3000, C=0.3)); c.fit(G[tr], y[tr]); pl = c.predict(G[te])
        pm = mlp(G[tr], y[tr], G[te], 5)
        print(f'{mode:8s} LR acc {accuracy_score(y[te], pl):.3f} f1 {f1_score(y[te], pl, average="macro"):.3f} | MLP+mixup acc {accuracy_score(y[te], pm):.3f} f1 {f1_score(y[te], pm, average="macro"):.3f}')
    print('\n=== leave-one-client-out (client = site/path; held-out client: healthy reference only) ===')
    for mode in ['raw', 'healthy', 'oracle']:
        G = normalize(mode); accs_l, accs_m = [], []
        for c in range(10):
            tr = is_train & (client != c); te = is_test & (client == c)
            clf = make_pipeline(StandardScaler(), PCA(n_components=100), LogisticRegression(max_iter=3000, C=0.3)); clf.fit(G[tr], y[tr]); accs_l.append(accuracy_score(y[te], clf.predict(G[te])))
            accs_m.append(accuracy_score(y[te], mlp(G[tr], y[tr], G[te], 5)))
        print(f'{mode:8s} LR mean {np.mean(accs_l):.3f} min {np.min(accs_l):.3f} | MLP+mixup mean {np.mean(accs_m):.3f} min {np.min(accs_m):.3f}')
