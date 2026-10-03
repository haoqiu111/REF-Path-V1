"""Synchronous-pair invariance (SPI), a negative result of the paper. Guilin's three microphones are recorded simultaneously, so the SAME
1-s window seen by two SOURCE microphones is a natural positive pair. We add an InfoNCE term that pulls the
embeddings of synchronous windows from different source microphones together (path invariance learned from
measured pairs), on top of the mixup MLP with healthy-reference standardization. The held-out microphone never
contributes pairs. Also reports a 'measured transfer-function' check: the per-window PSD ratio between source mics
(the physical path gain) vs the reference-standardization offset.

python scripts/spi.py [--lam 0.5] [--seed 0]
"""
import os, sys, time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.eval.protocol import make_tasks, normalize, Task
from sklearn.metrics import accuracy_score, f1_score
dev = 'cuda'
z = np.load(os.path.join(ROOT, 'cache', 'guilin_feats.npz'), allow_pickle=True); meta = z['meta']
F = np.hstack([z['psd'], z['cyclo'].mean(3).reshape(len(meta), -1)]).astype(np.float32)
classes = sorted(np.unique(meta['state']).tolist()); y = np.array([classes.index(s) for s in meta['state']])
# window identity: recordings are enumerated (state, rpm, load, seg) x 3 mics with 119 windows each -> same window index
# across mics of the same recording. Build a key (state, rpm, load, seg, window_in_recording).
n_win = 119; rec = np.arange(len(meta)) // n_win; win = np.arange(len(meta)) % n_win
rec_key = np.array([f"{meta['state'][i]}|{meta['rpm'][i]}|{meta['load'][i]}|{meta['seg'][i]}" for i in range(0, len(meta), n_win)])
pair_key = np.array([f'{rec_key[r]}|{w}' for r, w in zip(rec, win)])


def split_ref(task):
    rid = np.where(task.ref)[0]; rn = np.zeros_like(task.ref); rn[rid[::2]] = True; return Task(task.name, task.train, task.test, rn)


class Net(nn.Module):
    def __init__(self, d_in, n_cls, h=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_in, h), nn.GELU(), nn.Dropout(0.5), nn.Linear(h, h), nn.GELU(), nn.Dropout(0.5))
        self.head = nn.Linear(h, n_cls); self.proj = nn.Linear(h, 64)
    def forward(self, x): z = self.enc(x); return self.head(z), F_.normalize(self.proj(z), dim=1)


def run(task, lam, seed, mixup=0.4, epochs=40, bs=256):
    torch.manual_seed(seed); np.random.seed(seed); tn = split_ref(task)
    G = normalize(F, meta, tn, 'healthy'); tr, te = np.where(task.train)[0], np.where(task.test)[0]
    mu, sd = G[tr].mean(0), G[tr].std(0) + 1e-6; G = (G - mu) / sd
    X = torch.tensor(G).to(dev); Y = torch.tensor(y).to(dev)
    # synchronous partner for every training window: another SOURCE mic, same pair_key
    groups = {}
    for i in tr: groups.setdefault(pair_key[i], []).append(i)
    partner = np.array([np.random.choice([j for j in groups[pair_key[i]] if j != i]) if len(groups[pair_key[i]]) > 1 else i for i in tr])
    net = Net(G.shape[1], len(classes)).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    trn = torch.tensor(tr).to(dev); ptn = torch.tensor(partner).to(dev)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(perm), bs):
            sel = perm[i:i + bs]; idx = trn[sel]; pidx = ptn[sel]; xb = X[idx]; yb = F_.one_hot(Y[idx], len(classes)).float()
            logits_p, zp = net(X[pidx])                                          # partner view (other mic, same instant)
            if mixup > 0:
                j = torch.randperm(len(idx), device=dev); lamb = torch.distributions.Beta(mixup, mixup).sample((len(idx), 1)).to(dev); xb = lamb * xb + (1 - lamb) * xb[j]; yb = lamb * yb + (1 - lamb) * yb[j]
            logits, za = net(xb)
            loss = -(yb * F_.log_softmax(logits, 1)).sum(1).mean()
            if lam > 0:
                _, za0 = net(X[idx]); sim = za0 @ zp.T / 0.1; lab = torch.arange(len(idx), device=dev)
                loss = loss + lam * (F_.cross_entropy(sim, lab) + F_.cross_entropy(sim.T, lab)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): p = net(X[torch.tensor(te).to(dev)])[0].argmax(1).cpu().numpy()
    return accuracy_score(y[te], p), f1_score(y[te], p, average='macro')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--lam', type=float, default=0.5); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
    tasks = make_tasks(meta, 'mic_x_speed'); out = {}
    for lam in [0.0, a.lam]:
        res = {}
        for t in tasks:
            acc, f1 = run(t, lam, a.seed); res[t.name] = dict(acc=acc, f1=f1)
        accs = [v['acc'] for v in res.values()]; print(f'lambda={lam}: MEAN acc {np.mean(accs):.3f} min {np.min(accs):.3f}  ' + ' '.join(f'{k[4:]}={v["acc"]:.2f}' for k, v in res.items()), flush=True); out[str(lam)] = res
    json.dump(dict(args=vars(a), classes=classes, results=out), open(os.path.join(ROOT, 'outputs', f'guilin_mic_x_speed_spi_s{a.seed}.json'), 'w'), indent=1)
    # physical check: measured path gain between two source mics (mean log-PSD difference on synchronous healthy windows)
    # versus the difference of the reference means used by standardization (they should agree up to noise)
    psd = z['psd']; mics = ['CRY333', 'BK4966_2', 'BK4966_3']; healthy = meta['state'] == 'healthy'
    for r in [800, 1000, 1200]:
        dom = lambda m: (meta['path'] == m) & (meta['rpm'] == r) & (meta['load'] == 0) & (meta['seg'] == 1)
        a_, b_ = np.where(dom(mics[1]) & healthy)[0], np.where(dom(mics[2]) & healthy)[0]
        tf = (psd[a_] - psd[b_]).mean(0); ref = psd[a_].mean(0) - psd[b_].mean(0)
        print(f'rpm {r}: measured synchronous log-PSD ratio B&K2/B&K3 vs reference-mean difference: corr {np.corrcoef(tf, ref)[0,1]:.4f}, mean |diff| {np.abs(tf-ref).mean():.4f} dB*0.1')
