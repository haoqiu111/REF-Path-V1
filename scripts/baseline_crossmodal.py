"""Re-implementation of the core of Peng et al. (MSSP 244:113744, 2026), "unified cross-modal adaptive contrastive
learning" for acoustic bearing diagnosis: two views of the SAME window — the 1-D waveform (1-D CNN) and its
time-frequency image (log-Mel, 2-D CNN) — aligned with an InfoNCE loss, fused by concatenation, and trained with a
dual-task head (coarse healthy/faulty + fine class) with dynamic task re-weighting (uncertainty weighting).
No target data is used (their setting is within-condition); evaluated here under the Reference-Only mic x speed
protocol on Guilin, raw and with our healthy-reference standardization of the log-Mel branch as an add-on.

python scripts/baseline_crossmodal.py [--epochs 15] [--seed 0] [--refnorm]
"""
import os, sys, time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.eval.protocol import make_tasks, domain_key, Task
from refpath.features.torch_frontend import kaldi_fbank
from sklearn.metrics import accuracy_score, f1_score
dev = 'cuda'


class Net(nn.Module):
    def __init__(self, n_cls, d=128):
        super().__init__()
        self.wave = nn.Sequential(nn.Conv1d(1, 16, 64, 8, 28), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
                                  nn.Conv1d(16, 32, 16, 2, 7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
                                  nn.Conv1d(32, 64, 8, 1, 3), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, d))
        self.tf = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, d))
        self.proj_w = nn.Linear(d, 64); self.proj_t = nn.Linear(d, 64)
        self.fine = nn.Linear(2 * d, n_cls); self.coarse = nn.Linear(2 * d, 2)
        self.log_var = nn.Parameter(torch.zeros(3))          # uncertainty weights for (fine, coarse, contrastive)

    def forward(self, w, m):
        zw, zt = self.wave(w.unsqueeze(1)), self.tf(m.unsqueeze(1)); z = torch.cat([zw, zt], 1)
        return self.fine(z), self.coarse(z), F_.normalize(self.proj_w(zw), dim=1), F_.normalize(self.proj_t(zt), dim=1)


def info_nce(a, b, tau=0.1):
    logits = a @ b.T / tau; lab = torch.arange(len(a), device=a.device)
    return (F_.cross_entropy(logits, lab) + F_.cross_entropy(logits.T, lab)) / 2


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--refnorm', action='store_true'); ap.add_argument('--bs', type=int, default=64); a = ap.parse_args()
    z = np.load(os.path.join(ROOT, 'cache', 'guilin_wave16k_raw.npz'), allow_pickle=True); meta = z['meta']; wave = z['wave']
    classes = sorted(np.unique(meta['state']).tolist()); y = np.array([classes.index(s) for s in meta['state']]); H = classes.index('healthy')
    keys = domain_key(meta); healthy = meta['state'] == 'healthy'
    # log-Mel for all windows (cached)
    mel_path = os.path.join(ROOT, 'cache', 'guilin_logmel_raw.npy')
    if not os.path.exists(mel_path):
        M = np.zeros((len(meta), 98, 128), np.float16)
        with torch.no_grad():
            for i in range(0, len(meta), 256):
                M[i:i + 256] = kaldi_fbank(torch.tensor(wave[i:i + 256]).float().to(dev)).cpu().numpy()
        np.save(mel_path, M)
    M = np.load(mel_path)
    tasks = make_tasks(meta, 'mic_x_speed'); res = {}; t0 = time.time()
    for t in tasks:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        tr, te = np.where(t.train)[0], np.where(t.test)[0]
        # per-domain gain from the healthy reference (deployable), as in the main BEATs branch
        g = np.ones(len(meta), np.float32); Mz = M.astype(np.float32)
        for k in np.unique(keys):
            d = keys == k; ref = d & t.ref if (d & t.test).any() else d & t.train & healthy
            if ref.sum() == 0: ref = d & healthy
            g[d] = 0.05 / (np.sqrt(np.mean(wave[np.where(ref)[0][:200]].astype(np.float32) ** 2)) + 1e-8)
            if a.refnorm:
                rid = np.where(ref)[0][::2]; mu = Mz[rid].mean((0, 1)); sd = Mz[rid].std((0, 1)) + 1e-3; Mz[d] = (Mz[d] - mu) / sd
        if not a.refnorm:
            mu, sd = Mz[tr].mean(), Mz[tr].std() + 1e-6; Mz = (Mz - mu) / sd
        net = Net(len(classes)).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
        for ep in range(a.epochs):
            net.train(); perm = np.random.permutation(tr)
            for i in range(0, len(perm), a.bs):
                idx = perm[i:i + a.bs]
                w = torch.tensor(wave[idx].astype(np.float32) * g[idx, None]).to(dev); m = torch.tensor(Mz[idx]).to(dev); yb = torch.tensor(y[idx]).to(dev)
                fine, coarse, pw, pt = net(w, m)
                losses = torch.stack([F_.cross_entropy(fine, yb), F_.cross_entropy(coarse, (yb != H).long()), info_nce(pw, pt)])
                loss = (torch.exp(-net.log_var) * losses + net.log_var).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); preds = []
        with torch.no_grad():
            for i in range(0, len(te), 256):
                idx = te[i:i + 256]; w = torch.tensor(wave[idx].astype(np.float32) * g[idx, None]).to(dev); m = torch.tensor(Mz[idx]).to(dev)
                preds.append(net(w, m)[0].argmax(1).cpu().numpy())
        p = np.concatenate(preds); acc = accuracy_score(y[te], p); f1 = f1_score(y[te], p, average='macro'); res[t.name] = dict(acc=acc, f1=f1)
        print(f'{t.name:22s} acc {acc:.3f} f1 {f1:.3f} ({time.time()-t0:.0f}s)', flush=True)
    accs = [v['acc'] for v in res.values()]; print(f'MEAN acc {np.mean(accs):.3f} min {np.min(accs):.3f}')
    json.dump(dict(args=vars(a), classes=classes, results=res), open(os.path.join(ROOT, 'outputs', f'guilin_mic_x_speed_crossmodal{"_refnorm" if a.refnorm else ""}_s{a.seed}.json'), 'w'), indent=1)
