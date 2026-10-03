"""Exploratory cross-device check on the MSU multi-session, multi-device acoustic dataset (Zenodo 18971675, CC BY 4.0).
Not used in the paper: recording-level labels of this corpus are not separable even within one device.
Subsets (tool/task): shopvac/fill_level (0,30,50,70,100 %), shopvac/health (normal/faulty), vacuum/clog (0..100 %),
vacuum/power (0-8), sander/wear (new/moderate/worn_faulty). Eight recording devices = sensing paths. 48 kHz mono WAV.

Reference-Only cross-device protocol: for every subset and every device d that has at least one healthy recording
(label 0 / normal / new): train on ALL other devices (all labels), reference = first --ref_seconds of one healthy
recording of d, test = all windows of d except the reference recording. Features: log-PSD (128 bands) + cyclostationary
map with a FIXED cyclic-frequency axis (no shaft speed available; alpha grid = orders x 20 Hz -> 10-500 Hz).
Methods: linear raw / healthy-reference / oracle, and the mixup MLP with healthy-reference standardization.

python scripts/msu_experiment.py [--root <folder with Dataset/>] [--subset vacuum/clog] [--seconds 120]
"""
import os, sys, glob, time, argparse, json, csv, numpy as np, soundfile as sf, scipy.signal as sg, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath import paths
from refpath.features.frontend import log_psd, cyclo_volume
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
dev = 'cuda'; HEALTHY = {'0', '0%', 'normal', 'new', 'level_0', '0_percent'}


def find_manifests(root):
    return sorted(glob.glob(os.path.join(root, '**', '*manifest*.csv'), recursive=True))


def load_manifest(fn):
    rows = list(csv.DictReader(open(fn, encoding='utf-8-sig')))
    base = os.path.dirname(fn); tool_dir = os.path.basename(base).lower()
    out = []
    for r in rows:
        rel = r.get('output_relative_path') or r.get('public_relative_path') or r.get('relative_path') or r.get('source_relative_path')
        if not rel: continue
        cand = [os.path.join(base, rel), os.path.join(base, rel.replace('/', os.sep))]
        absp = next((c for c in cand if os.path.exists(c)), None)
        if absp is None: continue
        out.append(dict(tool=(r.get('tool') or tool_dir).lower(), task=(r.get('task') or 'wear').lower().replace(' ', '_'), label=str(r.get('label', '')).strip().lower(),
                        device=str(r.get('device', '')).strip().lower().replace(' ', '').replace('_', ''), split=str(r.get('split', '')).lower(), session_id=str(r.get('session', r.get('session_id', ''))), abs=absp,
                        start_time_s='', end_time_s=''))
    return out


def is_healthy(label):
    l = str(label).strip().lower().replace('%', ''); return l in HEALTHY or l == '0'


def features(fn, seconds, fs_target=32768, win=1.0, hop=0.5, start_s=None, end_s=None):
    x, fs = sf.read(fn, dtype='float32');
    if x.ndim > 1: x = x.mean(1)
    if start_s is not None: x = x[int(start_s * fs):int((end_s or len(x) / fs) * fs)]
    x = x[:int(seconds * fs)] if seconds else x
    if fs != fs_target: x = sg.resample_poly(x, fs_target, fs).astype(np.float32); fs = fs_target
    W, H = int(win * fs), int(hop * fs); out = []
    for s in range(0, len(x) - W + 1, H):
        w = x[s:s + W].astype(np.float64); out.append((s / fs, log_psd(w, fs), cyclo_volume(w, fs, 20.0).mean(2).astype(np.float16).ravel()))
    return out


def mlp_fit_predict(Xtr, ytr, Xte, n_cls, seed=0, epochs=40, mixup=0.4):
    torch.manual_seed(seed); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    X = torch.tensor(Xtr, dtype=torch.float32).to(dev); Y = torch.tensor(ytr).to(dev)
    m = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, n_cls)).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]; xb = X[idx]; yb = F_.one_hot(Y[idx], n_cls).float()
            j = torch.randperm(len(idx), device=dev); lam = torch.distributions.Beta(mixup, mixup).sample((len(idx), 1)).to(dev); xb = lam * xb + (1 - lam) * xb[j]; yb = lam * yb + (1 - lam) * yb[j]
            loss = -(yb * F_.log_softmax(m(xb), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(torch.tensor(Xte, dtype=torch.float32).to(dev)).argmax(1).cpu().numpy()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=paths.MSU_ROOT); ap.add_argument('--subset', default=''); ap.add_argument('--seconds', type=int, default=120); ap.add_argument('--ref_seconds', type=float, default=60); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
    cache = os.path.join(ROOT, 'cache', 'msu_feats.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); meta, F = z['meta'], z['F']
    else:
        rows = [r for fn in find_manifests(a.root) for r in load_manifest(fn)]; print('manifest rows with files:', len(rows), flush=True)
        metas, feats = [], []; t0 = time.time()
        for i, r in enumerate(rows):
            st = float(r['start_time_s']) if r.get('start_time_s') else None; en = float(r['end_time_s']) if r.get('end_time_s') else None
            for start, psd, cyc in features(r['abs'], a.seconds, start_s=st, end_s=en):
                metas.append((r['tool'], r['task'], str(r['label']), r['device'], r.get('split', ''), str(r.get('session_id', '')), r['abs'], start)); feats.append(np.concatenate([psd, cyc.astype(np.float32)]))
            if i % 20 == 0: print(f'{i}/{len(rows)} {time.time()-t0:.0f}s windows={len(metas)}', flush=True)
        meta = np.array(metas, dtype=[('tool', 'U16'), ('task', 'U16'), ('label', 'U24'), ('device', 'U24'), ('split', 'U8'), ('session', 'U16'), ('file', 'U400'), ('start', 'f4')]); F = np.array(feats, np.float32)
        np.savez(cache, meta=meta, F=F)
    subsets = sorted({(m['tool'], m['task']) for m in meta}); print('subsets:', subsets, 'windows', len(meta))
    results = {}
    for tool, task in subsets:
        if a.subset and f'{tool}/{task}' != a.subset: continue
        S = (meta['tool'] == tool) & (meta['task'] == task); ms = meta[S]; Fs = F[S]; labels = sorted(np.unique(ms['label']).tolist()); y = np.array([labels.index(l) for l in ms['label']])
        healthy = np.array([is_healthy(l) for l in ms['label']]); devices = sorted(np.unique(ms['device']).tolist())
        print(f'\n=== {tool}/{task}: labels {labels}, devices {devices}, healthy label present: {healthy.any()} ===', flush=True)
        if not healthy.any(): print('  no healthy label -> skipped'); continue
        res = {}
        for d in devices:
            D = ms['device'] == d; hd = D & healthy
            if hd.sum() == 0 or (D & ~healthy).sum() == 0: continue
            ref_file = ms['file'][np.where(hd)[0][0]]; ref = D & (ms['file'] == ref_file) & (ms['start'] < a.ref_seconds); ref_norm = ref.copy(); ref_norm[np.where(ref)[0][1::2]] = False
            te = D & (ms['file'] != ref_file); tr = ~D
            if te.sum() == 0: continue
            G_raw = Fs.copy(); G_h = Fs.copy(); G_o = Fs.copy()
            for dd in devices:                                 # per-device standardization
                sel = ms['device'] == dd
                if dd == d: rh, ro = ref_norm, te
                else: rh, ro = sel & healthy, sel
                if rh.sum() > 1: G_h[sel] = (Fs[sel] - Fs[rh].mean(0)) / (Fs[rh].std(0) + 1e-6)
                if ro.sum() > 1: G_o[sel] = (Fs[sel] - Fs[ro].mean(0)) / (Fs[ro].std(0) + 1e-6)
            out = {}
            for name, G in [('raw', G_raw), ('healthy', G_h), ('oracle', G_o)]:
                c = make_pipeline(StandardScaler(), PCA(n_components=min(60, tr.sum() - 1)), LogisticRegression(max_iter=3000, C=0.3)); c.fit(G[tr], y[tr]); p = c.predict(G[te]); out[f'linear_{name}'] = accuracy_score(y[te], p)
            p = mlp_fit_predict(G_h[tr], y[tr], G_h[te], len(labels), a.seed); out['mlp_healthy'] = accuracy_score(y[te], p); out['mlp_healthy_f1'] = f1_score(y[te], p, average='macro'); out['n_test'] = int(te.sum())
            res[d] = out; print(f'  held-out device {d:18s} ' + ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in out.items()), flush=True)
        if res:
            for k in ['linear_raw', 'linear_healthy', 'linear_oracle', 'mlp_healthy']: print(f'  MEAN {k}: {np.mean([r[k] for r in res.values()]):.3f} (min {np.min([r[k] for r in res.values()]):.3f}, n={len(res)})')
        results[f'{tool}/{task}'] = res
    json.dump(results, open(os.path.join(ROOT, 'outputs', f'msu_cross_device_s{a.seed}.json'), 'w'), indent=1); print('saved outputs/msu_cross_device_s%d.json' % a.seed)
