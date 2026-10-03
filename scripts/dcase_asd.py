"""DCASE 2026 Task 2 baseline for REF-Path: frozen BEATs embeddings + kNN anomaly score, evaluated per machine with the
official metrics (source AUC, target AUC, pAUC@0.1, and the harmonic mean), separately for near / far / both channels
and with/without healthy-reference (train-normal) standardization.

python scripts/dcase_asd.py [--machines fan,bearingEmu] [--seconds 10]
"""
import os, sys, time, argparse, glob, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.paths import beats_checkpoint
from refpath.data.dcase import dcase_index, read_wav, MACHINES
from refpath.models.beats.BEATs import BEATs, BEATsConfig
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
CKPT = beats_checkpoint()                    # $REFPATH_BEATS_CKPT or <repo>/checkpoints/*.pt


def fix_len(x, n=160000):
    return x[:n] if len(x) >= n else np.pad(x, ((0, n - len(x)), (0, 0)), mode='wrap')


def embed_all(idx, dev='cuda', bs=32):
    out = os.path.join(ROOT, 'cache', 'dcase_beats.npz')
    if os.path.exists(out):
        return np.load(out, allow_pickle=True)['emb']
    c = torch.load(CKPT, map_location='cpu', weights_only=False); m = BEATs(BEATsConfig(c['cfg'])); m.load_state_dict(c['model']); m = m.to(dev).eval()
    emb = np.zeros((len(idx), 2, 768), np.float32); t0 = time.time()
    for i in range(0, len(idx), bs):
        X = np.stack([fix_len(read_wav(r['file'])[0]) for r in idx[i:i + bs]])         # (b, 160000, 2) crop/pad to 10 s
        X = X / (X.std(axis=1, keepdims=True) + 1e-8) * 0.05
        with torch.no_grad():
            for ch in range(2):
                f, _ = m.extract_features(torch.tensor(X[:, :, ch]).to(dev)); emb[i:i + bs, ch] = f.mean(1).cpu().numpy()
        if i % 640 == 0: print(f'{i}/{len(idx)} {time.time()-t0:.0f}s', flush=True)
    np.savez(out, emb=emb); return emb


def pauc(y, s, max_fpr=0.1):
    return roc_auc_score(y, s, max_fpr=max_fpr)


def knn_scores(train, test, k=5):
    nn = NearestNeighbors(n_neighbors=k).fit(train); d, _ = nn.kneighbors(test); return d.mean(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--machines', default=','.join(MACHINES)); a = ap.parse_args()
    idx = dcase_index(machines=a.machines.split(',')); emb = embed_all(idx)
    mach = np.array([r['machine'] for r in idx]); split = np.array([r['split'] for r in idx]); dom = np.array([r['domain'] for r in idx]); y = np.array([r['label'] for r in idx])
    print(f'{"machine":12s} {"feat":10s} {"norm":8s}  AUC_src  AUC_tgt  pAUC   hmean')
    rows = []
    for m in a.machines.split(','):
        for feat, F in [('near', emb[:, 0]), ('far', emb[:, 1]), ('near-far', emb[:, 0] - emb[:, 1]), ('concat', np.concatenate([emb[:, 0], emb[:, 1]], 1))]:
            for norm in ['raw', 'healthy']:
                tr = (mach == m) & (split == 'train'); te = (mach == m) & (split == 'test')
                G = F.copy()
                if norm == 'healthy':
                    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6; G = (F - mu) / sd
                s = knn_scores(G[tr], G[te])
                yt, dt = y[te], dom[te]
                auc_s = roc_auc_score(yt[dt == 'source'], s[dt == 'source']); auc_t = roc_auc_score(yt[dt == 'target'], s[dt == 'target']); p = pauc(yt, s)
                h = 3 / (1 / auc_s + 1 / auc_t + 1 / p); rows.append((m, feat, norm, auc_s, auc_t, p, h))
                print(f'{m:12s} {feat:10s} {norm:8s}  {auc_s:.3f}    {auc_t:.3f}    {p:.3f}  {h:.3f}')
    with open(os.path.join(ROOT, 'outputs', 'dcase_beats_knn.csv'), 'w') as f:
        f.write('machine,feat,norm,auc_source,auc_target,pauc,hmean\n'); [f.write(','.join(map(str, r)) + '\n') for r in rows]
    import collections; agg = collections.defaultdict(list)
    for r in rows: agg[(r[1], r[2])].append(r[6])
    print('\nharmonic-mean score averaged over machines:'); [print(f'  {k[0]:10s} {k[1]:8s} {np.mean(v):.3f}') for k, v in agg.items()]
