"""Cache frozen-BEATs embeddings (mean-pooled, 768-d) for every 1-s window of a dataset, then evaluate the
Reference-Only protocols with linear / kNN probes and the three normalizations. Baseline 'BEATs-kNN' of the paper.
Usage: python scripts/beats_embed.py guilin [--seconds 60] [--hop 1.0]
"""
import os, sys, time, argparse, glob, numpy as np, torch, scipy.signal as sg
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.paths import beats_checkpoint
from refpath.data.datasets import get_recordings, meta_table
from refpath.eval.protocol import make_tasks, normalize
from refpath.models.beats.BEATs import BEATs, BEATsConfig
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

CKPT = beats_checkpoint()                    # $REFPATH_BEATS_CKPT or <repo>/checkpoints/*.pt


def load_beats(dev='cuda'):
    c = torch.load(CKPT, map_location='cpu', weights_only=False); m = BEATs(BEATsConfig(c['cfg'])); m.load_state_dict(c['model']); return m.to(dev).eval()


def embed(ds, seconds, hop, bs=128, dev='cuda'):
    out = os.path.join(ROOT, 'cache', f'{ds}_beats.npz')
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True); return z['meta'], z['emb']
    m = load_beats(dev); recs = get_recordings(ds); metas, embs = [], []; t0 = time.time()
    for i, r in enumerate(recs):
        x = r.read(seconds); up, down = (125, 256) if r.fs == 32768 else (4, 5)      # 32768->16000, 20000->16000
        x = sg.resample_poly(x, up, down).astype(np.float32); x = x / (x.std() + 1e-8) * 0.05    # fixed input scale
        W, H = 16000, int(hop * 16000); wins = np.stack([x[s:s + W] for s in range(0, len(x) - W + 1, H)])
        with torch.no_grad():
            for j in range(0, len(wins), bs):
                f, _ = m.extract_features(torch.tensor(wins[j:j + bs]).to(dev)); embs.append(f.mean(1).cpu().numpy())
        metas += [r] * len(wins)
        if i % 50 == 0: print(f'{ds}: {i}/{len(recs)} {time.time()-t0:.0f}s', flush=True)
    meta = meta_table(metas); emb = np.concatenate(embs); np.savez(out, meta=meta, emb=emb); return meta, emb


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('--seconds', type=int, default=60); ap.add_argument('--hop', type=float, default=1.0); a = ap.parse_args()
    meta, emb = embed(a.dataset, a.seconds, a.hop); print('embeddings', emb.shape)
    rows = []
    for kind in ['lomo', 'loso', 'mic_x_speed']:
        tasks = [t for t in make_tasks(meta, kind) if t.train.sum() > 0 and t.test.sum() > 0]
        for mode in ['raw', 'healthy', 'oracle']:
            for probe in ['LR', 'kNN']:
                accs = []
                for t in tasks:
                    F = normalize(emb, meta, t, mode)
                    c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3)) if probe == 'LR' else make_pipeline(StandardScaler(), KNeighborsClassifier(10))
                    c.fit(F[t.train], meta['state'][t.train]); acc = accuracy_score(meta['state'][t.test], c.predict(F[t.test])); accs.append(acc); rows.append((kind, t.name, mode, probe, acc))
                print(f'{kind:12s} {mode:8s} BEATs-{probe:4s} mean acc {np.mean(accs):.3f}  min {np.min(accs):.3f}', flush=True)
    with open(os.path.join(ROOT, 'outputs', f'{a.dataset}_beats_frozen.csv'), 'w') as f:
        f.write('protocol,task,normalization,probe,acc\n'); [f.write(','.join(map(str, r)) + '\n') for r in rows]
