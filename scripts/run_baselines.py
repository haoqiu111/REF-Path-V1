"""Linear baselines: caches window features and evaluates the Reference-Only protocols with linear models.
Usage:  python scripts/run_baselines.py guilin   |   python scripts/run_baselines.py ahu
Outputs: outputs/<dataset>_baselines.csv and cache/<dataset>_feats.npz
"""
import os, sys, time, argparse, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.data.datasets import get_recordings, meta_table
from refpath.features.frontend import window_iter, log_psd, cyclo_volume
from refpath.eval.protocol import make_tasks, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score


def build_features(ds, seconds=60, win_s=1.0, hop_s=0.5):
    out = os.path.join(ROOT, 'cache', f'{ds}_feats.npz')
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True); return z['meta'], z['psd'], z['cyclo']
    recs = get_recordings(ds); metas, psd, cyc = [], [], []; t0 = time.time()
    for i, r in enumerate(recs):
        x = r.read(seconds); fr = r.rpm / 60
        for _, w in window_iter(x, r.fs, win_s, hop_s):
            psd.append(log_psd(w, r.fs)); cyc.append(cyclo_volume(w, r.fs, fr)); metas.append(r)
        if i % 50 == 0: print(f'{ds}: {i}/{len(recs)} recordings, {time.time()-t0:.0f}s', flush=True)
    meta = meta_table(metas); psd = np.array(psd); cyc = np.array(cyc)
    np.savez(out, meta=meta, psd=psd, cyclo=cyc); return meta, psd, cyc


def evaluate(F, meta, task, npc=60):
    tr, te = task.train, task.test
    c = make_pipeline(StandardScaler(), PCA(n_components=min(npc, tr.sum() - 1, F.shape[1])), LogisticRegression(max_iter=4000, C=0.3))
    c.fit(F[tr], meta['state'][tr]); p = c.predict(F[te])
    return accuracy_score(meta['state'][te], p), f1_score(meta['state'][te], p, average='macro')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('--seconds', type=int, default=60); a = ap.parse_args()
    meta, psd, cyc = build_features(a.dataset, a.seconds)
    cyc_flat = cyc.reshape(len(cyc), -1); cyc_pool = cyc.mean(3).reshape(len(cyc), -1)   # pool tau for linear baseline
    feats = {'logPSD': psd, 'cyclo(f;o)': cyc_pool, 'PSD+cyclo': np.hstack([psd, cyc_pool])}
    rows = []
    for kind in ['in_domain', 'lomo', 'loso', 'lolo', 'mic_x_speed']:
        tasks = [t for t in make_tasks(meta, kind) if t.train.sum() > 0 and t.test.sum() > 0]
        if not tasks: continue
        for mode in ['raw', 'healthy', 'oracle']:
            for fname, F0 in feats.items():
                accs = []
                for t in tasks:
                    acc, f1 = evaluate(normalize(F0, meta, t, mode), meta, t); accs.append(acc)
                    rows.append((kind, t.name, mode, fname, acc, f1))
                print(f'{kind:12s} {mode:8s} {fname:11s} mean acc {np.mean(accs):.3f}  min {np.min(accs):.3f}  ({len(tasks)} tasks)', flush=True)
    os.makedirs(os.path.join(ROOT, 'outputs'), exist_ok=True)
    with open(os.path.join(ROOT, 'outputs', f'{a.dataset}_baselines.csv'), 'w') as f:
        f.write('protocol,task,normalization,features,acc,macro_f1\n')
        for r in rows: f.write(','.join(map(str, r)) + '\n')
    print('saved', os.path.join(ROOT, 'outputs', f'{a.dataset}_baselines.csv'))
