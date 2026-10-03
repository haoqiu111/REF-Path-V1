"""Ablation of the ORDER normalization itself: recompute the cyclostationary map with a FIXED cyclic-frequency axis
(alpha = orders * 1000/60 Hz for every recording, i.e. no speed normalization) and rerun the healthy-reference linear
baseline and the mixup MLP on the mic x speed protocol. Compares against the order-normalized features in cache."""
import os, sys, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.data.datasets import get_recordings, meta_table
from refpath.features.frontend import window_iter, cyclo_volume
from refpath.eval.protocol import make_tasks, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
out = os.path.join(ROOT, 'cache', 'guilin_cyclo_fixedalpha.npy')
z = np.load(os.path.join(ROOT, 'cache', 'guilin_feats.npz'), allow_pickle=True); meta = z['meta']; psd = z['psd']
if not os.path.exists(out):
    recs = get_recordings('guilin'); cyc = []; t0 = time.time()
    for i, r in enumerate(recs):
        x = r.read(60)
        for _, w in window_iter(x, r.fs, 1.0, 0.5): cyc.append(cyclo_volume(w, r.fs, 1000 / 60).mean(2))   # fixed alpha axis
        if i % 50 == 0: print(f'{i}/{len(recs)} {time.time()-t0:.0f}s', flush=True)
    cyc = np.array(cyc, np.float32); assert len(cyc) == len(meta); np.save(out, cyc)
cyc_fixed = np.load(out).reshape(len(meta), -1); cyc_order = z['cyclo'].mean(3).reshape(len(meta), -1)
sys.path.insert(0, os.path.join(ROOT, 'scripts')); import importlib; tm = importlib.import_module('train_mlp')
class A: pca = 0; hidden = 256; drop = 0.5; in_drop = 0.2; mixup = 0.4; epochs = 40; bs = 256; lr = 1e-3; wd = 1e-2; seed = 0; ref_split = True; holdout_class = ''
classes = sorted(np.unique(meta['state']).tolist()); tasks = make_tasks(meta, 'mic_x_speed')
for name, C in [('order-normalized (default)', cyc_order), ('fixed cyclic-frequency axis', cyc_fixed)]:
    F = np.hstack([psd, C]).astype(np.float32); accs_l, accs_m = [], []
    for t in tasks:
        G = normalize(F, meta, t, 'healthy'); c = make_pipeline(StandardScaler(), PCA(n_components=60), LogisticRegression(max_iter=4000, C=0.3)); c.fit(G[t.train], meta['state'][t.train]); accs_l.append(accuracy_score(meta['state'][t.test], c.predict(G[t.test])))
        acc, _ = tm.run_task(F, meta, t, classes, A()); accs_m.append(acc)
    print(f'{name:32s} linear healthy-ref mean {np.mean(accs_l):.3f} min {np.min(accs_l):.3f} | MLP+mixup mean {np.mean(accs_m):.3f} min {np.min(accs_m):.3f}', flush=True)
