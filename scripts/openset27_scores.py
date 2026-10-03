"""Open-set scores on the 27-cell protocol, recomputed per window from the stored physics-MLP posteriors and freshly computed
feature-space k-NN distances (features and standardization exactly as in train_mlp2: healthy-reference standardization with
the split 60-s reference, then training-set z-scoring; k = 5 neighbours among the source training windows of the known classes).
Scores (all label-free):
  soft      : 1 - max posterior
  knn       : mean distance to the 5 nearest source training windows (far = unknown)
  rank      : rank average of soft and knn over the test batch (transductive)
  pct       : deployable rank average: each score converted to a percentile w.r.t. the SOURCE training distribution
              (soft: posteriors of training windows; knn: leave-one-out distances of training windows), then averaged
  two-sided : deployable two-sided distance score |pct(knn) - 0.5| (an unknown fault may be anomalously FAR or anomalously
              CLOSE to the training set, e.g. cage fracture), averaged with pct(soft)
Metrics per held-out class: AUROC and unknown recall at 5 % false-unknown rate on known windows (TPR@FPR5), mean over 27 cells
and 3 seeds. pct_ref / two_sided_ref: the same with the k-NN percentile taken against the target's odd healthy reference windows (deployable, reference-only).
Writes outputs/openset27_scores.json.
python scripts/openset27_scores.py [--k 5] [--max_train 40000]"""
import os, sys, json, glob, argparse, time, numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import rankdata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); OUT = os.path.join(ROOT, 'outputs')
from refpath.eval.protocol import make_tasks, normalize, Task
ap = argparse.ArgumentParser(); ap.add_argument('--k', type=int, default=5); ap.add_argument('--max_train', type=int, default=40000); ap.add_argument('--seed_sub', type=int, default=0); a = ap.parse_args()
z = np.load(os.path.join(ROOT, 'cache', 'guilin_full_feats.npz'), allow_pickle=True); meta = z['meta']; F = np.concatenate([z['psd'], z['cyclo'].reshape(len(meta), -1)], 1)
classes = sorted(np.unique(meta['state']).tolist()); y = np.array([classes.index(s) for s in meta['state']])
tasks = make_tasks(meta, 'mic_x_speed_x_load', ref_seconds=60); CL = ['inner_pitting', 'outer_pitting', 'inner_outer_pitting', 'cage_fracture']
rng = np.random.RandomState(a.seed_sub)


def pct(x, ref):     # percentile of x within ref (deployable: ref = source training distribution)
    r = np.sort(ref); return np.searchsorted(r, x, side='right') / len(r)
def tpr_at(fpr_target, known_mask, s):
    fpr, tpr, _ = roc_curve(~known_mask, s); return float(np.interp(fpr_target, fpr, tpr))


res = {c: {} for c in CL}; ARR = {}; t0 = time.time()
for c in CL:
    hc = classes.index(c)
    P = [np.load(os.path.join(OUT, f'full_mic_x_speed_x_load_mlp_holdout_{c}_s{s}_probs.npz')) for s in range(3)]
    for t in tasks:
        rid = np.where(t.ref)[0]; rn = np.zeros_like(t.ref); rn[rid[::2]] = True; tn = Task(t.name, t.train, t.test, rn)
        tr = np.where(t.train)[0]; tr = tr[y[tr] != hc]; te = np.where(t.test)[0]
        if len(tr) > a.max_train: tr = np.sort(rng.choice(tr, a.max_train, replace=False))
        rref = np.where(rn)[0]; rc = rid[1::2]; idx = np.concatenate([tr, te, rref, rc])   # even reference windows: standardization; odd reference windows: label-free calibration of the scores
        G = normalize(F[idx].astype(np.float32), meta[idx], Task(t.name, t.train[idx], t.test[idx], tn.ref[idx]), 'healthy')
        Gtr, Gte, Grc = G[:len(tr)], G[len(tr):len(tr) + len(te)], G[len(tr) + len(te) + len(rref):]; mu, sd = Gtr.mean(0), Gtr.std(0) + 1e-6; Gtr = (Gtr - mu) / sd; Gte = (Gte - mu) / sd; Grc = (Grc - mu) / sd
        nn_ = NearestNeighbors(n_neighbors=a.k + 1).fit(Gtr); d_tr = nn_.kneighbors(Gtr)[0][:, 1:].mean(1)      # leave-one-out distances of training windows
        d_te = nn_.kneighbors(Gte, n_neighbors=a.k)[0].mean(1); d_rc = nn_.kneighbors(Grc, n_neighbors=a.k)[0].mean(1); known = y[te] != hc
        out = {}
        for s in range(3):
            pr = P[s][f'{t.name}/prob']; assert (P[s][f'{t.name}/te'] == te).all()
            soft = 1 - pr.max(1); pk = pct(d_te, d_tr)
            # deployable percentile of the softmax score: reference = softmax scores that the same model gives on the training windows are not stored;
            # use the calibration reference windows' scores (healthy, label-free) as the deployable reference for soft
            rc = P[s][f'{t.name}/ref_prob']; soft_ref = 1 - rc.max(1); ps = pct(soft, soft_ref)
            pkr = pct(d_te, d_rc)          # deployable percentile against the target's own healthy reference windows: SATURATES (all later windows are farther than the reference)
            # deployable robust z-scores against the reference windows (unbounded, so no saturation): z = (x - median_ref) / max(IQR_ref, 5 % of median_ref)
            def rz(x, ref):
                med = np.median(ref); iqr = np.subtract(*np.percentile(ref, [75, 25])); return (x - med) / max(iqr, 0.05 * abs(med) + 1e-6)
            zs, zk = rz(soft, soft_ref), rz(d_te, d_rc)
            S = {'soft': soft, 'knn': d_te, 'rank': rankdata(soft) + rankdata(d_te), 'pct': ps + pk, 'two_sided': ps + 2 * np.abs(pk - 0.5), 'two_sided_knn_only': np.abs(pk - 0.5),
                 'pct_ref': ps + pkr, 'two_sided_ref': ps + 2 * np.abs(pkr - 0.5), 'two_sided_ref_knn_only': np.abs(pkr - 0.5), 'knn_ref_pct': pkr,
                 'zsum_ref': zs + zk, 'ztwo_ref': zs + np.abs(zk), 'zknn_ref': zk, 'zsoft_ref': zs}
            if s == 0: ARR.setdefault(c, {})[t.name] = dict(d_te=d_te.astype(np.float32), d_rc=d_rc.astype(np.float32), known=known, soft=soft.astype(np.float32), soft_ref=soft_ref.astype(np.float32))
            for k_, v in S.items(): out.setdefault(k_, []).append((roc_auc_score(~known, v), tpr_at(0.05, known, v)))
        res[c][t.name] = {k_: dict(auroc=float(np.mean([x[0] for x in v])), tpr5=float(np.mean([x[1] for x in v]))) for k_, v in out.items()}
    print(f'{c:20s} ' + ' '.join(f"{k_}: AUROC {np.mean([res[c][t][k_]['auroc'] for t in res[c]]):.3f} TPR5 {np.mean([res[c][t][k_]['tpr5'] for t in res[c]]):.3f} |" for k_ in ['soft', 'knn', 'rank', 'pct', 'pct_ref', 'zsum_ref', 'ztwo_ref', 'zknn_ref', 'zsoft_ref']) + f'  ({time.time()-t0:.0f}s)', flush=True)
json.dump(res, open(os.path.join(OUT, 'openset27_scores.json'), 'w'), indent=1); print('saved outputs/openset27_scores.json')
np.savez(os.path.join(OUT, 'openset27_arrays_seed0.npz'), **{f'{c}/{t}/{k}': v for c in ARR for t in ARR[c] for k, v in ARR[c][t].items()}); print('saved per-window arrays (seed 0)')
