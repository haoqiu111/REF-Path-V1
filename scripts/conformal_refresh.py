"""Conformal reference self-refresh (deployable), physics MLP, nine-cell mic x speed protocol (60-s clips,
segments 1 / 3 / 5 = commissioning reference / refresh period / test). A mic x speed task has THREE target domains
(the three loads); every reference operation is per domain, calibration sets are pooled over the three (as in Table 5).
Reference variants (standardization = even windows, calibration = odd windows):
  OLD      : segment-1 healthy reference (what Table 5 reports; ages by ~2 segments at test time)
  IDEAL    : segment-3 true healthy windows (an engineer re-records a healthy reference right before the test period)
  SELF-H   : self-refresh, machine healthy during the refresh period: segment-3 windows of the healthy recording are
             accepted into the new reference iff (conformal p-value w.r.t. the OLD calibration set > alpha_acc) AND
             (predicted class = healthy); no labels used; a domain is refreshed only if >= MIN_ACCEPT windows are accepted
  SELF-ALL : self-refresh with an adversarial candidate pool = segment-3 windows of ALL five recordings of the domain
             (machine may be faulty during the refresh period); same filter; contamination = share of accepted windows
             that are faulty (ground truth used only for reporting)
  NAIVE-ALL: the pooled candidates without the filter (why the filter is needed)
For each variant: accuracy on segment 5, and false-alarm / detection at alpha in {0.01, 0.05, 0.10} with split conformal
on the variant's calibration windows. The classifier is trained once per task (source domains, own healthy statistics)
and never retrained; only the target standardization / calibration change.
python scripts/conformal_refresh.py [--alpha_acc 0.01] [--device cuda|cpu] [--seed 0] [--tasks ...] [--debug]"""
import os, sys, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.eval.protocol import make_tasks, domain_key
ap = argparse.ArgumentParser(); ap.add_argument('--alpha_acc', type=float, default=0.01); ap.add_argument('--device', default='cuda'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--tasks', default=''); ap.add_argument('--debug', action='store_true'); ap.add_argument('--min_accept', type=int, default=10); a = ap.parse_args()
dev = a.device; ALPHAS = [0.01, 0.05, 0.10]; MIN_ACCEPT = a.min_accept
z = np.load(os.path.join(ROOT, 'cache', 'guilin_feats.npz'), allow_pickle=True); meta = z['meta']
F = np.concatenate([z['psd'], z['cyclo'].mean(3).reshape(len(meta), -1)], 1).astype(np.float32)     # same features as the physics MLP (psd + tau-averaged map)
classes = sorted(np.unique(meta['state']).tolist()); H = classes.index('healthy'); y = np.array([classes.index(s) for s in meta['state']]); healthy = meta['state'] == 'healthy'
keys = domain_key(meta); tasks = make_tasks(meta, 'mic_x_speed')
if a.tasks: tasks = [t for t in tasks if t.name in a.tasks.split(',')]


def standardize(F, ref_mask, train_mask):
    """healthy-reference standardization: domains with reference windows in ref_mask use them; the others (source) use their own healthy training windows"""
    G = F.copy()
    for k in np.unique(keys):
        d = keys == k
        ref = d & ref_mask if (d & ref_mask).any() else d & train_mask & healthy
        if ref.sum() < 2: continue                          # domains excluded from the task (same mic or speed as the target) are never used
        G[d] = (F[d] - F[ref].mean(0)) / (F[ref].std(0) + 1e-6)
    return G


def train(G, tr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = G[tr].mean(0), G[tr].std(0) + 1e-6; X = torch.tensor((G[tr] - mu) / sd).to(dev); Y = torch.tensor(y[tr]).to(dev)
    model = nn.Sequential(nn.Dropout(0.2), nn.Linear(G.shape[1], 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, len(classes))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]; xb, yb = X[idx], F_.one_hot(Y[idx], len(classes)).float()
            j = torch.randperm(len(idx), device=dev); lam = torch.distributions.Beta(0.4, 0.4).sample((len(idx), 1)).to(dev); xb = lam * xb + (1 - lam) * xb[j]; yb = lam * yb + (1 - lam) * yb[j]
            loss = -(yb * F_.log_softmax(model(xb), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    def predict(Gsub):
        with torch.no_grad(): return F_.softmax(model(torch.tensor((Gsub - mu) / sd).to(dev)), 1).cpu().numpy()
    return predict


def conformal_pvals(s_cal, s):
    return np.array([(np.sum(s_cal >= v) + 1) / (len(s_cal) + 1) for v in s])


def evaluate(predict, F, ref_mask, tr_mask, te, cal_idx):
    G = standardize(F, ref_mask, tr_mask); pt = predict(G[te]); pc = predict(G[cal_idx])
    s_te, s_cal = 1 - pt[:, H], 1 - pc[:, H]; pv = conformal_pvals(s_cal, s_te); yt = y[te]
    out = dict(acc=float((pt.argmax(1) == yt).mean()), n_cal=int(len(cal_idx)))
    if a.debug: print('   debug: test pred hist', np.bincount(pt.argmax(1), minlength=5), 'cal pred hist', np.bincount(pc.argmax(1), minlength=5), 'cal p(healthy) mean %.3f' % pc[:, H].mean(), '| G[te] |mean| %.3f' % np.abs(G[te]).mean())
    for al in ALPHAS: alarm = pv <= al; out[f'far@{al}'] = float(alarm[yt == H].mean()); out[f'det@{al}'] = float(alarm[yt != H].mean())
    return out


res = {}
for t in tasks:
    te = np.where(t.test)[0]; tr = np.where(t.train)[0]; tgt_doms = np.unique(keys[te])                    # three target domains (loads)
    rid = np.where(t.ref)[0]; ref_even = np.zeros(len(meta), bool); ref_even[rid[::2]] = True; cal_old = rid[1::2]   # OLD: segment-1 healthy, per domain
    G_old = standardize(F, ref_even, t.train); predict = train(G_old, tr, a.seed)
    r = {'OLD': evaluate(predict, F, ref_even, t.train, te, cal_old)}
    seg3 = np.isin(keys, tgt_doms) & (meta['seg'] == 3); h3 = np.where(seg3 & healthy)[0]; m3 = np.zeros(len(meta), bool); m3[h3[::2]] = True   # IDEAL: segment-3 true healthy
    r['IDEAL'] = evaluate(predict, F, m3, t.train, te, h3[1::2])
    pc_old = predict(G_old[cal_old]); s_cal_old = 1 - pc_old[:, H]                                           # self-refresh candidates scored under the OLD pipeline
    for name, cand_all in [('SELF-H', h3), ('SELF-ALL', np.where(seg3)[0])]:
        p_c = predict(G_old[cand_all]); pv = conformal_pvals(s_cal_old, 1 - p_c[:, H]); acc_mask = (pv > a.alpha_acc) & (p_c.argmax(1) == H); acc_idx = cand_all[acc_mask]
        mm = ref_even.copy(); cal = []; refreshed = 0
        for d in tgt_doms:
            ai = acc_idx[keys[acc_idx] == d]
            if len(ai) >= MIN_ACCEPT: mm[keys == d] = False; mm[ai[::2]] = True; cal.append(ai[1::2]); refreshed += 1
            else: cal.append(cal_old[keys[cal_old] == d])
        r[name] = evaluate(predict, F, mm, t.train, te, np.concatenate(cal))
        r[name].update(n_candidates=int(len(cand_all)), n_accepted=int(len(acc_idx)), domains_refreshed=int(refreshed), contamination=float((y[acc_idx] != H).mean()) if len(acc_idx) else float('nan'),
                       faulty_rejected=float(1 - (acc_mask & (y[cand_all] != H)).sum() / max(1, (y[cand_all] != H).sum())))
    cand = np.where(seg3)[0]; mm = np.zeros(len(meta), bool); mm[cand[::2]] = True; r['NAIVE-ALL'] = evaluate(predict, F, mm, t.train, te, cand[1::2])
    res[t.name] = r
    print(f"{t.name:20s} acc old/ideal/selfH/selfALL/naive {r['OLD']['acc']:.3f}/{r['IDEAL']['acc']:.3f}/{r['SELF-H']['acc']:.3f}/{r['SELF-ALL']['acc']:.3f}/{r['NAIVE-ALL']['acc']:.3f} | FAR@0.05 {r['OLD']['far@0.05']:.3f}/{r['IDEAL']['far@0.05']:.3f}/{r['SELF-H']['far@0.05']:.3f}/{r['SELF-ALL']['far@0.05']:.3f}/{r['NAIVE-ALL']['far@0.05']:.3f} | accepted {r['SELF-H']['n_accepted']}/{r['SELF-H']['n_candidates']} , {r['SELF-ALL']['n_accepted']}/{r['SELF-ALL']['n_candidates']} contam {r['SELF-ALL']['contamination']:.3f} refreshed {r['SELF-ALL']['domains_refreshed']}/3", flush=True)

print('\nMEAN over %d tasks (FAR mean (max) / DET):' % len(res))
for name in ['OLD', 'IDEAL', 'SELF-H', 'SELF-ALL', 'NAIVE-ALL']:
    accs = [res[t][name]['acc'] for t in res]; line = f'{name:10s} acc {np.nanmean(accs):.3f} '
    for al in ALPHAS:
        fa = [res[t][name][f'far@{al}'] for t in res]; de = [res[t][name][f'det@{al}'] for t in res]; line += f'| a={al:.2f}: {np.nanmean(fa):.3f} ({np.nanmax(fa):.3f}) / {np.nanmean(de):.3f} '
    if name.startswith('SELF'): line += f"| accepted {np.mean([res[t][name]['n_accepted'] for t in res]):.0f}/{np.mean([res[t][name]['n_candidates'] for t in res]):.0f}, contamination {np.nanmean([res[t][name]['contamination'] for t in res]):.3f}, faulty rejected {np.nanmean([res[t][name]['faulty_rejected'] for t in res]):.3f}, domains refreshed {np.mean([res[t][name]['domains_refreshed'] for t in res]):.1f}/3"
    print(line)
json.dump(dict(args=vars(a), results=res), open(os.path.join(ROOT, 'outputs', f'conformal_refresh_acc{a.alpha_acc}_s{a.seed}{"_" + a.tasks.replace(":", "").replace("@", "_") if a.tasks else ""}.json'), 'w'), indent=1); print('saved')
