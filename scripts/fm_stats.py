"""27-cell statistics for the foundation-model branches vs the physics MLP, and fusion over seeds.
python scripts/fm_stats.py [--protocol mic_x_speed_x_load]
Reads outputs/full_<P>_mlp_s{0,1,2}(_probs.npz|.json), full_<P>_pathlora_refnorm_refgain_180s_s*.json,
full_<P>_dasheng_*_s*.json; prints per-branch mean/min over seeds, paired Wilcoxon over cells (seed-averaged),
fusion (product, entropy gate) per seed on the wave-window alignment. Writes outputs/fm_stats_<P>.json"""
import os, sys, glob, json, argparse, numpy as np
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'outputs')
ap = argparse.ArgumentParser(); ap.add_argument('--protocol', default='mic_x_speed_x_load'); ap.add_argument('--wave_seconds', type=int, default=180); a = ap.parse_args(); P = a.protocol
def load(pattern):
    R = {}
    for fn in sorted(glob.glob(os.path.join(OUT, pattern))):
        s = int(fn[:-5].rsplit('_s', 1)[1]); R[s] = {k: v['acc'] for k, v in json.load(open(fn))['results'].items()}
    return R
branches = {'physics_mlp': load(f'full_{P}_mlp_s[0-9].json'), 'beats_lora': load(f'full_{P}_pathlora_refnorm_refgain_{a.wave_seconds}s_s[0-9].json'),
            'dasheng_lora': load(f'full_{P}_dasheng_lora_refnorm_{a.wave_seconds}s_s[0-9].json'), 'dasheng_frozen': load(f'full_{P}_dasheng_frozen_refnorm_{a.wave_seconds}s_s[0-9].json'),
            'dasheng_lora_norefnorm': load(f'full_{P}_dasheng_lora_norefnorm_{a.wave_seconds}s_s[0-9].json')}
branches = {k: v for k, v in branches.items() if v}; cells = sorted(next(iter(branches['physics_mlp'].values())).keys())
summ = {}
print(f'== {P}: {len(cells)} cells ==')
for k, R in branches.items():
    M = np.array([[R[s][c] for c in cells] for s in sorted(R)]); summ[k] = dict(seeds=sorted(R), mean=float(M.mean()), std_over_seeds=float(M.mean(1).std()), min_cell=float(M.mean(0).min()), per_cell=M.mean(0).tolist())
    print(f'{k:24s} seeds={sorted(R)} mean={M.mean():.3f} (±{M.mean(1).std():.3f} over seeds) min-cell={M.mean(0).min():.3f}')
ref = np.array(summ['physics_mlp']['per_cell'])
for k in branches:
    if k == 'physics_mlp': continue
    x = np.array(summ[k]['per_cell']); d = x - ref
    try: pv = wilcoxon(x, ref).pvalue
    except ValueError: pv = 1.0
    summ[k]['vs_physics'] = dict(delta=float(d.mean()), wins=int((d > 0).sum()), losses=int((d < 0).sum()), p=float(pv)); print(f'  {k} vs physics: Δ={d.mean():+.3f}, wins {int((d>0).sum())}/{len(cells)}, Wilcoxon p={pv:.4f}')
# fusion per seed (wave-window alignment)
mf = np.load(os.path.join(ROOT, 'cache', 'guilin_full_feats.npz'), allow_pickle=True)['meta']; mw = np.load(os.path.join(ROOT, 'cache', f'guilin_full_wave16k_raw_{a.wave_seconds}s.npz'), allow_pickle=True)['meta']
key = lambda m: [f"{r['path']}|{r['rpm']}|{r['load']}|{r['state']}|{r['seg']}|{float(r['start']):.1f}" for r in m]
pos_f = {k: i for i, k in enumerate(key(mf))}; f_of_w = np.array([pos_f[k] for k in key(mw)])
def hist_ent(p):
    f = np.bincount(p.argmax(1), minlength=p.shape[1]) / len(p); f = f[f > 0]; return -(f * np.log(f)).sum() / np.log(p.shape[1])
def fuse(seed, fm_name):
    fa = os.path.join(OUT, f'full_{P}_mlp_s{seed}_probs.npz'); fb = os.path.join(OUT, f'full_{P}_{fm_name}_s{seed}_probs.npz')
    if not (os.path.exists(fa) and os.path.exists(fb)): return None
    A, B = np.load(fa), np.load(fb); rows = []
    for t in cells:
        pb, yb, teb = B[f'{t}/prob'], B[f'{t}/y'], B[f'{t}/te']; pos = {ix: i for i, ix in enumerate(A[f'{t}/te'])}; pa = A[f'{t}/prob'][np.array([pos[f_of_w[ix]] for ix in teb])]
        la, lb = np.log(pa + 1e-9), np.log(pb + 1e-9); ha, hb = hist_ent(pa), hist_ent(pb); wa = np.exp(10 * ha) / (np.exp(10 * ha) + np.exp(10 * hb))
        rows.append([(pa.argmax(1) == yb).mean(), (pb.argmax(1) == yb).mean(), ((la + lb).argmax(1) == yb).mean(), ((wa * la + (1 - wa) * lb).argmax(1) == yb).mean(), max((pa.argmax(1) == yb).mean(), (pb.argmax(1) == yb).mean())])
    return np.array(rows)
for fm in ['pathlora_refnorm_refgain_180s', 'dasheng_lora_refnorm_180s']:
    Rs = [r for r in (fuse(s, fm) for s in range(3)) if r is not None]
    if not Rs: continue
    M = np.mean(Rs, 0); names = ['physics', 'FM', 'product', 'gate', 'oracle']; print(f'fusion physics x {fm} ({len(Rs)} seeds, wave-aligned windows): ' + ' '.join(f'{n}={v:.3f}' for n, v in zip(names, M.mean(0))) + ' | min-cell ' + ' '.join(f'{n}={v:.3f}' for n, v in zip(names, M.min(0))))
    try: print(f'   product vs physics Wilcoxon p={wilcoxon(M[:,2], M[:,0]).pvalue:.4f}; product vs FM p={wilcoxon(M[:,2], M[:,1]).pvalue:.4f}; gate vs product p={wilcoxon(M[:,3], M[:,2]).pvalue:.4f}')
    except ValueError: pass
    summ[f'fusion_{fm}'] = dict(seeds=len(Rs), mean={n: float(v) for n, v in zip(names, M.mean(0))}, min_cell={n: float(v) for n, v in zip(names, M.min(0))}, per_cell=M.tolist())
json.dump(dict(cells=cells, summary=summ), open(os.path.join(OUT, f'fm_stats_{P}.json'), 'w'), indent=1); print('saved outputs/fm_stats_%s.json' % P)
