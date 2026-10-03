"""Mean ± std over seeds and paired Wilcoxon signed-rank tests over the 9 mic × speed tasks.
Reads outputs/*.json (per-task acc). Usage: python scripts/stats.py"""
import os, glob, json, numpy as np
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'outputs')
def load(pattern):
    runs = []
    for fn in sorted(glob.glob(os.path.join(OUT, pattern))):
        j = json.load(open(fn)); r = j['results']
        if 'results' in j and isinstance(next(iter(r.values())), dict) and 'acc' in next(iter(r.values())): runs.append({k: v['acc'] for k, v in r.items()})
    return runs
groups = {
    'physics MLP + mixup': 'guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_s?.json',
    'BEATs LoRA ref-norm (ref gain)': 'guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm_refsplit_refgain_s?.json',
    'BEATs LoRA ref-norm (per-recording gain)': 'guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm_seed?.json',
    'Cyclo3DNet': 'guilin_mic_x_speed_cyclo3d_ml_aug_refstd_psd*.json',
    'MLP no mixup': 'guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.0_refsplit.json',
    'MLP PSD only': 'guilin_mic_x_speed_mlp_psd_pca0_mix0.4_refsplit.json',
    'MLP cyclo only': 'guilin_mic_x_speed_mlp_cyclo_pca0_mix0.4_refsplit.json',
    'MLP full 3D': 'guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_full3d.json',
    'cross-modal (Peng-style) raw': 'guilin_mic_x_speed_crossmodal_s?.json',
}
tasks = None; table = {}
for name, pat in groups.items():
    runs = load(pat)
    if not runs: print(f'{name:34s} (no runs)'); continue
    tasks = sorted(runs[0].keys()); M = np.array([[r[t] for t in tasks] for r in runs]); table[name] = M
    print(f'{name:34s} n_seeds={len(runs)}  mean {M.mean():.3f} ± {M.mean(1).std():.3f}   min-task {M.mean(0).min():.3f}')
lit = sorted(glob.glob(os.path.join(OUT, 'guilin_mic_x_speed_baselines_lit_s*.json'))); litruns = {}
for fn in lit:
    j = json.load(open(fn))
    for w, r in j['results'].items(): litruns.setdefault(w, []).append([r[t]['acc'] for t in tasks])
for w, runs in litruns.items():
    M = np.array(runs); table[f'lit:{w}'] = M; print(f'{"lit:"+w:34s} n_seeds={len(runs)}  mean {M.mean():.3f} ± {M.mean(1).std():.3f}   min-task {M.mean(0).min():.3f}')
# linear baselines from csv
import csv
for fn, key in [('guilin_baselines.csv', 'linear healthy-ref PSD+cyclo')]:
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, fn))) if r['protocol'] == 'mic_x_speed' and r['features'] == 'PSD+cyclo' and r['normalization'] == 'healthy']
    d = {r['task']: float(r['acc']) for r in rows}; table[key] = np.array([[d[t] for t in tasks]]); print(f'{key:34s} mean {table[key].mean():.3f}')
print('\nPaired Wilcoxon signed-rank over the 9 tasks (seed-mean per task):')
ref = 'physics MLP + mixup'
if ref in table:
    a = table[ref].mean(0)
    for name, M in table.items():
        if name == ref: continue
        b = M.mean(0)
        try: p = wilcoxon(a, b).pvalue
        except ValueError: p = float('nan')
        print(f'  {ref} vs {name:34s}: Δmean {a.mean()-b.mean():+.3f}, wins {int((a>b).sum())}/9, p={p:.3f}')
