"""Seed ensembles and temperature-calibrated fusion on the 27-cell protocol (wave-aligned windows, first 180 s of segment 5).
Step 1  Seed ensembles: geometric mean of the three seeds' posteriors of each learned branch (physics MLP, BEATs LoRA,
        Dasheng LoRA) -> deployable variance control. Reported alone and inside product / entropy-gate fusion.
Step 2  Temperature-calibrated product fusion, deployable: sum_b log p_b / T_b with T chosen WITHOUT the target task:
        nested leave-one-task-out (T for task t maximises the mean fused accuracy over the other 26 tasks, whose test
        windows belong to other domains = legitimate labelled source-side data). The best-on-all-27 T ('oracle T') is
        reported only as a reference. Grid: T ratios in [1/4, 4] (log-spaced), physics branch fixed at T = 1.
python scripts/ensemble_temperature.py   -> outputs/ensemble_temperature_mic_x_speed_x_load.json"""
import os, json, itertools, numpy as np
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'outputs'); P = 'mic_x_speed_x_load'
mf = np.load(os.path.join(ROOT, 'cache', 'guilin_full_feats.npz'), allow_pickle=True)['meta']; mw = np.load(os.path.join(ROOT, 'cache', 'guilin_full_wave16k_raw_180s.npz'), allow_pickle=True)['meta']
key = lambda m: [f"{r['path']}|{r['rpm']}|{r['load']}|{r['state']}|{r['seg']}|{float(r['start']):.1f}" for r in m]
pos_f = {k: i for i, k in enumerate(key(mf))}; f_of_w = np.array([pos_f[k] for k in key(mw)])
cells = json.load(open(os.path.join(OUT, f'fm_stats_{P}.json')))['cells']
ld = lambda n: np.load(os.path.join(OUT, f'full_{P}_{n}_probs.npz'))
A = [ld(f'mlp_s{s}') for s in range(3)]; B = [ld(f'pathlora_refnorm_refgain_180s_s{s}') for s in range(3)]; D = [ld(f'dasheng_lora_refnorm_180s_s{s}') for s in range(3)]; DF = ld('dasheng_frozen_refnorm_180s_s0')

def ent(p):
    f = np.bincount(p.argmax(1), minlength=p.shape[1]) / len(p); f = f[f > 0]; return -(f * np.log(f)).sum() / np.log(p.shape[1])
def geo_mean(ps):
    L = np.mean([np.log(p + 1e-9) for p in ps], 0); E = np.exp(L - L.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)
acc = lambda p, y: float((p.argmax(1) == y).mean())
def gate(ps):
    H = np.array([ent(p) for p in ps]); w = np.exp(10 * H) / np.exp(10 * H).sum(); return sum(wi * np.log(p + 1e-9) for wi, p in zip(w, ps))
def product(ps, T=None):
    T = T or [1.0] * len(ps); return sum(np.log(p + 1e-9) / t for p, t in zip(ps, T))

# ---- per-cell posteriors on the wave grid ----
C = {}
for t in cells:
    yb, teb = B[0][f'{t}/y'], B[0][f'{t}/te']
    pa = []
    for a in A:
        pos = {ix: i for i, ix in enumerate(a[f'{t}/te'])}; pa.append(a[f'{t}/prob'][np.array([pos[f_of_w[ix]] for ix in teb])])
    pb = [b[f'{t}/prob'] for b in B]; pd = [d[f'{t}/prob'] for d in D]; pf = DF[f'{t}/prob']
    for x in pb + pd + [pf]: assert len(x) == len(yb)
    C[t] = dict(y=yb, phys=pa, beats=pb, dash=pd, dfz=pf, E_phys=geo_mean(pa), E_beats=geo_mean(pb), E_dash=geo_mean(pd))

res = {}
def row(name, per_cell):
    per_cell = np.array(per_cell); res[name] = dict(mean=float(per_cell.mean()), min=float(per_cell.min()), per_cell=per_cell.tolist()); return per_cell
print(f'== {P}: 27 cells, wave-aligned windows ==')
# ---- step 1: seed ensembles ----
phys_seed = row('physics, seed-averaged', [np.mean([acc(p, C[t]['y']) for p in C[t]['phys']]) for t in cells])
beats_seed = row('BEATs, seed-averaged', [np.mean([acc(p, C[t]['y']) for p in C[t]['beats']]) for t in cells])
dash_seed = row('Dasheng-LoRA, seed-averaged', [np.mean([acc(p, C[t]['y']) for p in C[t]['dash']]) for t in cells])
phys_E = row('physics, 3-seed ensemble', [acc(C[t]['E_phys'], C[t]['y']) for t in cells])
beats_E = row('BEATs, 3-seed ensemble', [acc(C[t]['E_beats'], C[t]['y']) for t in cells])
dash_E = row('Dasheng-LoRA, 3-seed ensemble', [acc(C[t]['E_dash'], C[t]['y']) for t in cells])
row('Dasheng frozen probe', [acc(C[t]['dfz'], C[t]['y']) for t in cells])
for n, a_, b_ in [('physics', phys_seed, phys_E), ('BEATs', beats_seed, beats_E), ('Dasheng-LoRA', dash_seed, dash_E)]:
    d = b_ - a_; print(f'{n:14s} seed-avg {a_.mean():.3f} (min {a_.min():.3f}) -> ensemble {b_.mean():.3f} (min {b_.min():.3f})  Δ={d.mean():+.3f} wins {(d>0).sum()}/27 p={wilcoxon(b_, a_).pvalue:.4f}')
    res[f'ensemble gain {n}'] = dict(delta=float(d.mean()), wins=int((d > 0).sum()), p=float(wilcoxon(b_, a_).pvalue))
# fusion with ensembles (compared with the mean over seed pairs)
pair_prod = row('product physics x BEATs, seed pairs', [np.mean([acc(product([pa, pb]), C[t]['y']) for pa, pb in zip(C[t]['phys'], C[t]['beats'])]) for t in cells])
pair_gate = row('gate physics x BEATs, seed pairs', [np.mean([acc(gate([pa, pb]), C[t]['y']) for pa, pb in zip(C[t]['phys'], C[t]['beats'])]) for t in cells])
E_prod = row('product E_phys x E_beats', [acc(product([C[t]['E_phys'], C[t]['E_beats']]), C[t]['y']) for t in cells])
E_gate = row('gate E_phys x E_beats', [acc(gate([C[t]['E_phys'], C[t]['E_beats']]), C[t]['y']) for t in cells])
E3_prod = row('product E_phys x E_beats x Dasheng-frozen', [acc(product([C[t]['E_phys'], C[t]['E_beats'], C[t]['dfz']]), C[t]['y']) for t in cells])
E3_gate = row('gate E_phys x E_beats x Dasheng-frozen', [acc(gate([C[t]['E_phys'], C[t]['E_beats'], C[t]['dfz']]), C[t]['y']) for t in cells])
E3d_prod = row('product E_phys x E_beats x E_dash', [acc(product([C[t]['E_phys'], C[t]['E_beats'], C[t]['E_dash']]), C[t]['y']) for t in cells])
E3d_gate = row('gate E_phys x E_beats x E_dash', [acc(gate([C[t]['E_phys'], C[t]['E_beats'], C[t]['E_dash']]), C[t]['y']) for t in cells])
for n, v in [('product seed pairs', pair_prod), ('product ensembles', E_prod), ('gate seed pairs', pair_gate), ('gate ensembles', E_gate), ('product 3-way (frozen Dasheng)', E3_prod), ('gate 3-way (frozen Dasheng)', E3_gate), ('product 3-way (E_dash)', E3d_prod), ('gate 3-way (E_dash)', E3d_gate)]:
    print(f'{n:32s} {v.mean():.3f} (min {v.min():.3f})  vs E_phys: wins {(v>phys_E).sum()}/27 p={wilcoxon(v, phys_E).pvalue:.4f}')
    res[n + ' vs E_phys'] = dict(wins=int((v > phys_E).sum()), p=float(wilcoxon(v, phys_E).pvalue))

# ---- step 2: temperature-calibrated product, nested leave-one-task-out ----
grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 17))
def temp_search(branches, Tgrid_list, label):
    """branches: dict cell -> list of posteriors [phys, b2, (b3)]; T for phys = 1; returns nested-CV per-cell acc"""
    combos = list(itertools.product(*Tgrid_list)); n = len(cells)
    M = np.zeros((len(combos), n))                       # acc of each T-combo on each cell
    for j, T in enumerate(combos):
        for i, t in enumerate(cells): M[j, i] = acc(product(branches[t], [1.0] + list(T)), C[t]['y'])
    nested = np.zeros(n); chosen = []
    for i in range(n):
        others = np.delete(np.arange(n), i); j = M[:, others].mean(1).argmax(); nested[i] = M[j, i]; chosen.append(combos[j])
    j_or = M.mean(1).argmax(); oracle = M[j_or]
    print(f'{label:40s} nested-CV {nested.mean():.3f} (min {nested.min():.3f}) wins vs E_phys {(nested>phys_E).sum()}/27 p={wilcoxon(nested, phys_E).pvalue:.4f} | oracle-T {oracle.mean():.3f} (T={tuple(round(x,2) for x in combos[j_or])}) | chosen T range {[tuple(round(x,2) for x in c) for c in sorted(set(chosen))]}')
    res[label] = dict(nested_mean=float(nested.mean()), nested_min=float(nested.min()), nested_per_cell=nested.tolist(), wins_vs_E_phys=int((nested > phys_E).sum()), p_vs_E_phys=float(wilcoxon(nested, phys_E).pvalue),
                      p_vs_untempered=None, oracle_mean=float(oracle.mean()), oracle_T=[float(x) for x in combos[j_or]], chosen_T=[[float(x) for x in c] for c in chosen], untempered_mean=float(M[[k for k, c in enumerate(combos) if all(abs(x - 1) < 1e-9 for x in c)][0]].mean()))
    return nested
n2 = temp_search({t: [C[t]['E_phys'], C[t]['E_beats']] for t in cells}, [grid], 'T-product E_phys x E_beats')
res['T-product E_phys x E_beats']['p_vs_untempered'] = float(wilcoxon(n2, E_prod).pvalue); print(f'   vs untempered product ({E_prod.mean():.3f}): p={wilcoxon(n2, E_prod).pvalue:.4f}; vs gate ({E_gate.mean():.3f}): p={wilcoxon(n2, E_gate).pvalue:.4f}')
g9 = np.exp(np.linspace(np.log(0.25), np.log(16.0), 13))
n3 = temp_search({t: [C[t]['E_phys'], C[t]['E_beats'], C[t]['dfz']] for t in cells}, [g9, g9], 'T-product E_phys x E_beats x Dasheng-frozen')
res['T-product E_phys x E_beats x Dasheng-frozen']['p_vs_untempered'] = float(wilcoxon(n3, E3_prod).pvalue); print(f'   vs untempered 3-way product ({E3_prod.mean():.3f}): p={wilcoxon(n3, E3_prod).pvalue:.4f}; vs 3-way gate ({E3_gate.mean():.3f}): p={wilcoxon(n3, E3_gate).pvalue:.4f}')
# single-seed robustness: temperature per seed pair (no ensembles)
ss = []
for s in range(3):
    ss.append(temp_search({t: [C[t]['phys'][s], C[t]['beats'][s]] for t in cells}, [grid], f'T-product single seed pair {s}'))
ss = np.mean(ss, 0); res['T-product single seed pairs, mean'] = dict(mean=float(ss.mean()), min=float(ss.min()), per_cell=ss.tolist(), wins_vs_phys_seed=int((ss > phys_seed).sum()), p_vs_phys_seed=float(wilcoxon(ss, phys_seed).pvalue))
print(f'T-product, single seed pairs averaged: {ss.mean():.3f} (min {ss.min():.3f}) vs physics seed-avg {phys_seed.mean():.3f}: wins {(ss>phys_seed).sum()}/27 p={wilcoxon(ss, phys_seed).pvalue:.4f}')
json.dump(dict(cells=cells, results=res), open(os.path.join(OUT, f'ensemble_temperature_{P}.json'), 'w'), indent=1); print('saved outputs/ensemble_temperature_%s.json' % P)
