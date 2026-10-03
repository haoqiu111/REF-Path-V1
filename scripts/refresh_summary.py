"""Aggregate the conformal self-refresh runs (scripts/conformal_refresh.py) over seeds and acceptance thresholds.
python scripts/refresh_summary.py -> outputs/conformal_refresh_summary.json + markdown table"""
import os, glob, json, re, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'outputs')
AL = [0.01, 0.05, 0.10]; V = ['OLD', 'IDEAL', 'SELF-H', 'SELF-ALL', 'NAIVE-ALL']
runs = {}
for fn in sorted(glob.glob(os.path.join(OUT, 'conformal_refresh_acc*_s?.json'))):
    m = re.search(r'acc([\d.]+)_s(\d)', fn); runs.setdefault(float(m.group(1)), {})[int(m.group(2))] = json.load(open(fn))['results']
summary = {}
for acc_a, seeds in sorted(runs.items()):
    S = {}
    for v in V:
        agg = {}
        for k in ['acc'] + [f'{q}@{a}' for q in ('far', 'det') for a in AL]:
            per_seed = [np.mean([r[t][v][k] for t in r]) for r in seeds.values()]; agg[k] = float(np.mean(per_seed)); agg[k + '_seeds'] = [float(x) for x in per_seed]
            if k.startswith('far'): agg[k + '_max'] = float(np.mean([np.max([r[t][v][k] for t in r]) for r in seeds.values()]))
        if v.startswith('SELF'):
            for k in ['n_accepted', 'n_candidates', 'contamination', 'faulty_rejected']: agg[k] = float(np.mean([np.mean([r[t][v][k] for t in r]) for r in seeds.values()]))
            agg['contamination_by_mic'] = {mic: float(np.mean([np.mean([r[t][v]['contamination'] for t in r if mic in t]) for r in seeds.values()])) for mic in ['CRY333', 'BK4966_2', 'BK4966_3']}
        S[v] = agg
    summary[str(acc_a)] = dict(n_seeds=len(seeds), variants=S)
    print(f'== alpha_acc = {acc_a}, {len(seeds)} seed(s) ==')
    print('| reference | acc | FAR@0.01 (max) | det | FAR@0.05 (max) | det | FAR@0.10 (max) | det | accepted / candidates | contamination (CRY333 / B&K#2 / B&K#3) | faulty rejected |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for v in V:
        s = S[v]; extra = (f"{s['n_accepted']:.0f} / {s['n_candidates']:.0f} | {s['contamination']:.3f} ({s['contamination_by_mic']['CRY333']:.3f} / {s['contamination_by_mic']['BK4966_2']:.3f} / {s['contamination_by_mic']['BK4966_3']:.3f}) | {s['faulty_rejected']:.3f}" if v.startswith('SELF') else '— | — | —')
        print(f"| {v} | {s['acc']:.3f} | " + ' | '.join(f"{s[f'far@{a}']:.3f} ({s[f'far@{a}_max']:.3f}) | {s[f'det@{a}']:.3f}" for a in AL) + f' | {extra} |')
json.dump(summary, open(os.path.join(OUT, 'conformal_refresh_summary.json'), 'w'), indent=1); print('saved outputs/conformal_refresh_summary.json')
