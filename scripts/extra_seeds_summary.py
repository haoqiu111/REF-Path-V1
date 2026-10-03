"""Seed statistics of two multi-seed experiments, collected into outputs/extra_seeds_summary.json (read by verify_results.py):
  spi_lam<l>  : synchronous-pair invariance (scripts/spi.py), mean / min accuracy over the nine cells per seed
  openset27   : unknown-fault rejection on the 27 cells (scripts/train_mlp2.py --holdout_class), per held-out class the mean over
                cells of every AUROC and of the known-class accuracy, then mean and std over seeds
python scripts/extra_seeds_summary.py"""
import os, glob, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'outputs')
S = {}
runs = [json.load(open(fn))['results'] for fn in sorted(glob.glob(os.path.join(OUT, 'guilin_mic_x_speed_spi_s?.json')))]
for lam in sorted({k for r in runs for k in r}, key=float):
    A = np.array([[v['acc'] for v in r[lam].values()] for r in runs if lam in r])           # (seeds, cells)
    S[f'spi_lam{lam}'] = dict(seeds=len(A), mean=float(A.mean(1).mean()), std=float(A.mean(1).std()), min_per_seed=A.min(1).tolist(), mean_per_seed=A.mean(1).tolist())
S['openset27'] = {}
for c in ['inner_pitting', 'outer_pitting', 'inner_outer_pitting', 'cage_fracture']:
    runs = [json.load(open(fn))['results'] for fn in sorted(glob.glob(os.path.join(OUT, f'full_mic_x_speed_x_load_mlp_holdout_{c}_s?.json')))]
    S['openset27'][c] = {}
    for k in ['auroc_softmax', 'auroc_knn', 'auroc_rank', 'auroc_fisher', 'auroc_minp', 'acc_known']:
        m = [float(np.mean([v[k] for v in r.values()])) for r in runs]
        S['openset27'][c][k] = dict(mean=float(np.mean(m)), std=float(np.std(m)), n=len(m))
json.dump(S, open(os.path.join(OUT, 'extra_seeds_summary.json'), 'w'), indent=1); print('saved outputs/extra_seeds_summary.json')
for k in [k for k in S if k.startswith('spi')]: print(f"{k}: MEAN acc {S[k]['mean']:.3f} ± {S[k]['std']:.3f} ({S[k]['seeds']} seeds)")
for c, v in S['openset27'].items(): print(f"{c}: AUROC softmax {v['auroc_softmax']['mean']:.3f} kNN {v['auroc_knn']['mean']:.3f} rank {v['auroc_rank']['mean']:.3f} ({v['auroc_rank']['n']} seeds)")
