"""Calibrate-after-adapt conformal guarantees on the healthy reference (Reference-Only setting).

For each task we have a model's probabilities on (i) the calibration half of the target-domain healthy reference
windows (label = healthy, disjoint from the windows used for standardization) and (ii) the test windows.

1. Healthy-acceptance / false-alarm control: nonconformity s = 1 - p(healthy). Conformal p-value of a test window
   against the calibration scores; declare "fault" when p-value <= alpha. Guarantee: P(false alarm on a healthy
   window) <= alpha (exchangeability of healthy windows within the domain). We report the empirical false-alarm rate
   on test healthy windows and the detection rate on faulty windows for several alpha.
2. Prediction sets: split-conformal on the same calibration set cannot cover fault classes (no faulty calibration
   data), so we report the class-conditional guarantee for the healthy class and a "mondrian" variant would need
   faulty references -> reported as not applicable under Reference-Only.

python scripts/conformal.py outputs/<model>_refsplit_probs.npz [--healthy_index 1]
"""
import sys, numpy as np, argparse
ap = argparse.ArgumentParser(); ap.add_argument('probs'); ap.add_argument('--healthy_index', type=int, default=1); a = ap.parse_args()
Z = np.load(a.probs); H = a.healthy_index; alphas = [0.01, 0.05, 0.10]
tasks = sorted({k.split('/')[0] for k in Z.files})
print(f'{"task":22s} n_cal ' + ' '.join(f'| a={al:.2f}: FAR   DET ' for al in alphas))
agg = {al: [] for al in alphas}
for t in tasks:
    pc = Z[f'{t}/ref_prob']; pt = Z[f'{t}/prob']; y = Z[f'{t}/y']
    s_cal = 1 - pc[:, H]; s_te = 1 - pt[:, H]; n = len(s_cal)
    pv = np.array([(np.sum(s_cal >= s) + 1) / (n + 1) for s in s_te])         # conformal p-values (larger score = more anomalous)
    line = f'{t:22s} {n:5d} '
    for al in alphas:
        alarm = pv <= al; far = alarm[y == H].mean(); det = alarm[y != H].mean(); agg[al].append((far, det)); line += f'| {far:.3f} {det:.3f}      '
    print(line)
for al in alphas:
    r = np.array(agg[al]); print(f'alpha={al:.2f}: mean false-alarm {r[:,0].mean():.3f} (max {r[:,0].max():.3f})  mean detection {r[:,1].mean():.3f} (min {r[:,1].min():.3f})')
