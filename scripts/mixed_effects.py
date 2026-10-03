"""Statistics that respect the factor structure of the 27-cell protocol.
Cells share microphones, speeds and loads, so per-cell paired differences are not independent. For each comparison
we fit a linear mixed model on the per-cell difference d_c = acc_A(c) - acc_B(c):
    d_c = beta0 + u_mic(c) + u_rpm(c) + u_load(c) + e_c     (crossed random intercepts, statsmodels MixedLM, one group + variance components)
and test beta0 = 0 (Wald z). We also report a factor-block summary (sign of the mean difference within each of the 9
factor levels) and, for reference, the paired Wilcoxon over cells.
python scripts/mixed_effects.py -> outputs/mixed_effects_mic_x_speed_x_load.json"""
import os, json, warnings, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'outputs'); P = 'mic_x_speed_x_load'
fm = json.load(open(os.path.join(OUT, f'fm_stats_{P}.json'))); et = json.load(open(os.path.join(OUT, f'ensemble_temperature_{P}.json'))); cells = fm['cells']; S = fm['summary']; R = et['results']
lin = json.load(open(os.path.join(OUT, f'full_{P}_linear_healthy_s0.json')))['results']
V = {'linear': np.array([lin[c]['acc'] for c in cells]), 'physics': np.array(S['physics_mlp']['per_cell']), 'beats': np.array(S['beats_lora']['per_cell']), 'dasheng_norefnorm': np.array(S['dasheng_lora_norefnorm']['per_cell']), 'dasheng': np.array(S['dasheng_lora']['per_cell']),
     'E_phys': np.array(R['physics, 3-seed ensemble']['per_cell']), 'E_beats': np.array(R['BEATs, 3-seed ensemble']['per_cell']), 'beats_seedavg': np.array(R['BEATs, seed-averaged']['per_cell']),
     'product_pairs': np.array(R['product physics x BEATs, seed pairs']['per_cell']), 'gate_pairs': np.array(R['gate physics x BEATs, seed pairs']['per_cell']),
     'product_E': np.array(R['product E_phys x E_beats']['per_cell']), 'Tproduct_E': np.array(R['T-product E_phys x E_beats']['nested_per_cell']), 'Tproduct_3': np.array(R['T-product E_phys x E_beats x Dasheng-frozen']['nested_per_cell']),
     'gate_E': np.array(R['gate E_phys x E_beats']['per_cell']), 'gate_3': np.array(R['gate E_phys x E_beats x Dasheng-frozen']['per_cell'])}
fac = [c.split(':')[1].split('@') for c in cells]; df0 = pd.DataFrame(dict(mic=[f[0] for f in fac], rpm=[f[1] for f in fac], load=[f[2] for f in fac]))
comparisons = [('physics', 'linear', 'physics MLP vs linear healthy-ref'), ('beats', 'physics', 'BEATs LoRA vs physics'), ('dasheng', 'dasheng_norefnorm', 'Dasheng: refnorm vs no refnorm'),
               ('E_beats', 'beats_seedavg', 'BEATs: 3-seed ensemble vs seed average'), ('product_pairs', 'physics', 'product (seed pairs) vs physics'), ('gate_pairs', 'physics', 'gate (seed pairs) vs physics'),
               ('product_E', 'E_phys', 'product of ensembles vs E_phys'), ('Tproduct_E', 'E_phys', 'T-product (nested) vs E_phys'), ('Tproduct_E', 'product_E', 'T-product vs untempered product'),
               ('Tproduct_3', 'E_phys', '3-way T-product (nested) vs E_phys'), ('gate_E', 'E_phys', 'gate of ensembles vs E_phys'), ('gate_3', 'E_phys', '3-way gate vs E_phys'), ('gate_3', 'Tproduct_3', '3-way gate vs 3-way T-product')]
out = []
print(f'{"comparison":42s} {"Δ":>7s} {"mixed z":>8s} {"p_mixed":>8s} {"95% CI":>16s} {"p_wilcoxon":>10s} blocks(+/9)')
for a_, b_, name in comparisons:
    d = V[a_] - V[b_]; df = df0.copy(); df['d'] = d; df['g'] = 1
    try:
        m = smf.mixedlm('d ~ 1', df, groups=df['g'], vc_formula={'mic': '0 + C(mic)', 'rpm': '0 + C(rpm)', 'load': '0 + C(load)'}).fit(reml=True, method=['lbfgs', 'powell'])
        beta, se = float(m.params['Intercept']), float(m.bse['Intercept']); z = beta / se; from scipy.stats import norm; p_mixed = float(2 * norm.sf(abs(z))); ci = (beta - 1.96 * se, beta + 1.96 * se); vc = {n: float(v) for n, v in zip(getattr(m.model, 'exog_vc', None).names if hasattr(m.model, 'exog_vc') and hasattr(m.model.exog_vc, 'names') else [f'vc{i}' for i in range(len(np.atleast_1d(m.vcomp)))], np.atleast_1d(m.vcomp))}
    except Exception as e:
        beta, se, z, p_mixed, ci, vc = float(d.mean()), float('nan'), float('nan'), float('nan'), (float('nan'), float('nan')), {'error': str(e)}
    blocks = sum(int(d[(df[k] == v).values].mean() > 0) for k in ['mic', 'rpm', 'load'] for v in df[k].unique())
    pw = float(wilcoxon(V[a_], V[b_]).pvalue) if np.any(d != 0) else 1.0
    print(f'{name:42s} {d.mean():+7.3f} {z:8.2f} {p_mixed:8.4f} [{ci[0]:+.3f}, {ci[1]:+.3f}] {pw:10.4f} {blocks}/9')
    out.append(dict(comparison=name, delta=float(d.mean()), beta=beta, se=se, z=z, p_mixed=p_mixed, ci95=list(ci), p_wilcoxon=pw, blocks_positive=blocks, wins=int((d > 0).sum()), var_components=vc))
json.dump(out, open(os.path.join(OUT, f'mixed_effects_{P}.json'), 'w'), indent=1); print('saved outputs/mixed_effects_%s.json' % P)
