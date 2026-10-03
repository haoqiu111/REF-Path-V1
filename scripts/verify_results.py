"""Consistency check between the numbers reported in the paper and the files in outputs/.
Recomputes every reported number from the raw per-task results (CSV / JSON / probability files) and compares it with the
reported value (the 'expected' column below). Tolerance 0.0015 on accuracies. No training, no GPU, no dataset needed.

  * 'Table 2: ...' checks parse Table 2 of results/expected_tables.md and compare every cell with a recomputation
    (scripts/make_fig_sota.py, function collect()).
  * 'Sec. 4.2: ...' checks cover the healthy-share sweep of the transductive methods and the paired comparison over cells.
  * The remaining checks carry their expected value in this file: ablations (Table 4), reference refresh (Table 5; the script
    variants OLD / IDEAL / SELF-H / SELF-ALL / NAIVE-ALL are the rows C / RR / SR-H / SR-A / UF), numbers quoted in the text of
    Section 4, and supplementary runs that the paper does not print (e.g. the first-round baselines of scripts/baselines_lit.py).

python scripts/verify_results.py            -> table of PASS / FAIL / MISSING, last line '<n> checks: <n> pass, 0 fail, 0 missing'
A result file that is absent is reported as MISSING. Exit code 1 if any FAIL."""
import os, sys, glob, json, csv, re, subprocess, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'outputs'); PY = sys.executable
TOL = 0.0015; rows = []


def report(name, expected, got, tol=TOL):
    if got is None: status = 'MISSING'
    else: status = 'PASS' if abs(got - expected) <= tol else 'FAIL'
    rows.append((status, name, expected, got)); return status


def json_mean(pattern, key='acc', sub=None):
    """mean over tasks of each file matching pattern -> list (one per seed/file); sub = nested key (e.g. SPI lambda)"""
    vals = []
    pats = pattern if isinstance(pattern, list) else [pattern]
    for fn in sorted(sum((glob.glob(os.path.join(OUT, pt)) for pt in pats), [])):
        r = json.load(open(fn))['results']
        if sub is not None: r = r[sub]
        vals.append(np.mean([v[key] if isinstance(v, dict) else v for v in r.values()]))
    return vals


def seeds(name, pattern, expected_mean, expected_std=None, sub=None):
    v = json_mean(pattern, sub=sub)
    if not v: report(name, expected_mean, None); return
    report(f'{name} (n={len(v)})', expected_mean, float(np.mean(v)))
    if expected_std is not None and len(v) > 1: report(f'{name} std over seeds', expected_std, float(np.std(v)), tol=0.003)


# ---- nine cells, 60-s clips: linear brackets from the run_baselines.py CSV ----------------------------------------------
def csv_mean(fn, protocol, norm, feat_regex):
    if not os.path.exists(fn): return None
    acc = [float(r['acc']) for r in csv.DictReader(open(fn)) if r['protocol'] == protocol and r['normalization'] == norm and re.fullmatch(feat_regex, r.get('features', r.get('probe', '')))]
    return float(np.mean(acc)) if acc else None
B = os.path.join(OUT, 'guilin_baselines.csv')
report('9 cells: linear raw, log-PSD', 0.585, csv_mean(B, 'mic_x_speed', 'raw', r'logPSD'))
report('9 cells: linear + reference, log-PSD', 0.819, csv_mean(B, 'mic_x_speed', 'healthy', r'logPSD'))
report('9 cells: linear + reference, log-PSD + map (deployable baseline)', 0.838, csv_mean(B, 'mic_x_speed', 'healthy', r'PSD\+cyclo'))
report('9 cells: linear oracle, log-PSD', 0.944, csv_mean(B, 'mic_x_speed', 'oracle', r'logPSD'))
report('9 cells: linear oracle, log-PSD + map (not in the paper)', 0.935, csv_mean(B, 'mic_x_speed', 'oracle', r'PSD\+cyclo'))
F = os.path.join(OUT, 'guilin_beats_frozen.csv')
report('9 cells: frozen BEATs + LR, raw', 0.717, csv_mean(F, 'mic_x_speed', 'raw', r'LR'))
report('9 cells: frozen BEATs + LR, + reference', 0.642, csv_mean(F, 'mic_x_speed', 'healthy', r'LR'))

# ---- nine cells, 60-s clips: learned models, ablations (Table 4), single-factor protocols -----------------------------
seeds('9 cells: Cyclo3DNet (3 seeds)', 'guilin_mic_x_speed_cyclo3d_ml_aug_refstd_psd*.json', 0.764, 0.009)
seeds('9 cells: BEATs-LoRA, raw', 'guilin_mic_x_speed_pathlora_plain_ml_aug.json', 0.756)
seeds('9 cells: BEATs hypernetwork LoRA (PSD descriptor)', 'guilin_mic_x_speed_pathlora_ml_aug.json', 0.698)
seeds('9 cells: BEATs hypernetwork LoRA (BEATs descriptor)', 'guilin_mic_x_speed_pathlora_ml_aug_descbeats.json', 0.597)
seeds('9 cells: BEATs-LoRA + reference, per-recording gain (3 seeds)', ['guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm.json', 'guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm_seed[12].json'], 0.799, 0.005)
seeds('9 cells: BEATs-LoRA + reference (3 seeds)', 'guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm_refsplit_refgain_s?.json', 0.826, 0.009)
seeds('9 cells: physics branch (3 seeds)', 'guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_s?.json', 0.870, 0.002)
seeds('ablation: PSD only', 'guilin_mic_x_speed_mlp_psd_pca0_mix0.4_refsplit.json', 0.811)
seeds('ablation: map only', 'guilin_mic_x_speed_mlp_cyclo_pca0_mix0.4_refsplit.json', 0.855)
seeds('ablation: physics branch, full (seed 0)', 'guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit.json', 0.873)
seeds('ablation: w/o mixup', 'guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.0_refsplit.json', 0.815)
seeds('ablation: + PCA-60', 'guilin_mic_x_speed_mlp_psd+cyclo_pca60_mix0.4_refsplit.json', 0.804)
seeds('ablation: + sub-window axis', 'guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_full3d.json', 0.857)
seeds('ablation: map only, + sub-window axis', 'guilin_mic_x_speed_mlp_cyclo_pca0_mix0.4_refsplit_full3d.json', 0.828)
seeds('held-out microphone: physics branch', 'guilin_lomo_mlp_psd+cyclo_pca0_mix0.4_refsplit.json', 0.933)
seeds('held-out speed: physics branch', 'guilin_loso_mlp_psd+cyclo_pca0_mix0.4_refsplit.json', 0.970)
seeds('held-out microphone: BEATs-LoRA + reference', 'guilin_lomo_pathlora_plain_ml_aug_refnorm_refsplit_refgain.json', 0.923)
seeds('held-out speed: BEATs-LoRA + reference', 'guilin_loso_pathlora_plain_ml_aug_refnorm_refsplit_refgain.json', 0.910)
seeds('SPI: lambda = 0 (no pair term), seed 0', 'guilin_mic_x_speed_spi_s0.json', 0.865, sub='0.0')
seeds('SPI: lambda = 0.5, seed 0', 'guilin_mic_x_speed_spi_s0.json', 0.821, sub='0.5')
seeds('first-round baselines: cross-modal, raw (seed 0)', 'guilin_mic_x_speed_crossmodal_s0.json', 0.582)
seeds('first-round baselines: cross-modal + reference (seed 0)', 'guilin_mic_x_speed_crossmodal_refnorm_s0.json', 0.581)
seeds('AHU: physics branch', 'ahu_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit.json', 0.56, 0.0)

# ---- first-round baselines of scripts/baselines_lit.py (3 seeds, one JSON per seed with nested method -> task); the paper
#      reports the re-implementations of scripts/sota_feat.py / sota_wave.py instead (Table 2 checks below) ----------------
lit = [json.load(open(fn))['results'] for fn in sorted(glob.glob(os.path.join(OUT, 'guilin_mic_x_speed_baselines_lit_s?.json')))]
def lit_mean(method):
    v = []
    for r in lit:
        if method in r: v.append(np.mean([t['acc'] if isinstance(t, dict) else t for t in r[method].values()]))
    return float(np.mean(v)) if v else None
for m, e in [('src_only', 0.585), ('adabn_ref', 0.564), ('adabn_test', 0.898), ('tent', 0.898), ('dann', 0.794), ('coral', 0.850), ('cscoh_cnn', 0.482), ('cscoh_cnn_ord', 0.511), ('cscoh_cnn_ref', 0.624)]:
    report(f'first-round baselines: {m} (n={len(lit)})', e, lit_mean(m))

# ---- fusion on the nine cells (3 seed pairs): product and prediction-histogram-entropy gate ---------------------------------
def hist_ent(p):
    f = np.bincount(p.argmax(1), minlength=p.shape[1]) / len(p); f = f[f > 0]; return -(f * np.log(f)).sum() / np.log(p.shape[1])
geo, gate = [], []
for s_ in range(3):
    a = os.path.join(OUT, f'guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_s{s_}_probs.npz'); b = os.path.join(OUT, f'guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm_refsplit_refgain_s{s_}_probs.npz')
    if not (os.path.exists(a) and os.path.exists(b)): continue
    A, Bp = np.load(a), np.load(b); g1, g2 = [], []
    for t in sorted({k.split('/')[0] for k in A.files}):
        pa, tea = A[f'{t}/prob'], A[f'{t}/te']; pb, yb, teb = Bp[f'{t}/prob'], Bp[f'{t}/y'], Bp[f'{t}/te']
        fiw = (teb // 60) * 119 + 2 * (teb % 60)                     # wave window j of a 60-s recording == feats window 2j (hop 0.5 s)
        pos = {ix: i for i, ix in enumerate(tea)}; pa = pa[np.array([pos[ix] for ix in fiw])]
        la, lb = np.log(pa + 1e-9), np.log(pb + 1e-9); ha, hb = hist_ent(pa), hist_ent(pb); wa = np.exp(10 * ha) / (np.exp(10 * ha) + np.exp(10 * hb))
        g1.append(((la + lb).argmax(1) == yb).mean()); g2.append(((wa * la + (1 - wa) * lb).argmax(1) == yb).mean())
    geo.append(np.mean(g1)); gate.append(np.mean(g2))
report('9 cells: REF-Path fusion (3 seed pairs)', 0.896, float(np.mean(geo)) if geo else None)
report('9 cells: REF-Path fusion, std over seed pairs', 0.006, float(np.std(geo)) if geo else None, tol=0.003)
report('9 cells: REF-Path-G, entropy gate (3 seed pairs)', 0.912, float(np.mean(gate)) if gate else None)
report('9 cells: REF-Path-G, std over seed pairs', 0.005, float(np.std(gate)) if gate else None, tol=0.003)

# ---- full recordings, 27 and 9 cells ----------------------------------------------------------------------------------
for P, col in [('mic_x_speed_x_load', '27'), ('mic_x_speed', '9')]:
    for tag, e in {'27': [('linear_raw', 0.583), ('linear_healthy', 0.733), ('linear_oracle', 0.917), ('mlp_ml', 0.782), ('mlp_coral1.0', 0.791), ('mlp_ml_coral1.0', 0.804), ('mlp_mix0.0', 0.683), ('mlp_psd', 0.773)],
                   '9': [('linear_raw', 0.612), ('linear_healthy', 0.833), ('linear_oracle', 0.946), ('mlp_ml', 0.802), ('mlp_coral1.0', 0.840), ('mlp_ml_coral1.0', 0.839), ('mlp_mix0.0', 0.749), ('mlp_psd', 0.833)]}[col]:
        seeds(f'full recordings, {col} cells: {tag}', f'full_{P}_{tag}_s0.json', e)
    seeds(f'full recordings, {col} cells: physics branch (3 seeds)', f'full_{P}_mlp_s?.json', {'27': 0.804, '9': 0.863}[col], {'27': 0.022, '9': 0.020}[col])
seeds('27 cells: BEATs-LoRA + reference, 180 s (3 seeds)', 'full_mic_x_speed_x_load_pathlora_refnorm_refgain_180s_s?.json', 0.797, 0.031)
seeds('27 cells: Dasheng-LoRA + reference (3 seeds)', 'full_mic_x_speed_x_load_dasheng_lora_refnorm_180s_s?.json', 0.760, 0.010)
seeds('27 cells: Dasheng frozen probe + reference', 'full_mic_x_speed_x_load_dasheng_frozen_refnorm_180s_s0.json', 0.784)
seeds('27 cells: Dasheng-LoRA, raw (seed 0)', 'full_mic_x_speed_x_load_dasheng_lora_norefnorm_180s_s0.json', 0.640)
fs = os.path.join(OUT, 'fm_stats_mic_x_speed_x_load.json')
if os.path.exists(fs):
    S = json.load(open(fs))['summary']; fb = S.get('fusion_pathlora_refnorm_refgain_180s', {}).get('mean', {}); fd = S.get('fusion_dasheng_lora_refnorm_180s', {}).get('mean', {})
    report('27 cells: product physics x BEATs (seed pairs)', 0.848, fb.get('product')); report('27 cells: gate physics x BEATs (seed pairs)', 0.871, fb.get('gate')); report('27 cells: oracle branch choice', 0.866, fb.get('oracle'))
    report('27 cells: product physics x Dasheng (seed pairs)', 0.851, fd.get('product')); report('27 cells: gate physics x Dasheng (seed pairs)', 0.884, fd.get('gate'))
else:
    for n in ['product physics x BEATs', 'gate physics x BEATs', 'product physics x Dasheng', 'gate physics x Dasheng']: report('27 cells: ' + n, 0, None)

# ---- print -----------------------------------------------------------------------------------------------------------
w = max(len(r[1]) for r in rows)
print(f'{"status":8s} {"check":{w}s} {"expected":>9s} {"recomputed":>11s}')
for st, name, e, g in rows: print(f'{st:8s} {name:{w}s} {e:9.3f} {"-" if g is None else f"{g:11.3f}"}')
n_fail = sum(r[0] == 'FAIL' for r in rows); n_miss = sum(r[0] == 'MISSING' for r in rows)
print(f'\n{len(rows)} checks so far: {len(rows) - n_fail - n_miss} pass, {n_fail} fail, {n_miss} missing')
_n_first = len(rows)

# ---- seed ensembles, nested temperature calibration, reference refresh (Table 5), extra seeds ----
def jload(name):
    fn = os.path.join(OUT, name); return json.load(open(fn)) if os.path.exists(fn) else None
ET = jload('ensemble_temperature_mic_x_speed_x_load.json'); ET = ET['results'] if ET else {}
for name, key, e in [('27 cells: BEATs 3-seed ensemble', 'BEATs, 3-seed ensemble', 0.820), ('27 cells: Dasheng-LoRA 3-seed ensemble', 'Dasheng-LoRA, 3-seed ensemble', 0.783), ('27 cells: product of ensembles', 'product E_phys x E_beats', 0.867),
                     ('27 cells: gate of ensembles', 'gate E_phys x E_beats', 0.882), ('27 cells: REF-Path-G, 3-way gate (frozen Dasheng)', 'gate E_phys x E_beats x Dasheng-frozen', 0.894)]:
    report(name, e, ET.get(key, {}).get('mean'))
for name, key, e in [('27 cells: fusion, 2 branches (nested temperature)', 'T-product E_phys x E_beats', 0.870), ('27 cells: REF-Path fusion, 3 branches (nested temperature)', 'T-product E_phys x E_beats x Dasheng-frozen', 0.876)]:
    report(name, e, ET.get(key, {}).get('nested_mean'))
CR = jload('conformal_refresh_summary.json'); V = CR['0.01']['variants'] if CR else {}
for name, var, key, e in [('Table 5: OLD FAR@0.05 (3 seeds)', 'OLD', 'far@0.05', 0.211), ('Table 5: IDEAL FAR@0.05', 'IDEAL', 'far@0.05', 0.152), ('Table 5: SELF-H FAR@0.05', 'SELF-H', 'far@0.05', 0.188), ('Table 5: SELF-ALL FAR@0.05', 'SELF-ALL', 'far@0.05', 0.159),
                          ('Table 5: SELF-ALL contamination', 'SELF-ALL', 'contamination', 0.039), ('Table 5: NAIVE-ALL acc', 'NAIVE-ALL', 'acc', 0.202)]:
    report(name, e, V.get(var, {}).get(key), tol=0.002)
seeds('27 cells: Dasheng-LoRA, raw (3 seeds)', 'full_mic_x_speed_x_load_dasheng_lora_norefnorm_180s_s?.json', 0.635, 0.022)
XS = jload('extra_seeds_summary.json') or {}
report('SPI: lambda = 0 (3 seeds)', 0.860, XS.get('spi_lam0.0', {}).get('mean')); report('SPI: lambda = 0.5 (3 seeds)', 0.816, XS.get('spi_lam0.5', {}).get('mean'))
for c, e in [('inner_pitting', 0.799), ('outer_pitting', 0.898), ('inner_outer_pitting', 0.927), ('cage_fracture', 0.437)]:
    report(f'open-set, 27 cells: rank-average AUROC {c} (3 seeds)', e, XS.get('openset27', {}).get(c, {}).get('auroc_rank', {}).get('mean'))
w = max(len(r[1]) for r in rows); print('\n-- seed ensembles, reference refresh, extra seeds --')
for st, name, e, g in rows[_n_first:]: print(f'{st:8s} {name:{w}s} {e:9.3f} {"-" if g is None else f"{g:11.3f}"}')
n_fail = sum(r[0] == 'FAIL' for r in rows); n_miss = sum(r[0] == 'MISSING' for r in rows); print(f'{len(rows)} checks so far: {len(rows) - n_fail - n_miss} pass, {n_fail} fail, {n_miss} missing')
_n_printed = len(rows)

# ---- open-set per-window scores on the 27 cells (Fig. 9, Section 4.3) ----
OS = jload('openset27_scores.json') or {}
def os_mean(c, k, q='auroc'): return float(np.mean([OS[c][t][k][q] for t in OS[c]])) if c in OS else None
for c, k, e in [('inner_outer_pitting', 'soft', 0.705), ('inner_outer_pitting', 'knn', 0.999), ('inner_outer_pitting', 'rank', 0.927), ('inner_outer_pitting', 'zsum_ref', 0.903), ('cage_fracture', 'soft', 0.632), ('cage_fracture', 'rank', 0.437), ('cage_fracture', 'zsum_ref', 0.572), ('outer_pitting', 'zsum_ref', 0.916), ('inner_pitting', 'zsum_ref', 0.801)]:
    report(f'open-set scores, 27 cells: {c} {k} AUROC', e, os_mean(c, k))
report('open-set scores, 27 cells: compound rank-average TPR@5%FPR', 0.691, os_mean('inner_outer_pitting', 'rank', 'tpr5'), tol=0.005); report('open-set scores, 27 cells: compound z-sum TPR@5%FPR', 0.676, os_mean('inner_outer_pitting', 'zsum_ref', 'tpr5'), tol=0.005)
# ---- Table 2 of the paper (same-protocol comparison, results/expected_tables.md) against a recomputation from outputs/ ----
import io as _io
_p4 = os.path.join(ROOT, 'results', 'expected_tables.md'); _rows = os.path.join(OUT, 'sota_table_rows.json')
if os.path.exists(_p4) and os.path.exists(_rows) and '## Table 2.' in _io.open(_p4, encoding='utf-8').read():
    sys.path.insert(0, os.path.join(ROOT, 'scripts')); import make_fig_sota as MS
    TT = MS.collect(); lab2key = json.load(open(_rows)); block = _io.open(_p4, encoding='utf-8').read().split('## Table 2.')[1].split('## Table 3.')[0]
    num = lambda c: None if c.replace('*', '').strip() in ('n/a', '') else float(c.replace('*', '').split('±')[0])
    special = {'Linear classifier': ('linear_raw', 'linear_ref'), 'Physics branch': (None, 'physics'), 'REF-Path (ours)': (None, 'refpath'), 'REF-Path-G': (None, 'refpath_gate'), 'Oracle': (None, 'oracle')}
    for ln in block.split(chr(10)):
        if not ln.startswith('|') or ln.startswith('|---') or ln.startswith('| Method'): continue
        c = [x.strip() for x in ln.strip().strip('|').split('|')]; label = re.sub(r'\s*\[[^\]]*\]', '', c[0]).replace('*', '').strip()
        if label in special: kr, kf = special[label]
        elif label in lab2key: kr, kf = lab2key[label] + '_raw', lab2key[label] + '_ref'
        else: continue
        for P, (cr, cf) in ((MS.P9, (c[2], c[3])), (MS.P27, (c[4], c[5]))):
            for k, cell in ((kr, cr), (kf, cf)):
                v = num(cell)
                if v is not None and k is not None: report(f'Table 2: {label[:34]} {k} {"9" if P == MS.P9 else "27"}', v, (TT[P].get(k) or {}).get('mean'))
# ---- Section 4.2: transductive adaptation against the healthy share of the test stream (Fig. 3(c)), straight from the per-seed files ----
import glob as _glob
def _imb(P, h):
    v = []
    for fn in _glob.glob(os.path.join(OUT, f'sota_feat_{P}_s?.json')):
        r = json.load(open(fn))['results']
        for m in ('adabntest', 'tent', 'sar'):
            k = f'{m}_raw' + ('' if h == 0.2 else f'_h{h}')
            if k in r: v.append(np.mean([x['acc'] for x in r[k].values()]))
    return float(np.mean(v)) if len(v) == 9 else None
if not (os.path.exists(_p4) and os.path.exists(_rows)): print('Table 2 checks skipped: results/expected_tables.md or outputs/sota_table_rows.json not found (run scripts/make_tables.py)')
for P, exp in (('mic_x_speed', {0.2: 0.925, 0.5: 0.909, 0.8: 0.797, 0.95: 0.705, 1.0: 0.624}), ('mic_x_speed_x_load', {0.2: 0.937, 0.5: 0.933, 0.8: 0.836, 0.95: 0.710, 1.0: 0.566})):
    for h, e in exp.items(): report(f'Sec. 4.2: AdaBN/TENT/SAR raw, healthy share {h}, {"9" if P == "mic_x_speed" else "27"} cells', e, _imb(P, h))
# ---- Section 4.2: paired comparison of the fusion with the strongest Reference-Only competitor over cells ----
if 'TT' in globals():
    def _pair(P, key):
        a, b = TT[P]['refpath'], TT[P][key]; d = dict(zip(b['tasks'], b['per_task'])); x = np.array(a['per_task']); y = np.array([d[t] for t in a['tasks']])
        return float((x > y).sum()), float(x.mean() - y.mean()), float(y.min()), float(x.min()), float((y < 0.7).sum()), float((x < 0.7).sum())
    for P, key, e in ((MS.P9, 'sagm_ref', (4, 0.031, 0.63, 0.78, 2, 0)), (MS.P27, 'aft_ref', (24, 0.099, 0.44, 0.64, 8, 2))):
        for name, ev, rv in zip(('cells won', 'mean gain', 'worst cell of the competitor', 'worst cell of REF-Path', 'competitor cells < 0.7', 'REF-Path cells < 0.7'), e, _pair(P, key)):
            report(f'Sec. 4.2: fusion vs {key}: {name}', ev, rv, tol=0.006)
w = max(len(r[1]) for r in rows); print('\n-- open-set scores, Table 2, Section 4.2 --')
for st, name, e, g in rows[_n_printed:]: print(f'{st:8s} {name:{w}s} {e:9.3f} {"-" if g is None else f"{g:11.3f}"}')
n_fail =sum(r[0] == 'FAIL' for r in rows); n_miss = sum(r[0] == 'MISSING' for r in rows); print(f'{len(rows)} checks: {len(rows) - n_fail - n_miss} pass, {n_fail} fail, {n_miss} missing'); sys.exit(1 if n_fail else 0)
