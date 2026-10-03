"""Same-protocol comparison of REF-Path with the re-implemented methods (raw and + healthy-reference versions) on the
nine-cell and the 27-cell protocols. Collects every result into outputs/sota_table.json and draws figures/fig_sota:
(a), (b) paired bars raw / +reference per method on the two protocols; (c) nine-cell accuracy grids of the brackets, the
strongest prior deployable method, the two branches and the REF-Path fusion (seed-averaged).
python scripts/make_fig_sota.py"""
import os, sys, json, glob, csv, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'outputs'); FIG = os.path.join(ROOT, 'figures'); sys.path.insert(0, ROOT)
from refpath import plot_style as PS
P9, P27 = 'mic_x_speed', 'mic_x_speed_x_load'; HS = [0.2, 0.5, 0.8, 0.95, 1.0]; OURS_BRANCH = ('beatslora_ref', 'dasheng_ref')
# (key, label, transductive?)
METHODS = [
    ('erm', 'ERM, BN-MLP', False), ('adabnref', 'AdaBN on the reference', False), ('adabntest', 'AdaBN on the test batch', True), ('tent', 'TENT', True), ('sar', 'SAR', True),
    ('dann', 'DANN', False), ('coral', 'Deep CORAL', False), ('sagm', 'SAGM', False), ('ddg', 'Dual disentanglement', False), ('fgdg', 'Frequency-guided DG', False),
    ('cscoh', 'CSCoh-CNN', False), ('xmodal', 'Cross-modal contrastive', False), ('aft', 'Acoustic-frame Transformer', False), ('genrep', 'Frozen BEATs + k-NN', False),
    ('beatslora', 'BEATs + LoRA', False), ('dasheng', 'Dasheng + LoRA', False)]


def per_task(files, key=None):
    """list of result dicts {task: acc} (one per seed)"""
    out = []
    for fn in sorted(files):
        r = json.load(open(fn))['results']
        if key is not None:
            if key not in r: continue
            r = r[key]
        out.append({t: (v['acc'] if isinstance(v, dict) else v) for t, v in r.items()})
    return out


def stat(runs):
    if not runs: return None
    tasks = sorted(runs[0]); M = np.array([[r[t] for t in tasks] for r in runs])
    return dict(mean=float(M.mean()), std=float(M.mean(1).std()), n=len(runs), min=float(M.mean(0).min()), tasks=tasks, per_task=M.mean(0).tolist())


def collect():
    T = {P9: {}, P27: {}}
    for P in (P9, P27):
        for key, _, _ in METHODS:
            for norm in ('raw', 'ref'):
                if key in ('xmodal', 'aft', 'genrep'): runs = per_task(glob.glob(os.path.join(OUT, f'sota_wave_{key}_{P}_s?.json')), f'{key}_{norm}')
                elif key == 'beatslora':
                    if P == P9: runs = per_task(glob.glob(os.path.join(OUT, 'guilin_mic_x_speed_pathlora_plain_ml_aug.json'))) if norm == 'raw' else per_task(glob.glob(os.path.join(OUT, 'guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm_refsplit_refgain_s?.json')))
                    else: runs = [] if norm == 'raw' else per_task(glob.glob(os.path.join(OUT, f'full_{P}_pathlora_refnorm_refgain_180s_s?.json')))
                elif key == 'dasheng':
                    runs = [] if P == P9 else per_task(glob.glob(os.path.join(OUT, f'full_{P}_dasheng_lora_{"norefnorm" if norm == "raw" else "refnorm"}_180s_s?.json')))
                else: runs = per_task(glob.glob(os.path.join(OUT, f'sota_feat_{P}_s?.json')), f'{key}_{norm}')
                T[P][f'{key}_{norm}'] = stat(runs)
        # transductive adaptation against the healthy share h of the test stream (0.2 = the class-balanced test set)
        T[P]['imb'] = {}
        for norm in ('raw', 'ref'):
            for key in ('adabntest', 'tent', 'sar'):
                T[P]['imb'][f'{key}_{norm}'] = {'0.2': T[P][f'{key}_{norm}']['mean'], **{str(h): stat(per_task(glob.glob(os.path.join(OUT, f'sota_feat_{P}_s?.json')), f'{key}_{norm}_h{h}'))['mean'] for h in HS[1:]}}
    # brackets and ours
    def csv9(norm, feat):
        d = {r['task']: float(r['acc']) for r in csv.DictReader(open(os.path.join(OUT, 'guilin_baselines.csv'))) if r['protocol'] == P9 and r['features'] == feat and r['normalization'] == norm}
        return stat([d])
    T[P9]['linear_raw'] = csv9('raw', 'logPSD'); T[P9]['linear_ref'] = csv9('healthy', 'PSD+cyclo'); T[P9]['oracle'] = csv9('oracle', 'logPSD')
    for norm, k in (('raw', 'linear_raw'), ('healthy', 'linear_ref'), ('oracle', 'oracle')): T[P27][k] = stat(per_task([os.path.join(OUT, f'full_{P27}_linear_{norm}_s0.json')]))
    T[P9]['physics'] = stat(per_task(glob.glob(os.path.join(OUT, 'guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_s?.json'))))
    T[P27]['physics'] = stat(per_task(glob.glob(os.path.join(OUT, f'full_{P27}_mlp_s?.json'))))
    # REF-Path fusion, nine cells: geometric fusion of the physics and BEATs branches, three seed pairs (as in verify_results.py)
    def hist_ent(p):
        f = np.bincount(p.argmax(1), minlength=p.shape[1]) / len(p); f = f[f > 0]; return -(f * np.log(f)).sum() / np.log(p.shape[1])
    geo, gate = [], []
    for s_ in range(3):
        A = np.load(os.path.join(OUT, f'guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_s{s_}_probs.npz')); B = np.load(os.path.join(OUT, f'guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm_refsplit_refgain_s{s_}_probs.npz')); g1, g2 = {}, {}
        for t in sorted({k.split('/')[0] for k in A.files}):
            pa, tea = A[f'{t}/prob'], A[f'{t}/te']; pb, yb, teb = B[f'{t}/prob'], B[f'{t}/y'], B[f'{t}/te']
            fiw = (teb // 60) * 119 + 2 * (teb % 60); pos = {ix: i for i, ix in enumerate(tea)}; pa = pa[np.array([pos[ix] for ix in fiw])]
            la, lb = np.log(pa + 1e-9), np.log(pb + 1e-9); ha, hb = hist_ent(pa), hist_ent(pb); wa = np.exp(10 * ha) / (np.exp(10 * ha) + np.exp(10 * hb))
            g1[t] = float(((la + lb).argmax(1) == yb).mean()); g2[t] = float(((wa * la + (1 - wa) * lb).argmax(1) == yb).mean())
        geo.append(g1); gate.append(g2)
    T[P9]['refpath'] = stat(geo); T[P9]['refpath_gate'] = stat(gate)
    E = json.load(open(os.path.join(OUT, f'ensemble_temperature_{P27}.json'))); cells = E['cells']
    def from_et(name, k='nested_per_cell'):
        v = E['results'][name][k]; return dict(mean=float(np.mean(v)), std=0.0, n=1, min=float(np.min(v)), tasks=cells, per_task=list(map(float, v)))
    T[P27]['refpath'] = from_et('T-product E_phys x E_beats x Dasheng-frozen'); T[P27]['refpath2'] = from_et('T-product E_phys x E_beats')
    T[P27]['refpath_gate'] = from_et('gate E_phys x E_beats x Dasheng-frozen', 'per_cell')
    T[P27]['physics_ens'] = from_et('physics, 3-seed ensemble', 'per_cell')
    return T


def best_prior(T, P, deploy_only=True):
    cands = []
    for key, _, trans in METHODS:
        if deploy_only and trans: continue
        for norm in ('raw', 'ref'):
            s = T[P].get(f'{key}_{norm}')
            if s and f'{key}_{norm}' not in OURS_BRANCH: cands.append((s['mean'], f'{key}_{norm}'))
    return max(cands)[1] if cands else None


def main():
    T = collect(); json.dump(T, open(os.path.join(OUT, 'sota_table.json'), 'w'), indent=1)
    for P in (P9, P27):
        print(f'== {P}')
        for k, v in T[P].items():
            if v and 'mean' in v: print(f'  {k:16s} {v["mean"]:.3f} ± {v["std"]:.3f} (n={v["n"]}, min {v["min"]:.3f})')
        print('  best deployable prior:', best_prior(T, P), '| best incl. transductive:', best_prior(T, P, False))
    draw(T)


def draw(T):
    PS.apply(8); fig = plt.figure(figsize=(7.6, 6.5)); gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1], hspace=0.3, wspace=0.06, left=0.2, right=0.985, top=0.965, bottom=0.075)
    # one row order for both panels: best version of every method, averaged over the protocols it was run on
    def score(k): return np.mean([max(s['mean'] for s in (T[P].get(f'{k}_raw'), T[P].get(f'{k}_ref')) if s) for P in (P9, P27) if T[P].get(f'{k}_raw') or T[P].get(f'{k}_ref')])
    order = sorted([(score(k), k, lab + (' *' if tr else '')) for k, lab, tr in METHODS] + [(max(T[P9]['linear_ref']['mean'], T[P27]['linear_ref']['mean']) / 2 + min(T[P9]['linear_ref']['mean'], T[P27]['linear_ref']['mean']) / 2, 'linear', 'Linear classifier')])
    names = [o[2] for o in order] + ['REF-Path (ours)', 'Oracle (not deployable)']; y = np.arange(len(names))
    for col, (P, title) in enumerate([(P9, '(a) microphone × speed held out, 9 cells'), (P27, '(b) microphone × speed × load held out, 27 cells')]):
        ax = fig.add_subplot(gs[0, col])
        for i, (_, k, _) in enumerate(order):
            a, b = T[P].get(f'{k}_raw'), T[P].get(f'{k}_ref')
            if a: ax.barh(i + 0.19, a['mean'], 0.36, xerr=a['std'], color=PS.GREYS[3], error_kw=dict(lw=0.5), label='raw input' if i == 0 else None); ax.text(a['mean'] + a['std'] + 0.008, i + 0.19, f"{a['mean']:.2f}", va='center', fontsize=5.5, color='#666')
            if b: ax.barh(i - 0.19, b['mean'], 0.36, xerr=b['std'], color=PS.GREYS[1], error_kw=dict(lw=0.5), label='+ healthy reference' if i == 0 else None); ax.text(b['mean'] + b['std'] + 0.008, i - 0.19, f"{b['mean']:.2f}", va='center', fontsize=5.5)
            if not a and not b: ax.text(0.31, i, 'not run on the 60 s clips', va='center', fontsize=5.5, color='#666', style='italic')
        r = T[P]['refpath']; ax.barh(len(order), r['mean'], 0.6, color=PS.BLUE); ax.text(r['mean'] + 0.008, len(order), f"{r['mean']:.3f}", va='center', fontsize=6, color=PS.BLUE, fontweight='bold')
        o = T[P]['oracle']; ax.barh(len(order) + 1, o['mean'], 0.6, color=PS.BLUE_LIGHT); ax.text(o['mean'] + 0.008, len(order) + 1, f"{o['mean']:.3f}", va='center', fontsize=6)
        ax.set_yticks(y); ax.set_yticklabels(names if col == 0 else [''] * len(names), fontsize=6.5); ax.set_ylim(-0.7, len(names) - 0.3); ax.set_xlim(0.3, 1.04); ax.set_xlabel('mean window accuracy', labelpad=1); ax.set_title(title, loc='left', fontsize=8); PS.grid(ax, 'x')
        ax.axvline(r['mean'], color=PS.BLUE, lw=0.6, ls=':', zorder=0)
        if col == 0: ax.legend(fontsize=6.5, loc='lower right', frameon=True, framealpha=0.9, edgecolor='none')
    gs2 = gs[1, :].subgridspec(1, 7, width_ratios=[2.3, 0.75, 1, 1, 1, 1, 1], wspace=0.08)
    # (c) transductive adaptation against the healthy share of the unlabeled test stream
    ax = fig.add_subplot(gs2[0, 0]); x = np.arange(len(HS))
    for P, ls, mk, tag in ((P9, '-', 'o', '9 cells'), (P27, '--', 's', '27 cells')):
        V = np.array([[T[P]['imb'][f'{k}_raw'][str(h)] for h in HS] for k in ('adabntest', 'tent', 'sar')])
        ax.plot(x, V.mean(0), ls=ls, marker=mk, ms=3, lw=1.0, color=PS.GREYS[1], label=f'AdaBN / TENT / SAR *, {tag}')
        ax.plot(x, [T[P]['refpath']['mean']] * len(HS), ls=ls, lw=1.2, color=PS.BLUE, label=f'REF-Path, {tag}')
    ax.set_xticks(x); ax.set_xticklabels([f'{h:g}' for h in HS], fontsize=6); ax.set_ylim(0.5, 1.0); ax.set_xlabel('healthy share of the test stream', fontsize=6.5, labelpad=1); ax.set_ylabel('accuracy', fontsize=6.5)
    ax.tick_params(labelsize=6); ax.legend(fontsize=5, loc='lower left', frameon=False, handlelength=2.6); ax.set_title('(c) test-stream composition', loc='left', fontsize=7); PS.grid(ax, 'y')
    # (d) nine-cell grids
    MICS = ['CRY333', 'BK4966_2', 'BK4966_3']; MICN = ['CRY333', 'B&K #2', 'B&K #3']; RPMS = [800, 1000, 1200]
    def grid(s):
        G = np.full((3, 3), np.nan)
        for t, v in zip(s['tasks'], s['per_task']):
            m, rpm = t[4:].split('@'); G[MICS.index(m), RPMS.index(int(rpm))] = v
        return G
    bp = best_prior(T, P9); lab = {k + '_' + n: l + (', raw' if n == 'raw' else ' + ref.') for k, l, _ in METHODS for n in ('raw', 'ref')}
    panels = [(lab.get(bp, bp), T[P9][bp]), ('Physics branch', T[P9]['physics']), ('BEATs branch', T[P9]['beatslora_ref']), ('REF-Path', T[P9]['refpath']), ('Oracle', T[P9]['oracle'])]
    for k, (name, s) in enumerate(panels):
        ax = fig.add_subplot(gs2[0, k + 2]); G = grid(s); ax.imshow(G, vmin=0.4, vmax=1.0, cmap=PS.CMAP_ACC)
        for i in range(3):
            for j in range(3): ax.text(j, i, f'{G[i, j]:.2f}', ha='center', va='center', fontsize=5.5, color='w' if G[i, j] > 0.8 else 'k')
        ax.set_xticks(range(3)); ax.set_xticklabels(RPMS, fontsize=5.5); ax.set_yticks(range(3)); ax.set_yticklabels(MICN if k == 0 else [''] * 3, fontsize=6)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True); ax.set_yticks(np.arange(-0.5, 3, 1), minor=True); ax.grid(which='minor', color='w', lw=1.0); ax.tick_params(which='minor', length=0); ax.tick_params(which='major', length=1.5)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(('(d) ' if k == 0 else '') + f'{name}\n{s["mean"]:.3f}', fontsize=6.5)
        if k == 2: ax.set_xlabel('held-out speed (rpm); rows: held-out microphone', fontsize=6.5, labelpad=2)
    fig.text(0.2, 0.012, '* transductive: adapts on the unlabeled test stream', ha='left', fontsize=6.5)
    out = os.path.join(FIG, 'fig_sota'); fig.savefig(out + '.png', dpi=220); fig.savefig(out + '.pdf'); print('saved', out)


if __name__ == '__main__':
    main()
