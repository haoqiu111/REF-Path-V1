"""Sensing-path figure.
Row 1: (a), (b) inter-microphone path gain G_m(f) = log-PSD_m(f) - log-PSD_ref(f) (ref = B&K #2) for every machine state at
       1000 rpm (Guilin): a state-independent path gives identical rows (Proposition 2); (c) parabolic-mirror / direct-microphone
       path gain of the AHU corpus at 900 rpm, healthy against nine fault conditions: a focusing path is state-dependent.
Row 2: raw order-normalized cyclostationary maps V(f, o) of outer-race pitting (1000 rpm, load 0) through the three microphones.
Row 3: healthy-reference contrasts z = (V - mu_R) / sigma_R of the same windows: the fault signature that survives the field change.
The statistics (state-variance fractions over three speeds, cross-field correlations) are written to outputs/fields_stats.json."""
import os, sys, json, glob, re, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import scipy.io as sio, scipy.signal as sg
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.features.frontend import log_psd, cyclo_volume
from refpath.data.datasets import AHU_ROOT, GUILIN_CACHE
CACHE = GUILIN_CACHE; fs = 32768; MICS = ['CRY333', 'B&K #2', 'B&K #3']; REF = 1
STATES = [('nm', 'healthy'), ('Ids', 'inner'), ('Ods', 'outer'), ('IOds', 'inner+outer'), ('lc', 'cage')]; RPMS = [800, 1000, 1200]
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
mean_psd = lambda x, n=30: np.mean([log_psd(x[i * fs:(i + 1) * fs].astype(np.float64), fs) for i in range(n)], 0)
maps = lambda x, fr, n: np.stack([cyclo_volume(x[i * fs:(i + 1) * fs].astype(np.float64), fs, fr).mean(2) for i in range(n)])

G = {}; stats = {}
for rpm in RPMS:
    P = {name: np.stack([mean_psd(np.load(os.path.join(CACHE, f'{st}_{rpm}_0_1.npy'), mmap_mode='r')[ch]) for ch in range(3)]) for st, name in STATES}
    for m in (0, 2): G[(rpm, m)] = np.stack([P[name][m] - P[name][REF] for _, name in STATES])
for m in (0, 2):
    A = np.concatenate([G[(rpm, m)] for rpm in RPMS], 1); stats[f'{MICS[m]} vs {MICS[REF]}'] = dict(state_var_fraction=float(A.var(0).mean() / A.var()), gain_range_dB=[float(10 * A.min() / np.log(10)), float(10 * A.max() / np.log(10))])
fr = 1000 / 60; Xo = np.load(os.path.join(CACHE, 'Ods_1000_0_1.npy'), mmap_mode='r'); Xh = np.load(os.path.join(CACHE, 'nm_1000_0_1.npy'), mmap_mode='r')
raw, con = [], []
for ch in range(3):
    Vf = maps(Xo[ch], fr, 20).mean(0); Hs = maps(Xh[ch], fr, 30); raw.append(Vf); con.append((Vf - Hs.mean(0)) / (Hs.std(0) + 1e-6))
raw, con = np.array(raw), np.array(con)
cc = lambda A: [float(x) for x in np.corrcoef(A.reshape(3, -1))[np.triu_indices(3, 1)]]
stats['raw_map_corr_between_mics'] = cc(raw); stats['contrast_map_corr_between_mics'] = cc(con)
stats['raw_map_mean_abs_offset_dB'] = [float(10 * np.abs(np.median(raw[i] - raw[j], 1)).mean() / np.log(10)) for i, j in [(0, 1), (0, 2), (1, 2)]]

fig = plt.figure(figsize=(10.5, 7.6)); gs = fig.add_gridspec(3, 3, height_ratios=[0.85, 1.2, 1.2], hspace=0.5, wspace=0.22, left=0.095, right=0.95, top=0.95, bottom=0.07)
for k, m in enumerate((0, 2)):
    ax = fig.add_subplot(gs[0, k]); A = 10 * G[(1000, m)] / np.log(10)
    im = ax.imshow(A, aspect='auto', cmap='coolwarm', vmin=-8, vmax=8, extent=[0, fs / 2000, 4.5, -0.5], interpolation='nearest')
    ax.set_yticks(range(5)); ax.set_yticklabels([n for _, n in STATES] if k == 0 else [''] * 5); ax.set_title(f'({"ab"[k]}) gain {MICS[m]} / {MICS[REF]}, 1000 rpm (dB)', fontsize=8, loc='left'); ax.set_xlabel('f (kHz)', labelpad=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.ax.tick_params(labelsize=7)
# (c) AHU parabolic mirror / direct microphone path gain at 900 rpm
ax = fig.add_subplot(gs[0, 2]); base = AHU_ROOT; fsa = 20000
for fn in sorted(glob.glob(os.path.join(base, 'PM', '**', '*_r900_*.mat'), recursive=True)):
    lab = re.search(r'_(N|B0\d|IR0\d|OR0\d)\.mat$', fn).group(1); dm = fn.replace(os.sep + 'PM' + os.sep, os.sep + 'DM' + os.sep).replace('/PM/', '/DM/').replace('PM_L05', 'DM_L05')
    if not os.path.exists(dm): continue
    f, Pp = sg.welch(sio.loadmat(fn)['Datas'].ravel().astype(float), fsa, nperseg=2048); _, Pd = sg.welch(sio.loadmat(dm)['Datas'].ravel().astype(float), fsa, nperseg=2048)
    ax.plot(f / 1000, 10 * np.log10(Pp / Pd), color='#c33' if lab == 'N' else '#999', lw=1.2 if lab == 'N' else 0.5, alpha=1 if lab == 'N' else 0.7, zorder=3 if lab == 'N' else 2)
ax.plot([], [], color='#c33', label='healthy'); ax.plot([], [], color='#999', label='9 fault conditions'); ax.legend(fontsize=6.5, loc='upper left', frameon=False, ncol=2)
ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('f (kHz)', labelpad=1); ax.set_title('(c) AHU mirror / direct gain, 900 rpm (dB)', fontsize=8, loc='left')
vmin, vmax = np.percentile(raw, 2), np.percentile(raw, 98)
for ch in range(3):
    ax = fig.add_subplot(gs[1, ch]); ax.imshow(raw[ch], aspect='auto', cmap='viridis', extent=[0.5, 25, fs / 2000, 0], vmin=vmin, vmax=vmax)
    ax.set_title(f'({"def"[ch]}) {MICS[ch]}: raw log V(f, o), outer-race pitting', fontsize=8, loc='left'); ax.set_ylabel('f (kHz)' if ch == 0 else '')
    ax = fig.add_subplot(gs[2, ch]); ax.imshow(con[ch], aspect='auto', cmap='RdBu_r', vmin=-4, vmax=4, extent=[0.5, 25, fs / 2000, 0])
    ax.set_title(f'({"ghi"[ch]}) {MICS[ch]}: healthy-reference contrast z(f, o)', fontsize=8, loc='left'); ax.set_ylabel('f (kHz)' if ch == 0 else ''); ax.set_xlabel('cyclic order o', labelpad=1)
out = os.path.join(ROOT, 'figures', 'fig_fields'); fig.savefig(out + '.png', dpi=200); fig.savefig(out + '.pdf')
json.dump(stats, open(os.path.join(ROOT, 'outputs', 'fields_stats.json'), 'w'), indent=1); print(json.dumps(stats, indent=1)); print('saved', out)
