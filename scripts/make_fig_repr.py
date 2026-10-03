"""Representation figure (Guilin, B&K #2 microphone, 1000 rpm, no load): top row, log-PSD of every state against the healthy
spectrum; bottom row, the raw order-normalized cyclostationary map of the healthy state and the healthy-reference contrast
V(f, o) - V_healthy(f, o) of the four fault states."""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); FIG = os.path.join(ROOT, 'figures'); sys.path.insert(0, ROOT)
from refpath.features.frontend import log_psd, cyclo_volume, DEFAULT_ORDERS
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fs = 32768; fr = 1000 / 60
states = [('nm', 'healthy'), ('Ids', 'inner pitting'), ('Ods', 'outer pitting'), ('IOds', 'compound'), ('lc', 'cage fracture')]
fig, axes = plt.subplots(2, 5, figsize=(11, 4.6), gridspec_kw=dict(height_ratios=[1, 1.55], hspace=0.55, wspace=0.42, left=0.055, right=0.9, top=0.93, bottom=0.11))
ref = None; ext = [DEFAULT_ORDERS[0], DEFAULT_ORDERS[-1], 0, fs / 2000]
for k, (st, name) in enumerate(states):
    X = np.load(os.path.join(ROOT, 'cache', 'guilin', f'{st}_1000_0_1.npy'), mmap_mode='r')[1]
    Ls, Vs = [], []
    for s in range(0, 20 * fs, fs):
        w = np.asarray(X[s:s + fs], np.float64); Ls.append(log_psd(w, fs)); Vs.append(cyclo_volume(w, fs, fr))
    L = np.mean(Ls, 0); V = np.mean(Vs, 0).mean(2)
    if ref is None: ref = (L.copy(), V.copy())
    ax = axes[0, k]; fb = np.linspace(0, fs / 2, 128); ax.plot(fb / 1000, L, color='#36c', lw=1); ax.plot(fb / 1000, ref[0], color='#999', lw=0.7, ls='--'); ax.set_title(name); ax.set_xlabel('carrier frequency (kHz)', labelpad=1)
    if k == 0: ax.set_ylabel('log-PSD')
    ax = axes[1, k]
    if k == 0:
        im0 = ax.imshow(V, aspect='auto', origin='lower', cmap='viridis', extent=ext); ax.set_title('raw V(f, o), healthy', fontsize=8); ax.set_ylabel('carrier band (kHz)')
    else:
        im = ax.imshow(V - ref[1], aspect='auto', origin='lower', cmap='RdBu_r', vmin=-1.5, vmax=1.5, extent=ext); ax.set_title('contrast to healthy', fontsize=8)
    ax.set_xlabel('cyclic order o')
    for o in (1.0, 3.0, 5.0): ax.axvline(o, color='k', lw=0.4, ls=':')
cax = fig.add_axes([0.915, 0.13, 0.012, 0.42]); fig.colorbar(im, cax=cax, label='V(f, o) − V_healthy(f, o)  (log)')
fig.savefig(os.path.join(FIG, 'fig_representation.png'), dpi=200); fig.savefig(os.path.join(FIG, 'fig_representation.pdf')); plt.close(fig); print('saved fig_representation')
