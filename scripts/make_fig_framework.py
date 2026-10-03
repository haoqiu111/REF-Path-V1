"""Figure 1: framework of REF-Path as an editable PowerPoint drawing (figures/fig_framework.pptx), rendered by PowerPoint
itself to figures/fig_framework.pdf (vector) and .png. Illustrations are computed from the Guilin recordings (spectrogram
stacks of source windows and of the target healthy reference, raw and reference-standardized cyclostationary map and
mel input). Elements: inputs, the two reference-standardized branches (modules), the training losses with their
gradient paths, the fusion, and the reference-calibrated safeguards with the label-free refresh loop (control path).
python scripts/make_fig_framework.py"""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import signal as sg
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from refpath.features.frontend import cyclo_volume
import pptx_draw as D
FIG = os.path.join(ROOT, 'figures'); TH = os.path.join(FIG, '_fw'); os.makedirs(TH, exist_ok=True); FS = 32768


def wave(state, rpm, mic, start=5):
    X = np.load(os.path.join(ROOT, 'cache', 'guilin', f'{state}_{rpm}_0_1.npy'), mmap_mode='r')[mic]
    return np.asarray(X[start * FS:(start + 1) * FS], np.float64)


def spec(w):
    f, t, S = sg.spectrogram(w, FS, nperseg=1024, noverlap=768); S = np.log(S[f <= 8000] + 1e-12); return S[::2]


def save(img, name, cmap, vmin=None, vmax=None, size=(1.7, 1.25)):
    fig = plt.figure(figsize=size); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax); ax.axis('off')
    p = os.path.join(TH, name + '.png'); fig.savefig(p, dpi=200); plt.close(fig); return p


def thumbs():
    T = {}
    src = [('Ids', 800, 1), ('Ods', 1000, 2), ('lc', 1000, 1)]      # labelled source windows: other microphones and speeds
    T['src'] = [save(spec(wave(*a)), f'src{i}', 'magma') for i, a in enumerate(src)]
    T['ref'] = [save(spec(wave('nm', 1200, 0, s)), f'ref{i}', 'gray') for i, s in enumerate((3, 9, 15))]   # target path: healthy only
    fr = 1200 / 60; V = np.mean([cyclo_volume(wave('Ods', 1200, 0, s), FS, fr) for s in range(20, 26)], 0).mean(2)
    Vh = np.mean([cyclo_volume(wave('nm', 1200, 0, s), FS, fr) for s in range(0, 12)], 0).mean(2)
    Vs = np.std([cyclo_volume(wave('nm', 1200, 0, s), FS, fr).mean(2) for s in range(0, 12)], 0) + 0.05
    T['map'] = save(V, 'map_raw', 'viridis', size=(1.3, 1.0)); T['map_z'] = save((V - Vh) / Vs, 'map_std', 'RdBu_r', -6, 6, size=(1.3, 1.0))
    M = spec(wave('Ods', 1200, 0, 22)); Mh = np.stack([spec(wave('nm', 1200, 0, s)) for s in range(0, 12)]); mu, sd = Mh.mean((0, 2), keepdims=True)[0], Mh.std((0, 2), keepdims=True)[0] + 1e-3
    T['mel'] = save(M, 'mel_raw', 'magma', size=(1.3, 1.0)); T['mel_z'] = save((M - mu) / sd, 'mel_std', 'RdBu_r', -3, 3, size=(1.3, 1.0))
    return T


def draw(T, path):
    W, H = 13.33, 6.2; c = D.Canvas(W, H); G, B, O = D.GREEN, D.BLUE, D.ORANGE
    yU, yM, yL = 1.2, 2.9, 4.6                        # physics lane, reference lane, foundation-model lane
    # ---------------- regions
    c.frame(0.12, 0.15, 2.7, 5.5); c.frame(2.95, 0.15, 6.55, 5.5); c.frame(9.63, 0.15, 3.58, 1.95); c.frame(9.63, 2.2, 3.58, 1.4); c.frame(9.63, 3.7, 3.58, 1.95)
    # ---------------- inputs (illustrations)
    c.text(0.2, 0.2, 2.5, 0.3, 'Source windows (labelled)', 12)
    c.stack(T['src'], 0.42, 0.78, 1.75, 1.3)
    c.text(0.2, 2.25, 2.5, 0.3, 'other paths, other speeds', 9.5, color=D.GREY)
    c.stack(T['ref'], 0.42, 3.75, 1.75, 1.3)
    c.text(0.2, 5.2, 2.5, 0.3, 'Target healthy reference (60 s)', 12)
    # ---------------- forward trunk from the source windows to the two feature extractors
    c.path([(2.38, 1.2), (3.12, yU)], G); c.path([(2.7, yU), (2.7, yL), (3.12, yL)], G)
    # ---------------- features (illustrations) and reference standardization
    c.picture(T['map'], 3.15, yU - 0.45, 1.15, 0.9, border='BFBFBF'); c.text(2.98, yU + 0.47, 1.5, 0.42, ['Cyclostationary map', 'V(f, o), log-PSD'], 9.5)
    c.picture(T['mel'], 3.15, yL - 0.45, 1.15, 0.9, border='BFBFBF'); c.text(2.98, yL - 0.9, 1.5, 0.42, ['Mel filterbank', 'F(t, m)'], 9.5)
    c.arrow(4.32, yU, 4.72, yU, G); c.arrow(4.32, yL, 4.72, yL, G)
    c.node(4.95, yU, 0.44, '÷'); c.node(4.95, yL, 0.44, '÷')
    c.text(4.35, yU - 0.62, 1.2, 0.3, 'Eq. (1)', 10, color=D.BLUE_D); c.text(4.35, yL + 0.3, 1.2, 0.3, 'Eq. (4)', 10, color=D.BLUE_D)
    c.box(4.2, yM - 0.33, 1.5, 0.66, ['Healthy reference', 'μ_{d} , σ_{d}'], fill='E2F0D9', line=D.GREEN_D, size=10.5)
    c.arrow(4.95, yM - 0.33, 4.95, yU + 0.24, D.GREEN_D, 2.0, dash='dash'); c.arrow(4.95, yM + 0.33, 4.95, yL - 0.24, D.GREEN_D, 2.0, dash='dash')
    c.path([(1.3, 3.6), (1.3, yM), (4.2, yM)], D.GREEN_D, 2.0, dash='dash')      # reference path
    c.arrow(5.18, yU, 5.5, yU, G); c.arrow(5.18, yL, 5.5, yL, G)
    c.picture(T['map_z'], 5.52, yU - 0.45, 1.15, 0.9, border='BFBFBF'); c.picture(T['mel_z'], 5.52, yL - 0.45, 1.15, 0.9, border='BFBFBF')
    c.text(5.4, yU + 0.47, 1.4, 0.3, 'standardized z', 9.5); c.text(5.4, yL - 0.75, 1.4, 0.3, "standardized F'", 9.5)
    c.arrow(6.69, yU, 7.0, yU, G); c.arrow(6.69, yL, 7.0, yL, G)
    # ---------------- modules
    c.slabs(7.02, yU - 0.62, 0.62, 1.25, n=3, fill=O, line=D.ORANGE_D); c.text(6.45, yU - 1.0, 2.1, 0.32, 'Physics branch (MLP)', 11.5)
    c.slabs(7.02, yL - 0.62, 0.62, 1.25, n=3, fill='BDD7EE', line=D.BLUE_D); c.text(6.55, yL - 1.0, 1.9, 0.32, 'Foundation-model branch', 11.5)
    c.text(6.6, yL + 0.66, 1.25, 0.3, 'frozen encoder', 9.5); c.cube(7.75, yL + 0.3, 0.62, 0.36, 'LoRA', fill=O, line=D.ORANGE_D, size=9.5, depth=0.3)
    # ---------------- fusion (control of the over-confident branch by the temperature) and output
    c.path([(8.02, yU), (8.85, yU), (8.85, yM - 0.24)], G); c.path([(8.02, yL), (8.85, yL), (8.85, yM + 0.24)], G)
    c.node(8.85, yM, 0.46, '×'); c.text(7.55, yM - 0.2, 1.05, 0.4, ['Fusion', 'Eq. (5)'], 10.5)
    c.text(8.9, yU + 0.3, 0.6, 0.3, 'p_{phys}', 10); c.text(8.9, yL - 0.6, 0.6, 0.3, 'p_{FM}', 10); c.text(9.0, yM + 0.26, 0.7, 0.3, 'T_{b}', 10, color=D.BLUE_D)
    c.arrow(9.08, yM, 9.85, yM, G)
    # ---------------- losses and gradient paths
    c.arrow(8.85, yU, 10.0, yU, G); c.arrow(8.85, yL, 10.0, yL, G)
    c.bar(10.02, yU - 0.72, 0.9, 1.44, ['Loss', 'L_{CE}'], 'F8CBAD', 11.5); c.text(11.0, yU - 0.72, 2.15, 1.44, ['cross-entropy on', 'source windows,', 'cross-domain mixup'], 10)
    c.bar(10.02, yL - 0.72, 0.9, 1.44, ['Loss', 'L_{CE} + L_{ML}'], 'BDD7EE', 11.5); c.text(11.0, yL - 0.72, 2.15, 1.44, ['label smoothing,', 'multi-label primitives,', 'random gain'], 10)
    c.path([(10.47, yU + 0.74), (10.47, yU + 0.92), (7.35, yU + 0.92), (7.35, yU + 0.66)], B); c.text(8.55, yU + 0.9, 1.3, 0.26, 'gradient', 9.5, color=D.BLUE_D)
    c.path([(10.47, yL + 0.74), (10.47, yL + 0.92), (8.06, yL + 0.92), (8.06, yL + 0.68)], B); c.text(8.75, yL + 0.64, 1.3, 0.26, 'gradient', 9.5, color=D.BLUE_D)
    # ---------------- outputs and reference-calibrated safeguards
    c.box(9.87, yM - 0.5, 1.0, 1.0, ['Fault', 'class'], fill='FFFFFF', line=D.GREEN_D, size=11)
    c.box(11.0, yM - 0.56, 2.1, 0.34, 'Conformal alarm, Eq. (6)', fill='FFF2CC', line='BF9000', size=9.5)
    c.box(11.0, yM - 0.17, 2.1, 0.34, 'Unknown-fault score', fill='FFF2CC', line='BF9000', size=9.5)
    c.box(11.0, yM + 0.22, 2.1, 0.34, 'Reference refresh', fill='FFF2CC', line='BF9000', size=9.5)
    c.path([(13.1, yM + 0.39), (13.27, yM + 0.39), (13.27, 6.0), (1.3, 6.0), (1.3, 5.56)], 'BF9000', 2.0, dash='dash')      # control loop: label-free refresh of the reference
    c.text(3.0, 5.7, 5.0, 0.26, 'label-free refresh: accepted healthy windows renew the reference', 9.5, color='7F6000')
    # ---------------- legend
    c.arrow(8.3, 5.83, 8.8, 5.83, G, 4.0); c.text(8.86, 5.71, 0.7, 0.25, 'forward', 9, align='l')
    c.arrow(9.6, 5.83, 10.1, 5.83, B, 4.0); c.text(10.16, 5.71, 0.7, 0.25, 'gradient', 9, align='l')
    c.arrow(10.95, 5.83, 11.45, 5.83, D.GREEN_D, 2.0, dash='dash'); c.text(11.51, 5.71, 1.6, 0.25, 'reference / control', 9, align='l')
    c.save(path)


if __name__ == '__main__':
    T = thumbs(); out = os.path.join(FIG, 'fig_framework.pptx'); draw(T, out); print('saved', out, *D.export(out))
