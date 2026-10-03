"""Signal front-ends for REF-Path.

1D  -> waveform windows
2D  -> log-PSD (carrier / resonance content; the f axis carries fault-type information and is kept)
3D  -> order-normalized cyclostationary volume V(f, o, tau):
         f   = carrier frequency band (STFT bin groups)
         o   = cyclic order = cyclic frequency / shaft frequency (speed normalization lives on this axis)
         tau = sub-window index inside the analysis window (time evolution)
       Path gain acts multiplicatively on the f axis, speed acts on the o axis, fault content is the (f, o) pattern.

All functions are numpy (used for baselines and caching); torch versions in torch_frontend.py mirror them.
"""
from __future__ import annotations
import numpy as np
import scipy.signal as sg

DEFAULT_ORDERS = np.linspace(0.5, 25, 100)


def log_psd(x, fs, n_bands=128, nperseg=2048):
    _, P = sg.welch(x, fs, nperseg=nperseg)
    P = P[1:1 + (len(P) - 1) // n_bands * n_bands]
    return np.log10(P.reshape(n_bands, -1).mean(1) + 1e-12).astype(np.float32)


def cyclo_volume(x, fs, fr, orders=DEFAULT_ORDERS, n_fbands=32, nperseg=256, hop=16, n_tau=4):
    """Order-normalized cyclostationary volume from one window.

    Returns V with shape (n_fbands, len(orders), n_tau) = log |cyclic spectrum of the STFT band envelopes|,
    evaluated at cyclic frequencies alpha = orders * fr for each of n_tau sub-windows.
    fr = shaft frequency in Hz (rpm / 60).
    """
    f, t, S = sg.stft(x, fs, nperseg=nperseg, noverlap=nperseg - hop)
    E = np.abs(S)                                   # band envelopes, envelope sampling rate fe = fs / hop
    E = E[: (E.shape[0] // n_fbands) * n_fbands].reshape(n_fbands, -1, E.shape[1]).mean(1)   # (n_fbands, T)
    fe = fs / hop
    T = E.shape[1]; L = T // n_tau
    V = np.empty((n_fbands, len(orders), n_tau), np.float32)
    for k in range(n_tau):
        seg = E[:, k * L:(k + 1) * L]
        A = np.abs(np.fft.rfft(seg - seg.mean(1, keepdims=True), axis=1))
        alpha = np.fft.rfftfreq(seg.shape[1], 1 / fe)
        for i in range(n_fbands):
            V[i, :, k] = np.interp(orders * fr, alpha, A[i])
    return np.log(V + 1e-9)


def window_iter(x, fs, win_s=1.0, hop_s=0.5):
    W, H = int(win_s * fs), int(hop_s * fs)
    for s in range(0, len(x) - W + 1, H):
        yield s, x[s:s + W]


def extract_window_features(x, fs, fr, kinds=('psd', 'cyclo')):
    out = {}
    if 'psd' in kinds:
        out['psd'] = log_psd(x, fs)
    if 'cyclo' in kinds:
        out['cyclo'] = cyclo_volume(x, fs, fr)
    return out
