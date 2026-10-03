"""Torch (GPU, differentiable) versions of the front-ends plus a torchaudio-free Kaldi-compatible fbank for BEATs.

`kaldi_fbank` re-implements torchaudio.compliance.kaldi.fbank with the exact settings BEATs uses, so that the BEATs
front end does not depend on a torchaudio build that matches the installed torch:
  sample rate 16 kHz, 25 ms Povey window, 10 ms shift, 128 mel bins, dither 0, remove DC offset,
  pre-emphasis 0.97, no energy, log-mel with floor eps. BEATs then normalizes: (fbank - 15.41663) / (6.55582 * 2).
"""
from __future__ import annotations
import math, torch, torch.nn.functional as F_


# --------------------------------------------------------------------------------------------- Kaldi fbank
def _mel(f): return 1127.0 * torch.log(1.0 + f / 700.0)
def _imel(m): return 700.0 * (torch.exp(m / 1127.0) - 1.0)


def _mel_banks(n_mels, n_fft_bins, fs, low=20.0, high=None):
    high = high if high is not None else fs / 2
    nyq = fs / 2; fft_bin_width = fs / ((n_fft_bins - 1) * 2)
    mlo, mhi = _mel(torch.tensor(low)), _mel(torch.tensor(high)); delta = (mhi - mlo) / (n_mels + 1)
    bins = torch.arange(n_mels).unsqueeze(1)
    left, center, right = mlo + bins * delta, mlo + (bins + 1) * delta, mlo + (bins + 2) * delta
    mel = _mel(fft_bin_width * torch.arange(n_fft_bins)).unsqueeze(0)
    up = (mel - left) / (center - left); down = (right - mel) / (right - center)
    return torch.clamp(torch.min(up, down), min=0.0)          # (n_mels, n_fft_bins)


def kaldi_fbank(wave, fs=16000, n_mels=128, frame_ms=25.0, shift_ms=10.0, preemph=0.97, eps=1.1920928955078125e-07):
    """wave: (B, T) float in [-1, 1] scale (BEATs feeds 16-bit-scaled waveforms: wave * 2**15). Returns (B, n_frames, n_mels)."""
    dev = wave.device; B = wave.shape[0]
    win = int(fs * frame_ms / 1000); shift = int(fs * shift_ms / 1000); n_fft = 1 << (win - 1).bit_length()
    n_frames = 1 + (wave.shape[1] - win) // shift
    idx = torch.arange(win, device=dev).unsqueeze(0) + shift * torch.arange(n_frames, device=dev).unsqueeze(1)
    frames = wave[:, idx]                                              # (B, n_frames, win)
    frames = frames - frames.mean(-1, keepdim=True)                    # remove DC offset
    frames = torch.cat([frames[..., :1] * (1 - preemph), frames[..., 1:] - preemph * frames[..., :-1]], -1)
    povey = torch.hann_window(win, periodic=False, device=dev).pow(0.85)
    frames = frames * povey
    spec = torch.fft.rfft(frames, n=n_fft).abs().pow(2)                # (B, n_frames, n_fft/2+1)
    banks = _mel_banks(n_mels, n_fft // 2 + 1, fs).to(dev)
    return torch.log(torch.clamp(spec @ banks.T, min=eps))


def beats_input(wave16k):
    """Normalized fbank as BEATs expects: (B, n_frames, 128)."""
    fb = kaldi_fbank(wave16k * (2 ** 15))
    return (fb - 15.41663) / (6.55582 * 2)


# --------------------------------------------------------------------------------------------- cyclo volume
def cyclo_volume_torch(x, fs, fr, orders, n_fbands=32, nperseg=256, hop=16, n_tau=4):
    """x: (B, T) ; fr: (B,) shaft frequency Hz ; orders: (O,) tensor. Returns (B, n_fbands, O, n_tau) log volume.
    Differentiable in x (used as a learnable front-end when needed)."""
    B = x.shape[0]; dev = x.device
    win = torch.hann_window(nperseg, device=dev)
    S = torch.stft(x, nperseg, hop_length=hop, win_length=nperseg, window=win, return_complex=True).abs()   # (B, F, T)
    Fb = S.shape[1] // n_fbands * n_fbands
    E = S[:, :Fb].reshape(B, n_fbands, -1, S.shape[2]).mean(2)                                              # (B, nf, T)
    fe = fs / hop; T = E.shape[2]; L = T // n_tau
    out = []
    for k in range(n_tau):
        seg = E[:, :, k * L:(k + 1) * L]; seg = seg - seg.mean(-1, keepdim=True)
        A = torch.fft.rfft(seg, dim=-1).abs()                                                                # (B, nf, L/2+1)
        alpha = torch.fft.rfftfreq(seg.shape[-1], 1 / fe).to(dev)                                            # (L/2+1,)
        target = orders.unsqueeze(0) * fr.unsqueeze(1)                                                        # (B, O)
        pos = target / (alpha[1] - alpha[0])                                                                  # fractional bin index
        i0 = pos.floor().clamp(0, len(alpha) - 2).long(); w = (pos - i0.float()).clamp(0, 1)
        g0 = torch.gather(A, 2, i0.unsqueeze(1).expand(-1, n_fbands, -1)); g1 = torch.gather(A, 2, (i0 + 1).unsqueeze(1).expand(-1, n_fbands, -1))
        out.append(g0 * (1 - w.unsqueeze(1)) + g1 * w.unsqueeze(1))
    return torch.log(torch.stack(out, -1) + 1e-9)


def log_psd_torch(x, n_bands=128, nperseg=2048):
    win = torch.hann_window(nperseg, device=x.device)
    S = torch.stft(x, nperseg, hop_length=nperseg // 2, window=win, return_complex=True).abs().pow(2).mean(-1)   # (B, F)
    S = S[:, 1:1 + (S.shape[1] - 1) // n_bands * n_bands].reshape(x.shape[0], n_bands, -1).mean(-1)
    return torch.log10(S + 1e-12)
