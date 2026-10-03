"""Path-conditioned LoRA for a frozen BEATs encoder.

A small hypernetwork maps a *path descriptor* d (e.g. the log-PSD ratio between the deployment healthy reference and
a canonical reference, 128-d) to the LoRA factors of every attention q/v projection:
    W' = W + (alpha / r) * B(d) A,   B(d) = B0 + U_b(d)  (A shared, B path-conditioned; per layer)
So a new sensing path only needs its healthy reference to obtain its adapter (zero-shot), and the adapter
degenerates to a plain LoRA when d = 0 (canonical path). The physical prior (parabolic-mirror gain curve or measured
PSD ratio) enters as an additive frequency-wise bias on the patch-embedding input.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F_


class PathLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, d_dim=128, hyper_hidden=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        i, o = base.in_features, base.out_features
        self.A0 = nn.Parameter(torch.randn(r, i) * 0.01)
        self.B0 = nn.Parameter(torch.zeros(o, r))
        self.hyper = nn.Sequential(nn.Linear(d_dim, hyper_hidden), nn.GELU(), nn.Linear(hyper_hidden, o * r))
        nn.init.zeros_(self.hyper[-1].weight); nn.init.zeros_(self.hyper[-1].bias)
        self._d = None                      # set by PathLoRAModel.set_path before the forward pass

    def forward(self, x):                   # x: (B, T, i)
        y = self.base(x)
        d = self._d
        if d is None:
            return y + self.scale * F_.linear(F_.linear(x, self.A0), self.B0)
        i, o = self.base.in_features, self.base.out_features
        Bm = self.B0.unsqueeze(0) + self.hyper(d).view(-1, o, self.r)      # (B, o, r): path-conditioned up-projection
        z = F_.linear(x, self.A0)                                          # shared down-projection
        if x.dim() == 3 and x.shape[0] != d.shape[0] and x.shape[1] == d.shape[0]:   # fairseq layout (T, B, C)
            return y + self.scale * torch.einsum('tbr,bor->tbo', z, Bm)
        return y + self.scale * torch.einsum('btr,bor->bto', z, Bm)


def inject_path_lora(module: nn.Module, target_names=('q_proj', 'v_proj'), **kw):
    """Replace nn.Linear submodules whose attribute name is in target_names with PathLoRALinear. Returns list of wrappers."""
    wrapped = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            w = PathLoRALinear(child, **kw); setattr(module, name, w); wrapped.append(w)
        else:
            wrapped += inject_path_lora(child, target_names, **kw)
    return wrapped


class PathLoRAModel(nn.Module):
    """Frozen BEATs + path-conditioned LoRA + classification / primitive heads."""
    def __init__(self, beats, n_classes, d_dim=128, r=8, alpha=16, n_prim=0, freq_bias=True):
        super().__init__()
        self.beats = beats
        for p in self.beats.parameters():
            p.requires_grad_(False)
        self.loras = inject_path_lora(self.beats.encoder, d_dim=d_dim, r=r, alpha=alpha)
        dim = self.beats.cfg.encoder_embed_dim
        self.freq_bias = nn.Linear(d_dim, 128) if freq_bias else None     # prior: frequency-wise fbank bias from descriptor
        if self.freq_bias is not None:
            nn.init.zeros_(self.freq_bias.weight); nn.init.zeros_(self.freq_bias.bias)
        self.head = nn.Linear(dim, n_classes)
        self.prim_head = nn.Linear(dim, n_prim) if n_prim else None

    def set_path(self, d):
        for w in self.loras:
            w._d = d

    def _encode(self, fbank, d):
        self.set_path(d)
        if self.freq_bias is not None:
            fbank = fbank + self.freq_bias(d).unsqueeze(1)
        x = self.beats.patch_embedding(fbank.unsqueeze(1)).reshape(fbank.shape[0], self.beats.cfg.embed_dim, -1).transpose(1, 2)
        x = self.beats.layer_norm(x)
        if self.beats.post_extract_proj is not None:
            x = self.beats.post_extract_proj(x)
        x = self.beats.dropout_input(x)
        x, _ = self.beats.encoder(x, padding_mask=None)
        z = x.mean(1); out = {'logits': self.head(z), 'embedding': z}
        if self.prim_head is not None:
            out['prim_logits'] = self.prim_head(z)
        return out

    def _fbank(self, wave16k, ref_mu=None, ref_sd=None):
        fbank = self.beats.preprocess(wave16k)                                # (B, T, 128)
        if ref_mu is not None:                                                # healthy-reference standardization on the mel axis
            fbank = (fbank - ref_mu.unsqueeze(1)) / (ref_sd.unsqueeze(1) + 1e-3) * 0.5
        return fbank

    def forward(self, wave16k, d, ref_mu=None, ref_sd=None):
        """wave16k: (B, T) at 16 kHz; d: (B, d_dim) path descriptor; ref_mu/ref_sd: (B, 128) per-mel-bin statistics of the
        healthy reference of the window's own domain (reference-conditioned normalization)."""
        return self._encode(self._fbank(wave16k, ref_mu, ref_sd), d)

    def forward_mix(self, wave16k, d, ref_mu, ref_sd, j, lam):
        """Cross-domain mixup in the reference-normalized fbank space: mix window b with window j[b] by lam[b]."""
        fbank = self._fbank(wave16k, ref_mu, ref_sd)
        fbank = lam[:, None, None] * fbank + (1 - lam[:, None, None]) * fbank[j]
        return self._encode(fbank, lam[:, None] * d + (1 - lam[:, None]) * d[j])

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
