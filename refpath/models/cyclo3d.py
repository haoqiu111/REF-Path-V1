"""Cyclo3DNet: factorized 3D network on order-normalized cyclostationary volumes with
reference-conditioned path gain (RCN) and an optional multi-label decomposition head.

Input volume V: (B, 1, F, O, T) = (carrier band, cyclic order, sub-window).
Reference R: (B, 1, F, O, T) = mean cyclo volume of the healthy reference windows of the SAME domain.

Physics-motivated structure:
  * path gain acts on the F axis  -> FiLM parameters (gamma_f, beta_f) predicted from the reference's F-profile
  * speed acts on the O axis      -> already normalized by order axis; residual speed effects handled by a
                                     small FiLM on O predicted from the reference's O-profile
  * fault content = (F, O) pattern -> factorized convs: (F,1,1) x (1,O,1) x (1,1,T) kernels
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F_


class FactorizedBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1, 1)):
        super().__init__()
        self.f = nn.Conv3d(cin, cout, (3, 1, 1), stride=(stride[0], 1, 1), padding=(1, 0, 0))
        self.o = nn.Conv3d(cout, cout, (1, 3, 1), stride=(1, stride[1], 1), padding=(0, 1, 0))
        self.t = nn.Conv3d(cout, cout, (1, 1, 3), stride=(1, 1, stride[2]), padding=(0, 0, 1))
        self.bn = nn.BatchNorm3d(cout)
        self.skip = nn.Conv3d(cin, cout, 1, stride=stride) if (cin != cout or stride != (1, 1, 1)) else nn.Identity()

    def forward(self, x):
        y = F_.gelu(self.f(x)); y = F_.gelu(self.o(y)); y = self.t(y)
        return F_.gelu(self.bn(y) + self.skip(x))


class ReferenceConditioner(nn.Module):
    """Predicts FiLM parameters on the F axis (path gain) and O axis (residual speed effect) from the reference volume."""
    def __init__(self, n_f, n_o, hidden=64):
        super().__init__()
        self.f_net = nn.Sequential(nn.Linear(n_f, hidden), nn.GELU(), nn.Linear(hidden, 2 * n_f))
        self.o_net = nn.Sequential(nn.Linear(n_o, hidden), nn.GELU(), nn.Linear(hidden, 2 * n_o))
        nn.init.zeros_(self.f_net[-1].weight); nn.init.zeros_(self.f_net[-1].bias)
        nn.init.zeros_(self.o_net[-1].weight); nn.init.zeros_(self.o_net[-1].bias)

    def forward(self, ref):                       # ref: (B, 1, F, O, T)
        pf = ref.mean(dim=(1, 3, 4)); po = ref.mean(dim=(1, 2, 4))
        gf, bf = self.f_net(pf).chunk(2, -1); go, bo = self.o_net(po).chunk(2, -1)
        return gf, bf, go, bo


class Cyclo3DNet(nn.Module):
    def __init__(self, n_f=32, n_o=100, n_t=4, n_classes=5, width=32, use_ref=True, multilabel=None, n_psd=0):
        """multilabel: optional list of (class_name -> set of primitive labels) handled by the training script;
        here we expose an extra sigmoid head of size n_prim if multilabel is an int (number of primitives)."""
        super().__init__()
        self.use_ref = use_ref
        self.cond = ReferenceConditioner(n_f, n_o) if use_ref else None
        self.stem = nn.Conv3d(1, width, 1)
        self.blocks = nn.Sequential(
            FactorizedBlock(width, width), FactorizedBlock(width, 2 * width, (2, 2, 1)),
            FactorizedBlock(2 * width, 2 * width), FactorizedBlock(2 * width, 4 * width, (2, 2, 2)),
            FactorizedBlock(4 * width, 4 * width))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.psd_net = nn.Sequential(nn.Linear(n_psd, 2 * width), nn.GELU(), nn.Dropout(0.2), nn.Linear(2 * width, 2 * width), nn.GELU()) if n_psd else None
        emb = 4 * width + (2 * width if n_psd else 0)
        self.head = nn.Linear(emb, n_classes)
        self.prim_head = nn.Linear(emb, multilabel) if multilabel else None

    def normalize_with_ref(self, v, ref, ref_std=None):
        """Reference-conditioned normalization: standardize by the healthy reference (mean and, if given, std) of the
        same domain, then FiLM on F and O axes. Falls back to per-sample standardization without reference."""
        if not self.use_ref:
            return (v - v.mean(dim=(2, 3, 4), keepdim=True)) / (v.std(dim=(2, 3, 4), keepdim=True) + 1e-5)
        x = (v - ref) / (ref_std + 1e-3) if ref_std is not None else v - ref
        gf, bf, go, bo = self.cond(ref)
        x = x * (1 + gf)[:, None, :, None, None] + bf[:, None, :, None, None]
        x = x * (1 + go)[:, None, None, :, None] + bo[:, None, None, :, None]
        return x

    def forward(self, v, ref=None, ref_std=None, psd=None):
        x = self.normalize_with_ref(v, ref, ref_std)
        z = self.pool(self.blocks(self.stem(x))).flatten(1)
        if self.psd_net is not None:
            z = torch.cat([z, self.psd_net(psd)], 1)
        out = {'logits': self.head(z), 'embedding': z}
        if self.prim_head is not None:
            out['prim_logits'] = self.prim_head(z)
        return out
