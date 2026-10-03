"""Late fusion of the physics branch (MLP on healthy-standardized PSD + cyclo, feats cache, 1-s windows hop 0.5 s)
and the BEATs branch (PathLoRA, wave16k cache, 1-s windows hop 1.0 s) per Reference-Only task.

Window alignment: both caches enumerate the same recordings in the same order; feats has 119 windows per 60-s
recording, wave16k has 60, and wave window j == feats window 2j of the same recording.

python scripts/fuse.py outputs/guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_s0_probs.npz outputs/guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm_refsplit_refgain_s0_probs.npz
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
zf = np.load(os.path.join(ROOT, 'cache', 'guilin_feats.npz'), allow_pickle=True); mf = zf['meta']
zw = np.load(os.path.join(ROOT, 'cache', 'guilin_wave16k_raw.npz'), allow_pickle=True); mw = zw['meta']
nf, nw = 119, 60
assert len(mf) % nf == 0 and len(mw) % nw == 0 and len(mf) // nf == len(mw) // nw
rec_w = np.arange(len(mw)) // nw; j_w = np.arange(len(mw)) % nw; feats_index_of_wave = rec_w * nf + 2 * j_w
assert (mf['state'][feats_index_of_wave] == mw['state']).all() and (mf['path'][feats_index_of_wave] == mw['path']).all()

A = np.load(sys.argv[1]); B = np.load(sys.argv[2])
tasks = sorted({k.split('/')[0] for k in A.files})
print(f'{"task":22s}  physics  beats   fuse(avg)  fuse(geo)')
res = []
for t in tasks:
    pa, ya, tea = A[f'{t}/prob'], A[f'{t}/y'], A[f'{t}/te']          # feats indexing
    pb, yb, teb = B[f'{t}/prob'], B[f'{t}/y'], B[f'{t}/te']          # wave indexing
    pos = {ix: i for i, ix in enumerate(tea)}
    sel = np.array([pos[feats_index_of_wave[ix]] for ix in teb])     # physics prob for each wave test window
    pa2 = pa[sel]; assert (ya[sel] == yb).all()
    acc_a, acc_b = (pa2.argmax(1) == yb).mean(), (pb.argmax(1) == yb).mean()
    avg = ((pa2 + pb) / 2).argmax(1); geo = (np.log(pa2 + 1e-9) + np.log(pb + 1e-9)).argmax(1)
    acc_avg, acc_geo = (avg == yb).mean(), (geo == yb).mean()
    # reference-gated fusion: reliability of a branch in the target domain = mean prob it assigns to 'healthy' on the
    # target's healthy reference windows (label-free at deployment, since the reference is known healthy)
    H = 0   # class index of 'healthy' (classes sorted: cage_fracture, healthy, ...) -> find from probs shape via name order
    classes = ['cage_fracture', 'healthy', 'inner_outer_pitting', 'inner_pitting', 'outer_pitting']; H = classes.index('healthy')
    ra = A[f'{t}/ref_prob'][:, H].mean() if f'{t}/ref_prob' in A.files else 1.0
    rb = B[f'{t}/ref_prob'][:, H].mean() if f'{t}/ref_prob' in B.files else 1.0
    wa, wb = np.exp(8 * ra), np.exp(8 * rb); wa, wb = wa / (wa + wb), wb / (wa + wb)
    gated = (wa * np.log(pa2 + 1e-9) + wb * np.log(pb + 1e-9)).argmax(1); acc_gate = (gated == yb).mean()
    hard = pa2.argmax(1) if ra >= rb else pb.argmax(1); acc_hard = (hard == yb).mean()
    res.append((acc_a, acc_b, acc_avg, acc_geo, acc_gate, acc_hard))
    print(f'{t:22s}  {acc_a:.3f}    {acc_b:.3f}   {acc_avg:.3f}      {acc_geo:.3f}     gate {acc_gate:.3f} (w_phys {wa:.2f}, ref-healthy {ra:.2f}/{rb:.2f})  hard {acc_hard:.3f}')
r = np.array(res); print(f'{"MEAN":22s}  ' + '    '.join(f'{v:.3f}' for v in r.mean(0)) + '   [physics, beats, avg, geo, gate, hard-select]')
print(f'{"MIN":22s}  ' + '    '.join(f'{v:.3f}' for v in r.min(0)))
