"""DCASE 2026: reference-normalized BEATs features. For each machine and channel, the train-normal clips are the
healthy reference: fbank is standardized per mel bin by the reference mean/std before the frozen BEATs encoder
(the same mechanism as in the Guilin BEATs branch). Compared with raw BEATs features via kNN anomaly scores.

python scripts/dcase_refnorm.py
"""
import os, sys, time, glob, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.paths import beats_checkpoint
from refpath.data.dcase import dcase_index, read_wav, MACHINES
from refpath.models.beats.BEATs import BEATs, BEATsConfig
from refpath.features.torch_frontend import beats_input
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.join(ROOT, 'scripts')); from dcase_asd import fix_len, knn_scores, pauc
CKPT = beats_checkpoint(); dev = 'cuda'


def encode_fbank(m, fb):
    x = m.patch_embedding(fb.unsqueeze(1)).reshape(fb.shape[0], m.cfg.embed_dim, -1).transpose(1, 2)
    x = m.layer_norm(x)
    if m.post_extract_proj is not None: x = m.post_extract_proj(x)
    x, _ = m.encoder(x, padding_mask=None); return x.mean(1)


if __name__ == '__main__':
    out = os.path.join(ROOT, 'cache', 'dcase_beats_refnorm.npz')
    idx = dcase_index(); mach = np.array([r['machine'] for r in idx]); split = np.array([r['split'] for r in idx]); dom = np.array([r['domain'] for r in idx]); y = np.array([r['label'] for r in idx])
    if os.path.exists(out):
        emb = np.load(out)['emb']
    else:
        c = torch.load(CKPT, map_location='cpu', weights_only=False); m = BEATs(BEATsConfig(c['cfg'])); m.load_state_dict(c['model']); m = m.to(dev).eval()
        emb = np.zeros((len(idx), 2, 768), np.float32); t0 = time.time()
        for mc in MACHINES:
            sel = np.where(mach == mc)[0]; ref = np.where((mach == mc) & (split == 'train'))[0]
            # reference statistics per channel from up to 200 train-normal clips
            stats = []
            for ch in range(2):
                fbs = []
                with torch.no_grad():
                    for i in range(0, min(len(ref), 200), 16):
                        X = np.stack([fix_len(read_wav(idx[r]['file'])[0])[:, ch] for r in ref[i:i + 16]]); X = X / (X.std(1, keepdims=True) + 1e-8) * 0.05
                        fbs.append(beats_input(torch.tensor(X).to(dev)).reshape(-1, 128))
                fb = torch.cat(fbs); stats.append((fb.mean(0), fb.std(0)))
            for i in range(0, len(sel), 16):
                X = np.stack([fix_len(read_wav(idx[r]['file'])[0]) for r in sel[i:i + 16]]); X = X / (X.std(1, keepdims=True) + 1e-8) * 0.05
                with torch.no_grad():
                    for ch in range(2):
                        fb = beats_input(torch.tensor(X[:, :, ch]).to(dev)); mu, sd = stats[ch]
                        fb = (fb - mu) / (sd + 1e-3) * 0.5
                        emb[sel[i:i + 16], ch] = encode_fbank(m, fb).cpu().numpy()
            print(mc, f'{time.time()-t0:.0f}s', flush=True)
        np.savez(out, emb=emb)
    print(f'{"machine":12s} {"feat":10s}  AUC_src  AUC_tgt  pAUC   hmean'); import collections; agg = collections.defaultdict(list)
    for mc in MACHINES:
        for feat, F in [('near', emb[:, 0]), ('far', emb[:, 1]), ('near-far', emb[:, 0] - emb[:, 1]), ('concat', np.concatenate([emb[:, 0], emb[:, 1]], 1))]:
            tr = (mach == mc) & (split == 'train'); te = (mach == mc) & (split == 'test')
            s = knn_scores(F[tr], F[te]); yt, dt = y[te], dom[te]
            a_s = roc_auc_score(yt[dt == 'source'], s[dt == 'source']); a_t = roc_auc_score(yt[dt == 'target'], s[dt == 'target']); p = pauc(yt, s); h = 3 / (1 / a_s + 1 / a_t + 1 / p)
            agg[feat].append(h); print(f'{mc:12s} {feat:10s}  {a_s:.3f}    {a_t:.3f}    {p:.3f}  {h:.3f}')
    print('\nharmonic-mean over machines (reference-normalized BEATs):'); [print(f'  {k:10s} {np.mean(v):.3f}') for k, v in agg.items()]
