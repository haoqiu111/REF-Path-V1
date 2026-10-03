"""DCASE 2026 secondary experiment: the standard ASD recipe — BEATs + LoRA fine-tuned on an auxiliary classification
task (machine type × channel, 14 classes) using train-normal clips, then per-machine kNN anomaly scores on the
fine-tuned embeddings (near / far / concat), with and without far-channel reference (concatenation).

python scripts/dcase_finetune.py [--epochs 2]
"""
import os, sys, time, glob, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from refpath.paths import beats_checkpoint
from refpath.data.dcase import dcase_index, read_wav, MACHINES
from refpath.models.beats.BEATs import BEATs, BEATsConfig
from refpath.models.path_lora import PathLoRAModel
from dcase_asd import fix_len, knn_scores, pauc
from sklearn.metrics import roc_auc_score
CKPT = beats_checkpoint(); dev = 'cuda'

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--bs', type=int, default=16); a = ap.parse_args()
    idx = dcase_index(); mach = np.array([r['machine'] for r in idx]); split = np.array([r['split'] for r in idx]); dom = np.array([r['domain'] for r in idx]); y = np.array([r['label'] for r in idx])
    tr = np.where(split == 'train')[0]; mid = {m: i for i, m in enumerate(MACHINES)}
    c = torch.load(CKPT, map_location='cpu', weights_only=False); b = BEATs(BEATsConfig(c['cfg'])); b.load_state_dict(c['model'])
    model = PathLoRAModel(b, n_classes=len(MACHINES) * 2, freq_bias=False).to(dev); model.set_path(None)
    opt = torch.optim.AdamW(model.trainable_parameters(), 2e-4, weight_decay=1e-2); t0 = time.time()
    cache = {}
    def wav(i):
        if i not in cache: cache[i] = fix_len(read_wav(idx[i]['file'])[0]).astype(np.float32)
        return cache[i]
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(tr)
        for k in range(0, len(perm), a.bs):
            ids = perm[k:k + a.bs]; X = np.stack([wav(i) for i in ids]); X = X / (X.std(1, keepdims=True) + 1e-8) * 0.05
            ch = np.random.randint(0, 2, len(ids)); xb = torch.tensor(X[np.arange(len(ids)), :, ch]).to(dev)
            yb = torch.tensor([mid[mach[i]] * 2 + int(c_) for i, c_ in zip(ids, ch)]).to(dev)
            xb = xb[:, :80000]                                                    # 5-s random-free crop for speed
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model._encode(model._fbank(xb), torch.zeros(len(ids), 128, device=dev))
            loss = F_.cross_entropy(out['logits'].float(), yb); opt.zero_grad(); loss.backward(); opt.step()
            if (k // a.bs) % 100 == 0: print(f'ep {ep} step {k // a.bs} loss {loss.item():.3f} {time.time()-t0:.0f}s', flush=True)
    model.eval(); emb = np.zeros((len(idx), 2, 768), np.float32)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for k in range(0, len(idx), 32):
            X = np.stack([wav(i) for i in range(k, min(k + 32, len(idx)))]); X = X / (X.std(1, keepdims=True) + 1e-8) * 0.05
            for ch in range(2):
                emb[k:k + 32, ch] = model._encode(model._fbank(torch.tensor(X[:, :, ch]).to(dev)), torch.zeros(X.shape[0], 128, device=dev))['embedding'].float().cpu().numpy()
    np.savez(os.path.join(ROOT, 'cache', 'dcase_beats_finetuned.npz'), emb=emb)
    import collections; agg = collections.defaultdict(list); print(f'{"machine":12s} {"feat":10s}  AUC_src  AUC_tgt  pAUC   hmean')
    for mc in MACHINES:
        for feat, Fm in [('near', emb[:, 0]), ('far', emb[:, 1]), ('near-far', emb[:, 0] - emb[:, 1]), ('concat', np.concatenate([emb[:, 0], emb[:, 1]], 1))]:
            trm = (mach == mc) & (split == 'train'); te = (mach == mc) & (split == 'test')
            Fz = (Fm - Fm[trm].mean(0)) / (Fm[trm].std(0) + 1e-6)
            s = knn_scores(Fz[trm], Fz[te]); yt, dt = y[te], dom[te]
            a_s = roc_auc_score(yt[dt == 'source'], s[dt == 'source']); a_t = roc_auc_score(yt[dt == 'target'], s[dt == 'target']); p = pauc(yt, s); h = 3 / (1 / a_s + 1 / a_t + 1 / p)
            agg[feat].append(h); print(f'{mc:12s} {feat:10s}  {a_s:.3f}    {a_t:.3f}    {p:.3f}  {h:.3f}')
    print('\nharmonic mean over machines (fine-tuned BEATs, auxiliary machine×channel classification):'); [print(f'  {k:10s} {np.mean(v):.3f}') for k, v in agg.items()]
