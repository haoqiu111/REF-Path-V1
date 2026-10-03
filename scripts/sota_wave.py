"""Same-protocol re-implementations of waveform / spectrogram methods, every method in a 'raw' and a 'ref' version.

  aft     acoustic-frame embedded lightweight Transformer: mel frames -> frame embedding -> 3-layer Transformer -> [CLS]
          (phoneme-inspired acoustic-frame paradigm, Peng et al. MSSP 248:114030, 2026)
  xmodal  cross-modal contrastive dual-view network: 1-D CNN on the waveform + 2-D CNN on the log-Mel image, InfoNCE between
          the views, coarse / fine heads with uncertainty weighting (Peng et al. MSSP 244:113744, 2026)
  genrep  frozen BEATs mean-pooled embeddings + k-NN (training-free generic representations, Saengthong & Shinozaki ICASSP 2025)

raw : one global input gain from the source training windows, mel normalised with source statistics; no target information.
ref : input gain from each domain's healthy reference (RMS 0.05) and per-mel-bin standardization with the statistics of the
      domain's healthy reference (even reference windows), exactly as in the BEATs branch of the paper; genrep: embeddings
      standardized with the healthy reference of each domain.

Protocols: mic_x_speed (9 cells, 60-s clips) and mic_x_speed_x_load (27 cells, first 180 s of every recording).
python scripts/sota_wave.py --method aft --protocol mic_x_speed --seed 0
-> outputs/sota_wave_<method>_<protocol>_s<seed>.json   {results: {method_norm: {task: {acc, f1}}}}
"""
import os, sys, time, argparse, json, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from refpath.eval.protocol import make_tasks, domain_key, normalize, Task
from refpath.features.torch_frontend import beats_input
from refpath.paths import beats_checkpoint
from sklearn.metrics import accuracy_score, f1_score
dev = 'cuda'


class AFT(nn.Module):
    def __init__(self, n_cls, d=128, depth=3, heads=4, n_frames=98):
        super().__init__()
        self.embed = nn.Sequential(nn.Conv1d(128, d, 3, padding=1), nn.GELU(), nn.Conv1d(d, d, 3, padding=1))    # acoustic-frame embedding
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); self.pos = nn.Parameter(torch.zeros(1, n_frames + 1, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth); self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, n_cls))
    def forward(self, fb):                                  # (B, T, 128)
        h = self.embed(fb.transpose(1, 2)).transpose(1, 2); h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1) + self.pos[:, :h.shape[1] + 1]
        return self.head(self.enc(h)[:, 0])


def split_ref(t):
    rid = np.where(t.ref)[0]; rn = np.zeros_like(t.ref); rn[rid[::2]] = True; return Task(t.name, t.train, t.test, rn)


@torch.no_grad()
def mel_of(wave, idx, gain, bs=512):
    out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]; w = torch.tensor(wave[j].astype(np.float32) * gain[j, None]).to(dev); out.append(beats_input(w))
    return torch.cat(out)                                    # (n, 98, 128) on GPU


def embed_beats(wave, scale, bs=128):
    from refpath.models.beats.BEATs import BEATs, BEATsConfig
    c = torch.load(beats_checkpoint(), map_location='cpu', weights_only=False); m = BEATs(BEATsConfig(c['cfg'])); m.load_state_dict(c['model']); m = m.to(dev).eval(); E = []
    with torch.no_grad():
        for i in range(0, len(wave), bs):
            f, _ = m.extract_features(torch.tensor(wave[i:i + bs].astype(np.float32) * scale).to(dev)); E.append(f.mean(1).cpu().numpy())
    return np.concatenate(E)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--method', required=True, choices=['aft', 'xmodal', 'genrep']); ap.add_argument('--protocol', default='mic_x_speed'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=0); ap.add_argument('--bs', type=int, default=64); ap.add_argument('--tasks', default=''); ap.add_argument('--tag', default=''); a = ap.parse_args()
    if a.protocol == 'mic_x_speed':
        z = np.load(os.path.join(ROOT, 'cache', 'guilin_wave16k_raw.npz'), allow_pickle=True); meta, wave = z['meta'], z['wave']; tasks = make_tasks(meta, a.protocol); epochs = a.epochs or 15; emb_cache = 'guilin_beats_globalgain.npy'
    else:
        z = np.load(os.path.join(ROOT, 'cache', 'guilin_full_wave16k_raw_180s.npz'), allow_pickle=True); meta, wave = z['meta'], z['wave']; tasks = make_tasks(meta, a.protocol, ref_seconds=60); epochs = a.epochs or 8; emb_cache = 'guilin_full_beats_180s_globalgain.npy'
    if a.tasks: tasks = [t for t in tasks if t.name in a.tasks.split(',')]
    classes = sorted(np.unique(meta['state']).tolist()); y = np.array([classes.index(s) for s in meta['state']]); NC = len(classes); H = classes.index('healthy')
    keys = domain_key(meta); healthy = meta['state'] == 'healthy'; res = {}; t0 = time.time()
    print(f'{a.method} {a.protocol}: {len(tasks)} tasks, {len(meta)} windows, epochs {epochs}', flush=True)
    if a.method == 'genrep':
        p = os.path.join(ROOT, 'cache', emb_cache)
        if not os.path.exists(p):
            rms = float(np.sqrt(np.mean(wave[::50].astype(np.float32) ** 2))); np.save(p, embed_beats(wave, 0.05 / rms))
        E = np.load(p).astype(np.float32)
        from sklearn.neighbors import KNeighborsClassifier; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
        for t in tasks:
            tn = split_ref(t); tr, te = np.where(t.train)[0], np.where(t.test)[0]
            for norm in ('raw', 'ref'):
                G = E if norm == 'raw' else normalize(E, meta, tn, 'healthy')
                c = make_pipeline(StandardScaler(), KNeighborsClassifier(10)).fit(G[tr], y[tr]); pr = c.predict(G[te])
                res.setdefault(f'genrep_{norm}', {})[t.name] = dict(acc=float(accuracy_score(y[te], pr)), f1=float(f1_score(y[te], pr, average='macro')))
            print(f'{t.name:26s} ' + ' '.join(f'{k}={v[t.name]["acc"]:.3f}' for k, v in res.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)
    else:
        import train_pathlora as TP
        if a.method == 'xmodal': from baseline_crossmodal import Net as XNet, info_nce
        for t in tasks:
            tn = split_ref(t); tr, te = np.where(t.train)[0], np.where(t.test)[0]; idx = np.concatenate([tr, te]); ntr = len(tr)
            for norm in ('raw', 'ref'):
                torch.manual_seed(a.seed); np.random.seed(a.seed)
                if norm == 'raw':
                    g = np.full(len(meta), 0.05 / (np.sqrt(np.mean(wave[tr[::10]].astype(np.float32) ** 2)) + 1e-8), np.float32)
                    M = mel_of(wave, idx, g); mu, sd = M[:ntr].mean(), M[:ntr].std() + 1e-6; M = (M - mu) / sd
                else:
                    g = TP.reference_gain(wave, meta, tn); M = mel_of(wave, idx, g)
                    for k in np.unique(keys[idx]):
                        d = keys == k; ref = d & tn.ref if (d & t.test).any() else d & t.train & healthy
                        rid = np.where(ref)[0][:200]; R = mel_of(wave, rid, g).reshape(-1, 128); m_, s_ = R.mean(0), R.std(0) + 1e-3
                        sel = torch.tensor(np.where(keys[idx] == k)[0]).to(dev); M[sel] = (M[sel] - m_) / s_ * 0.5
                Yt = torch.tensor(y[idx]).to(dev)
                net = (AFT(NC) if a.method == 'aft' else XNet(NC)).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3 if a.method == 'xmodal' else 5e-4, weight_decay=1e-2)
                for ep in range(epochs):
                    net.train(); perm = torch.randperm(ntr, device=dev)
                    for i in range(0, ntr, a.bs):
                        b = perm[i:i + a.bs]
                        if len(b) < 4: continue
                        if a.method == 'aft':
                            loss = F_.cross_entropy(net(M[b]), Yt[b], label_smoothing=0.05)
                        else:
                            bi = idx[b.cpu().numpy()]; w = torch.tensor(wave[bi].astype(np.float32) * g[bi, None]).to(dev)
                            fine, coarse, pw, pt = net(w, M[b]); losses = torch.stack([F_.cross_entropy(fine, Yt[b]), F_.cross_entropy(coarse, (Yt[b] != H).long()), info_nce(pw, pt)])
                            loss = (torch.exp(-net.log_var) * losses + net.log_var).sum()
                        opt.zero_grad(); loss.backward(); opt.step()
                net.eval(); preds = []
                with torch.no_grad():
                    for i in range(ntr, len(idx), 256):
                        if a.method == 'aft': preds.append(net(M[i:i + 256]).argmax(1).cpu().numpy())
                        else:
                            bi = idx[i:i + 256]; w = torch.tensor(wave[bi].astype(np.float32) * g[bi, None]).to(dev); preds.append(net(w, M[i:i + 256])[0].argmax(1).cpu().numpy())
                pr = np.concatenate(preds); res.setdefault(f'{a.method}_{norm}', {})[t.name] = dict(acc=float(accuracy_score(y[te], pr)), f1=float(f1_score(y[te], pr, average='macro')))
                del net, M; torch.cuda.empty_cache()
            print(f'{t.name:26s} ' + ' '.join(f'{k}={v[t.name]["acc"]:.3f}' for k, v in res.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)
    print('\nMEAN / MIN over tasks')
    for k, v in res.items():
        accs = [x['acc'] for x in v.values()]; print(f'{k:16s} acc {np.mean(accs):.3f} (min {np.min(accs):.3f})')
    name = f'sota_wave_{a.method}_{a.protocol}{a.tag}_s{a.seed}.json'; json.dump(dict(args=vars(a), classes=classes, results=res), open(os.path.join(ROOT, 'outputs', name), 'w'), indent=1); print('saved outputs/' + name)
