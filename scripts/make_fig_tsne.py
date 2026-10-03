"""t-SNE of the feature spaces before and after healthy-reference standardization, source domains against the held-out
target domain, on two corpora: Guilin (physics-MLP embedding, median-gain cell B&K #3 @ 1200 rpm; titles give the target
accuracy of the MLP) and AHU (physics features, parabolic mirror @ 600 rpm held out; titles give the nearest-source-class-centroid
accuracy of the target windows in the original feature space), raw vs standardized. Needs a GPU unless REFPATH_DEVICE=cpu.
Writes figures/fig_tsne.pdf/png and outputs/tsne_stats.json."""
import os, sys, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA; from sklearn.manifold import TSNE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath import plot_style as PS; PS.apply(8); plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'axes.spines.top': True, 'axes.spines.right': True})
from refpath.eval.protocol import make_tasks, normalize, Task
rng = np.random.RandomState(0); stats = {}


def split_ref(t):
    rid = np.where(t.ref)[0]; rn = np.zeros_like(t.ref); rn[rid[::2]] = True; return Task(t.name, t.train, t.test, rn)
def centroid_acc(Gs, ys, Gt, yt):
    mu, sd = Gs.mean(0), Gs.std(0) + 1e-6; Gs, Gt = (Gs - mu) / sd, (Gt - mu) / sd
    C = np.stack([Gs[ys == c].mean(0) for c in np.unique(ys)]); pred = np.unique(ys)[np.argmin(((Gt[:, None, :] - C[None]) ** 2).sum(2), 1)]; return float((pred == yt).mean())
def embed(Gs, Gt):
    X = np.concatenate([Gs, Gt]); mu, sd = Gs.mean(0), Gs.std(0) + 1e-6; X = (X - mu) / sd
    Z = TSNE(n_components=2, perplexity=30, init='pca', random_state=0).fit_transform(PCA(50, random_state=0).fit_transform(X)); return Z[:len(Gs)], Z[len(Gs):]
def sub(idx, n):
    return np.sort(rng.choice(idx, n, replace=False)) if len(idx) > n else idx


panels = []
# ---- Guilin: penultimate embedding of the physics MLP trained on raw vs healthy-reference-standardized features ----
import torch, torch.nn as nn, torch.nn.functional as F_
def train_embed(G, tr, y, n_cls, seed=0, epochs=40, dev=os.environ.get('REFPATH_DEVICE', 'cuda')):
    torch.manual_seed(seed); mu, sd = G[tr].mean(0), G[tr].std(0) + 1e-6; X = torch.tensor((G[tr] - mu) / sd).to(dev); Y = torch.tensor(y[tr]).to(dev)
    enc = nn.Sequential(nn.Dropout(0.2), nn.Linear(G.shape[1], 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 256), nn.GELU()).to(dev); head = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, n_cls)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        enc.train(); head.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]; xb, yb = X[idx], F_.one_hot(Y[idx], n_cls).float(); j = torch.randperm(len(idx), device=dev); lam = torch.distributions.Beta(0.4, 0.4).sample((len(idx), 1)).to(dev)
            xb = lam * xb + (1 - lam) * xb[j]; yb = lam * yb + (1 - lam) * yb[j]; loss = -(yb * F_.log_softmax(head(enc(xb)), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); head.eval()
    def run(Gsub):
        with torch.no_grad(): z = enc(torch.tensor((Gsub - mu) / sd).to(dev)); return z.cpu().numpy(), head(z).argmax(1).cpu().numpy()
    return run
z = np.load(os.path.join(ROOT, 'cache', 'guilin_feats.npz'), allow_pickle=True); meta = z['meta']; F = np.concatenate([z['psd'], z['cyclo'].mean(3).reshape(len(meta), -1)], 1).astype(np.float32)
t = [x for x in make_tasks(meta, 'mic_x_speed') if x.name == 'mxs:BK4966_3@1200'][0]; tn = split_ref(t)     # the cell with the median gain of standardization over the nine cells
classes = sorted(np.unique(meta['state']).tolist()); y = np.array([classes.index(s) for s in meta['state']]); NAMES = {'healthy': 'healthy', 'inner_pitting': 'inner', 'outer_pitting': 'outer', 'inner_outer_pitting': 'inner+outer', 'cage_fracture': 'cage'}
tr_all = np.where(t.train)[0]; src, tgt = sub(tr_all, 1800), sub(np.where(t.test)[0], 700)
for mode in ['raw', 'healthy']:
    G = F if mode == 'raw' else normalize(F, meta, tn, 'healthy'); run = train_embed(G, tr_all, y, len(classes)); Es, _ = run(G[src]); Et, pt = run(G[tgt]); acc = float((pt == y[tgt]).mean())
    panels.append(dict(corpus='Guilin, physics-MLP embedding, target = B&K #3 @ 1200 rpm', mode=mode, Gs=Es, ys=y[src], Gt=Et, yt=y[tgt], labels=[NAMES[c] for c in classes], acc=acc, metric='target accuracy of the MLP'))
# ---- AHU ----
z = np.load(os.path.join(ROOT, 'cache', 'ahu_feats.npz'), allow_pickle=True); meta = z['meta']; F = np.concatenate([z['psd'], z['cyclo'].mean(3).reshape(len(meta), -1)], 1).astype(np.float32)
t = [x for x in make_tasks(meta, 'mic_x_speed') if x.name == 'mxs:PM@600'][0]; tn = split_ref(t)
grp = np.array([s.split('_')[0] for s in meta['state']]); gl = ['healthy', 'ball', 'inner', 'outer']; y = np.array([gl.index(g) for g in grp])
src, tgt = np.where(t.train)[0], np.where(t.test)[0]
for mode in ['raw', 'healthy']:
    G = F if mode == 'raw' else normalize(F, meta, tn, 'healthy')
    panels.append(dict(corpus='AHU, physics features, target = parabolic mirror @ 600 rpm', mode=mode, Gs=G[src], ys=y[src], Gt=G[tgt], yt=y[tgt], labels=gl, acc=None, metric='target nearest-centroid acc.'))
fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.9))
from matplotlib.lines import Line2D
for k, P in enumerate(panels):
    ax = axes[k // 2, k % 2]; acc = P['acc'] if P['acc'] is not None else centroid_acc(P['Gs'], P['ys'], P['Gt'], P['yt']); Zs, Zt = embed(P['Gs'], P['Gt'])
    for c, name in enumerate(P['labels']):
        col = PS.TSNE_COLORS[c % len(PS.TSNE_COLORS)]
        ax.scatter(*Zs[P['ys'] == c].T, s=6, color=col, alpha=0.75, linewidths=0, marker='o')
        ax.scatter(*Zt[P['yt'] == c].T, s=11, color=col, alpha=0.95, linewidths=0, marker='^')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"({'abcd'[k]}) " + P['corpus'].split(',')[0] + (', raw' if P['mode'] == 'raw' else ', reference-standardized') + f": {P['metric']} {acc:.2f}", fontsize=8, loc='left')
    if k % 2 == 0:
        ax.text(-0.03, 0.5, ', '.join(P['corpus'].split(', ')[1:]), transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=7)
        handles = [Line2D([], [], marker='s', ls='', color=PS.TSNE_COLORS[c % len(PS.TSNE_COLORS)], label=n, markersize=6) for c, n in enumerate(P['labels'])] + [Line2D([], [], marker='o', ls='', color='#666', label='source', markersize=5), Line2D([], [], marker='^', ls='', color='#666', label='target', markersize=6)]
        ax.legend(handles=handles, fontsize=6, loc='best', ncol=2 if len(P['labels']) > 5 else 1, frameon=True, framealpha=0.9, edgecolor='#ccc')
    stats[f"{P['corpus']} | {P['mode']}"] = dict(metric=P['metric'], value=acc, n_source=int(len(P['ys'])), n_target=int(len(P['yt'])))
fig.tight_layout(); out = os.path.join(ROOT, 'figures', 'fig_tsne'); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
json.dump(stats, open(os.path.join(ROOT, 'outputs', 'tsne_stats.json'), 'w'), indent=1); print(json.dumps(stats, indent=1)); print('saved', out)
