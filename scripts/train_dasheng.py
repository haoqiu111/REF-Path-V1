"""Second audio foundation model. The backbone is kept in eval mode during LoRA training and reference-statistics extraction:
in train mode the per-bin input BatchNorm of Dasheng would use batch statistics (i.e. batch normalization would replace the
reference normalization) and drift its running statistics. Refnorm scale 0.1 = eval-mode global std.
Dasheng-base (86 M, Apache-2.0) with the same recipe as the BEATs branch:
reference gain, reference-normalized (per-mel-bin) spectrogram after Dasheng's own front-end + init_bn, LoRA on the
fused qkv projections of all 12 blocks, classification + primitive heads, 3 epochs. Frozen-Dasheng + LR probe is also
reported (--frozen).

python scripts/train_dasheng.py --protocol mic_x_speed_x_load --seed 0 [--frozen] [--seconds 180]
"""
import os, sys, time, argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from refpath.eval.protocol import make_tasks, domain_key, Task
from refpath.models.path_lora import inject_path_lora
import train_pathlora as TP
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
dev = 'cuda'
PRIM = {'healthy': [], 'inner_pitting': ['inner'], 'outer_pitting': ['outer'], 'inner_outer_pitting': ['inner', 'outer'], 'cage_fracture': ['cage']}; PRIMS = ['inner', 'outer', 'cage']


class DashengRef(nn.Module):
    def __init__(self, n_cls, n_prim, r=8, lora=True, scale=0.1):
        super().__init__(); self.scale = scale
        from dasheng import dasheng_base
        self.m = dasheng_base()
        for p in self.m.parameters(): p.requires_grad_(False)
        self.loras = inject_path_lora(self.m.blocks, target_names=('qkv',), r=r, alpha=16) if lora else []
        for w in self.loras: w._d = None                      # plain LoRA (no hypernetwork)
        self.head = nn.Linear(768, n_cls); self.prim = nn.Linear(768, n_prim)
        self.m.eval()                                     # backbone stays in eval mode: init_bn is a per-mel-bin BatchNorm2d(64) whose train-mode batch statistics would replace the reference statistics and drift its running stats

    def train(self, mode=True):
        super().train(mode); self.m.eval(); return self

    def spec(self, x):                                        # (B, 1, 64, T) after front_end + init_bn
        return self.m.forward_to_spec(x)

    def forward_spec(self, S, mu=None, sd=None):
        if mu is not None: S = (S - mu[:, None, :, None]) / (sd[:, None, :, None] + 1e-3) * self.scale   # keep Dasheng's eval-mode input scale (global std 0.11 on Guilin)
        z = self.m.forward_spectrogram(S).mean(1)
        return self.head(z), self.prim(z), z

    def trainable(self): return [p for p in self.parameters() if p.requires_grad]


def ref_spec_stats(model, wave, meta, task, dev='cuda'):
    keys = domain_key(meta); healthy = meta['state'] == 'healthy'; MU = np.zeros((len(meta), 64), np.float32); SD = np.ones((len(meta), 64), np.float32)
    for k in np.unique(keys):
        d = keys == k; ref = d & task.ref if (d & task.test).any() else d & task.train & healthy
        if ref.sum() == 0: ref = d & healthy
        idx = np.where(ref)[0][:120]
        with torch.no_grad(): S = model.spec(torch.tensor(wave[idx]).float().to(dev))[:, 0]      # (n, 64, T)
        MU[d] = S.mean((0, 2)).cpu().numpy(); SD[d] = S.std((0, 2)).cpu().numpy()
    return MU, SD


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--protocol', default='mic_x_speed_x_load'); ap.add_argument('--seconds', type=int, default=180); ap.add_argument('--ref_seconds', type=float, default=60)
    ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--bs', type=int, default=32); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--frozen', action='store_true'); ap.add_argument('--no_refnorm', action='store_true'); ap.add_argument('--tasks', default=''); ap.add_argument('--tag', default=''); ap.add_argument('--scale', type=float, default=0.1)
    a = ap.parse_args()
    z = np.load(os.path.join(ROOT, 'cache', f'guilin_full_wave16k_raw_{a.seconds}s.npz'), allow_pickle=True); meta, wave = z['meta'], z['wave']
    classes = sorted(np.unique(meta['state']).tolist()); y = np.array([classes.index(s) for s in meta['state']]); P = np.array([[1.0 if p in PRIM[s] else 0.0 for p in PRIMS] for s in meta['state']], np.float32)
    tasks = make_tasks(meta, a.protocol, ref_seconds=a.ref_seconds)
    if a.tasks: tasks = [t for t in tasks if t.name in a.tasks.split(',')]
    res = {}; probs = {}; t0 = time.time()
    for t in tasks:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        rid = np.where(t.ref)[0]; rn = np.zeros_like(t.ref); rn[rid[::2]] = True; tn = Task(t.name, t.train, t.test, rn); rc = np.where(t.ref)[0][1::2]
        g = TP.reference_gain(wave, meta, tn); tr, te = np.where(t.train)[0], np.where(t.test)[0]
        model = DashengRef(len(classes), len(PRIMS), lora=not a.frozen, scale=a.scale).to(dev)
        W = lambda idx: torch.tensor(wave[idx].astype(np.float32) * g[idx, None]).to(dev)
        MU, SD = (None, None) if a.no_refnorm else ref_spec_stats(model, (wave.astype(np.float32) * g[:, None]).astype(np.float16), meta, tn)
        fwd = lambda idx: model.forward_spec(model.spec(W(idx)), None if MU is None else torch.tensor(MU[idx]).to(dev), None if SD is None else torch.tensor(SD[idx]).to(dev))
        if a.frozen:
            model.eval(); Z = {}
            with torch.no_grad():
                for name, idx_all in [('tr', tr), ('te', te)]:
                    Z[name] = np.concatenate([fwd(idx_all[i:i + 256])[2].cpu().numpy() for i in range(0, len(idx_all), 256)])
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3)); c.fit(Z['tr'], y[tr]); p = c.predict(Z['te']); prob = c.predict_proba(Z['te']); rprob = None
        else:
            opt = torch.optim.AdamW(model.trainable(), a.lr, weight_decay=1e-2); steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
            for ep in range(a.epochs):
                model.train(); perm = np.random.permutation(tr)
                for i in range(0, len(perm), a.bs):
                    idx = perm[i:i + a.bs]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits, prim, _ = fwd(idx)
                    loss = F_.cross_entropy(logits.float(), torch.tensor(y[idx]).to(dev), label_smoothing=0.05) + 0.5 * F_.binary_cross_entropy_with_logits(prim.float(), torch.tensor(P[idx]).to(dev))
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            model.eval(); outs = []; routs = []
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for i in range(0, len(te), 256): outs.append(F_.softmax(fwd(te[i:i + 256])[0].float(), 1).cpu().numpy())
                for i in range(0, len(rc), 256): routs.append(F_.softmax(fwd(rc[i:i + 256])[0].float(), 1).cpu().numpy())
            prob = np.concatenate(outs); rprob = np.concatenate(routs); p = prob.argmax(1)
        acc = accuracy_score(y[te], p); f1 = f1_score(y[te], p, average='macro'); res[t.name] = dict(acc=acc, f1=f1)
        probs[t.name] = dict(prob=prob.astype(np.float32), y=y[te], te=te, ref_prob=rprob, ref_idx=rc)
        print(f'{t.name:28s} acc {acc:.3f} f1 {f1:.3f}  ({time.time()-t0:.0f}s)', flush=True); del model; torch.cuda.empty_cache()
    accs = [r['acc'] for r in res.values()]; print(f'MEAN acc {np.mean(accs):.3f}  min {np.min(accs):.3f}')
    name = f'full_{a.protocol}_dasheng{"_frozen" if a.frozen else "_lora"}{"_norefnorm" if a.no_refnorm else "_refnorm"}_{a.seconds}s{a.tag}_s{a.seed}.json'
    json.dump(dict(args=vars(a), classes=classes, results=res), open(os.path.join(ROOT, 'outputs', name), 'w'), indent=1); print('saved outputs/' + name)
    np.savez(os.path.join(ROOT, 'outputs', name.replace('.json', '_probs.npz')), **{k + '/' + kk: v for k, d in probs.items() for kk, v in d.items() if v is not None})
