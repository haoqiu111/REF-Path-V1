"""Frozen BEATs + (path-conditioned) LoRA under the Reference-Only protocols.

python scripts/train_pathlora.py guilin --protocol mic_x_speed --epochs 4 [--tasks mxs:CRY333@1200] [--no_hyper]

Path descriptor d (128-d) of a domain = mean log-PSD of that domain's healthy reference windows minus the canonical
reference (mean over all training-domain healthy windows). Training windows get the descriptor of their own domain;
test windows get the descriptor computed from the test domain's healthy reference segment only.
Waveforms are resampled to 16 kHz, 1-s windows, hop 1 s, cached as float16.
"""
import os, sys, time, argparse, json, glob, numpy as np, torch, torch.nn.functional as F_, scipy.signal as sg
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.paths import beats_checkpoint
from refpath.data.datasets import get_recordings, meta_table
from refpath.features.frontend import log_psd
from refpath.eval.protocol import make_tasks, domain_key
from refpath.models.beats.BEATs import BEATs, BEATsConfig
from refpath.models.path_lora import PathLoRAModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
CKPT = beats_checkpoint()                    # $REFPATH_BEATS_CKPT or <repo>/checkpoints/*.pt
PRIM = {'healthy': [], 'inner_pitting': ['inner'], 'outer_pitting': ['outer'], 'inner_outer_pitting': ['inner', 'outer'],
        'cage_fracture': ['cage'], 'ball_pitting': ['ball'], 'ball_crack': ['ball'], 'ball_wear': ['ball'],
        'inner_crack': ['inner'], 'inner_wear': ['inner'], 'outer_crack': ['outer'], 'outer_wear': ['outer']}
PRIMS = ['inner', 'outer', 'ball', 'cage']


def build_wave_cache(ds, seconds=60, hop=1.0):
    out = os.path.join(ROOT, 'cache', f'{ds}_wave16k_raw.npz')
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True); return z['meta'], z['wave'], z['psd']
    recs = get_recordings(ds); metas, waves, psds = [], [], []; t0 = time.time()
    for i, r in enumerate(recs):
        x = r.read(seconds); up, down = (125, 256) if r.fs == 32768 else (4, 5)
        x16 = sg.resample_poly(x, up, down).astype(np.float32) * 10.0          # raw amplitude x10 (float16-safe); gain applied later from the REFERENCE
        W, H = 16000, int(hop * 16000)
        for s in range(0, len(x16) - W + 1, H):
            waves.append(x16[s:s + W].astype(np.float16)); psds.append(log_psd(x16[s:s + W], 16000, n_bands=128, nperseg=1024)); metas.append(r)
        if i % 50 == 0: print(f'{ds}: {i}/{len(recs)} {time.time()-t0:.0f}s', flush=True)
    meta = meta_table(metas); wave = np.stack(waves); psd = np.stack(psds); np.savez(out, meta=meta, wave=wave, psd=psd); return meta, wave, psd


def reference_gain(wave, meta, task):
    """Per-window scale factor so that the healthy REFERENCE of each domain has RMS 0.05 (deployable: uses reference only;
    source domains use their training healthy windows). The gain is never computed from the test recording itself."""
    keys = domain_key(meta); healthy = meta['state'] == 'healthy'; g = np.ones(len(meta), np.float32)
    for k in np.unique(keys):
        d = keys == k
        ref = d & task.ref if (d & task.test).any() else d & train_healthy(task, healthy)
        if ref.sum() == 0: ref = d & healthy
        rms = np.sqrt(np.mean(wave[np.where(ref)[0][:200]].astype(np.float32) ** 2)) + 1e-8; g[d] = 0.05 / rms
    return g


def train_healthy(task, healthy):
    return task.train & healthy


def descriptors(psd, meta, task):
    keys = domain_key(meta); healthy = meta['state'] == 'healthy'; D = np.zeros_like(psd)
    canon = psd[task.train & healthy].mean(0)
    for k in np.unique(keys):
        d = keys == k
        ref = d & task.ref if (d & task.test).any() else d & task.train & healthy
        if ref.sum() == 0: ref = d & healthy
        D[d] = psd[ref].mean(0) - canon
    return D.astype(np.float32)


def ref_fbank_stats(wave, meta, task, dev='cuda'):
    """Per-domain mean/std of BEATs-normalized fbank over the healthy reference windows -> (N, 128) each."""
    from refpath.features.torch_frontend import beats_input
    keys = domain_key(meta); healthy = meta['state'] == 'healthy'; MU = np.zeros((len(meta), 128), np.float32); SD = np.ones((len(meta), 128), np.float32)
    for k in np.unique(keys):
        d = keys == k
        ref = d & task.ref if (d & task.test).any() else d & task.train & healthy
        if ref.sum() == 0: ref = d & healthy
        idx = np.where(ref)[0][:200]
        with torch.no_grad():
            fb = beats_input(torch.tensor(wave[idx]).float().to(dev)).reshape(-1, 128)
        MU[d] = fb.mean(0).cpu().numpy(); SD[d] = fb.std(0).cpu().numpy()
    return MU, SD


def run_task(meta, wave, psd, task, classes, args, dev='cuda', desc_src=None):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    task_n = task; ref_cal = task.ref
    if args.ref_split:   # even reference windows -> normalization statistics, odd -> calibration / reliability
        from refpath.eval.protocol import Task
        rid = np.where(task.ref)[0]; ref_norm = np.zeros_like(task.ref); ref_norm[rid[::2]] = True; ref_cal = np.zeros_like(task.ref); ref_cal[rid[1::2]] = True
        task_n = Task(task.name, task.train, task.test, ref_norm)
    gain = reference_gain(wave, meta, task_n); wave = (wave.astype(np.float32) * gain[:, None]).astype(np.float16)
    MU, SD = ref_fbank_stats(wave, meta, task_n) if args.ref_norm else (None, None)
    D = descriptors(desc_src if desc_src is not None else psd, meta, task_n); D = D / (D.std() + 1e-6)
    if args.no_hyper: D = np.zeros_like(D)
    y = np.array([classes.index(s) for s in meta['state']]); P = np.array([[1.0 if p in PRIM[s] else 0.0 for p in PRIMS] for s in meta['state']], np.float32)
    tr, te = np.where(task.train)[0], np.where(task.test)[0]
    c = torch.load(CKPT, map_location='cpu', weights_only=False); b = BEATs(BEATsConfig(c['cfg'])); b.load_state_dict(c['model'])
    model = PathLoRAModel(b, n_classes=len(classes), n_prim=len(PRIMS) if args.multilabel else 0, r=args.rank, d_dim=D.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * ((len(tr) + args.bs - 1) // args.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    Wt = torch.tensor(wave[tr]); Dt = torch.tensor(D[tr]); yt = torch.tensor(y[tr]); Pt = torch.tensor(P[tr])
    MUt = torch.tensor(MU[tr]) if MU is not None else None; SDt = torch.tensor(SD[tr]) if SD is not None else None
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb = Wt[idx].float().to(dev); db = Dt[idx].to(dev); yb = yt[idx].to(dev); pb = Pt[idx].to(dev)
            if args.aug: xb = xb * (10 ** (0.2 * torch.randn(xb.shape[0], 1, device=dev)))      # gain aug
            mub = MUt[idx].to(dev) if MUt is not None else None; sdb = SDt[idx].to(dev) if SDt is not None else None
            yb1 = F_.one_hot(yb, len(classes)).float(); lam = None
            if args.mixup > 0:
                j = torch.randperm(len(idx), device=dev); lam = torch.distributions.Beta(args.mixup, args.mixup).sample((len(idx),)).to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if lam is None:
                    out = model(xb, db, mub, sdb)
                else:
                    out = model.forward_mix(xb, db, mub, sdb, j, lam)
                    yb1 = lam[:, None] * yb1 + (1 - lam[:, None]) * yb1[j]; pb = lam[:, None] * pb + (1 - lam[:, None]) * pb[j]
            loss = -(yb1 * F_.log_softmax(out['logits'].float(), 1)).sum(1).mean()
            if args.multilabel: loss = loss + 0.5 * F_.binary_cross_entropy_with_logits(out['prim_logits'].float(), pb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval(); preds = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(te), 256):
            idx = te[i:i + 256]; preds.append(F_.softmax(model(torch.tensor(wave[idx]).float().to(dev), torch.tensor(D[idx]).to(dev), torch.tensor(MU[idx]).to(dev) if MU is not None else None, torch.tensor(SD[idx]).to(dev) if SD is not None else None)['logits'].float(), 1).cpu().numpy())
    prob = np.concatenate(preds); p = prob.argmax(1)
    rf = np.where(ref_cal)[0]; rpreds = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(rf), 256):
            idx = rf[i:i + 256]; rpreds.append(F_.softmax(model(torch.tensor(wave[idx]).float().to(dev), torch.tensor(D[idx]).to(dev), torch.tensor(MU[idx]).to(dev) if MU is not None else None, torch.tensor(SD[idx]).to(dev) if SD is not None else None)['logits'].float(), 1).cpu().numpy())
    run_task.last = dict(prob=prob, y=y[te], te=te, ref_prob=np.concatenate(rpreds), ref_idx=rf); del model, b; torch.cuda.empty_cache()
    return accuracy_score(y[te], p), f1_score(y[te], p, average='macro'), confusion_matrix(y[te], p, labels=range(len(classes)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset'); ap.add_argument('--protocol', default='mic_x_speed'); ap.add_argument('--tasks', default='')
    ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--bs', type=int, default=32); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--rank', type=int, default=8); ap.add_argument('--multilabel', action='store_true'); ap.add_argument('--aug', action='store_true')
    ap.add_argument('--no_hyper', action='store_true', help='plain LoRA: zero descriptor'); ap.add_argument('--desc', default='psd', help='psd | beats (frozen-BEATs reference embedding difference)'); ap.add_argument('--ref_norm', action='store_true'); ap.add_argument('--ref_split', action='store_true'); ap.add_argument('--mixup', type=float, default=0.0); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    meta, wave, psd = build_wave_cache(a.dataset); print('windows', wave.shape)
    desc_src = None
    if a.desc == 'beats':
        zb = np.load(os.path.join(ROOT, 'cache', f'{a.dataset}_beats.npz'), allow_pickle=True); assert (zb['meta']['state'] == meta['state']).all(); desc_src = zb['emb'].astype(np.float32)
    classes = sorted(np.unique(meta['state']).tolist()); tasks = make_tasks(meta, a.protocol)
    if a.tasks: tasks = [t for t in tasks if t.name in a.tasks.split(',')]
    res = {}; probs = {}; t0 = time.time()
    for t in tasks:
        acc, f1, cm = run_task(meta, wave, psd, t, classes, a, desc_src=desc_src)
        res[t.name] = dict(acc=acc, f1=f1, cm=cm.tolist()); probs[t.name] = run_task.last; print(f'{t.name:22s} acc {acc:.3f} f1 {f1:.3f}  ({time.time()-t0:.0f}s)', flush=True)
    accs = [r['acc'] for r in res.values()]; print(f'MEAN acc {np.mean(accs):.3f}  min {np.min(accs):.3f}')
    name = f'{a.dataset}_{a.protocol}_pathlora{"_plain" if a.no_hyper else ""}{"_ml" if a.multilabel else ""}{"_aug" if a.aug else ""}{"_desc" + a.desc if a.desc != "psd" else ""}{"_refnorm" if a.ref_norm else ""}{"_mix" + str(a.mixup) if a.mixup > 0 else ""}{"_refsplit" if a.ref_split else ""}{a.tag}.json'
    json.dump(dict(args=vars(a), classes=classes, results=res), open(os.path.join(ROOT, 'outputs', name), 'w'), indent=1); print('saved outputs/' + name)
    np.savez(os.path.join(ROOT, 'outputs', name.replace('.json', '_probs.npz')), **{k + '/' + kk: v for k, d in probs.items() for kk, v in d.items()})
