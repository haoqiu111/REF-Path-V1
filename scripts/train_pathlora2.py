"""BEATs branch (reference-normalized fbank + reference gain + LoRA) on the FULL-cache protocols.
Builds cache/guilin_full_wave16k_raw.npz from cache/guilin_full/*.npy using the first --seconds of each recording
(default 180 s; 1-s windows, 1-s hop), then reuses train_pathlora.run_task.

python scripts/train_pathlora2.py --protocol mic_x_speed_x_load --epochs 3 --seed 0 [--seconds 180] [--tasks ...]
"""
import os, sys, time, argparse, json, glob, numpy as np, scipy.signal as sg
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from refpath.data.datasets import GUILIN_STATE_MAP, GUILIN_PATHS
from refpath.eval.protocol import make_tasks
import train_pathlora as TP
FULL = os.path.join(ROOT, 'cache', 'guilin_full'); fs = 32768


def build(seconds):
    out = os.path.join(ROOT, 'cache', f'guilin_full_wave16k_raw_{seconds}s.npz')
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True); return z['meta'], z['wave']
    metas, waves = [], []; t0 = time.time(); files = sorted(glob.glob(os.path.join(FULL, '*.npy')))
    for i, fn in enumerate(files):
        st, rpm, load, seg = os.path.basename(fn)[:-4].split('_'); rpm, load, seg = int(rpm), int(load), int(seg)
        X = np.load(fn, mmap_mode='r')
        for ch in range(3):
            x = np.asarray(X[ch, :seconds * fs], np.float32) / 10.0
            x16 = sg.resample_poly(x, 125, 256).astype(np.float32) * 10.0
            for s in range(0, len(x16) - 16000 + 1, 16000):
                waves.append(x16[s:s + 16000].astype(np.float16)); metas.append(('guilin', GUILIN_PATHS[ch], rpm, load, GUILIN_STATE_MAP[st], seg, s / 16000))
        if i % 20 == 0: print(f'wave cache {i}/{len(files)} {time.time()-t0:.0f}s', flush=True)
    meta = np.array(metas, dtype=[('dataset', 'U8'), ('path', 'U12'), ('rpm', 'i4'), ('load', 'i4'), ('state', 'U24'), ('seg', 'i4'), ('start', 'f4')])
    wave = np.stack(waves); np.savez(out, meta=meta, wave=wave); return meta, wave


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', default='mic_x_speed_x_load'); ap.add_argument('--tasks', default=''); ap.add_argument('--seconds', type=int, default=180); ap.add_argument('--ref_seconds', type=float, default=60)
    ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--bs', type=int, default=32); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    # fixed choices: plain LoRA (no hypernetwork), reference-normalized fbank, reference gain, split reference, multilabel, gain aug
    for k, v in dict(multilabel=True, aug=True, no_hyper=True, ref_norm=True, ref_split=True, mixup=0.0, desc='psd').items(): setattr(a, k, v)
    meta, wave = build(a.seconds); psd = np.zeros((len(meta), 128), np.float32)     # descriptor unused (no_hyper)
    classes = sorted(np.unique(meta['state']).tolist()); tasks = make_tasks(meta, a.protocol, ref_seconds=a.ref_seconds)
    if a.tasks: tasks = [t for t in tasks if t.name in a.tasks.split(',')]
    print(f'{len(tasks)} tasks, {len(meta)} windows', flush=True); res = {}; probs = {}; t0 = time.time()
    for t in tasks:
        acc, f1, cm = TP.run_task(meta, wave, psd, t, classes, a); res[t.name] = dict(acc=acc, f1=f1, cm=cm.tolist()); probs[t.name] = TP.run_task.last
        print(f'{t.name:28s} acc {acc:.3f} f1 {f1:.3f}  ({time.time()-t0:.0f}s)', flush=True)
    accs = [r['acc'] for r in res.values()]; print(f'MEAN acc {np.mean(accs):.3f}  min {np.min(accs):.3f}')
    name = f'full_{a.protocol}_pathlora_refnorm_refgain_{a.seconds}s{a.tag}_s{a.seed}.json'
    json.dump(dict(args=vars(a), classes=classes, results=res), open(os.path.join(ROOT, 'outputs', name), 'w'), indent=1); print('saved outputs/' + name)
    np.savez(os.path.join(ROOT, 'outputs', name.replace('.json', '_probs.npz')), **{k + '/' + kk: v for k, d in probs.items() for kk, v in d.items()})
