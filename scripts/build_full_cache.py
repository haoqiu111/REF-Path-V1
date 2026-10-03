"""Stream the FULL 10-minute recordings of segments 1, 3, 5 from the Guilin zips into per-file float16 arrays
(cache/guilin_full/<state>_<rpm>_<load>_<seg>.npy, shape (3, 600*32768)), then extract window features with
multiprocessing: log-PSD (128) and the tau-averaged order-normalized cyclostationary map (32 x 100), 1-s windows,
0.5-s hop -> cache/guilin_full_feats.npz with meta fields incl. window start time.

python scripts/build_full_cache.py [--workers 12] [--seconds 600]
"""
import os, sys, time, argparse, glob, numpy as np
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from refpath.data.datasets import _guilin_index, _guilin_stream, GUILIN_STATE_MAP, GUILIN_PATHS, GUILIN_CRY_SENS
from refpath.features.frontend import log_psd, cyclo_volume
FULL = os.path.join(ROOT, 'cache', 'guilin_full'); fs = 32768


def stream_one(args):
    z, name, st, rpm, load, seg, seconds = args
    out = os.path.join(FULL, f'{st}_{rpm}_{load}_{seg}.npy')
    if os.path.exists(out): return out
    X = _guilin_stream(z, name, seconds)                  # (3, n) float32, ch0 already in Pa
    np.save(out, (X * 10.0).astype(np.float16))            # x10 keeps float16 precision (values ~0.1-1)
    return out


def feats_one(fn):
    st, rpm, load, seg = os.path.basename(fn)[:-4].split('_'); rpm, load, seg = int(rpm), int(load), int(seg)
    X = np.load(fn, mmap_mode='r'); fr = rpm / 60; W, H = fs, fs // 2; n = X.shape[1]
    rows = []
    for ch in range(3):
        x = np.asarray(X[ch], np.float32) / 10.0
        for s in range(0, n - W + 1, H):
            w = x[s:s + W]
            rows.append((GUILIN_PATHS[ch], rpm, load, GUILIN_STATE_MAP[st], seg, s / fs, log_psd(w, fs), cyclo_volume(w, fs, fr).mean(2).astype(np.float16)))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=10); ap.add_argument('--seconds', type=int, default=600); a = ap.parse_args()
    os.makedirs(FULL, exist_ok=True); t0 = time.time()
    idx = [(z, name, st, rpm, load, seg, a.seconds) for (z, name, st, rpm, load, seg) in _guilin_index() if seg in (1, 3, 5)]
    print('streaming', len(idx), 'files', flush=True)
    with Pool(4) as p:                                      # zip streaming is disk-bound; 4 readers
        for i, out in enumerate(p.imap_unordered(stream_one, idx)):
            if i % 15 == 0: print(f'  {i}/{len(idx)} {time.time()-t0:.0f}s', flush=True)
    files = sorted(glob.glob(os.path.join(FULL, '*.npy'))); print('feature extraction over', len(files), 'files', flush=True)
    meta, psd, cyc = [], [], []
    with Pool(a.workers) as p:
        for i, rows in enumerate(p.imap(feats_one, files)):
            for r in rows: meta.append(r[:6]); psd.append(r[6]); cyc.append(r[7])
            if i % 10 == 0: print(f'  {i}/{len(files)} {time.time()-t0:.0f}s windows={len(meta)}', flush=True)
    meta = np.array([('guilin',) + m for m in meta], dtype=[('dataset', 'U8'), ('path', 'U12'), ('rpm', 'i4'), ('load', 'i4'), ('state', 'U24'), ('seg', 'i4'), ('start', 'f4')])
    np.savez(os.path.join(ROOT, 'cache', 'guilin_full_feats.npz'), meta=meta, psd=np.array(psd, np.float32), cyclo=np.array(cyc, np.float16))
    print('saved', len(meta), 'windows', f'{time.time()-t0:.0f}s')
