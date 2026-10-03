"""Dataset registry for REF-Path.

Every dataset exposes a list of Recording objects with a uniform metadata schema so that protocols can be
expressed as boolean masks over (dataset, path, rpm, load, state, segment).

  path   : sensing path id, e.g. 'PM'/'DM' (AHU), 'CRY333'/'BK4966_2'/'BK4966_3' (Guilin), 'near'/'far' (DCASE)
  rpm    : shaft speed in rev/min (int)
  load   : load level (int, dataset specific; 0 if unknown)
  state  : fault label string; 'nm'/'N' style healthy labels are mapped to 'healthy'
  seg    : recording segment index (int) used for time-disjoint train/test splits
"""
from __future__ import annotations
import os, re, glob, subprocess, struct
from dataclasses import dataclass, field
import numpy as np
import scipy.io as sio
from refpath import paths

HEALTHY = 'healthy'


@dataclass
class Recording:
    dataset: str
    path: str
    rpm: int
    load: int
    state: str
    seg: int
    fs: int
    loader: object = field(repr=False)          # callable -> np.ndarray (n_samples,) float32
    uid: str = ''

    def read(self, seconds=None):
        x = self.loader()
        if seconds is not None:
            x = x[: int(seconds * self.fs)]
        return x.astype(np.float32)


# ----------------------------------------------------------------------------------------------------
# AHU parabolic acoustic mirror dataset (PM / DM), 20 kHz, 25 s, one .mat per (path, state, rpm)
# ----------------------------------------------------------------------------------------------------
AHU_ROOT = paths.AHU_ROOT                    # folder that contains PM/ and DM/
AHU_STATE_MAP = {'N': HEALTHY, 'B01': 'ball_pitting', 'B02': 'ball_crack', 'B03': 'ball_wear',
                 'IR01': 'inner_pitting', 'IR02': 'inner_crack', 'IR03': 'inner_wear',
                 'OR01': 'outer_pitting', 'OR02': 'outer_crack', 'OR03': 'outer_wear'}


def ahu_recordings(root=AHU_ROOT, n_seg=5):
    """Each 25-s file is virtually cut into n_seg 5-s segments so that time-disjoint splits are possible."""
    recs = []
    for path in ['PM', 'DM']:
        for fn in sorted(glob.glob(os.path.join(root, path, '**', '*.mat'), recursive=True)):
            m = re.search(r'_r(\d+)_(N|B0\d|IR0\d|OR0\d)\.mat$', fn)
            rpm, st = int(m.group(1)), AHU_STATE_MAP[m.group(2)]
            fs = 20000
            for s in range(n_seg):
                def _load(fn=fn, s=s, fs=fs, n_seg=n_seg):
                    x = sio.loadmat(fn)['Datas'].ravel().astype(np.float32)
                    L = len(x) // n_seg
                    return x[s * L:(s + 1) * L]
                recs.append(Recording('ahu', path, rpm, 5, st, s + 1, fs, _load, uid=f'ahu/{path}/{rpm}/{st}/{s+1}'))
    return recs


# ----------------------------------------------------------------------------------------------------
# Guilin multi-condition 3-microphone dataset (.bkc inside 5 zips), 32768 Hz, 10 min per file
# ----------------------------------------------------------------------------------------------------
GUILIN_ZIPDIR = paths.GUILIN_ZIPDIR          # folder with the five zip archives
GUILIN_CACHE = os.path.join(paths.CACHE, 'guilin')
GUILIN_STATE_MAP = {'nm': HEALTHY, 'Ids': 'inner_pitting', 'Ods': 'outer_pitting', 'IOds': 'inner_outer_pitting', 'lc': 'cage_fracture'}
GUILIN_PATHS = ['CRY333', 'BK4966_2', 'BK4966_3']
GUILIN_CRY_SENS = 0.02398  # V/Pa


def _guilin_index(zipdir=GUILIN_ZIPDIR):
    idx = []
    for z in sorted(glob.glob(os.path.join(zipdir, '*.zip'))):
        names = subprocess.run(['unzip', '-Z1', z], capture_output=True, text=True).stdout.split('\n')
        for name in names:
            m = re.match(r'.*/(nm|Ids|Ods|IOds|lc)(\d+)r-(\d+)%?-32Khz\.(\d)\.bkc$', name)
            if m:
                idx.append((z, name, m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return idx


def _guilin_stream(z, name, seconds, fs=32768, block=4096, nch=3):
    """Stream the first `seconds` of a .bkc entry directly from the zip (no full extraction)."""
    nblocks = int(np.ceil(seconds * fs / block))
    need = 15437 + 92 + 262144 + nblocks * nch * block * 4
    p = subprocess.Popen(['unzip', '-p', z, name], stdout=subprocess.PIPE)
    buf = bytearray()
    while len(buf) < need:
        chunk = p.stdout.read(min(1 << 20, need - len(buf)))
        if not chunk:
            break
        buf += chunk
    p.stdout.close(); p.kill()
    b = bytes(buf)
    i = b.find(b'\x06\x00\x2e\x00P\x00a\x00r\x00t\x000\x00')
    off = struct.unpack('<Q', b[i + 14:i + 22])[0]
    h = np.frombuffer(b[off:off + 32], dtype='<u4'); assert h[6] == block and h[7] == nch, h
    nb = min(nblocks, (len(b) - off - 92) // (nch * block * 4))
    a = np.frombuffer(b[off + 92:off + 92 + nb * nch * block * 4], dtype='<f4')
    X = a.reshape(nb, nch, block).transpose(1, 0, 2).reshape(nch, -1)[:, :int(seconds * fs)].copy()
    X[0] /= GUILIN_CRY_SENS
    return X


def guilin_build_cache(seconds=60, segs=(1, 3, 5), cache=GUILIN_CACHE, verbose=True):
    os.makedirs(cache, exist_ok=True)
    idx = _guilin_index()
    for k, (z, name, st, rpm, load, seg) in enumerate(idx):
        if seg not in segs:
            continue
        out = os.path.join(cache, f'{st}_{rpm}_{load}_{seg}.npy')
        if os.path.exists(out):
            continue
        np.save(out, _guilin_stream(z, name, seconds).astype(np.float32))
        if verbose and k % 20 == 0:
            print(f'guilin cache {k}/{len(idx)}', flush=True)


def guilin_recordings(cache=GUILIN_CACHE, segs=(1, 3, 5)):
    recs = []
    for fn in sorted(glob.glob(os.path.join(cache, '*.npy'))):
        st, rpm, load, seg = os.path.basename(fn)[:-4].split('_')
        rpm, load, seg = int(rpm), int(load), int(seg)
        if seg not in segs:
            continue
        for ch, path in enumerate(GUILIN_PATHS):
            def _load(fn=fn, ch=ch):
                return np.load(fn, mmap_mode='r')[ch]
            recs.append(Recording('guilin', path, rpm, load, GUILIN_STATE_MAP[st], seg, 32768, _load,
                                  uid=f'guilin/{path}/{rpm}/{load}/{st}/{seg}'))
    return recs


# ----------------------------------------------------------------------------------------------------
def get_recordings(name):
    if name == 'ahu':
        return ahu_recordings()
    if name == 'guilin':
        if not glob.glob(os.path.join(GUILIN_CACHE, '*.npy')):
            guilin_build_cache()
        return guilin_recordings()
    raise KeyError(name)


def meta_table(recs):
    """Structured numpy record array of metadata (for fast boolean masking)."""
    return np.array([(r.dataset, r.path, r.rpm, r.load, r.state, r.seg) for r in recs],
                    dtype=[('dataset', 'U8'), ('path', 'U12'), ('rpm', 'i4'), ('load', 'i4'), ('state', 'U24'), ('seg', 'i4')])
