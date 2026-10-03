"""DCASE 2026 Task 2 (noise-aware first-shot ASD) development set.

Layout (after extraction to $REFPATH_DCASE, default <repo>/data/DCASE2026):
  <machine>/train/section_00_{source|target}_train_normal_XXXX_noAttribute.wav   (990 source + 10 target)
  <machine>/test/section_00_{source|target}_test_{normal|anomaly}_XXXX_noAttribute.wav (50 each)
Each wav: 10 s, 16 kHz, 2 channels: ch0 = near (machine) microphone, ch1 = far (noise-reference) microphone.

In REF-Path terms a *sensing path* here is {near, far} and the healthy reference of a domain is the train-normal set.
"""
from __future__ import annotations
import os, glob, re, numpy as np, soundfile as sf
from refpath import paths

DCASE_ROOT = paths.DCASE_ROOT
MACHINES = ['fan', 'bearingEmu', 'gearboxEmu', 'sliderEmu', 'valveEmu', 'ToyCar', 'ToyCarEmu']
PATHS = ['near', 'far']


def dcase_index(root=DCASE_ROOT, machines=MACHINES):
    """Returns list of dicts: machine, split, domain, label (0 normal / 1 anomaly), path to wav."""
    idx = []
    for m in machines:
        for fn in sorted(glob.glob(os.path.join(root, m, '*', '*.wav'))):
            r = re.search(r'section_(\d+)_(source|target)_(train|test)_(normal|anomaly)_(\d+)', os.path.basename(fn))
            idx.append(dict(machine=m, section=int(r.group(1)), domain=r.group(2), split=r.group(3),
                            label=int(r.group(4) == 'anomaly'), idx=int(r.group(5)), file=fn))
    return idx


def read_wav(fn, channel=None):
    x, fs = sf.read(fn, dtype='float32')
    assert fs == 16000 and x.ndim == 2 and x.shape[1] == 2, (fn, fs, x.shape)
    return (x if channel is None else x[:, channel]), fs
