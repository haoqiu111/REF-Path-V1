"""Locations of the raw datasets, the caches and the pretrained BEATs checkpoint.

Everything is resolved relative to the repository root unless an environment variable overrides it:

  REFPATH_DATA        root folder of the raw datasets                       (default: <repo>/data)
  REFPATH_GUILIN      folder with the five Guilin zip archives              (default: $REFPATH_DATA/Guilin)
  REFPATH_AHU         AHU folder that contains PM/ and DM/                  (default: $REFPATH_DATA/AHU/Raw data)
  REFPATH_DCASE       DCASE 2026 Task 2 development set, one folder/machine (default: $REFPATH_DATA/DCASE2026)
  REFPATH_WINDBLADE   folder with iid.zip (and dirichlet_eps0.2.zip)        (default: $REFPATH_DATA/wind_blade)
  REFPATH_MSU         optional MSU multi-device corpus                      (default: $REFPATH_DATA/MSU)
  REFPATH_BEATS_CKPT  BEATs checkpoint file, or a folder that contains it   (default: <repo>/checkpoints)

Caches are written to <repo>/cache and results to <repo>/outputs (see docs/DATA.md).
"""
import os, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('REFPATH_DATA', os.path.join(ROOT, 'data'))
CACHE = os.path.join(ROOT, 'cache')

GUILIN_ZIPDIR = os.environ.get('REFPATH_GUILIN', os.path.join(DATA, 'Guilin'))
AHU_ROOT = os.environ.get('REFPATH_AHU', os.path.join(DATA, 'AHU', 'Raw data'))
DCASE_ROOT = os.environ.get('REFPATH_DCASE', os.path.join(DATA, 'DCASE2026'))
WINDBLADE_ROOT = os.environ.get('REFPATH_WINDBLADE', os.path.join(DATA, 'wind_blade'))
MSU_ROOT = os.environ.get('REFPATH_MSU', os.path.join(DATA, 'MSU'))


def beats_checkpoint():
    """Path of the pretrained BEATs checkpoint (BEATs_iter3_plus_AS2M.pt). REFPATH_BEATS_CKPT may name the file itself or a
    folder; in a folder the first *.pt file is used."""
    p = os.environ.get('REFPATH_BEATS_CKPT', os.path.join(ROOT, 'checkpoints'))
    if os.path.isfile(p) or p.lower().endswith('.pt'):
        return p
    found = sorted(glob.glob(os.path.join(p, '*.pt')))
    return found[0] if found else os.path.join(p, 'BEATs_iter3_plus_AS2M.pt')
