# Data preparation

No dataset and no pretrained checkpoint is redistributed with this repository. All corpora are public and have to be
obtained from their authors. `scripts/verify_results.py` does **not** need any of them: it works on the result files
in `outputs/`.

## 1. Folder layout

By default everything is expected below the repository root:

```
REF-Path/
├── data/
│   ├── Guilin/                 the five zip archives of the Guilin dataset, as downloaded (do not extract)
│   ├── AHU/
│   │   └── Raw data/
│   │       ├── PM/…/*.mat      parabolic-mirror microphone, e.g. PM_L05_r900_OR01.mat
│   │       └── DM/…/*.mat      direct microphone,           e.g. DM_L05_r900_OR01.mat
│   ├── DCASE2026/
│   │   └── <machine>/{train,test}/*.wav      fan, bearingEmu, gearboxEmu, sliderEmu, valveEmu, ToyCar, ToyCarEmu
│   ├── wind_blade/
│   │   └── iid.zip             contains iid/X.npy, iid/y.npy, iid/splits.npz
│   └── MSU/Dataset/…           optional (exploratory corpus, not used in the paper)
├── checkpoints/
│   └── BEATs_iter3_plus_AS2M.pt
└── cache/                      created by the scripts (about 30 GB for the full reproduction)
```

Every location can be changed with an environment variable (`refpath/paths.py`):

| Variable | Meaning | Default |
|---|---|---|
| `REFPATH_DATA` | root folder of the raw datasets | `<repo>/data` |
| `REFPATH_GUILIN` | folder with the five Guilin zip archives | `$REFPATH_DATA/Guilin` |
| `REFPATH_AHU` | AHU folder that contains `PM/` and `DM/` | `$REFPATH_DATA/AHU/Raw data` |
| `REFPATH_DCASE` | DCASE 2026 Task 2 development set, one folder per machine | `$REFPATH_DATA/DCASE2026` |
| `REFPATH_WINDBLADE` | folder with `iid.zip` | `$REFPATH_DATA/wind_blade` |
| `REFPATH_MSU` | optional MSU corpus (folder that contains `Dataset/`) | `$REFPATH_DATA/MSU` |
| `REFPATH_BEATS_CKPT` | BEATs checkpoint file, or a folder that contains it | `<repo>/checkpoints` |
| `REFPATH_DEVICE` | device of `scripts/make_fig_tsne.py` only (`cuda` or `cpu`) | `cuda` |
| `PY` | Python interpreter used by the shell scripts | `python` |

Caches are always written to `<repo>/cache`; link the folder to another disk if needed.

## 2. Datasets

### Guilin multi-condition acoustic bearing dataset (benchmark)

C. Xu, X. Wang, J. Huang, Y. Lu, H. Su, J. Lu. A multi-condition acoustic dataset of ball bearings for fault diagnosis.
Data in Brief 67:112919, 2026. <https://doi.org/10.1016/j.dib.2026.112919>

- 6200 deep-groove ball bearings in five states: normal (`nm`), inner-race pitting (`Ids`), outer-race pitting (`Ods`),
  compound inner and outer pitting (`IOds`), cage fracture (`lc`).
- Three synchronous microphones (channel 0: CRY333 electret, channels 1 and 2: two B&K 4966), three speeds (800, 1000,
  1200 rpm), three loads (0, 15, 30 %), 32,768 Hz, five 10-min segments per condition: 225 recordings in five zip archives.
- The recordings are BK Connect containers (`*.bkc`). They are **not** extracted: `refpath/data/datasets.py` lists the
  archives with `unzip -Z1` and streams the needed part of an entry with `unzip -p` (`_guilin_stream`: 92-byte part
  header, float32 samples in blocks of 4096 per channel). The `unzip` command therefore has to be on the `PATH`
  (Linux/macOS: package `unzip`; Windows: Git Bash provides it). The CRY333 channel is converted from volts to pascals
  with its sensitivity of 23.98 mV/Pa.
- Any archive name is accepted; entries are recognized by the pattern `<state><rpm>r-<load>%-32Khz.<segment>.bkc`.

### AHU Parabolic Acoustic Mirror Bearing Dataset (analysis)

L. Peng, F. Liu. AHU Parabolic Acoustic Mirror Bearing Dataset. IEEE DataPort, 2026. <https://doi.org/10.21227/jm8w-zw54>

- Two sensing paths (parabolic mirror `PM`, direct microphone `DM`), a healthy state and nine faults (ball, inner ring,
  outer ring × pitting, crack, wear), three speeds (600, 900, 1200 rpm), 20 kHz, 25 s per file, MATLAB key `Datas`.
- Files are found recursively below `PM/` and `DM/`; the name has to end in `_r<rpm>_<code>.mat` with the code `N`,
  `B01`–`B03`, `IR01`–`IR03` or `OR01`–`OR03`.
- Each 25-s file is cut into five virtual 5-s segments (training 1–3, reference 1, test 5).

### DCASE 2026 Task 2 development set (cross-corpus check)

T. Nishida et al. Description and discussion on DCASE 2026 Challenge Task 2: Noise-aware unsupervised anomalous sound
detection for machine condition monitoring. arXiv:2606.01578. <https://doi.org/10.48550/arXiv.2606.01578>

- Extract the seven development archives so that every machine type has its own folder with `train/` and `test/`.
- File names follow `section_00_{source|target}_{train|test}_{normal|anomaly}_<id>_….wav`; 10 s, 16 kHz, two channels
  (channel 0: near microphone, channel 1: far microphone).

### Wind-turbine blade acoustic corpus (cross-corpus check)

- `iid.zip` with `iid/X.npy` (24,062 windows × 3 channels × 5,120 samples, −10 dB signal-to-noise ratio), `iid/y.npy`
  (five classes) and `iid/splits.npz` (`dict_users_train`, `dict_users_test`: ten clients). `dirichlet_eps0.2.zip` is
  optional (`python scripts/windblade.py --split dirichlet_eps0.2`).

### MSU multi-device corpus (optional, not used in the paper)

Zenodo record 18971675 (CC BY 4.0). `scripts/msu_experiment.py` runs an exploratory cross-device check when
`$REFPATH_MSU/Dataset` exists; the stage is skipped otherwise.

## 3. Pretrained encoders

- **BEATs**: download `BEATs_iter3_plus_AS2M.pt` (BEATs iter3+, AudioSet-2M, pre-trained) from
  <https://github.com/microsoft/unilm/tree/master/beats> and place it in `checkpoints/`, or point `REFPATH_BEATS_CKPT`
  to the file. The model code is vendored in `refpath/models/beats/` (see `THIRD_PARTY_NOTICES.md`).
- **Dasheng**: `pip install dasheng==0.0.9`; `dasheng_base()` downloads its checkpoint on first use.

## 4. Caches

The caches are built on first use and re-used afterwards. Delete a file to rebuild it.

| Cache | Built by | Content | Size |
|---|---|---|---|
| `cache/guilin/<state>_<rpm>_<load>_<seg>.npy` | `scripts/run_baselines.py guilin` (first call of `get_recordings('guilin')`) | first 60 s of segments 1, 3, 5, three channels, float32 | 3.0 GB |
| `cache/guilin_feats.npz`, `cache/ahu_feats.npz` | `scripts/run_baselines.py guilin` / `ahu` | log-PSD (128) and cyclostationary volume (32 × 100 × 4) per 1-s window, 0.5-s hop | 2.5 GB, 0.14 GB |
| `cache/guilin_beats.npz` | `scripts/beats_embed.py guilin` | frozen BEATs embeddings | 0.08 GB |
| `cache/guilin_cyclo_fixedalpha.npy` | `scripts/ablate_order_axis.py` | cyclostationary map with the cyclic axis in hertz | 0.6 GB |
| `cache/guilin_wave16k_raw.npz`, `cache/ahu_wave16k_raw.npz` | `scripts/train_pathlora.py guilin` / `ahu` | 16-kHz 1-s windows, 1-s hop, float16 | 0.8 GB |
| `cache/guilin_logmel_raw.npy` | `scripts/baseline_crossmodal.py` | log-mel images | 0.6 GB |
| `cache/guilin_full/*.npy`, `cache/guilin_full_feats.npz` | `scripts/build_full_cache.py` | full 10-min recordings of segments 1, 3, 5 (float16) and their features | 15 GB, 3.4 GB |
| `cache/guilin_full_wave16k_raw_180s.npz` | `scripts/train_pathlora2.py` | 16-kHz windows of the first 180 s of every full recording | 2.3 GB |
| `cache/guilin_beats_globalgain.npy`, `cache/guilin_full_beats_180s_globalgain.npy` | `scripts/sota_wave.py --method genrep` | frozen BEATs embeddings for the BEATs-kNN baseline | 0.3 GB |
| `cache/dcase_beats*.npz`, `cache/wb_*`, `cache/msu_feats.npz` | `scripts/dcase_*.py`, `scripts/windblade.py`, `scripts/msu_experiment.py` | cross-corpus checks | 1.6 GB (+ 0.9 GB with MSU) |

Segments and windows: the nine-cell protocol uses the first 60 s of segments 1, 3 and 5 (48,195 one-second windows with
0.5-s hop); the 27-cell protocol uses the full recordings (485,595 windows). The foundation-model branches and the
waveform-level baselines use 1-s windows with 1-s hop, on the 27 cells from the first 180 s of every recording.
