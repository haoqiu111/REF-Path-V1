# REF-Path: Healthy-Reference Standardization for Acoustic Fault Diagnosis across Sensing Paths and Speeds

Code, benchmark protocols and result files of the paper *"REF-Path: Healthy-Reference Standardization for Acoustic
Fault Diagnosis across Sensing Paths and Speeds"* (under review at the Journal of Sound and Vibration, 2026).

## Summary

REF-Path addresses the **Reference-Only** setting of acoustic fault diagnosis: a new microphone (a new sensing path)
is installed at a new shaft speed, and the only target data is a short healthy recording made at commissioning.
No labelled and no faulty target window is available. The method rests on one input-level operation,
**healthy-reference standardization**: every input is standardized with the statistics of the healthy reference of
its own domain, in the same way at training and at deployment. The operation is applied to an order-normalized
cyclostationary map, whose carrier axis carries the path and whose cyclic-order axis carries the speed (physics
branch, a small MLP with cross-domain mixup), and to the mel input and the input gain of frozen audio foundation
models with LoRA adapters (BEATs, Dasheng). The branch posteriors are fused by a product, and a conformal alarm, a
label-free reference refresh and an unknown-fault score are calibrated on the same reference. On the
three-microphone Guilin corpus, 16 cross-domain, test-time and acoustic methods are re-implemented under the
identical protocol, each on raw inputs and with the healthy reference.

## Main results

Window accuracy on the Guilin corpus, mean ± std over three seeds (excerpt of Table 2 of the paper; all 16 methods
and Tables 1–6 are in [`results/expected_tables.md`](results/expected_tables.md)).

| Method | Target data | 9 cells, raw | 9 cells, + ref. | 27 cells, raw | 27 cells, + ref. |
|---|---|---|---|---|---|
| Linear classifier | R | 0.585 | 0.838 | 0.583 | 0.733 |
| ERM | R | 0.608 ± 0.015 | 0.862 ± 0.001 | 0.564 ± 0.007 | 0.756 ± 0.006 |
| Deep CORAL | R | 0.656 ± 0.014 | 0.850 ± 0.004 | 0.559 ± 0.007 | 0.759 ± 0.003 |
| SAGM | R | 0.623 ± 0.014 | 0.865 ± 0.002 | 0.558 ± 0.010 | 0.760 ± 0.004 |
| AFT | R | 0.626 ± 0.018 | 0.850 ± 0.009 | 0.568 ± 0.013 | 0.777 ± 0.004 |
| **REF-Path (fusion)** | R | n/a | **0.896 ± 0.006** | n/a | **0.876** |
| AdaBN-T | T | 0.924 ± 0.005 | 0.917 ± 0.005 | 0.936 ± 0.005 | 0.940 ± 0.002 |
| TENT | T | 0.925 ± 0.005 | 0.918 ± 0.005 | 0.939 ± 0.005 | 0.943 ± 0.002 |
| SAR | T | 0.925 ± 0.005 | 0.917 ± 0.005 | 0.936 ± 0.005 | 0.940 ± 0.002 |
| REF-Path-G (entropy gate) | R + T | n/a | 0.912 ± 0.005 | n/a | 0.894 |
| Oracle | O | n/a | 0.944 | n/a | 0.917 |

Target data: R, healthy reference only; T, the unlabeled test set (shuffled, class-balanced); O, all target classes
(not deployable). Raw versions use source statistics only. 9 cells: microphone and speed held out; 27 cells:
microphone, speed and load held out. SAGM and AFT with the reference are the strongest Reference-Only competitors
on the nine and on the 27 cells.

How to read the table:

- **Reference-Only methods.** REF-Path reaches 0.896 against 0.865 for SAGM with the reference on the nine cells,
  and 0.876 against 0.777 for AFT with the reference on the 27 cells. On the nine cells it wins 4 of the 9 cells and
  the paired difference over cells is not significant (p = 0.652), while the worst cell rises from 0.63 to 0.78. On
  the 27 cells it wins 24 of the 27 cells (p < 0.001) and the worst cell rises from 0.44 to 0.64.
- **Transductive test-time adaptation is more accurate on a class-balanced test set.** AdaBN-T, TENT and SAR adapt
  on the unlabeled test set. On the shuffled, class-balanced test set they reach 0.925 and 0.937 on raw inputs,
  which is **above** REF-Path (0.896 and 0.876) and, on the 27 cells, above the linear oracle. The advantage depends
  on the composition of the test stream. With all healthy test windows kept and the faulty ones subsampled, the
  three methods reach 0.909 / 0.933 at a healthy share of 0.5, 0.797 / 0.836 at 0.8 (below REF-Path), 0.705 / 0.710
  at 0.95 (below the linear reference baseline) and 0.624 / 0.566 on a healthy-only stream (nine cells / 27 cells).
  REF-Path does not read the test stream, so its accuracy is the same for every composition.
- **Every re-implemented method is reported twice**, on raw inputs and with healthy-reference standardization of
  its input. The reference lifts 9 of 12 non-transductive methods on the nine cells and 8 of 12 on the 27 cells;
  three methods lose accuracy with it on both protocols (AdaBN on the reference, FGDG, BEATs-kNN).

## Repository layout

```
REF-Path/
├── refpath/                  package
│   ├── paths.py              dataset / checkpoint locations (environment variables)
│   ├── data/                 dataset registry (Guilin BK Connect reader, AHU, DCASE 2026)
│   ├── features/             log-PSD, order-normalized cyclostationary map, Kaldi-compatible filterbank
│   ├── eval/protocol.py      Reference-Only tasks and per-domain standardization
│   ├── models/               Cyclo3DNet, LoRA wrapper, vendored BEATs model code
│   └── plot_style.py
├── scripts/                  training, baselines, analysis, figures, verification (one file per experiment)
├── reproduce.sh              full pipeline, stages 0–9; `bash reproduce.sh check` runs the verification only
├── run_sota_feat.sh          queue of the feature-level baselines
├── run_sota_wave.sh          queue of the waveform-level baselines
├── run_sota_tta.sh           queue of the transductive baselines alone
├── results/
│   ├── expected_tables.md    Tables 1–6 of the paper
│   └── tables_2_3.md         Tables 2–3 as regenerated by scripts/make_tables.py
├── outputs/                  result files of every experiment (JSON / CSV, and the posteriors that the checks need)
├── figures/                  the nine figures of the paper (PNG + PDF, Fig. 1 also as editable PPTX)
├── docs/DATA.md              data preparation, folder layout, caches
├── requirements.txt, CITATION.cff, LICENSE, THIRD_PARTY_NOTICES.md
```

## Installation

Python 3.12 or newer is required.

```bash
git clone <repository URL> && cd REF-Path
python -m venv .venv && source .venv/bin/activate        # or a conda environment
pip install torch torchaudio                              # choose the build for your CUDA version, see pytorch.org
pip install -r requirements.txt
```

The results were produced with Python 3.12, torch 2.13 (CUDA), numpy 2.4, scipy 1.18, scikit-learn 1.9,
matplotlib 3.10 and dasheng 0.0.9 on one RTX 5090. The training scripts need a CUDA GPU; the verification below runs
on a CPU with numpy and matplotlib only. The `unzip` command has to be on the `PATH` for the Guilin reader.

## Quick start: verify every reported number without training

```bash
python scripts/verify_results.py          # or: bash reproduce.sh check
```

The script recomputes every reported number from the per-task result files in `outputs/` and compares it with the
reported value (tolerance 0.0015 on accuracies): every cell of Table 2, the healthy-share sweep and the paired
comparison of Section 4.2, the ablations, the conformal reference refresh, the open-set scores and the numbers quoted
in the text. It needs no dataset, no checkpoint and no GPU and takes a few seconds. The last line must read

```
205 checks: 205 pass, 0 fail, 0 missing
```

A result file that is absent is reported as `MISSING`; the exit code is 1 if any check fails.
`python scripts/make_tables.py` regenerates Tables 2 and 3 from `outputs/` and reports whether they are identical to
`results/expected_tables.md`.

## Data preparation

The datasets and the pretrained encoders are public but are not redistributed here.

| Corpus | Role | Source |
|---|---|---|
| Guilin multi-condition acoustic bearing dataset | benchmark (9 and 27 cells) | C. Xu et al., Data in Brief 67:112919, 2026, <https://doi.org/10.1016/j.dib.2026.112919> |
| AHU Parabolic Acoustic Mirror Bearing Dataset | analysis of the path gain | IEEE DataPort, <https://doi.org/10.21227/jm8w-zw54> |
| DCASE 2026 Task 2 development set | cross-corpus check (near / far channel) | arXiv:2606.01578 |
| Wind-turbine blade acoustic corpus | cross-corpus check (−10 dB) | see `docs/DATA.md` |
| BEATs checkpoint `BEATs_iter3_plus_AS2M.pt` | foundation-model branch | <https://github.com/microsoft/unilm/tree/master/beats> |
| Dasheng-base | second foundation model | downloaded by the `dasheng` package |

Default layout (details, file-name patterns and the Guilin container reader are described in
[`docs/DATA.md`](docs/DATA.md)):

```
data/Guilin/*.zip                      the five archives, not extracted
data/AHU/Raw data/{PM,DM}/…/*.mat
data/DCASE2026/<machine>/{train,test}/*.wav
data/wind_blade/iid.zip
checkpoints/BEATs_iter3_plus_AS2M.pt
```

Environment variables override the defaults:

| Variable | Default |
|---|---|
| `REFPATH_DATA` (root of the datasets) | `./data` |
| `REFPATH_GUILIN`, `REFPATH_AHU`, `REFPATH_DCASE`, `REFPATH_WINDBLADE`, `REFPATH_MSU` (single corpora) | below `$REFPATH_DATA` |
| `REFPATH_BEATS_CKPT` (checkpoint file or its folder) | `./checkpoints` |
| `PY` (interpreter of the shell scripts) | `python` |

The caches are built by the scripts on first use and written to `./cache` (about 30 GB for the full reproduction):
`scripts/run_baselines.py` streams the first 60 s of segments 1, 3 and 5 from the Guilin archives and extracts the
features of the nine-cell protocol, `scripts/build_full_cache.py` does the same for the full 10-min recordings, and
`scripts/train_pathlora.py` / `scripts/train_pathlora2.py` build the 16-kHz waveform caches of the foundation-model
branches.

## Full reproduction

```bash
bash reproduce.sh            # all stages, strictly serial (one GPU job at a time)
bash reproduce.sh 6 8        # stages 6 to 8 only; stages are resumable
PY=/path/to/python bash reproduce.sh
```

| Stage | Content | Main scripts |
|---|---|---|
| 0 | caches of the 60-s clips, linear brackets, frozen-BEATs probes | `run_baselines.py`, `beats_embed.py` |
| 1 | physics branch on the nine cells: seeds, ablations, single-factor protocols, conformal alarm, reference refresh, open-set, synchronous-pair invariance | `rerun_physics_chain.sh`, `train_mlp.py`, `conformal*.py`, `spi.py` |
| 2 | Cyclo3DNet (negative result) | `train_cyclo3d.py` |
| 3 | BEATs branch on the nine cells: plain LoRA, hypernetwork variants, reference-normalized input | `train_pathlora.py` |
| 4 | fusion on the nine cells, first-round baselines, statistics | `fuse.py`, `baselines_lit.py`, `baseline_crossmodal.py`, `stats.py` |
| 5 | cross-corpus checks: wind-turbine blades, DCASE 2026 | `windblade.py`, `dcase_*.py` |
| 6 | cache of the full recordings; physics branch and linear brackets on the 27 and 9 cells | `build_full_cache.py`, `run_full_chain.sh` |
| 7 | BEATs and Dasheng on the 27 cells, seed ensembles, temperature-calibrated fusion, statistics, cost | `train_pathlora2.py`, `train_dasheng.py`, `fm_stats.py`, `ensemble_temperature.py`, `mixed_effects.py`, `factor_cost_table.py` |
| 8 | same-protocol re-implementations of the 16 prior methods, raw and + reference | `run_sota_feat.sh`, `run_sota_wave.sh` |
| 9 | figures, Tables 2–3, consistency check | `make_fig_*.py`, `make_tables.py`, `verify_results.py` |

Approximate GPU time on one RTX 5090: about 20 GPU-hours for stages 0–7 and about 4 hours for stage 8 (2.3 h for the
feature-level queue, 1.7 h for the waveform-level queue). Every script is deterministic for a given `--seed`; logs go
to `logs/`.

Notes on provenance:

- Two checks of `verify_results.py` ("BEATs-LoRA + reference, per-recording gain") use result files of an earlier
  variant of the input gain (`outputs/guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm*.json`). The current script
  derives the gain from the healthy reference only and does not regenerate these three files.
- `outputs/` also keeps result files of exploratory runs that the paper does not report (for example
  `msu_cross_device_s0.json`, `openset27_scores_v1.json`, `openset27_scores_v2.json`).

## Protocols and leakage rules

A **domain** is a tuple (sensing path, speed, load). The Guilin corpus has three microphones, three speeds and three
loads.

- **Nine cells** (`mic_x_speed`): one microphone and one speed are held out jointly. The source domains are the
  recordings of the two other microphones at the two other speeds; the target is the held-out microphone at the
  held-out speed (all loads). First 60 s of every segment, 48,195 one-second windows with 0.5-s hop.
- **27 cells** (`mic_x_speed_x_load`): the load is held out as well, which leaves eight source domains per cell.
  Full recordings, 485,595 windows; the waveform-level methods and the fusion use the first 180 s of the test segment.
- Single-factor protocols (`lomo`, `loso`, `lolo`) are available in `refpath/eval/protocol.py`.

Rules that keep the protocols free of leakage:

1. **Time-disjoint segments.** Training uses segments 1 and 3, testing segment 5. The reference is the first 60 s of
   the healthy recording of segment 1 of the target domain, so test windows always come from a later segment than
   the reference. (AHU: each 25-s file is cut into five virtual segments; training 1–3, reference 1, test 5.)
2. **Healthy reference only.** From the target domain the learner receives only healthy windows of the reference
   segment. No labelled and no faulty target window is ever used; few-shot adaptation is excluded because the corpus
   has one recording per condition, so a single labelled target window would leak the recording.
3. **Split reference.** The reference windows are split by interleaving: even windows provide the standardization
   statistics, odd windows are kept for the conformal calibration (178 windows per task).
4. **Own statistics per domain.** Source domains are standardized with their own healthy training windows; the input
   gain of the foundation-model branches is a reference statistic, never a test statistic.
5. **No target information in model selection.** The fusion temperatures on the 27 cells are chosen by nested
   leave-one-task-out selection on the other tasks.
6. **Transductive methods are marked.** AdaBN-T, TENT, SAR and the entropy gate (REF-Path-G) read the unlabeled test
   set and are reported separately; the oracle (statistics of all target test windows) is not deployable.

## Re-implemented baselines

All baselines run on the same source windows as REF-Path, with three seeds, in a raw version (`<method>_raw`, source
statistics only) and with healthy-reference standardization of the input (`<method>_ref`).

| Family | Methods | Command |
|---|---|---|
| Brackets | linear classifier (raw / + reference), oracle | `python scripts/run_baselines.py guilin` (nine cells); `python scripts/train_mlp2.py --model linear --norm {raw,healthy,oracle} --protocol <P>` (full recordings) |
| Feature level | ERM, AdaBN-R, DANN, Deep CORAL, SAGM, DDG, FGDG, CSCoh-CNN | `python scripts/sota_feat.py --protocol <P> --seed <s>`; queue: `bash run_sota_feat.sh` |
| Test-time adaptation (transductive) | AdaBN-T, TENT, SAR, healthy-share sweep | part of `sota_feat.py` (`--which adabntest,tent,sar,imb`); alone: `bash run_sota_tta.sh`, then `python scripts/merge_sota_tta.py` |
| Waveform level | AFT, XMC (cross-modal contrastive), BEATs-kNN | `python scripts/sota_wave.py --method {aft,xmodal,genrep} --protocol <P> --seed <s>`; queue: `bash run_sota_wave.sh` |
| Foundation models with LoRA | BEATs-LoRA, Dasheng-LoRA | nine cells: `python scripts/train_pathlora.py guilin --epochs 3 --multilabel --aug --no_hyper` (raw), add `--ref_norm --ref_split` (+ reference); 27 cells: `python scripts/train_pathlora2.py --seed <s>`, `python scripts/train_dasheng.py --seed <s>` (+ reference), `python scripts/train_dasheng.py --no_refnorm --seed <s>` (raw) |

`<P>` is `mic_x_speed` (nine cells) or `mic_x_speed_x_load` (27 cells). The feature-level and waveform-level scripts
write `outputs/sota_feat_<P>_s<seed>.json` and `outputs/sota_wave_<method>_<P>_s<seed>.json`;
`scripts/make_fig_sota.py` collects every result into `outputs/sota_table.json`. The transductive methods adapt on a
shuffled stream of the test windows; in recording order the class-balanced test set is sorted by class. BEATs-LoRA
and Dasheng-LoRA with the reference are branches of REF-Path.

## Figures

The nine figures of the paper are in `figures/` (PNG and PDF).

| Figure | File | Script | Needs |
|---|---|---|---|
| 1 Framework | `fig_framework` (+ `.pptx`) | `make_fig_framework.py` | `cache/guilin`, Microsoft PowerPoint on Windows, `python-pptx`, `pywin32` (optional) |
| 2 Representation | `fig_representation` | `make_fig_repr.py` | `cache/guilin` |
| 3 Same-protocol comparison | `fig_sota` | `make_fig_sota.py` | `outputs/` only |
| 4 Accuracy per cell, 27 cells | `fig_27cell_branches` | `make_fig_27cell.py` | `outputs/` only |
| 5 Confusion matrices | `fig_confusion_hard_cells` | `make_fig_confusion.py` | `outputs/` only |
| 6 Sensing paths as acoustic fields | `fig_fields` | `make_fig_fields.py` | `cache/guilin`, AHU raw data |
| 7 t-SNE | `fig_tsne` | `make_fig_tsne.py` | feature caches; trains two small MLPs (`REFPATH_DEVICE=cpu` to run without a GPU) |
| 8 Conformal false-alarm control | `fig_conformal_far` | `make_fig_safeguards.py` | `outputs/` only |
| 9 Unknown-fault rejection | `fig_openset27` | `make_fig_openset.py` | `outputs/` only |

```bash
for f in sota 27cell confusion safeguards openset; do python scripts/make_fig_$f.py; done    # from outputs/ only
for f in repr fields tsne; do python scripts/make_fig_$f.py; done                            # need the caches
python scripts/make_fig_framework.py                                                         # optional, Windows + PowerPoint
```

`make_fig_framework.py` draws Fig. 1 as an editable PowerPoint file with `python-pptx` and lets PowerPoint render the
PDF and the PNG; it is not part of the default pipeline (`WITH_FRAMEWORK=1 bash reproduce.sh 9 9` includes it).

## Result files

`outputs/` contains every JSON / CSV result file and the eight posterior files (`*_probs.npz`) that the verification
and the figure scripts read. The remaining 75 posterior files (per-window class posteriors of the other runs, about
200 MB) are attached to the GitHub release of this repository; unpack them into `outputs/` to re-run the fusion and
ensemble analyses (`fm_stats.py`, `ensemble_temperature.py`, `openset27_scores.py`) without retraining. These
analyses also read the window metadata of the caches. The release asset also holds `outputs/_tta_rerun/`, the
separate re-run of the transductive baselines that `scripts/merge_sota_tta.py` merged into
`outputs/sota_feat_<protocol>_s<seed>.json` (the values of the first run, which adapted in recording order, are kept
in these files under `results_ordered_stream`).

## Citation

The paper is under review. Until it is published, please cite:

```bibtex
@unpublished{refpath2026,
  title  = {{REF-Path}: Healthy-Reference Standardization for Acoustic Fault Diagnosis across Sensing Paths and Speeds},
  author = {Author names to be added},
  year   = {2026},
  note   = {Under review at the Journal of Sound and Vibration}
}
```

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The BEATs model code in
`refpath/models/beats/` is vendored from Microsoft unilm under its MIT License; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Datasets and pretrained checkpoints keep the terms of their
authors and are not redistributed.

## Acknowledgements

This work builds on data and models released by others:

- the Guilin multi-condition acoustic dataset of ball bearings (C. Xu, X. Wang, J. Huang, Y. Lu, H. Su, J. Lu, Data in
  Brief 67:112919, 2026), under the terms given on the dataset page;
- the AHU Parabolic Acoustic Mirror Bearing Dataset (L. Peng, F. Liu, IEEE DataPort, 2026), under the terms given on
  IEEE DataPort;
- the DCASE 2026 Challenge Task 2 development set (T. Nishida et al., arXiv:2606.01578), under the terms given on the
  task page;
- the wind-turbine blade acoustic corpus, under the terms of its authors;
- BEATs (S. Chen et al., ICML 2023; Microsoft unilm, MIT License);
- Dasheng (H. Dinkel et al., Interspeech 2024; Apache License 2.0).
