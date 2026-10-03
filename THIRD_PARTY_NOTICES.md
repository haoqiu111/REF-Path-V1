# Third-party notices

The REF-Path code is released under the MIT License (see `LICENSE`). The repository also contains or builds on the
third-party material listed below. Datasets and pretrained checkpoints are **not** redistributed; see `docs/DATA.md`.

## 1. BEATs model code (vendored)

Files: `refpath/models/beats/BEATs.py`, `backbone.py`, `modules.py`, `quantizer.py`

- Source: <https://github.com/microsoft/unilm/tree/master/beats>
  (S. Chen et al., "BEATs: Audio Pre-Training with Acoustic Tokenizers", ICML 2023)
- License: MIT, Copyright (c) 2022 Microsoft. The original license header is kept at the top of every file.
- According to these headers the code is based on fairseq (<https://github.com/pytorch/fairseq>, MIT License) and, for
  `quantizer.py`, on taming-transformers (<https://github.com/CompVis/taming-transformers>, MIT License).
- Modification: in `BEATs.py` the filterbank of `preprocess()` calls `refpath.features.torch_frontend.kaldi_fbank`
  instead of `torchaudio.compliance.kaldi.fbank` (the two lines are marked by comments). `quantizer.py` is kept for
  completeness and is not used by the experiments.

```
The MIT License (MIT)

Copyright (c) 2022 Microsoft

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Kaldi-compatible filterbank (re-implementation)

File: `refpath/features/torch_frontend.py`, function `kaldi_fbank`

A compact re-implementation of the algorithm of `torchaudio.compliance.kaldi.fbank` with the settings that BEATs
uses (16 kHz, 25 ms Povey window, 10 ms shift, 128 mel bins). torchaudio is distributed under the BSD 2-Clause
License (<https://github.com/pytorch/audio>).

## 3. Software used through its public interface (not vendored)

| Component | Use in this repository | License |
|---|---|---|
| Dasheng (`dasheng` package, <https://github.com/Richermans/dasheng>) | second audio foundation model, `scripts/train_dasheng.py`; the package downloads its own checkpoint | Apache License 2.0 |
| PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, pandas, statsmodels, SoundFile | see `requirements.txt` | their respective licenses |
| python-pptx, pywin32 (optional) | Fig. 1 only, `scripts/make_fig_framework.py` | MIT / PSF-2.0 |

## 4. Pretrained checkpoints and datasets (not included)

| Item | Authors | Terms |
|---|---|---|
| BEATs checkpoint `BEATs_iter3_plus_AS2M.pt` | Microsoft (unilm) | see <https://github.com/microsoft/unilm/tree/master/beats> |
| Dasheng-base checkpoint | Xiaomi / H. Dinkel et al. | see <https://github.com/Richermans/dasheng> |
| Guilin multi-condition acoustic bearing dataset | C. Xu et al., Data in Brief 67:112919, 2026 | see <https://doi.org/10.1016/j.dib.2026.112919> |
| AHU Parabolic Acoustic Mirror Bearing Dataset | L. Peng, F. Liu, IEEE DataPort, 2026 | see <https://doi.org/10.21227/jm8w-zw54> |
| DCASE 2026 Task 2 development set | T. Nishida et al., arXiv:2606.01578 | see the DCASE 2026 Task 2 page |
| Wind-turbine blade acoustic corpus | its authors | see the dataset page |

The files in `outputs/` are results computed by the authors of this repository (accuracies, posteriors of the models
on test windows and window indices); they contain no audio.
