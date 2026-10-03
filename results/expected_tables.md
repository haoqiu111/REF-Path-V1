# Tables of the paper

Tables 1 to 6 of "REF-Path: Healthy-Reference Standardization for Acoustic Fault Diagnosis across Sensing Paths and Speeds", with their notes. Bracketed numbers are the reference numbers of the paper. `scripts/verify_results.py` parses Table 2 of this file and compares every cell with a recomputation from `outputs/`.

## Table 1. Corpora and Reference-Only protocols

| Corpus | Sensing paths | Classes | Conditions | Protocol (tasks) | Role |
|---|---|---|---|---|---|
| Guilin, 60 s clips [10] | CRY333, B&K #2, B&K #3 | 5 | 3 speeds, 3 loads | mic, speed (9) | benchmark |
| Guilin, full | same | 5 | same | mic, speed, load (27) | benchmark |
| AHU [11] | mirror, direct | 10 | 3 speeds | path, speed (6) | analysis |
| DCASE26-T2 [12] | near, far | 7 | dev. set | near to far | check |
| WTB | 3 channels | 5 | iid clients | LOCO | check |

Protocol: held-out factors (number of tasks). Guilin: 800, 1000, 1200 rpm; 0, 15, 30% load; 32,768 Hz; training segments 1 and 3, test segment 5; reference: first 60 s of the healthy recording of segment 1. WTB: wind-turbine blades at −10 dB; LOCO: leave one client out.

## Table 2. Same-protocol comparison, raw and with the healthy reference (+ ref.)

| Method | Tgt. | 9 cells, raw | 9 cells, + ref. | 27 cells, raw | 27 cells, + ref. |
|---|---|---|---|---|---|
| Linear classifier | R | 0.585 | 0.838 | 0.583 | 0.733 |
| ERM | R | 0.608 ± 0.015 | 0.862 ± 0.001 | 0.564 ± 0.007 | 0.756 ± 0.006 |
| AdaBN-R [16] | R | 0.567 ± 0.014 | 0.480 ± 0.004 | 0.455 ± 0.015 | 0.407 ± 0.003 |
| DANN [1] | R | 0.508 ± 0.018 | 0.789 ± 0.005 | 0.511 ± 0.008 | 0.660 ± 0.011 |
| Deep CORAL [13] | R | 0.656 ± 0.014 | 0.850 ± 0.004 | 0.559 ± 0.007 | 0.759 ± 0.003 |
| SAGM [8] | R | 0.623 ± 0.014 | 0.865 ± 0.002 | 0.558 ± 0.010 | 0.760 ± 0.004 |
| DDG [7] | R | 0.525 ± 0.014 | 0.766 ± 0.036 | 0.526 ± 0.004 | 0.647 ± 0.008 |
| FGDG [9] | R | 0.723 ± 0.014 | 0.647 ± 0.019 | 0.681 ± 0.006 | 0.507 ± 0.008 |
| CSCoh-CNN [35] | R | 0.537 ± 0.054 | 0.577 ± 0.017 | 0.555 ± 0.024 | 0.516 ± 0.007 |
| XMC [18] | R | 0.474 ± 0.021 | 0.600 ± 0.016 | 0.445 ± 0.011 | 0.560 ± 0.026 |
| AFT [19] | R | 0.626 ± 0.018 | 0.850 ± 0.009 | 0.568 ± 0.013 | 0.777 ± 0.004 |
| BEATs-kNN [29] | R | 0.725 | 0.493 | 0.606 | 0.392 |
| BEATs-LoRA [33] | R | 0.756 | 0.826 ± 0.009 | n/a | 0.797 ± 0.031 |
| Dasheng-LoRA [28] | R | n/a | n/a | 0.635 ± 0.022 | 0.760 ± 0.010 |
| Physics branch | R | n/a | 0.871 ± 0.002 | n/a | 0.804 ± 0.022 |
| **REF-Path (ours)** | R | n/a | **0.896 ± 0.006** | n/a | **0.876** |
| AdaBN-T [16] | T | 0.924 ± 0.005 | 0.917 ± 0.005 | 0.936 ± 0.005 | 0.940 ± 0.002 |
| TENT [4] | T | 0.925 ± 0.005 | 0.918 ± 0.005 | 0.939 ± 0.005 | 0.943 ± 0.002 |
| SAR [5] | T | 0.925 ± 0.005 | 0.917 ± 0.005 | 0.936 ± 0.005 | 0.940 ± 0.002 |
| REF-Path-G | R + T | n/a | 0.912 ± 0.005 | n/a | 0.894 |
| Oracle | O | n/a | 0.944 | n/a | 0.917 |

Window accuracy, mean ± std over 3 seeds. Tgt.: target data of the + ref. version; R healthy reference, T unlabeled test set (shuffled, class-balanced), O all target classes. Raw versions use source statistics only. On the nine cells the raw and oracle linear rows use the log-PSD, and the + ref. row adds the map. With matched features the + ref. value is 0.819 and the oracle 0.935. REF-Path-G: entropy gate. BEATs-LoRA and Dasheng-LoRA with the reference are branches of REF-Path. On the 27 cells the waveform-level methods and the fusion use the first 180 s of the test segment.

## Table 3. Accuracy on the 27 cells per held-out microphone, speed (rpm) and load

| Method | CRY333 | B&K #2 | B&K #3 | 800 | 1000 | 1200 | 0% | 15% | 30% | Mean | Min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Linear + ref. | 0.694 | 0.843 | 0.662 | 0.841 | 0.768 | 0.590 | 0.696 | 0.813 | 0.691 | 0.733 | 0.367 |
| AFT + ref. | 0.774 | 0.823 | 0.733 | 0.817 | 0.823 | 0.690 | 0.825 | 0.804 | 0.701 | 0.777 | 0.436 |
| Physics branch | 0.789 | 0.902 | 0.720 | 0.898 | 0.868 | 0.646 | 0.796 | 0.851 | 0.765 | 0.804 | 0.458 |
| BEATs branch | 0.795 | 0.878 | 0.717 | 0.829 | 0.795 | 0.766 | 0.846 | 0.788 | 0.756 | 0.797 | 0.618 |
| Dasheng branch | 0.750 | 0.887 | 0.716 | 0.806 | 0.859 | 0.688 | 0.786 | 0.820 | 0.748 | 0.784 | 0.544 |
| REF-Path (ours) | **0.910** | **0.919** | **0.800** | **0.924** | **0.940** | **0.765** | **0.893** | **0.874** | **0.862** | **0.876** | **0.642** |
| REF-Path-G | 0.897 | 0.947 | 0.840 | 0.929 | 0.953 | 0.802 | 0.925 | 0.882 | 0.876 | 0.894 | 0.632 |

Each column averages the nine cells that share a held-out level; Min is the worst of the 27 cells. AFT + ref. is the strongest prior method. The Dasheng branch is the frozen probe that enters the fusion.

## Table 4. Component ablation of REF-Path

| Variant | Cells | Mean | Min |
|---|---|---|---|
| Physics branch, full (seed 0) | 9 | 0.873 | 0.615 |
| PSD only | 9 | 0.811 | 0.587 |
| Map only | 9 | 0.855 | 0.588 |
| Cyclic axis in Hz | 9 | 0.783 | 0.552 |
| + sub-window axis | 9 | 0.857 | 0.550 |
| + PCA-60 | 9 | 0.804 | 0.591 |
| w/o mixup | 9 | 0.815 | 0.613 |
| w/o mixup | 27 | 0.683 | n/a |
| Linear head | 9 | 0.838 | 0.577 |
| 10× windows | 9 | 0.863 ± 0.020 | 0.60 |
| BEATs-LoRA, raw | 9 | 0.756 | 0.402 |
| BEATs-LoRA + ref. | 9 | 0.826 ± 0.009 | 0.58–0.72 |
| Dasheng-LoRA, raw | 27 | 0.635 ± 0.022 | 0.20–0.34 |
| Dasheng-LoRA + ref. | 27 | 0.760 ± 0.010 | 0.448 |
| Fusion w/o ensembles and $T_b$ | 27 | 0.848 | 0.658 |
| Fusion, 2 branches | 27 | 0.870 | 0.610 |
| Fusion, 3 branches | 27 | 0.876 | 0.642 |

Window accuracy; mean ± std over 3 seeds where given. PSD: log-PSD; Map: cyclostationary map; PCA-60: 60 principal components before the classifier; 10× windows: full recordings instead of 60 s clips. The full physics branch reaches 0.871 ± 0.002 over 3 seeds.

## Table 5. Conformal alarms with the commissioning and the refreshed reference

| Ref. | Acc. | FAR, 0.01 | FAR, 0.05 | FAR, 0.10 | DR, 0.01 | DR, 0.05 | DR, 0.10 | Accepted | Cont. |
|---|---|---|---|---|---|---|---|---|---|
| C | 0.867 | 0.025 (0.085) | 0.211 (0.404) | 0.345 (0.546) | 0.991 | 1.000 | 1.000 | n/a | n/a |
| RR | 0.875 | 0.071 (0.321) | 0.152 (0.480) | 0.249 (0.563) | 0.998 | 1.000 | 1.000 | n/a | n/a |
| SR-H | 0.875 | 0.073 (0.332) | 0.188 (0.488) | 0.271 (0.572) | 0.998 | 1.000 | 1.000 | 350 / 357 | 0 |
| SR-A | 0.847 | 0.059 (0.302) | 0.159 (0.460) | 0.237 (0.571) | 0.952 | 0.969 | 0.976 | 370 / 1785 | 0.039 |
| UF | 0.202 | 0.000 | 0.000 | 0.000 | 0.013 | 0.072 | 0.144 | n/a | faults |

Physics branch, nine cells, three seeds. FAR: false-alarm rate on healthy test windows at level $\alpha$, mean (max over tasks). DR: detection rate on faulty windows. C: commissioning reference, segment 1. RR: re-recorded healthy reference, segment 3. SR-H, SR-A: self-refresh with the machine healthy or in any of the five states. UF: unfiltered refresh. Accepted: accepted / candidate windows. Cont.: share of faulty windows in the new reference.

## Table 6. Deployment cost per one-second window

| Branch | Parameters (M) | Features (ms) | Inference (ms) |
|---|---|---|---|
| Physics branch | 0.92 / 0.92 | 5.5, CPU | 0.06, CPU |
| BEATs-LoRA, one seed | 2.9 / 93.2 | incl. | 17.30, GPU |
| Dasheng, frozen probe | 0.004 / 85.5 | incl. | 6.17, GPU |

Parameters: trainable / total. RTX 5090 GPU and desktop CPU; batch size 1. incl.: the front end runs on the GPU inside the inference time.
