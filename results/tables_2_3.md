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
