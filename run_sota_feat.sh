#!/bin/bash
# Feature-level same-protocol re-implementations (ERM, AdaBN-R, AdaBN-T, TENT, SAR, DANN, Deep CORAL, SAGM, DDG, FGDG, CSCoh-CNN),
# raw and + healthy-reference versions, 9 cells then 27 cells, 3 seeds. Strictly serial.
cd "$(dirname "$0")"; PY="${PY:-python}"; export PYTHONIOENCODING=utf-8
for s in 0 1 2; do echo "### 9-cell seed $s $(date)"; "$PY" scripts/sota_feat.py --protocol mic_x_speed --seed $s 2>&1 | grep --line-buffered "mxs:\|acc \|saved\|Error\|Traceback"; done
for s in 0 1 2; do echo "### 27-cell seed $s $(date)"; "$PY" scripts/sota_feat.py --protocol mic_x_speed_x_load --seed $s 2>&1 | grep --line-buffered "mxsxl:\|acc \|saved\|Error\|Traceback"; done
echo "SOTA_FEAT DONE $(date)"
