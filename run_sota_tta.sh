#!/bin/bash
# Transductive test-time adaptation (AdaBN-T, TENT, SAR) on a shuffled test stream, and its dependence on the healthy share of the stream.
# Writes outputs/sota_feat_<protocol>_tta_s<seed>.json; scripts/merge_sota_tta.py merges them into outputs/sota_feat_<protocol>_s<seed>.json.
# Only needed to refresh these methods alone: run_sota_feat.sh already runs them with the same settings.
cd "$(dirname "$0")"; PY="${PY:-python}"; export PYTHONIOENCODING=utf-8
for P in mic_x_speed mic_x_speed_x_load; do for s in 0 1 2; do echo "### $P s$s $(date)"; "$PY" scripts/sota_feat.py --protocol $P --seed $s --which adabntest,tent,sar,imb --tag _tta 2>&1 | grep -v "^mxs"; done; done
echo "SOTA_TTA DONE $(date)"
