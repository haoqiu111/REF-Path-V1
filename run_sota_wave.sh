#!/bin/bash
# Waveform-level same-protocol re-implementations (AFT, cross-modal contrastive XMC, frozen BEATs + k-NN), raw and + reference versions.
# Waits for run_sota_feat.sh (one GPU job at a time): that queue must write its log to logs/sota_feat.log, as reproduce.sh does.
cd "$(dirname "$0")"; PY="${PY:-python}"; export PYTHONIOENCODING=utf-8
until grep -q "SOTA_FEAT DONE" logs/sota_feat.log 2>/dev/null; do sleep 60; done
run() { echo "### $* $(date)"; "$PY" scripts/sota_wave.py "$@" 2>&1 | grep --line-buffered "acc \|saved\|Error\|Traceback"; }
for P in mic_x_speed mic_x_speed_x_load; do
  run --method genrep --protocol $P
  for s in 0 1 2; do run --method aft --protocol $P --seed $s; run --method xmodal --protocol $P --seed $s; done
done
echo "SOTA_WAVE DONE $(date)"
