#!/bin/bash
# Physics branch and linear brackets on the full recordings (27-cell and 9-cell protocols). Sequential (one GPU job at a time).
# Run after scripts/build_full_cache.py, which builds cache/guilin_full_feats.npz.
cd "$(dirname "$0")/.."; PY="${PY:-python}"; export PYTHONIOENCODING=utf-8
run() { "$PY" scripts/train_mlp2.py "$@" 2>&1 | grep --line-buffered "MEAN\|saved\|Error\|Traceback"; }
for P in mic_x_speed_x_load mic_x_speed; do
  echo "### $P linear brackets"
  for N in raw healthy oracle; do run --model linear --norm $N --protocol $P; done
  echo "### $P MLP + mixup, 3 seeds (window-level and smoothed K=9)"
  for S in 0 1 2; do run --model mlp --protocol $P --seed $S; done
  echo "### $P variants: multilabel / coral+mixup / multilabel+coral"
  run --model mlp --protocol $P --multilabel
  run --model mlp --protocol $P --coral 1.0
  run --model mlp --protocol $P --multilabel --coral 1.0
  echo "### $P ablations: no mixup and PSD only"
  run --model mlp --protocol $P --mixup 0
  run --model mlp --protocol $P --feats psd
done
echo "### open-set with p-value combinations (27 cells)"
for C in inner_pitting outer_pitting inner_outer_pitting cage_fracture; do run --model mlp --protocol mic_x_speed_x_load --holdout_class $C; done
echo "### DONE"
