#!/bin/bash
# Physics branch on the nine-cell protocol (60-s clips): seeds, ablations, single-factor protocols, reference recency, conformal
# alarm, open-set runs, AHU. Run after scripts/run_baselines.py, which builds cache/guilin_feats.npz and cache/ahu_feats.npz.
cd "$(dirname "$0")/.."; PY="${PY:-python}"; export PYTHONIOENCODING=utf-8
echo "### MLP main protocol, 3 seeds (split reference)"
for s in 0 1 2; do "$PY" scripts/train_mlp.py guilin --feats psd,cyclo --mixup 0.4 --ref_split --seed $s --tag _s$s 2>&1 | grep "MEAN"; done
"$PY" scripts/train_mlp.py guilin --feats psd,cyclo --mixup 0.4 --ref_split 2>&1 | grep "MEAN"          # canonical (seed 0, no tag) for the conformal analysis and the figures
echo "### ablations"
"$PY" scripts/train_mlp.py guilin --feats psd --mixup 0.4 --ref_split 2>&1 | grep "MEAN"
"$PY" scripts/train_mlp.py guilin --feats cyclo --mixup 0.4 --ref_split 2>&1 | grep "MEAN"
"$PY" scripts/train_mlp.py guilin --feats psd,cyclo --mixup 0 --ref_split 2>&1 | grep "MEAN"
"$PY" scripts/train_mlp.py guilin --feats psd,cyclo --mixup 0.4 --pca 60 --ref_split 2>&1 | grep "MEAN"
echo "### single-factor protocols (held-out microphone, held-out speed)"
for p in lomo loso; do "$PY" scripts/train_mlp.py guilin --protocol $p --feats psd,cyclo --mixup 0.4 --ref_split 2>&1 | grep "MEAN"; done
echo "### reference recency"
"$PY" scripts/train_mlp.py guilin --feats psd,cyclo --mixup 0.4 --ref_split --ref_seg 3 2>&1 | grep "MEAN"
echo "### conformal"
"$PY" scripts/conformal.py "outputs/guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_probs.npz" | tail -3
"$PY" scripts/conformal.py "outputs/guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_refseg3_probs.npz" | tail -3
echo "### open-set"
for c in inner_pitting outer_pitting inner_outer_pitting cage_fracture; do echo "holdout $c"; "$PY" scripts/train_mlp.py guilin --feats psd,cyclo --mixup 0.4 --ref_split --holdout_class $c 2>&1 | grep "AUROC" | awk '{m+=$4; e+=$6; k+=$8; n++} END {printf "  AUROC maxprob %.3f entropy %.3f kNN %.3f\n", m/n, e/n, k/n}'; done
echo "### AHU MLP"
"$PY" scripts/train_mlp.py ahu --feats psd,cyclo --mixup 0.4 --ref_split 2>&1 | grep "MEAN"
echo "### DONE"
