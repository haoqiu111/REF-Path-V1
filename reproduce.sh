#!/bin/bash
# Reproduction of every number, table and figure of the REF-Path paper.
#
#   bash reproduce.sh            # run every stage in order (stages are resumable; see README.md for the GPU time)
#   bash reproduce.sh 3 5        # run stages 3..5 only
#   bash reproduce.sh check      # no training: recompute every reported number from outputs/ and compare (scripts/verify_results.py)
#
# Prerequisites (README.md, docs/DATA.md):
#   data:        $REFPATH_DATA (default ./data) with Guilin/*.zip (5 zips, .bkc containers), AHU/, DCASE2026/, wind_blade/,
#                and optionally MSU/Dataset (exploratory, not used in the paper)
#   checkpoint:  $REFPATH_BEATS_CKPT (default ./checkpoints/BEATs_iter3_plus_AS2M.pt); Dasheng is downloaded by its package
#   python:      PY=<interpreter> (default: python), Python >= 3.12 with the packages of requirements.txt; `unzip` on the PATH
# Rules: one GPU job at a time (the stages are strictly serial); every script is deterministic for a given --seed.
set -o pipefail
cd "$(dirname "$0")"; export PY="${PY:-python}"; export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
if [ "$1" == "check" ]; then "$PY" scripts/verify_results.py; exit $?; fi
mkdir -p outputs figures logs cache
log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { log "$*"; "$PY" "$@" 2>&1 | tee -a "logs/reproduce_$(date +%Y%m%d).log" | grep --line-buffered "MEAN\|saved\|AUROC\|FAR\|hmean\|Error\|Traceback"; }

FROM=${1:-0}; TO=${2:-9}
stage() { [ "$1" -ge "$FROM" ] && [ "$1" -le "$TO" ]; }

# 0. caches: 60-s clips of Guilin segments 1/3/5, AHU, features (log-PSD + order-normalized cyclostationary map); linear brackets
if stage 0; then
  log "### stage 0: caches and linear brackets (linear classifier and oracle rows of Table 2, nine cells)"
  run scripts/run_baselines.py guilin            # -> cache/guilin/, cache/guilin_feats.npz, outputs/guilin_baselines.csv
  run scripts/run_baselines.py ahu               # -> cache/ahu_feats.npz,    outputs/ahu_baselines.csv
  run scripts/beats_embed.py guilin              # frozen BEATs + LR / k-NN probes -> cache/guilin_beats.npz, outputs/guilin_beats_frozen.csv
fi

# 1. physics branch on the nine-cell protocol: MLP + mixup, ablations, single-factor protocols, reference recency, conformal, open-set
if stage 1; then
  log "### stage 1: physics branch, nine cells (Tables 2, 4, 5; Fig. 8)"
  bash scripts/rerun_physics_chain.sh
  run scripts/ablate_order_axis.py                                          # Table 4: cyclic axis in hertz (also builds cache/guilin_cyclo_fixedalpha.npy)
  run scripts/train_mlp.py guilin --feats psd,cyclo --mixup 0.4 --ref_split --full3d          # Table 4: sub-window axis kept
  run scripts/train_mlp.py guilin --feats cyclo --mixup 0.4 --ref_split --full3d
  for s in 1 2; do for c in inner_pitting outer_pitting inner_outer_pitting cage_fracture; do
    run scripts/train_mlp.py guilin --feats psd,cyclo --mixup 0.4 --ref_split --holdout_class $c --seed $s --tag _s$s; done; done   # open-set seeds, nine cells
  run scripts/openset_conformal.py                                          # rank-average open-set score, chronological reference split
  run scripts/conformal_weighted.py "outputs/guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_probs.npz"   # BH false-discovery rate, weighted conformal
  for s in 0 1 2; do run scripts/spi.py --lam 0.5 --seed $s; done                 # synchronous-pair invariance (negative result, 3 seeds)
  for s in 0 1 2; do run scripts/conformal_refresh.py --seed $s; done; run scripts/conformal_refresh.py --alpha_acc 0.05   # label-free reference refresh (Table 5)
  run scripts/refresh_summary.py
fi

# 2. Cyclo3DNet (factorized 3-D network, negative result), 3 seeds
if stage 2; then
  log "### stage 2: Cyclo3DNet (Section 4.3)"
  run scripts/train_cyclo3d.py guilin --protocol mic_x_speed --epochs 30 --multilabel --aug --ref_std --psd
  for s in 1 2; do run scripts/train_cyclo3d.py guilin --protocol mic_x_speed --epochs 30 --multilabel --aug --ref_std --psd --seed $s --tag _s$s; done
  run scripts/train_cyclo3d.py ahu --protocol mic_x_speed --epochs 30 --multilabel --aug --ref_std --psd
fi

# 3. BEATs branch: plain LoRA, hypernetwork variants (negative), reference-normalized fbank with reference gain (3 seeds), single-factor protocols
if stage 3; then
  log "### stage 3: BEATs branch, nine cells (Tables 2, 4)"
  run scripts/train_pathlora.py guilin --epochs 3 --multilabel --aug --no_hyper                              # plain LoRA, raw input
  run scripts/train_pathlora.py guilin --epochs 3 --multilabel --aug                                         # hypernetwork, PSD descriptor
  run scripts/train_pathlora.py guilin --epochs 3 --multilabel --aug --desc beats                            # hypernetwork, BEATs descriptor
  # (outputs/guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm{,_seed1,_seed2}.json, the 'per-recording gain' variant checked by
  #  verify_results.py, come from an earlier version of the gain computation; the script now derives the gain from the healthy reference only)
  for s in 0 1 2; do run scripts/train_pathlora.py guilin --epochs 3 --multilabel --aug --no_hyper --ref_norm --ref_split --seed $s --tag _refgain_s$s; done
  for p in lomo loso; do run scripts/train_pathlora.py guilin --protocol $p --epochs 3 --multilabel --aug --no_hyper --ref_norm --ref_split --tag _refgain; done
  run scripts/train_pathlora.py ahu --epochs 3 --multilabel --aug --no_hyper --ref_norm --ref_split
fi

# 4. fusion on the nine cells and first-round baselines under the identical protocol, 3 seeds
if stage 4; then
  log "### stage 4: fusion on the nine cells (Table 2), first-round baselines, statistics"
  for s in 0 1 2; do run scripts/fuse.py "outputs/guilin_mic_x_speed_mlp_psd+cyclo_pca0_mix0.4_refsplit_s${s}_probs.npz" outputs/guilin_mic_x_speed_pathlora_plain_ml_aug_refnorm_refsplit_refgain_s${s}_probs.npz; done
  for s in 0 1 2; do run scripts/baselines_lit.py --seed $s; done
  run scripts/baseline_crossmodal.py; run scripts/baseline_crossmodal.py --refnorm
  run scripts/stats.py                                                     # seeds and paired Wilcoxon tests
fi

# 5. cross-corpus checks: wind-turbine blades, DCASE 2026 near/far, MSU (exploratory, optional)
if stage 5; then
  log "### stage 5: cross-corpus checks (Section 4.3, generality and boundaries)"
  run scripts/windblade.py
  run scripts/dcase_asd.py; run scripts/dcase_refnorm.py; run scripts/dcase_finetune.py; run scripts/dcase_eval_official.py
  MSU="${REFPATH_MSU:-${REFPATH_DATA:-data}/MSU}"; [ -d "$MSU/Dataset" ] && run scripts/msu_experiment.py --root "$MSU"
fi

# 6. full recordings: 27-cell protocol, physics chain
if stage 6; then
  log "### stage 6: full-recording cache and physics branch on the full recordings (Tables 2-4)"
  run scripts/build_full_cache.py                                          # about 19 GB: cache/guilin_full + guilin_full_feats.npz
  bash scripts/run_full_chain.sh
fi

# 7. foundation models on the 27 cells: BEATs LoRA (3 seeds), Dasheng LoRA (3 seeds) / frozen / no reference; statistics and fusion
if stage 7; then
  log "### stage 7: BEATs and Dasheng on the 27 cells, seed ensembles and fusion (Tables 2-4, 6; Fig. 4, 9)"
  for s in 0 1 2; do run scripts/train_pathlora2.py --protocol mic_x_speed_x_load --epochs 3 --seed $s; done
  for s in 0 1 2; do run scripts/train_dasheng.py --protocol mic_x_speed_x_load --seed $s; done
  run scripts/train_dasheng.py --protocol mic_x_speed_x_load --frozen
  for s in 0 1 2; do run scripts/train_dasheng.py --protocol mic_x_speed_x_load --no_refnorm --seed $s; done
  for s in 1 2; do for c in inner_pitting outer_pitting inner_outer_pitting cage_fracture; do run scripts/train_mlp2.py --model mlp --protocol mic_x_speed_x_load --holdout_class $c --seed $s; done; done   # open-set seeds
  run scripts/openset27_scores.py                                          # per-window open-set scores -> outputs/openset27_scores.json
  run scripts/extra_seeds_summary.py                                       # seed statistics of the SPI and open-set runs -> outputs/extra_seeds_summary.json
  run scripts/fm_stats.py                                                  # seed statistics, Wilcoxon, product / gate fusion -> outputs/fm_stats_*.json
  run scripts/ensemble_temperature.py                                      # seed ensembles + nested temperature calibration (the REF-Path fusion on the 27 cells)
  run scripts/mixed_effects.py                                             # mixed model over cells
  run scripts/factor_cost_table.py --gpu                                   # per-factor accuracies and deployment cost (Table 6)
fi

# 8. same-protocol re-implementations of prior methods, raw and + healthy reference (Table 2, Fig. 3)
if stage 8; then
  log "### stage 8: same-protocol baselines"
  bash run_sota_feat.sh > logs/sota_feat.log; bash run_sota_wave.sh > logs/sota_wave.log
fi

# 9. figures, Tables 2-3 and the consistency check
if stage 9; then
  log "### stage 9: figures, tables, consistency check"
  for f in repr 27cell confusion fields tsne safeguards openset sota; do "$PY" scripts/make_fig_$f.py; done
  [ "${WITH_FRAMEWORK:-0}" == "1" ] && "$PY" scripts/make_fig_framework.py   # Fig. 1: needs Microsoft PowerPoint on Windows (python-pptx, pywin32)
  "$PY" scripts/make_tables.py; "$PY" scripts/verify_results.py | tail -1
fi
log "### done"
