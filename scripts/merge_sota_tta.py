"""Merge the re-run of the transductive baselines (outputs/sota_feat_<protocol>_tta_s<seed>.json, written by run_sota_tta.sh:
AdaBN / TENT / SAR on a shuffled test stream plus the healthy-share sweep) into outputs/sota_feat_<protocol>_s<seed>.json.
The first run adapted in recording order, where the class-balanced test set is sorted by class; those values are kept under
'results_ordered_stream'. A fresh run of sota_feat.py with the default --which writes the merged structure directly.
python scripts/merge_sota_tta.py"""
import os, json, glob
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
for fn in sorted(glob.glob(os.path.join(OUT, 'sota_feat_*_tta_s?.json'))):
    main = fn.replace('_tta_s', '_s'); A = json.load(open(main)); B = json.load(open(fn)); old = A.setdefault('results_ordered_stream', {})
    for k, v in B['results'].items():
        if k in A['results'] and k not in old and '_h' not in k: old[k] = A['results'][k]
        A['results'][k] = v
    A['tta_stream'] = 'shuffled'; json.dump(A, open(main, 'w'), indent=1); print('merged', os.path.basename(fn), '->', os.path.basename(main), len(B['results']), 'keys')
