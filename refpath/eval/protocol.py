"""Reference-Only cross-path / cross-speed protocols and normalizations.

A *domain* is a (dataset, path, rpm, load) tuple. The Reference-Only rule: for every TEST domain the model may use
only healthy windows from the reference segment of that domain (no labels, no faulty samples).

Split rule (time-disjoint): train segments vs. test segment are fixed per dataset:
    guilin : train segs {1, 3}, test seg 5, reference = healthy windows of seg 1 of the test domain
    ahu    : each 25-s file is cut into 5 virtual segments; train {1,2,3}, test 5, reference = healthy seg 1
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

SPLIT = {'guilin': dict(train=(1, 3), test=(5,), ref=(1,)), 'ahu': dict(train=(1, 2, 3), test=(5,), ref=(1,))}


@dataclass
class Task:
    name: str
    train: np.ndarray      # boolean mask over windows
    test: np.ndarray
    ref: np.ndarray        # healthy reference windows available at deployment (subset of test-domain windows)


def domain_key(meta):
    return np.array([f'{d}|{p}|{r}|{l}' for d, p, r, l in zip(meta['dataset'], meta['path'], meta['rpm'], meta['load'])])


def make_tasks(meta, kind, ref_seconds=None):
    """kind in {'in_domain', 'lomo', 'loso', 'lolo', 'mic_x_speed', 'mic_x_speed_x_load'} ; returns list[Task].
    ref_seconds: if given and meta has a 'start' field, only reference windows starting before ref_seconds are used
    (keeps the deployable 'short healthy reference' even when the full recordings are used for train/test)."""
    ds = meta['dataset'][0]; sp = SPLIT[ds]
    tr_seg = np.isin(meta['seg'], sp['train']); te_seg = np.isin(meta['seg'], sp['test']); ref_seg = np.isin(meta['seg'], sp['ref'])
    if ref_seconds is not None and 'start' in meta.dtype.names:
        ref_seg = ref_seg & (meta['start'] < ref_seconds)
    healthy = meta['state'] == 'healthy'
    paths, rpms, loads = np.unique(meta['path']), np.unique(meta['rpm']), np.unique(meta['load'])
    tasks = []

    def add(name, tr_dom, te_dom):
        tasks.append(Task(name, tr_seg & tr_dom, te_seg & te_dom, ref_seg & te_dom & healthy))

    if kind == 'in_domain':
        add('in_domain', np.ones(len(meta), bool), np.ones(len(meta), bool))
    elif kind == 'lomo':
        for p in paths:
            add(f'lomo:{p}', meta['path'] != p, meta['path'] == p)
    elif kind == 'loso':
        for r in rpms:
            add(f'loso:{r}', meta['rpm'] != r, meta['rpm'] == r)
    elif kind == 'lolo':
        for l in loads:
            add(f'lolo:{l}', meta['load'] != l, meta['load'] == l)
    elif kind == 'mic_x_speed':
        for p in paths:
            for r in rpms:
                add(f'mxs:{p}@{r}', (meta['path'] != p) & (meta['rpm'] != r), (meta['path'] == p) & (meta['rpm'] == r))
    elif kind == 'mic_x_speed_x_load':
        for p in paths:
            for r in rpms:
                for l in loads:
                    add(f'mxsxl:{p}@{r}@{l}', (meta['path'] != p) & (meta['rpm'] != r) & (meta['load'] != l), (meta['path'] == p) & (meta['rpm'] == r) & (meta['load'] == l))
    else:
        raise KeyError(kind)
    return tasks


def normalize(F, meta, task, mode):
    """Per-domain feature standardization.
    mode: 'raw' | 'healthy' (deployable: stats from task.ref windows of each test domain, train domains use their
          own healthy windows of train segments) | 'oracle' (all windows of the domain, class-balanced, optimistic).
    Train-domain statistics always come from training windows only.
    """
    if mode == 'raw':
        return F
    G = F.copy(); keys = domain_key(meta); healthy = meta['state'] == 'healthy'
    for k in np.unique(keys):
        d = keys == k
        if (d & task.test).any():
            ref = d & task.ref if mode == 'healthy' else d & task.test
        else:
            ref = d & task.train & healthy if mode == 'healthy' else d & task.train
        if ref.sum() < 2:
            continue
        G[d] = (F[d] - F[ref].mean(0)) / (F[ref].std(0) + 1e-6)
    return G
