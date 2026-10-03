"""(a) Factor-level accuracy table on the 27-cell protocol (mean over the 9 cells sharing a held-out
microphone / speed / load) for every branch and fusion; (b) deployment-cost table: trainable / total parameters, CPU
feature-extraction time per 1-s window, CPU and GPU inference time per window.
python scripts/factor_cost_table.py [--gpu]   (GPU timings only when no other GPU job is running)"""
import os, sys, json, time, argparse, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts')); OUT = os.path.join(ROOT, 'outputs')
ap = argparse.ArgumentParser(); ap.add_argument('--gpu', action='store_true'); a = ap.parse_args()
P = 'mic_x_speed_x_load'; fm = json.load(open(os.path.join(OUT, f'fm_stats_{P}.json'))); et = json.load(open(os.path.join(OUT, f'ensemble_temperature_{P}.json')))
cells = fm['cells']; S = fm['summary']; R = et['results']
rows = {'Linear, healthy-reference': None, 'Physics MLP (seed avg)': np.array(S['physics_mlp']['per_cell']), 'BEATs ref-norm LoRA (seed avg)': np.array(S['beats_lora']['per_cell']),
        'Dasheng ref-norm LoRA (seed avg)': np.array(S['dasheng_lora']['per_cell']), 'Dasheng frozen probe': np.array(S['dasheng_frozen']['per_cell']),
        'BEATs 3-seed ensemble': np.array(R['BEATs, 3-seed ensemble']['per_cell']), 'Product E_phys x E_beats': np.array(R['product E_phys x E_beats']['per_cell']),
        'T-product E_phys x E_beats (nested)': np.array(R['T-product E_phys x E_beats']['nested_per_cell']), 'T-product 3-way (nested)': np.array(R['T-product E_phys x E_beats x Dasheng-frozen']['nested_per_cell']),
        'Gate E_phys x E_beats (transductive)': np.array(R['gate E_phys x E_beats']['per_cell']), 'Gate 3-way (transductive)': np.array(R['gate E_phys x E_beats x Dasheng-frozen']['per_cell'])}
lin = json.load(open(os.path.join(OUT, f'full_{P}_linear_healthy_s0.json')))['results']; rows['Linear, healthy-reference'] = np.array([lin[c]['acc'] for c in cells])
def fac(c): m, r, l = c.split(':')[1].split('@'); return m, int(r), int(l)
groups = {'mic': ['CRY333', 'BK4966_2', 'BK4966_3'], 'rpm': [800, 1000, 1200], 'load': [0, 15, 30]}
hdr = ['method'] + [f'{k}={v}' for k, vs in groups.items() for v in vs] + ['mean', 'min']
lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
table = {}
for name, v in rows.items():
    vals = []
    for k, vs in groups.items():
        for val in vs:
            idx = [i for i, c in enumerate(cells) if fac(c)[{'mic': 0, 'rpm': 1, 'load': 2}[k]] == val]; vals.append(float(v[idx].mean()))
    table[name] = vals + [float(v.mean()), float(v.min())]; lines.append('| ' + name + ' | ' + ' | '.join(f'{x:.3f}' for x in table[name]) + ' |')
md = '\n'.join(lines); print(md)

# ---- cost table ----
from refpath.features.frontend import log_psd, cyclo_volume
fs = 32768; x = np.random.randn(fs).astype(np.float64); t0 = time.perf_counter(); n = 20
for _ in range(n): log_psd(x, fs); cyclo_volume(x, fs, 1000 / 60).mean(2)
t_feat = (time.perf_counter() - t0) / n
mlp = torch.nn.Sequential(torch.nn.Linear(3328, 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 5)).eval()
n_mlp = sum(p.numel() for p in mlp.parameters()); xb = torch.randn(1, 3328)
with torch.no_grad():
    for _ in range(50): mlp(xb)
    t0 = time.perf_counter()
    for _ in range(500): mlp(xb)
    t_mlp_cpu = (time.perf_counter() - t0) / 500
cost = {'physics MLP': dict(params_total=n_mlp, params_trainable=n_mlp, feat_ms_cpu=t_feat * 1e3, infer_ms_cpu=t_mlp_cpu * 1e3)}
if a.gpu:
    from refpath.models.path_lora import PathLoRAModel
    from refpath.features.torch_frontend import kaldi_fbank
    import train_dasheng as TD
    dev = 'cuda'
    try:
        import train_pathlora as TP
        from refpath.models.beats.BEATs import BEATs, BEATsConfig
        c_ = torch.load(TP.CKPT, map_location='cpu', weights_only=False); b_ = BEATs(BEATsConfig(c_['cfg'])); b_.load_state_dict(c_['model'])
        m = PathLoRAModel(b_, n_classes=5, n_prim=4, r=8, d_dim=128).to(dev).eval(); w = torch.randn(1, 16000).to(dev); d = torch.zeros(1, 128).to(dev); mu = torch.zeros(1, 128).to(dev); sd = torch.ones(1, 128).to(dev)
        tot = sum(p.numel() for p in m.parameters()); trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(10): m(w, d, mu, sd)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(50): m(w, d, mu, sd)
            torch.cuda.synchronize(); t_b = (time.perf_counter() - t0) / 50
        cost['BEATs + LoRA'] = dict(params_total=tot, params_trainable=trn, infer_ms_gpu=t_b * 1e3)
    except Exception as e: print('BEATs timing failed:', e)
    m = TD.DashengRef(5, 3, lora=False).to(dev).eval(); w = torch.randn(1, 16000).to(dev); tot = sum(p.numel() for p in m.parameters())
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(10): m.forward_spec(m.spec(w))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(50): m.forward_spec(m.spec(w))
        torch.cuda.synchronize(); t_d = (time.perf_counter() - t0) / 50
    cost['Dasheng frozen'] = dict(params_total=tot, params_trainable=768 * 5 + 5, infer_ms_gpu=t_d * 1e3)
    mlp = mlp.to(dev); xb = xb.to(dev)
    with torch.no_grad():
        for _ in range(50): mlp(xb)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(500): mlp(xb)
        torch.cuda.synchronize(); cost['physics MLP']['infer_ms_gpu'] = (time.perf_counter() - t0) / 500 * 1e3
print(json.dumps(cost, indent=1))
json.dump(dict(factor_table=table, header=hdr, markdown=md, cost=cost), open(os.path.join(OUT, f'factor_cost_{P}.json'), 'w'), indent=1); print('saved outputs/factor_cost_%s.json' % P)
