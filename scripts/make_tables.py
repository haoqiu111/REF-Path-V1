# -*- coding: utf-8 -*-
"""Tables 2 and 3 of the paper from outputs/sota_table.json (written by scripts/make_fig_sota.py) and
outputs/ensemble_temperature_mic_x_speed_x_load.json (written by scripts/ensemble_temperature.py).
  Table 2: methods x {raw, + healthy reference} x {9 cells, 27 cells}
  Table 3: the 27 cells averaged per held-out microphone / speed / load
Writes results/tables_2_3.md and outputs/sota_table_rows.json (row label -> result key, read by scripts/verify_results.py), and
reports whether the two tables are identical to those in results/expected_tables.md.
python scripts/make_tables.py   (run scripts/make_fig_sota.py first)"""
import io, os, re, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "outputs"); RES = os.path.join(ROOT, "results")
T = json.load(open(os.path.join(OUT, "sota_table.json"))); P9, P27 = "mic_x_speed", "mic_x_speed_x_load"
ROWS = [  # key, short label with the reference number of the paper, transductive
    ("erm", "ERM", False), ("adabnref", "AdaBN-R [16]", False), ("adabntest", "AdaBN-T [16]", True),
    ("tent", "TENT [4]", True), ("sar", "SAR [5]", True), ("dann", "DANN [1]", False), ("coral", "Deep CORAL [13]", False), ("sagm", "SAGM [8]", False),
    ("ddg", "DDG [7]", False), ("fgdg", "FGDG [9]", False), ("cscoh", "CSCoh-CNN [35]", False),
    ("xmodal", "XMC [18]", False), ("aft", "AFT [19]", False), ("genrep", "BEATs-kNN [29]", False),
    ("beatslora", "BEATs-LoRA [33]", False), ("dasheng", "Dasheng-LoRA [28]", False)]
OURS_BRANCH = {"beatslora_ref", "dasheng_ref"}      # the + reference versions of the foundation models are branches of REF-Path


def fmt(s, bold=False):
    if not s: return "n/a"
    t = f"{s['mean']:.3f} ± {s['std']:.3f}" if s.get("n", 1) > 1 else f"{s['mean']:.3f}"
    return f"**{t}**" if bold else t


def best(P, raw_only=False, include_trans=False):
    c = []
    for k, _, tr in ROWS:
        if tr and not include_trans: continue
        for n in (("raw",) if raw_only else ("raw", "ref")):
            key = f"{k}_{n}"; s = T[P].get(key)
            if s and key not in OURS_BRANCH: c.append((s["mean"], key))
    return max(c) if c else (float("nan"), None)


# ---------------------------------------------------------------- Table 2
def row(k, lab, tgt): return f"| {lab} | {tgt} | {fmt(T[P9].get(k + '_raw'))} | {fmt(T[P9].get(k + '_ref'))} | {fmt(T[P27].get(k + '_raw'))} | {fmt(T[P27].get(k + '_ref'))} |"


L = ["## Table 2. Same-protocol comparison, raw and with the healthy reference (+ ref.)", "",
     "| Method | Tgt. | 9 cells, raw | 9 cells, + ref. | 27 cells, raw | 27 cells, + ref. |", "|---|---|---|---|---|---|",
     f"| Linear classifier | R | {fmt(T[P9]['linear_raw'])} | {fmt(T[P9]['linear_ref'])} | {fmt(T[P27]['linear_raw'])} | {fmt(T[P27]['linear_ref'])} |"]
L += [row(k, lab, "R") for k, lab, tr in ROWS if not tr]
L.append(f"| Physics branch | R | n/a | {fmt(T[P9]['physics'])} | n/a | {fmt(T[P27]['physics'])} |")
L.append(f"| **REF-Path (ours)** | R | n/a | {fmt(T[P9]['refpath'], True)} | n/a | **{T[P27]['refpath']['mean']:.3f}** |")
L += [row(k, lab, "T") for k, lab, tr in ROWS if tr]
L.append(f"| REF-Path-G | R + T | n/a | {fmt(T[P9]['refpath_gate'])} | n/a | {T[P27]['refpath_gate']['mean']:.3f} |")
L.append(f"| Oracle | O | n/a | {fmt(T[P9]['oracle'])} | n/a | {fmt(T[P27]['oracle'])} |")
L += ["", "Window accuracy, mean ± std over 3 seeds. Tgt.: target data of the + ref. version; R healthy reference, T unlabeled test set (shuffled, class-balanced), O all target classes. Raw versions use source statistics only. On the nine cells the raw and oracle linear rows use the log-PSD, and the + ref. row adds the map. With matched features the + ref. value is 0.819 and the oracle 0.935. REF-Path-G: entropy gate. BEATs-LoRA and Dasheng-LoRA with the reference are branches of REF-Path. On the 27 cells the waveform-level methods and the fusion use the first 180 s of the test segment."]
table2 = "\n".join(L)

# ---------------------------------------------------------------- Table 3 (per held-out factor, 27 cells)
E = json.load(open(os.path.join(OUT, f"ensemble_temperature_{P27}.json"))); cells = E["cells"]
MICS = ["CRY333", "BK4966_2", "BK4966_3"]; RPMS = ["800", "1000", "1200"]; LOADS = ["0", "15", "30"]


def factor_row(name, tasks, vals, bold=False):
    d = dict(zip(tasks, vals)); cols = []
    for i, levels in enumerate((MICS, RPMS, LOADS)):
        for lv in levels: cols.append(np.mean([v for t, v in d.items() if t.split(":")[1].split("@")[i] == lv]))
    cols += [np.mean(vals), np.min(vals)]; f = (lambda x: f"**{x:.3f}**") if bold else (lambda x: f"{x:.3f}")
    return f"| {name} | " + " | ".join(f(c) for c in cols) + " |"


b27 = best(P27)[1]; lab_of = {k: l for k, l, _ in ROWS}
bname = re.sub(r"\s*\[[^\]]+\]", "", lab_of[b27.rsplit("_", 1)[0]]) + (", raw" if b27.endswith("_raw") else " + ref.")
F = ["## Table 3. Accuracy on the 27 cells per held-out microphone, speed (rpm) and load", "",
     "| Method | CRY333 | B&K #2 | B&K #3 | 800 | 1000 | 1200 | 0% | 15% | 30% | Mean | Min |", "|---|---|---|---|---|---|---|---|---|---|---|---|",
     factor_row("Linear + ref.", T[P27]["linear_ref"]["tasks"], T[P27]["linear_ref"]["per_task"]),
     factor_row(bname, T[P27][b27]["tasks"], T[P27][b27]["per_task"]),
     factor_row("Physics branch", cells, E["results"]["physics, seed-averaged"]["per_cell"]),
     factor_row("BEATs branch", cells, E["results"]["BEATs, seed-averaged"]["per_cell"]),
     factor_row("Dasheng branch", cells, E["results"]["Dasheng frozen probe"]["per_cell"]),
     factor_row("REF-Path (ours)", cells, E["results"]["T-product E_phys x E_beats x Dasheng-frozen"]["nested_per_cell"], True),
     factor_row("REF-Path-G", cells, E["results"]["gate E_phys x E_beats x Dasheng-frozen"]["per_cell"]), "",
     f"Each column averages the nine cells that share a held-out level; Min is the worst of the 27 cells. {bname} is the strongest prior method. The Dasheng branch is the frozen probe that enters the fusion."]
table3 = "\n".join(F)

# ---------------------------------------------------------------- outputs
os.makedirs(RES, exist_ok=True)
io.open(os.path.join(RES, "tables_2_3.md"), "w", encoding="utf-8", newline="\n").write(table2 + "\n\n" + table3 + "\n")
lab_plain = {k: re.sub(r"\s*\[[^\]]+\]", "", l) for k, l, _ in ROWS}
json.dump({v: k for k, v in lab_plain.items()}, open(os.path.join(OUT, "sota_table_rows.json"), "w"), indent=1)
print("saved results/tables_2_3.md and outputs/sota_table_rows.json")
exp = os.path.join(RES, "expected_tables.md")
if os.path.exists(exp):
    s = io.open(exp, encoding="utf-8").read().replace("\r\n", "\n")
    for name, tab in (("Table 2", table2), ("Table 3", table3)):
        if tab in s: print(f"{name}: identical to results/expected_tables.md")
        else:
            ref = s.split(f"## {name}.")[1].split("\n## ")[0].split("\n") if f"## {name}." in s else []
            diff = [ln for ln in tab.split("\n")[1:] if ln not in ref]
            print(f"{name}: {len(diff)} line(s) differ from results/expected_tables.md"); [print("   " + ln) for ln in diff]
