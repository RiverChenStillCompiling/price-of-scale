"""Hit rates when the positive store holds several relevant memories, against one evidence turn.

Thresholds of the label-tuned gates come from the null trials of the main run (calibration half); the positive
trials of each variant are evaluated on the evaluation half with the same gates.
Usage: python sensitivity.py trials_main.csv out.csv variant=trials.csv [...]
"""
import sys, zlib
import numpy as np, pandas as pd

main, out = sys.argv[1], sys.argv[2]
variants = dict(a.split("=", 1) for a in sys.argv[3:])
ALPHA = 0.05
read = lambda f: pd.read_csv(f, keep_default_na=False)
num = ["v1_z", "v1_thr", "max_sem_sim", "swap_p", "evt_exp_fire"]
def prep(t):
    for c in num:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["half"] = t.qid.map(lambda q: zlib.crc32(str(q).encode()) % 2)
    return t
m = prep(read(main))
cn = m[(m.half == 0) & (m.kind == "null")]
tau_z = cn.groupby("N").v1_z.quantile(1 - ALPHA)
tau_c = cn.groupby("N").max_sem_sim.quantile(1 - ALPHA)
gates = {"recall-v1": lambda d: (d.v1_z >= d.v1_thr) & (d.v1_top_ev.astype(int) == 1),
         "z-score, oracle per N": lambda d: (d.v1_z >= d.N.map(tau_z)) & (d.v1_top_ev.astype(int) == 1),
         "cosine, oracle per N": lambda d: (d.max_sem_sim >= d.N.map(tau_c)) & (d.sem_top_ev.astype(int) == 1),
         "EVT tail": lambda d: (d.evt_exp_fire == 1) & (d.evt_exp_top_ev.astype(int) == 1),
         "cue-swap": lambda d: (d.swap_p <= ALPHA) & (d.v1_top_ev.astype(int) == 1)}
rows = []
for name, f in {"one": main, **variants}.items():
    t = m if name == "one" else prep(read(f))
    p = t[(t.half == 1) & (t.kind == "pos")]
    for N, d in p.groupby("N"):
        for g, fire in gates.items():
            rows.append({"variant": name, "N": N, "gate": g, "hit": fire(d).mean(), "trials": len(d),
                         "questions": d.qid.nunique()})
r = pd.DataFrame(rows)
r.to_csv(out, index=False)
pd.set_option("display.width", 200)
print(r.pivot_table(index=["gate", "N"], columns="variant", values="hit").round(3).to_string())
print(r.pivot_table(index="N", columns="variant", values="questions").to_string())
