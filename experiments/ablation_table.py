"""Summarise the cue-swap ablation: false alarms and hits by K, alpha and N.
Usage: python ablation_table.py ablation.csv [more.csv ...]
"""
import sys
import numpy as np, pandas as pd

def boot_hi(per_q, B=1000, seed=0):
    r = np.random.default_rng(seed)
    v = per_q.values
    return float(np.quantile([v[r.integers(0, len(v), len(v))].mean() for _ in range(B)], .975))

for f in sys.argv[1:]:
    t = pd.read_csv(f, keep_default_na=False)
    rows = []
    for (K, a, N), d in t.groupby(["K", "alpha", "N"]):
        nul, pos = d[d.kind == "null"], d[d.kind == "pos"]
        rows.append({"K": K, "alpha": a, "N": N, "fa": nul.fire.mean(), "fa_hi": boot_hi(nul.groupby("qid").fire.mean()),
                     "hit": ((pos.fire == 1) & (pos.top_ev == 1)).mean()})
    r = pd.DataFrame(rows)
    print(f"== {f}  ({t.qid.nunique()} questions)")
    print(r.pivot_table(index=["alpha", "K"], columns="N", values=["fa", "hit"]).round(3).to_string())
    worst = r.groupby(["alpha", "K"]).apply(lambda x: pd.Series({"max_fa": x.fa.max(), "max_fa_hi": x.fa_hi.max()}), include_groups=False)
    print(worst.round(3).to_string())
    r.to_csv(f.replace(".csv", "_summary.csv"), index=False)
