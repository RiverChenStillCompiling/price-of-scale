"""Cross-encoder gate: hit and false-alarm rates with thresholds from the calibration half (fixed at N=8, or per N)
and hit rate at exactly 5% false alarms on the evaluation half, with bootstrap intervals over questions.
Usage: python rerank_table.py rr.csv out.csv"""
import sys, zlib
import numpy as np, pandas as pd

t = pd.read_csv(sys.argv[1], keep_default_na=False)
ALPHA = 0.05
t["half"] = t.qid.map(lambda q: zlib.crc32(str(q).encode()) % 2)
cal, ev = t[(t.half == 0) & (t.kind == "null")], t[t.half == 1]
t8 = np.quantile(cal[cal.N == 8].rr_max, 1 - ALPHA)
tN = cal.groupby("N").rr_max.quantile(1 - ALPHA)
rng = np.random.default_rng(0)
rows = []
for N, d in ev.groupby("N"):
    nul, pos = d[d.kind == "null"], d[d.kind == "pos"]
    tau = np.quantile(nul.rr_max, 1 - ALPHA)
    hit = lambda th, s: ((s.rr_max >= th) & (s.rr_top_ev == 1))
    per = pos.assign(h=hit(tau, pos)).groupby("qid").h.mean().values
    b = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(1000)]
    rows.append({"N": N, "fa_fixed": (nul.rr_max >= t8).mean(), "hit_fixed": hit(t8, pos).mean(),
                 "fa_oracle": (nul.rr_max >= tN[N]).mean(), "hit_oracle": hit(tN[N], pos).mean(),
                 "hit_matched": per.mean(), "lo": np.quantile(b, .025), "hi": np.quantile(b, .975),
                 "ev_in_top20": pos.ev_in_top.mean(), "rr_top_ev_if_in_top": pos[pos.ev_in_top == 1].rr_top_ev.mean(),
                 "cos_top_ev": pos.cos_top_ev.mean(), "questions": pos.qid.nunique()})
r = pd.DataFrame(rows)
r.to_csv(sys.argv[2], index=False)
print(r.round(3).to_string(index=False))
