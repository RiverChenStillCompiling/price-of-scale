"""Summary of the TriggerBench runs: hit rate (pos), firing rate on resolved negatives (neg) and false alarms (null).
Fixed and oracle thresholds are tuned on the null condition of the calibration half.
Usage: python triggerbench_table.py tb_trials.csv out.csv"""
import sys
import numpy as np, pandas as pd

t = pd.read_csv(sys.argv[1], keep_default_na=False)
ALPHA = 0.05
t["Nb"] = np.where(t["size"] == "natural", t["bin"], t.N.astype(str))
cal = t[(t.half == 0) & (t.cond == "null")]
tc = np.quantile(cal[(cal["size"] == "grown") & (cal.N == 8)].max_sem, 1 - ALPHA)
tz = cal.groupby("Nb").v1_z.quantile(1 - ALPHA)
ev = t[t.half == 1].copy()
fire = {"fixed cosine": (ev.max_sem >= tc, "sem_top_ev"), "recall-v1": (ev.v1_z >= ev.v1_thr, "v1_top_ev"),
        "z-score, oracle": (ev.v1_z >= ev.Nb.map(tz), "v1_top_ev"), "EVT tail": (ev.evt == 1, "evt_top_ev"),
        "cue-swap": (ev.swap_p <= ALPHA, "v1_top_ev")}
rows = []
for (nb, cond), d in ev.groupby(["Nb", "cond"]):
    for g, (f, top) in fire.items():
        v = f[d.index] & (d[top] == 1) if cond == "pos" else f[d.index]
        rows.append({"store": nb, "cond": cond, "gate": g, "rate": v.mean(), "n": len(d), "tasks": d.task.nunique(),
                     "N_median": d.N.median()})
r = pd.DataFrame(rows)
r.to_csv(sys.argv[2], index=False)
order = ["8", "base", "32", "128", "40K", "512", "2048"]
p = r.pivot_table(index=["cond", "gate"], columns="store", values="rate")
pd.set_option("display.width", 220)
print(p[[c for c in order if c in p.columns]].round(3).to_string())
print(r.groupby("store")[["N_median"]].first().T.to_string())
by_cat = ev[(ev.cond == "pos") & (ev["size"] == "natural")].groupby(["bin", "cat"]).apply(
    lambda d: pd.Series({"swap_hit": ((d.swap_p <= ALPHA) & (d.v1_top_ev == 1)).mean(), "sem_top_ev": d.sem_top_ev.mean(), "n": len(d)}))
print(by_cat.round(3).to_string())
