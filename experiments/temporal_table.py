"""Summary of the chronological evaluation: interruptions per opening turn and hit rates of the question cues.
Usage: python temporal_table.py trials_full.csv out_prefix temporal_*.csv"""
import sys, zlib
import numpy as np, pandas as pd

cal_csv, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
ALPHA = 0.05
t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
cal = pd.read_csv(cal_csv, keep_default_na=False)
cal = cal[(cal.qid.map(lambda q: zlib.crc32(str(q).encode()) % 2) == 0) & (cal.kind == "null")]
for c in ("max_sem_sim", "max_lex_sim"):
    cal[c] = pd.to_numeric(cal[c])
tc = np.quantile(cal[cal.N == 8].max_sem_sim, 1 - ALPHA)
tl = np.quantile(cal[cal.N == 8].max_lex_sim, 1 - ALPHA)
fire = {"fixed cosine": t.max_sem >= tc, "fixed tf-idf": t.max_lex >= tl, "recall-v1": t.v1_z >= t.v1_thr,
        "z-score, transferred": t.v1_z >= t.tau_z, "EVT tail": t.evt == 1, "cue-swap": t.swap_p <= ALPHA}
top = {"fixed cosine": "sem_top_ev", "fixed tf-idf": "lex_top_ev", "recall-v1": "v1_top_ev",
       "z-score, transferred": "v1_top_ev", "cue-swap": "v1_top_ev"}
rng = np.random.default_rng(0)
def ci(v, g):
    per = pd.Series(v.values.astype(float), index=g.values).groupby(level=0).mean()
    b = [per.values[rng.integers(0, len(per), len(per))].mean() for _ in range(1000)]
    return per.mean(), np.quantile(b, .025), np.quantile(b, .975)
rows = []
turn = t.kind == "turn"
for tN in sorted(t[turn].target.unique()):
    d = turn & (t.target == tN)
    for g, f in fire.items():
        m, lo, hi = ci(f[d], t.qid[d])
        rows.append({"part": "interruption", "N": tN, "gate": g, "value": m, "lo": lo, "hi": hi, "n": int(d.sum()),
                     "K_median": float(t.K[d].median())})
for kind in ("q_after", "q_end"):
    for ct in ("explicit", "implicit"):
        d = (t.kind == kind) & (t.cue_type == ct)
        for g in top:
            m, lo, hi = ci(fire[g][d] & (t[top[g]][d] == 1), t.qid[d])
            rows.append({"part": f"{kind}/{ct}", "N": float(t.N[d].median()), "gate": g, "value": m, "lo": lo, "hi": hi, "n": int(d.sum())})
rng = np.random.default_rng(1)  # separate stream, so the intervals above do not change
for kind in ("q_after", "q_end"):
    for ct in ("explicit", "implicit"):
        d = (t.kind == kind) & (t.cue_type == ct)
        m, lo, hi = ci(fire["EVT tail"][d] & (t.evt_top_ev[d] == 1), t.qid[d])
        rows.append({"part": f"{kind}/{ct}", "N": float(t.N[d].median()), "gate": "EVT tail", "value": m, "lo": lo, "hi": hi, "n": int(d.sum())})
r = pd.DataFrame(rows)
r.to_csv(out + ".csv", index=False)
pd.set_option("display.width", 220)
print(r[r.part == "interruption"].pivot(index="N", columns="gate", values="value").round(3).to_string())
print(r[r.part == "interruption"].groupby("N")[["n", "K_median"]].first().to_string())
print(r[r.part != "interruption"].pivot(index="gate", columns="part", values="value").round(3).to_string())
print(r[r.part != "interruption"].groupby("part")[["N", "n"]].first().to_string())
print(r[r.gate == "cue-swap"].round(3).to_string())
