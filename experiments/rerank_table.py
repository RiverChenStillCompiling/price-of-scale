"""Cross-encoder gate: hit and false-alarm rates with thresholds from the calibration half (fixed at N=8, or per N)
and hit rate at exactly 5% false alarms on the evaluation half (tie-randomised threshold, as in ablation_norm.py),
with bootstrap intervals over questions that re-estimate the threshold on each replicate.
Usage: python rerank_table.py rr.csv out.csv"""
import sys, zlib
import numpy as np, pandas as pd

t = pd.read_csv(sys.argv[1], keep_default_na=False)
ALPHA = 0.05
t["half"] = t.qid.map(lambda q: zlib.crc32(str(q).encode()) % 2)
cal, ev = t[(t.half == 0) & (t.kind == "null")], t[t.half == 1]
t8 = np.quantile(cal[cal.N == 8].rr_max, 1 - ALPHA)
tN = cal.groupby("N").rr_max.quantile(1 - ALPHA)
def threshold(null_scores, alpha):
    """(tau, rho): fire if s > tau, or with probability rho if s == tau, so that the null firing rate is alpha."""
    s = np.sort(np.asarray(null_scores, float))[::-1]
    k = alpha * len(s)
    tau = s[min(int(np.floor(k)), len(s) - 1)]
    above, at = (s > tau).sum(), (s == tau).sum()
    rho = 0.0 if at == 0 else float(np.clip((k - above) / at, 0, 1))
    return tau, rho

def fire_prob(scores, tau, rho):
    s = np.asarray(scores, float)
    return np.where(s > tau, 1.0, np.where(s == tau, rho, 0.0))

rng = np.random.default_rng(0)
rows = []
for N, d in ev.groupby("N"):
    nul, pos = d[d.kind == "null"], d[d.kind == "pos"]
    hit = lambda th, s: ((s.rr_max >= th) & (s.rr_top_ev == 1))
    tau, rho = threshold(nul.rr_max, ALPHA)
    qs = np.array(sorted(set(nul.qid) | set(pos.qid)))
    ni = {q: np.flatnonzero(nul.qid.values == q) for q in qs}
    pi = {q: np.flatnonzero(pos.qid.values == q) for q in qs}
    nv, pv, pok = nul.rr_max.values, pos.rr_max.values, (pos.rr_top_ev == 1).values
    b = []
    for _ in range(1000):
        pick = qs[rng.integers(0, len(qs), len(qs))]
        a, c = np.concatenate([ni[q] for q in pick]), np.concatenate([pi[q] for q in pick])
        b.append(fire_prob(pv[c], *threshold(nv[a], ALPHA)).dot(pok[c]) / len(c))
    rows.append({"N": N, "fa_fixed": (nul.rr_max >= t8).mean(), "hit_fixed": hit(t8, pos).mean(),
                 "fa_oracle": (nul.rr_max >= tN[N]).mean(), "hit_oracle": hit(tN[N], pos).mean(),
                 "fa_matched": fire_prob(nv, tau, rho).mean(), "hit_matched": fire_prob(pv, tau, rho).dot(pok) / len(pv),
                 "lo": np.quantile(b, .025), "hi": np.quantile(b, .975),
                 "ev_in_top20": pos.ev_in_top.mean(), "rr_top_ev_if_in_top": pos[pos.ev_in_top == 1].rr_top_ev.mean(),
                 "cos_top_ev": pos.cos_top_ev.mean(), "questions": pos.qid.nunique()})
r = pd.DataFrame(rows)
r.to_csv(sys.argv[2], index=False)
print(r.round(3).to_string(index=False))
