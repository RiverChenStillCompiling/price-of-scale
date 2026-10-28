"""Which component of the recall-v1 score buys hit rate, at exactly matched false alarms.

Scorers (best score over the store, and whether its arg-max is the evidence turn):
  raw cosine        max_sem_sim / sem_top_ev
  semantic z        max_sem_z   / sem_z_top_ev     per-cue robust z of the cosine, no eligibility
  semantic z, elig  semel_z     / semel_top_ev     plus the cosine >= 0.55 rule
  both z            both_z      / both_top_ev      lexical and semantic z, no eligibility
  recall-v1         v1_z        / v1_top_ev        both channels with eligibility
  raw tf-idf        max_lex_sim / lex_top_ev

Matched: on the evaluation half, each scorer gets the threshold whose (tie-randomised) false-alarm rate on null
trials is exactly alpha. Calibrated: threshold from the calibration half's null trials, reported on the evaluation
half. Hit rates are pooled over trials. Paired differences use a cluster bootstrap over questions, or over
conversations for LoCoMo, that resamples null and positive trials together and re-estimates every threshold on
each replicate, so the intervals include the uncertainty of the matched thresholds.
Usage: python ablation_norm.py trials.csv out_prefix
"""
import sys, zlib
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
ALPHA, B = 0.05, 2000
SCORERS = {"raw cosine": ("max_sem_sim", "sem_top_ev"), "semantic z": ("max_sem_z", "sem_z_top_ev"),
           "semantic z, elig": ("semel_z", "semel_top_ev"), "both z": ("both_z", "both_top_ev"),
           "recall-v1": ("v1_z", "v1_top_ev"), "raw tf-idf": ("max_lex_sim", "lex_top_ev")}
PAIRS = [("semantic z", "raw cosine"), ("semantic z, elig", "semantic z"), ("recall-v1", "semantic z, elig"),
         ("recall-v1", "raw cosine"), ("both z", "semantic z")]

t = pd.read_csv(src, keep_default_na=False)
for c in [c for v in SCORERS.values() for c in v]:
    t[c] = pd.to_numeric(t[c], errors="coerce")
t["half"] = t.qid.map(lambda q: zlib.crc32(str(q).encode()) % 2)
locomo = t.qid.astype(str).str.startswith("conv-").any()
t["cluster"] = t.qid.astype(str).str.rsplit("_", n=1).str[0] if locomo else t.qid

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

rows, hit_pt, curves, boot_data = [], {}, [], {}
for N, d in t.groupby("N"):
    ev, cal = d[d.half == 1], d[d.half == 0]
    nul, pos = ev[ev.kind == "null"], ev[ev.kind == "pos"]
    for name, (sc, top) in SCORERS.items():
        tau, rho = threshold(nul[sc], ALPHA)
        h = fire_prob(pos[sc], tau, rho) * (pos[top] == 1)
        ctau, crho = threshold(cal[cal.kind == "null"][sc], ALPHA)
        rows.append({"N": N, "scorer": name, "hit_matched": h.mean(),
                     "fa_calibrated": fire_prob(nul[sc], ctau, crho).mean(),
                     "hit_calibrated": (fire_prob(pos[sc], ctau, crho) * (pos[top] == 1)).mean()})
        for a in (0.01, 0.02, 0.05, 0.1, 0.2):
            ta, ra = threshold(nul[sc], a)
            curves.append({"N": N, "scorer": name, "alpha": a, "hit": (fire_prob(pos[sc], ta, ra) * (pos[top] == 1)).mean()})
    hit_pt[N] = {name: fire_prob(pos[sc], *threshold(nul[sc], ALPHA)).dot(pos[top] == 1) / len(pos)
                for name, (sc, top) in SCORERS.items()}
    boot_data[N] = (nul, pos)
res = pd.DataFrame(rows)

rng = np.random.default_rng(0)
diffs = []
for N in sorted(t.N.unique()):
    nul, pos = boot_data[N]
    cl = np.array(sorted(set(nul.cluster) | set(pos.cluster)))
    nidx = {c: np.flatnonzero(nul.cluster.values == c) for c in cl}
    pidx = {c: np.flatnonzero(pos.cluster.values == c) for c in cl}
    ns = {n: nul[sc].values for n, (sc, _) in SCORERS.items()}
    ps = {n: (pos[sc].values, (pos[top] == 1).values) for n, (sc, top) in SCORERS.items()}
    boots = {pair: [] for pair in PAIRS}
    for _ in range(B):
        pick = cl[rng.integers(0, len(cl), len(cl))]
        ni, pi = np.concatenate([nidx[c] for c in pick]), np.concatenate([pidx[c] for c in pick])
        hit = {}
        for n in {x for pair in PAIRS for x in pair}:
            tau, rho = threshold(ns[n][ni], ALPHA)
            v, ok = ps[n][0][pi], ps[n][1][pi]
            hit[n] = fire_prob(v, tau, rho).dot(ok) / len(pi)
        for a, b in PAIRS:
            boots[(a, b)].append(hit[a] - hit[b])
    for a, b in PAIRS:
        diffs.append({"N": N, "comparison": f"{a} - {b}", "diff": hit_pt[N][a] - hit_pt[N][b],
                      "lo": np.quantile(boots[(a, b)], .025), "hi": np.quantile(boots[(a, b)], .975), "clusters": len(cl)})
dif = pd.DataFrame(diffs)
res.to_csv(out + "_matched.csv", index=False)
dif.to_csv(out + "_paired.csv", index=False)
pd.DataFrame(curves).to_csv(out + "_curves.csv", index=False)
pd.set_option("display.width", 200)
print(res.pivot(index="N", columns="scorer", values="hit_matched").round(3).to_string())
print(dif.round(3).to_string(index=False))
