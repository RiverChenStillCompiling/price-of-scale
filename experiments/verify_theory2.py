"""Checks of Proposition 1 and Corollary 5 against the oracle gate.

(a) Global threshold, exact per-cue model: under the trial design (store = N-1 pool items drawn
    without replacement + 1 evidence item), Hit_c(tau) = mean_y 1{y >= tau} C(K - m(y), N-1) / C(K, N-1)
    with m(y) = #pool >= y. Report mean Hit_c, mean Gbar_c, (1-alpha) mean Gbar_c and Cov_c(alpha_c, Gbar_c)
    at the theory threshold and at the empirical oracle threshold (from the calibration-half trials).
(b) Per-cue thresholds (alpha_c = alpha for every cue): exact mean Hit, its local log-log slope, and the
    binormal prediction kappa_eff(N) = (1/s^2)(1 - d / sqrt(2 ln(N/lambda))) from per-cue standardised scores.
Usage: python verify_theory2.py scores.npz trials.csv out.json
"""
import sys, json, zlib, math
import numpy as np, pandas as pd
from scipy.special import gammaln
from scipy.optimize import brentq

scores, trials_csv, out = sys.argv[1:4]
ALPHA = 0.05
LAM = -math.log(1 - ALPHA)
Ns = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
z = np.load(scores)
sem, pool, pp, evid, ep, qids = z["sem"], z["pool"], z["pool_ptr"], z["evid"], z["evid_ptr"], z["qids"]
cues = {0: [], 1: []}
for q, qid in enumerate(qids):
    e = evid[ep[q]:ep[q + 1]]
    if not len(e) or str(qid).endswith("_abs"):
        continue
    s = sem[q].astype(np.float64)
    cues[zlib.crc32(str(qid).encode()) % 2].append((np.sort(s[pool[pp[q]:pp[q + 1]]]), s[e]))

def lchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def quiet(null_sorted, y, n):
    """P(all n items drawn without replacement from the pool are below y)."""
    K = len(null_sorted)
    m = K - np.searchsorted(null_sorted, y, side="left")
    return np.where(K - m >= n, np.exp(lchoose(K - m, n) - lchoose(K, n)), 0.0)

def alpha_c(n, tau, N):
    return 1 - float(quiet(n, np.array([tau]), N)[0])

def hit_c(n, e, tau, N):
    return float(np.mean((e >= tau) * quiet(n, e, N - 1)))

t = pd.read_csv(trials_csv, keep_default_na=False)
t["half"] = t.qid.map(lambda q: zlib.crc32(str(q).encode()) % 2)
res = {"global": {}, "per_cue": {}}
for N in Ns:
    cal = [c for c in cues[0] if len(c[0]) >= N]
    ev = [c for c in cues[1] if len(c[0]) >= N]
    if not ev:
        continue
    tau_th = brentq(lambda x: np.mean([alpha_c(n, x, N) for n, _ in cal]) - ALPHA, -1, 1.5, xtol=1e-6)
    tau_emp = float(np.quantile(t[(t.half == 0) & (t.kind == "null") & (t.N == N)].max_sem_sim, 1 - ALPHA))
    emp = t[(t.half == 1) & (t.N == N)]
    pos = emp[emp.kind == "pos"]
    row = {"empirical_hit": float(((pos.max_sem_sim >= tau_emp) & (pos.sem_top_ev == 1)).mean()),
           "empirical_fa": float((emp[emp.kind == "null"].max_sem_sim >= tau_emp).mean())}
    for name, tau in (("theory_tau", tau_th), ("empirical_tau", tau_emp)):
        a = np.array([alpha_c(n, tau, N) for n, _ in ev])
        g = np.array([(e >= tau).mean() for _, e in ev])
        h = np.array([hit_c(n, e, tau, N) for n, e in ev])
        row[name] = {"tau": tau, "mean_alpha_c": float(a.mean()), "exact_hit": float(h.mean()),
                     "upper_mean_G": float(g.mean()), "naive_lower": float((1 - a.mean()) * g.mean()),
                     "lower_mean_(1-a_c)G_c": float(((1 - a) * g).mean()), "cov_alpha_G": float(np.cov(a, g, bias=True)[0, 1])}
    res["global"][N] = row

    # per-cue thresholds with alpha_c = alpha exactly
    hs = []
    for n, e in ev:
        tau_c = brentq(lambda x: alpha_c(n, x, N) - ALPHA, n[0] - 1, n[-1] + 1e-9, xtol=1e-7)
        hs.append(hit_c(n, e, tau_c, N))
    res["per_cue"][N] = {"exact_hit": float(np.mean(hs))}

# binormal fit on per-cue standardised scores (evaluation half)
zs = np.concatenate([(e - n.mean()) / n.std() for n, e in cues[1]])
d, s = float(zs.mean()), float(zs.std())
res["binormal"] = {"d": d, "s": s, "kappa_limit": 1 / s ** 2}
Nk = [N for N in Ns if N in res["per_cue"]]
h = np.array([res["per_cue"][N]["exact_hit"] for N in Nk])
for a, b in zip(Nk[:-1], Nk[1:]):
    slope = -math.log(res["per_cue"][b]["exact_hit"] / res["per_cue"][a]["exact_hit"]) / math.log(b / a)
    mid = math.sqrt(a * b)
    res["per_cue"][b]["local_slope"] = slope
    res["per_cue"][b]["kappa_eff_binormal"] = (1 / s ** 2) * (1 - d / math.sqrt(2 * math.log(mid / LAM)))
json.dump(res, open(out, "w"), indent=1)

print(f"binormal fit: d={d:.3f} s={s:.3f}  1/s^2={1 / s ** 2:.3f}")
print("N     emp_hit | theory tau: exact  [naive_low, upper]  mixed_low  cov | emp tau: exact  upper  cov")
for N, r in res["global"].items():
    a, b = r["theory_tau"], r["empirical_tau"]
    print(f"{N:<5} {r['empirical_hit']:.3f}  | {a['exact_hit']:.3f} [{a['naive_lower']:.3f}, {a['upper_mean_G']:.3f}] "
          f"{a['lower_mean_(1-a_c)G_c']:.3f} {a['cov_alpha_G']:+.4f} | {b['exact_hit']:.3f} {b['upper_mean_G']:.3f} {b['cov_alpha_G']:+.4f}")
print("per-cue thresholds: N  exact_hit  local_slope  kappa_eff(binormal)")
for N, r in res["per_cue"].items():
    print(f"  {N:<5} {r['exact_hit']:.3f}  {r.get('local_slope', float('nan')):.3f}  {r.get('kappa_eff_binormal', float('nan')):.3f}")
