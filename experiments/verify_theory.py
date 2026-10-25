"""Check Proposition 1 against the per-N oracle gate.

For a global cosine threshold tau, a store of N items drawn without replacement from cue c's
null pool (K_c items, m_c(tau) of them >= tau) stays silent with probability C(K-m, N) / C(K, N).
tau_N solves  mean_c P(fire | c) = alpha  on the calibration half; the predicted hit on the
evaluation half is sandwiched between (1-alpha) * R and R, R = mean_c P(evidence >= tau_N | c).
Also fits the power-law exponent kappa of the hit decay.
Usage: python verify_theory.py scores.npz gates.csv out.json
"""
import sys, json, zlib
import numpy as np
from scipy.special import gammaln
from scipy.optimize import brentq

scores, gates_csv, out = sys.argv[1:4]
ALPHA = 0.05
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

def p_fire(null_sorted, tau, N):
    K = len(null_sorted)
    m = K - np.searchsorted(null_sorted, tau, side="left")
    if K - m < N:
        return 1.0
    return 1 - np.exp(lchoose(K - m, N) - lchoose(K, N))

res = {}
for N in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    cal = [c for c in cues[0] if len(c[0]) >= N]
    ev = [c for c in cues[1] if len(c[0]) >= N]
    f = lambda tau: np.mean([p_fire(n, tau, N) for n, _ in cal]) - ALPHA
    tau = brentq(f, -1, 1.5, xtol=1e-5)
    R = float(np.mean([(e >= tau).mean() for _, e in ev]))
    fa_eval = float(np.mean([p_fire(n, tau, N) for n, _ in ev]))
    res[N] = {"tau": tau, "R_upper": R, "lower": (1 - ALPHA) * R, "fa_eval_pred": fa_eval}

import pandas as pd
g = pd.read_csv(gates_csv)
o = g[g.gate == "cosine, re-tuned per N (oracle)"].set_index("N")
for N in res:
    res[N]["oracle_hit_empirical"] = float(o.loc[N, "hit"])
    res[N]["oracle_fa_empirical"] = float(o.loc[N, "false_alarm"])
Ns = np.array(list(res))
for key in ["R_upper", "oracle_hit_empirical"]:
    y = np.array([res[N][key] for N in Ns])
    big = Ns >= 128
    res[f"kappa_{key}_N>=128"] = float(-np.polyfit(np.log(Ns[big]), np.log(y[big]), 1)[0])
json.dump(res, open(out, "w"), indent=1)
for N in Ns:
    r = res[N]
    print(f"N={N:5d} tau={r['tau']:.3f}  predicted hit in [{r['lower']:.3f}, {r['R_upper']:.3f}]  "
          f"empirical {r['oracle_hit_empirical']:.3f}   FA pred {r['fa_eval_pred']:.3f} emp {r['oracle_fa_empirical']:.3f}")
print({k: round(v, 3) for k, v in res.items() if str(k).startswith("kappa")})
