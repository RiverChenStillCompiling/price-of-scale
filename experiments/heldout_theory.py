"""Out-of-sample checks of the finite-pool model.

(a) Across users: the threshold and the exact finite-pool prediction come only from the calibration half
    (its null trials and its cues' score distributions); the prediction is compared with the hit rate observed
    on the evaluation half, whose questions, users and histories take no part in it.
(b) Across store sizes: an effective power law fitted to observed hit rates at N in {32, 64, 128} is
    extrapolated to N in {256, ..., 2048}.
Usage: python heldout_theory.py scores.npz trials.csv out.json
"""
import sys, json, zlib, math
import numpy as np, pandas as pd
from scipy.special import gammaln

scores, trials_csv, out = sys.argv[1:4]
ALPHA, B = 0.05, 1000
Ns = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
z = np.load(scores)
sem, pool, pp, evid, ep, qids = z["sem"], z["pool"], z["pool_ptr"], z["evid"], z["evid_ptr"], z["qids"]
cues = {0: [], 1: []}
for q, qid in enumerate(qids):
    e = evid[ep[q]:ep[q + 1]]
    if len(e) and not str(qid).endswith("_abs"):
        s = sem[q].astype(np.float64)
        cues[zlib.crc32(str(qid).encode()) % 2].append((np.sort(s[pool[pp[q]:pp[q + 1]]]), s[e]))

def lchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def quiet(null_sorted, y, n):
    K = len(null_sorted)
    m = K - np.searchsorted(null_sorted, y, side="left")
    return np.where(K - m >= n, np.exp(lchoose(K - m, n) - lchoose(K, n)), 0.0)

def hit_c(n, e, tau, N):
    return float(np.mean((e >= tau) * quiet(n, e, N - 1)))

t = pd.read_csv(trials_csv, keep_default_na=False)
t["half"] = t.qid.map(lambda q: zlib.crc32(str(q).encode()) % 2)
rng = np.random.default_rng(0)
res = {"across_users": {}, "across_N": {}}
obs = {}
for N in Ns:
    cal_cues = [c for c in cues[0] if len(c[0]) >= N]
    if not cal_cues:
        continue
    tau = float(np.quantile(t[(t.half == 0) & (t.kind == "null") & (t.N == N)].max_sem_sim, 1 - ALPHA))
    pred_c = np.array([hit_c(n, e, tau, N) for n, e in cal_cues])
    pos = t[(t.half == 1) & (t.N == N) & (t.kind == "pos")]
    per_q = ((pos.max_sem_sim >= tau) & (pos.sem_top_ev == 1)).groupby(pos.qid).mean()
    pb = [pred_c[rng.integers(0, len(pred_c), len(pred_c))].mean() for _ in range(B)]
    ob = [per_q.values[rng.integers(0, len(per_q), len(per_q))].mean() for _ in range(B)]
    obs[N] = float(per_q.mean())
    res["across_users"][N] = {"tau": tau, "predicted": float(pred_c.mean()), "pred_lo": float(np.quantile(pb, .025)),
                              "pred_hi": float(np.quantile(pb, .975)), "observed": obs[N],
                              "obs_lo": float(np.quantile(ob, .025)), "obs_hi": float(np.quantile(ob, .975)),
                              "p_N": 1 - (1 - ALPHA) ** (1 / N), "pool_resolution": 1 / float(np.median([len(n) for n, _ in cal_cues]))}

fit_N = [n for n in (32, 64, 128) if n in obs]
slope, icpt = np.polyfit(np.log(fit_N), np.log([obs[n] for n in fit_N]), 1)
res["across_N"] = {"fitted_on": fit_N, "kappa_eff": float(-slope),
                   "extrapolated": {n: {"predicted": float(math.exp(icpt + slope * math.log(n))), "observed": obs[n]}
                                    for n in obs if n > 128}}
json.dump(res, open(out, "w"), indent=1)
print("across users (threshold and prediction from the calibration half only):")
for N, r in res["across_users"].items():
    print(f"  N={N:5d} predicted {r['predicted']:.3f} [{r['pred_lo']:.3f}, {r['pred_hi']:.3f}]  observed {r['observed']:.3f} "
          f"[{r['obs_lo']:.3f}, {r['obs_hi']:.3f}]  p_N={r['p_N']:.1e} vs 1/M={r['pool_resolution']:.1e}")
print(f"across N: kappa_eff={res['across_N']['kappa_eff']:.3f} fitted on {fit_N}")
for n, r in res["across_N"]["extrapolated"].items():
    print(f"  N={n:5d} extrapolated {r['predicted']:.3f}  observed {r['observed']:.3f}")
