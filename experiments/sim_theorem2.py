"""Theorem 2 and Example 3 on scores with known tails.

Exponential model: X ~ Exp(1), Y ~ Exp(mu) gives R(p) = p^mu exactly, so kappa = mu and ell = 1. The exact hit rate
mu * integral_0^{p_N} (1-u)^{N-1} u^{mu-1} du is compared with Monte Carlo and with the limit C(alpha, mu) lambda^mu N^-mu.
Binormal model: X ~ N(0,1), Y ~ N(d, s^2); the exact hit rate from Proposition 1 against N, and the local exponent.
Usage: python sim_theorem2.py out.json
"""
import sys, json, math
import numpy as np
from scipy import integrate, special, stats

out = sys.argv[1]
ALPHA = 0.05
LAM = -math.log(1 - ALPHA)
rng = np.random.default_rng(3)
Ns = [8, 64, 512, 4096, 32768, 262144]
res = {"exponential": {}, "binormal": {}}

def C(kappa):
    return integrate.quad(lambda v: kappa * v ** (kappa - 1) * (1 - ALPHA) ** v, 0, 1)[0]

for mu in (0.3, 0.6):
    rows = {}
    for N in Ns:
        p = 1 - (1 - ALPHA) ** (1 / N)
        exact = mu * special.beta(mu, N) * special.betainc(mu, N, p)
        limit = C(mu) * LAM ** mu * N ** -mu
        mc = None
        if N <= 4096:
            t = -math.log(p)                                  # F-bar(t) = e^{-t} = p_N
            M = 20000
            y = rng.exponential(1 / mu, M)
            mx = rng.exponential(1, (M, N - 1)).max(1) if N > 1 else np.full(M, -np.inf)
            mc = float(((y >= t) & (y > mx)).mean())
        rows[N] = {"exact": float(exact), "monte_carlo": mc, "limit": float(limit), "ratio_exact_to_limit": float(exact / limit)}
    res["exponential"][mu] = {"C": C(mu), "rows": rows}

for d, s in ((2.0, 1.0), (3.6, 1.3)):
    rows = {}
    for N in Ns:
        p = 1 - (1 - ALPHA) ** (1 / N)
        R = lambda u: stats.norm.sf((stats.norm.isf(u) - d) / s)
        exact = integrate.quad(lambda u: (1 - u) ** (N - 1) * (R(u * (1 + 1e-6)) - R(u)) / (u * 1e-6), 0, p, limit=200)[0]
        t = stats.norm.isf(p)
        h = lambda x: stats.norm.pdf(x) / stats.norm.sf(x)
        rows[N] = {"hit_exact": float(exact), "upper_R": float(R(p)), "kappa_eff": float(h((t - d) / s) / (s * h(t))), "t": float(t)}
    res["binormal"][f"d={d},s={s}"] = {"kappa_limit": 1 / s ** 2, "rows": rows}

json.dump(res, open(out, "w"), indent=1)
for mu, r in res["exponential"].items():
    print(f"exponential mu={mu}: C={r['C']:.4f}")
    for N, x in r["rows"].items():
        mc = "" if x["monte_carlo"] is None else f" MC {x['monte_carlo']:.4f}"
        print(f"  N={N:7d} exact {x['exact']:.5f}{mc} limit {x['limit']:.5f} ratio {x['ratio_exact_to_limit']:.4f}")
for k, r in res["binormal"].items():
    print(f"binormal {k}: 1/s^2={r['kappa_limit']:.3f}")
    for N, x in r["rows"].items():
        print(f"  N={N:7d} hit {x['hit_exact']:.4f} R(p_N) {x['upper_R']:.4f} kappa_eff {x['kappa_eff']:.3f} t {x['t']:.2f}")
