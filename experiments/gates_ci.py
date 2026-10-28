"""Cluster-aware intervals for the main gates, simultaneous over N, and threshold laws from upper quantiles.

Clusters are questions (LongMemEval) or conversations (LoCoMo). Rates are pooled over trials, the estimand of the
gate tables, and the bootstrap resamples clusters and takes the ratio of summed firings to summed trials.
Pointwise intervals are 95%; simultaneous
intervals use Bonferroni over the store sizes. The null maximum is summarised by its mean and its 95% quantile,
each fitted by a + b sqrt(ln N) and a + b ln N.
Usage: python gates_ci.py trials.csv out_prefix
"""
import sys, zlib
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
ALPHA, B = 0.05, 2000
t = pd.read_csv(src, keep_default_na=False)
num = ["v1_z", "v1_thr", "max_sem_sim", "max_lex_sim", "max_lex_z", "max_sem_z", "swap_p", "evt_exp_fire"]
for c in num:
    t[c] = pd.to_numeric(t[c], errors="coerce")
t["half"] = t.qid.map(lambda q: zlib.crc32(str(q).encode()) % 2)
locomo = t.qid.astype(str).str.startswith("conv-").any()
t["cluster"] = t.qid.astype(str).str.rsplit("_", n=1).str[0] if locomo else t.qid
Ns = sorted(t.N.unique())
cal, ev = t[t.half == 0], t[t.half == 1]
c0 = cal[(cal.kind == "null") & (cal.N == Ns[0])]
tau_cos = np.quantile(c0.max_sem_sim, 1 - ALPHA)
tau_z = {N: np.quantile(cal[(cal.kind == "null") & (cal.N == N)].v1_z, 1 - ALPHA) for N in Ns}
gates = {
    "fixed cosine": (lambda d: d.max_sem_sim >= tau_cos, "sem_top_ev"),
    "recall-v1": (lambda d: d.v1_z >= d.v1_thr, "v1_top_ev"),
    "EVT tail": (lambda d: d.evt_exp_fire == 1, "evt_exp_top_ev"),
    "z-score, oracle per N": (lambda d: d.v1_z >= d.N.map(tau_z), "v1_top_ev"),
    "cue-swap": (lambda d: d.swap_p <= ALPHA, "v1_top_ev"),
}
rng = np.random.default_rng(0)
z_sim = 1 - ALPHA / (2 * len(Ns))
rows = []
for N in Ns:
    d = ev[ev.N == N]
    nul, pos = d[d.kind == "null"], d[d.kind == "pos"]
    for name, (fire, top) in gates.items():
        for metric, frame, val in (("false_alarm", nul, fire(nul).astype(float)),
                                   ("hit", pos, (fire(pos) & (pos[top].astype(int) == 1)).astype(float))):
            per = pd.Series(val.values, index=frame.cluster.values).groupby(level=0).agg(["sum", "count"])
            k, n = per["sum"].values, per["count"].values
            idx = rng.integers(0, len(per), (B, len(per)))
            boots = k[idx].sum(1) / n[idx].sum(1)
            rows.append({"N": N, "gate": name, "metric": metric, "value": k.sum() / n.sum(),
                         "lo": np.quantile(boots, .025), "hi": np.quantile(boots, .975),
                         "sim_lo": np.quantile(boots, 1 - z_sim), "sim_hi": np.quantile(boots, z_sim), "clusters": len(per)})
r = pd.DataFrame(rows)
r.to_csv(out + "_ci.csv", index=False)

nul = t[t.kind == "null"]
grow = []
for col in ("max_sem_z", "max_lex_z"):
    for stat, f in (("mean", np.mean), ("q95", lambda x: np.quantile(x, 0.95))):
        y = nul.groupby("N")[col].agg(f)
        x = np.log(np.array(y.index, float))
        fits = {}
        for law, g in (("sqrt_lnN", np.sqrt(x)), ("lnN", x)):
            A = np.c_[np.ones_like(g), g]
            coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
            fits[law] = float(((A @ coef - y.values) ** 2).sum())
        grow.append({"channel": col, "statistic": stat, "first": float(y.iloc[0]), "last": float(y.iloc[-1]),
                     "rss_sqrt_lnN": fits["sqrt_lnN"], "rss_lnN": fits["lnN"]})
g = pd.DataFrame(grow)
g.to_csv(out + "_growth.csv", index=False)
pd.set_option("display.width", 220)
fa = r[r.metric == "false_alarm"]
summary = fa.groupby("gate").agg(max_fa=("value", "max"), max_hi=("hi", "max"), max_sim_hi=("sim_hi", "max"))
print(f"clusters: {r.clusters.max()} ({'conversations' if locomo else 'questions'})")
print(summary.round(3).to_string())
print(g.round(4).to_string(index=False))
