"""Gates, false alarms vs N, and how the null maximum grows.

Questions are split in half by hash: thresholds that need tuning are tuned on the
calibration half at the smallest N (the moment a deployed system would be tuned),
then everything is reported on the other half.
"""
import sys, zlib, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, outdir = sys.argv[1], sys.argv[2]
ALPHA = 0.05
t = pd.read_csv(src, keep_default_na=False)
t["half"] = t.qid.map(lambda q: zlib.crc32(str(q).encode()) % 2)
cal, ev = t[t.half == 0], t[t.half == 1]
Ns = sorted(t.N.unique())
n0 = Ns[0]

def q95(df, col):
    return float(np.quantile(df[col], 1 - ALPHA))

c0 = cal[(cal.kind == "null") & (cal.N == n0)]
tau_sem, tau_lex, tau_v1 = q95(c0, "max_sem_sim"), q95(c0, "max_lex_sim"), float(c0.v1_thr.iloc[0])
tau_sem_N = {N: q95(cal[(cal.kind == "null") & (cal.N == N)], "max_sem_sim") for N in Ns}
# matched-false-alarm versions: every score gets its own threshold per N on the calibration half
def per_N(col, sign=1):
    return {N: sign * q95(cal[(cal.kind == "null") & (cal.N == N)].assign(v=lambda d: sign * d[col]), "v") for N in Ns}
tau_v1_N = per_N("v1_z")
tau_evt_N = per_N("evt_exp_padj", -1)

def gates(df):
    return {
        "recall-v1 (N-scaled z)": (df.v1_z >= df.v1_thr, df.v1_top_ev == 1),
        "recall-v1, threshold frozen at N0": (df.v1_z >= tau_v1, df.v1_top_ev == 1),
        "fixed cosine (tuned at N0)": (df.max_sem_sim >= tau_sem, df.sem_top_ev == 1),
        "fixed tf-idf (tuned at N0)": (df.max_lex_sim >= tau_lex, df.lex_top_ev == 1),
        "cosine, re-tuned per N (oracle)": (df.max_sem_sim >= df.N.map(tau_sem_N), df.sem_top_ev == 1),
        "EVT tail, exponential (no tuning)": (df.evt_exp_fire == 1, df.evt_exp_top_ev == 1),
        "EVT tail, GPD (no tuning)": (df.evt_gpd_fire == 1, df.evt_gpd_top_ev == 1),
        "recall-v1 z, re-tuned per N": (df.v1_z >= df.N.map(tau_v1_N), df.v1_top_ev == 1),
        "EVT-exp p, re-tuned per N": (df.evt_exp_padj <= df.N.map(tau_evt_N), df.evt_exp_top_ev == 1),
        "cue-swap conformal (no tuning)": (df.swap_p <= ALPHA, df.v1_top_ev == 1),
    }

def boot(per_q, B=1000, seed=0):
    r = np.random.default_rng(seed)
    v = per_q.values
    m = [v[r.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return float(v.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))

rows = []
for N in Ns:
    e = ev[ev.N == N]
    nul, pos = e[e.kind == "null"], e[e.kind == "pos"]
    gn, gp = gates(nul), gates(pos)
    for g in gn:
        fa = boot(pd.Series(gn[g][0].values, index=nul.qid).groupby(level=0).mean())
        hit = boot(pd.Series((gp[g][0] & gp[g][1]).values, index=pos.qid).groupby(level=0).mean())
        rows.append({"N": N, "gate": g, "false_alarm": fa[0], "fa_lo": fa[1], "fa_hi": fa[2],
                     "hit": hit[0], "hit_lo": hit[1], "hit_hi": hit[2], "n_questions": nul.qid.nunique()})
res = pd.DataFrame(rows)
res.to_csv(f"{outdir}/gates.csv", index=False)

# ---- growth of the null maximum: sqrt(ln N) (sub-gaussian) vs ln N (sub-exponential) ----
nul = t[t.kind == "null"]
growth = {}
for col in ["max_sem_z", "max_lex_z", "max_sem_sim", "max_lex_sim"]:
    y = nul.groupby("N")[col].mean()
    x = np.log(np.array(y.index, float))
    fits = {}
    for name, f in {"sqrt_lnN": np.sqrt(x), "lnN": x}.items():
        A = np.c_[np.ones_like(f), f]
        coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
        rss = float(((A @ coef - y.values) ** 2).sum())
        fits[name] = {"a": float(coef[0]), "b": float(coef[1]), "rss": rss}
    growth[col] = {"mean_by_N": {int(k): float(v) for k, v in y.items()}, "fits": fits}
json.dump({"tau_sem": tau_sem, "tau_lex": tau_lex, "tau_v1": tau_v1, "alpha": ALPHA, "N0": int(n0),
           "growth": growth}, open(f"{outdir}/summary.json", "w"), indent=1)

# ---- figures ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for g, d in res.groupby("gate", sort=False):
    ax[0].plot(d.N, d.false_alarm, marker="o", label=g)
    ax[0].fill_between(d.N, d.fa_lo, d.fa_hi, alpha=.15)
    ax[1].plot(d.N, d.hit, marker="o", label=g)
    ax[1].fill_between(d.N, d.hit_lo, d.hit_hi, alpha=.15)
ax[0].axhline(ALPHA, ls=":", c="gray")
for a, yl in zip(ax, ["false-alarm rate (null cues)", "hit rate (cue with one relevant memory)"]):
    a.set_xscale("log", base=2); a.set_xlabel("memory store size N"); a.set_ylabel(yl); a.grid(alpha=.3)
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{outdir}/gates.png", dpi=160)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, cols in zip(ax, [["max_sem_z", "max_lex_z"], ["max_sem_sim", "max_lex_sim"]]):
    for col in cols:
        m = growth[col]["mean_by_N"]
        a.plot(np.log(list(m)), list(m.values()), marker="o", label=col)
    a.set_xlabel("ln N"); a.grid(alpha=.3); a.legend()
ax[0].set_ylabel("mean null maximum")
fig.tight_layout(); fig.savefig(f"{outdir}/null_max.png", dpi=160)
print(res.to_string(index=False))
print(json.dumps({k: v["fits"] for k, v in growth.items()}, indent=1))
