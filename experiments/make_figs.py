"""Paper figures and tables. Runs use the full-history background (directories with suffix _full).
Usage: python make_figs.py out_dir fig_dir [suffix]
"""
import sys, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O, F = sys.argv[1], sys.argv[2]
SUF = sys.argv[3] if len(sys.argv) > 3 else "_full"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman No9 L", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
G = {"fixed cosine (tuned at N0)": ("fixed cosine", "#9E9E9E", "s"),
     "fixed tf-idf (tuned at N0)": ("fixed tf-idf", "#C9B79C", "D"),
     "recall-v1 (N-scaled z)": (r"z-score, $\sqrt{2\ln 2N}$ threshold", "#4C78A8", "o"),
     "EVT tail, exponential (no tuning)": ("EVT tail", "#72B7B2", "^"),
     "cue-swap conformal (no tuning)": ("cue-swap (ours)", "#C96442", "*"),
     "cosine, re-tuned per N (oracle)": ("cosine, oracle per N", "#555555", "x"),
     "recall-v1 z, re-tuned per N": ("z-score, oracle per N", "#2F4B7C", "+")}

def gates(d):
    g = pd.read_csv(f"{O}/{d}{SUF}/gates.csv")
    return g[g.gate.isin(G)]

def lines(ax, g, metric, names, band=False):
    for name in names:
        lab, c, m = G[name]
        x = g[g.gate == name].sort_values("N")
        ax.plot(x.N, x[metric], marker=m, ms=3.5, lw=1.2, color=c, label=lab)
        if band:
            lo, hi = ("fa_lo", "fa_hi") if metric == "false_alarm" else ("hit_lo", "hit_hi")
            ax.fill_between(x.N, x[lo], x[hi], color=c, alpha=.12, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("memory store size $N$")
    ax.grid(alpha=.25, lw=.5)

# Figure 1: false alarms and hits vs N (M, bge-small, explicit cues)
g = gates("m")
untuned = ["fixed cosine (tuned at N0)", "fixed tf-idf (tuned at N0)", "recall-v1 (N-scaled z)",
           "EVT tail, exponential (no tuning)", "cue-swap conformal (no tuning)"]
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
lines(ax[0], g, "false_alarm", untuned, band=True)
ax[0].axhline(.05, ls=":", c="k", lw=.8)
ax[0].set_ylabel("false-alarm rate")
lines(ax[1], g, "hit", untuned + ["recall-v1 z, re-tuned per N"], band=True)
ax[1].set_ylabel("hit rate")
ax[1].legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{F}/fig1_scale.pdf"); plt.close(fig)

# Figure 2: growth of the null maximum per channel
fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.4))
for a, d, title in zip(ax, ["m", "m_implicit", "locomo"], ["LongMemEval-M", "LongMemEval-M, implicit cues", "LoCoMo"]):
    tr = pd.read_csv(f"{O}/trials_{d}{SUF}.csv", usecols=["N", "kind", "max_sem_z", "max_lex_z"], keep_default_na=False)
    tr = tr[tr.kind == "null"]
    for col, lab, c in (("max_sem_z", "semantic", "#4C78A8"), ("max_lex_z", "lexical", "#C96442")):
        g2 = tr.groupby("N")[col]
        x = np.log(np.array(g2.mean().index, float))
        a.plot(x, g2.mean().values, "o-", ms=3, lw=.9, color=c, label=f"{lab}, mean")
        a.plot(x, g2.quantile(0.95).values, "^--", ms=3, lw=.9, color=c, mfc="white", label=f"{lab}, 95% quantile")
    a.set_title(title); a.set_xlabel(r"$\ln N$"); a.grid(alpha=.25, lw=.5)
ax[0].set_ylabel("null maximum (z)")
ax[0].legend(frameon=False, loc="upper left", fontsize=6)
fig.tight_layout(); fig.savefig(f"{F}/fig2_tails.pdf"); plt.close(fig)

# Figure 3: Proposition 1 against the oracle gate
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
for a, d, title in zip(ax, ["m", "m_large"], ["bge-small", "bge-large"]):
    t = json.load(open(f"{O}/{d}{SUF}/theory2.json"))["global"]
    N = np.array([int(k) for k in t])
    emp = np.array([t[k]["empirical_hit"] for k in t])
    ex = np.array([t[k]["empirical_tau"]["exact_hit"] for k in t])
    up = np.array([t[k]["empirical_tau"]["upper_mean_G"] for k in t])
    lo = np.array([t[k]["empirical_tau"]["lower_mean_(1-a_c)G_c"] for k in t])
    a.fill_between(N, lo, up, color="#C96442", alpha=.18, lw=0, label="bounds (Cor. 5)")
    a.plot(N, ex, "-", color="#C96442", lw=1.2, label="exact model")
    a.plot(N, emp, "o", color="k", ms=3.5, label="observed")
    a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("memory store size $N$"); a.set_title(title)
    a.set_yticks([0.1, 0.2, 0.4, 0.8]); a.set_yticklabels(["0.1", "0.2", "0.4", "0.8"]); a.minorticks_off()
    a.grid(alpha=.25, lw=.5, which="both")
ax[0].set_ylabel(r"hit rate at false-alarm rate $0.05$"); ax[0].legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{F}/fig3_theory.pdf"); plt.close(fig)

# Figure 4: LLM judges vs the best similarity gates
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.9))
for a, d, title in zip(ax, ["m", "locomo"], ["LongMemEval-M", "LoCoMo"]):
    for f, lab, c in ((f"llm_q7_{d}.csv", "Qwen2.5-7B judge", "#72B7B2"), (f"llm_q14_{d}.csv", "Qwen2.5-14B judge", "#2F4B7C")):
        t = pd.read_csv(f"{O}/{f}", keep_default_na=False)
        r = pd.DataFrame({"fa": t[t.kind == "null"].groupby("N").fire.mean(), "hit": t[t.kind == "pos"].groupby("N").top_ev.mean()})
        a.plot(r.index, r.hit, "-o", ms=3, color=c, label=f"{lab}: hit")
        a.plot(r.index, r.fa, ":o", ms=3, color=c, label=f"{lab}: false alarm")
    gg = gates(d)
    x = gg[gg.gate == "recall-v1 z, re-tuned per N"].sort_values("N")
    a.plot(x.N, x.hit, "-+", color="#C96442", ms=5, label="z-score, oracle per N: hit")
    a.set_xscale("log", base=2); a.set_xlabel("memory store size $N$"); a.set_title(title); a.grid(alpha=.25, lw=.5)
ax[0].set_ylabel("rate")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=3, loc="lower center", fontsize=6.8)
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(f"{F}/fig4_llm.pdf"); plt.close(fig)

# Table 2: hits at (near) matched false alarms
rows = []
for d, lab in (("m", "explicit, bge-small"), ("m_large", "explicit, bge-large"),
               ("m_implicit", "implicit, bge-small"), ("m_implicit_large", "implicit, bge-large")):
    g = pd.read_csv(f"{O}/{d}{SUF}/gates.csv")
    for N in (8, 128, 2048):
        r = {"setting": lab, "N": N}
        for name, key in (("cosine, re-tuned per N (oracle)", "cos"), ("recall-v1 z, re-tuned per N", "z"),
                          ("cue-swap conformal (no tuning)", "swap")):
            x = g[(g.gate == name) & (g.N == N)].iloc[0]
            r[f"{key}_hit"], r[f"{key}_fa"] = x.hit, x.false_alarm
        rows.append(r)
pd.DataFrame(rows).round(3).to_csv(f"{F}/table2.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
