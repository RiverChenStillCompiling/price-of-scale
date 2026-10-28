"""Streaming-size trials on precomputed scores.

For each cue and store size N:
  null trial      store = N user turns from sessions that hold no answer -> any firing is a false alarm
  positive trial  store = N-1 such turns + 1 evidence turn                -> hit = fires and its top item is evidence
The null sample for recall-v1 is store + 300 background turns from the same user's history,
exactly as recall.js does (median / 1.4826*MAD, spread floors 0.04 lexical, 0.03 semantic).
swap_p: the cue-swap conformal p-value (see cue_swap_p). One row per trial; gates are applied afterwards in analyse.py.

BG_MODE chooses where the 300 background items come from (always disjoint from the store):
  pool  the null pool, which needs evidence labels to build (BG_POOL=1 is the same)
  evid  the null pool plus the evidence turns
  full  every user turn of the user's history, nothing removed: what a deployed assistant has
Ablation columns give the best score of one scorer at a time on the same store and background.
POS_MODE changes what the positive store holds besides N - m null turns (KINDS=pos skips the null trials):
  one      one evidence turn (default)
  all      every evidence turn of the question
  dup3     one evidence turn three times, a fact repeated
  session  every user turn of one evidence session
HIT_REL=1 counts a firing on any of these items as a hit.
"""
import os, sys, math, csv
import numpy as np

src, out = sys.argv[1], sys.argv[2]
Ns = [int(x) for x in sys.argv[3].split(",")]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 20
BG, LEX_FLOOR, SEM_FLOOR = 300, 0.1, 0.55
z = np.load(src)
lex, sem, shared = z["lex"], z["sem"], z["shared"]
pool, pp, evid, ep = z["pool"], z["pool_ptr"], z["evid"], z["evid_ptr"]
qids, qtypes = z["qids"], z["qtypes"]
sess = z["sess"]
rng = np.random.default_rng(20261025)
BG_MODE = os.environ.get("BG_MODE") or ("pool" if os.environ.get("BG_POOL") else "evid")
POS_MODE = os.environ.get("POS_MODE", "one")
KINDS = os.environ.get("KINDS", "null,pos").split(",")
HIT_REL = bool(os.environ.get("HIT_REL"))  # count any relevant item of the positive store as a hit
by_session = {}
for i, sid in enumerate(sess):
    by_session.setdefault(str(sid), []).append(i)
by_conv = {}
for i, sid in enumerate(sess):
    if ":" in str(sid):
        by_conv.setdefault(str(sid).split(":")[0], []).append(i)

def full_history(q, P, Ev):
    """All user turns of this user's history; evidence sessions are found from the stored evidence turns."""
    if ":" in str(sess[P[0]]):  # LoCoMo: every turn of the conversation
        return np.array(by_conv[str(sess[P[0]]).split(":")[0]], dtype=np.int64)
    extra = [i for s_ in {str(sess[e]) for e in Ev} for i in by_session[s_]]
    return np.unique(np.concatenate([P, np.array(extra, dtype=P.dtype)]))

ALPHA = 0.05

def tail_fit(x, gpd):
    """Peaks over the 90th percentile: exponential (xi=0) or GPD by probability-weighted moments.
    None when no tail can be fitted (fewer than 11 scores, or all scores tied)."""
    if len(x) < 11:
        return None
    u = np.quantile(x, 0.9)
    if (x > u).sum() < 10:
        u = np.sort(x)[-11]
    y = np.sort(x[x > u] - u)
    if not len(y):
        return None
    pu = len(y) / len(x)
    if not gpd:
        return u, pu, max(y.mean(), 1e-6), 0.0
    n = len(y)
    a0 = y.mean()
    a1 = np.mean((1 - (np.arange(1, n + 1) - 0.35) / n) * y)
    d = max(a0 - 2 * a1, 1e-9)
    xi = float(np.clip(2 - a0 / d, -0.25, 0.5))
    return u, pu, max(2 * a0 * a1 / d, 1e-6), xi

def tail_p(s, fit):
    if fit is None:
        return np.ones(len(s))
    u, pu, sig, xi = fit
    e = np.maximum(s - u, 0)
    if abs(xi) < 1e-6:
        return pu * np.exp(-e / sig)
    base = 1 + xi * e / sig
    return pu * np.where(base > 0, np.maximum(base, 1e-300) ** (-1 / xi), 0.0)

def evt_gate(ls, ss, lnull, snull, N, gpd):
    """Fire when the smallest tail p-value over items and both channels clears alpha / (2N)."""
    p = np.minimum(tail_p(ls, tail_fit(lnull, gpd)), tail_p(ss, tail_fit(snull, gpd)))
    i = int(np.argmin(p))
    return p[i] <= ALPHA / (2 * N), i, p[i] * 2 * N

def null(xs, floor):
    c = np.median(xs)
    return c, max(floor, 1.4826 * np.median(np.abs(xs - c)))

K_SWAP = 199
SWAP_SAME = bool(os.environ.get("SWAP_SAME"))
groups = [str(x).rsplit("_", 1)[0] for x in qids]
group_members = {}
for j, g in enumerate(groups):
    group_members.setdefault(g, []).append(j)

def v1_best(Lr, Sr, Hr, store, b):
    """recall-v1's best eligible z over the store, for each row of score matrices (cues x items)."""
    ls, ss = Lr[:, store], Sr[:, store]
    lall, sall = np.concatenate([ls, Lr[:, b]], 1), np.concatenate([ss, Sr[:, b]], 1)
    lc, sc = np.median(lall, 1, keepdims=True), np.median(sall, 1, keepdims=True)
    lsd = np.maximum(0.04, 1.4826 * np.median(np.abs(lall - lc), 1, keepdims=True))
    ssd = np.maximum(0.03, 1.4826 * np.median(np.abs(sall - sc), 1, keepdims=True))
    lz = np.where((ls >= LEX_FLOOR) & (Hr[:, store] >= 2), (ls - lc) / lsd, -np.inf)
    sz = np.where(ss >= SEM_FLOOR, (ss - sc) / ssd, -np.inf)
    return np.maximum(lz, sz).max(1)

def cue_swap_p(q, store, b, t_cue):
    """Conformal p-value of this cue's best z among other cues put against the same store.
    Other cues have no evidence in this store, so they stand in for the user's past messages."""
    if SWAP_SAME:  # only cues from the same conversation / user, as a deployment would have
        cands = np.array([j for j in group_members[groups[q]] if j != q])
    else:
        cands = np.delete(np.arange(len(qids)), q)
    others = rng.choice(cands, size=min(K_SWAP, len(cands)), replace=False)
    cols = np.concatenate([store, b])
    n = len(store)
    sub = lambda m: m[np.ix_(others, cols)]
    t = v1_best(sub(lex).astype(np.float32), sub(sem).astype(np.float32), sub(shared), np.arange(n), np.arange(n, len(cols)))
    return (1 + np.sum(t >= t_cue)) / (len(others) + 1)

cols = ["qid", "qtype", "N", "kind", "rep",
        "v1_z", "v1_top_ev", "v1_thr",
        "max_lex_z", "max_sem_z", "max_lex_sim", "max_sem_sim",
        "lex_top_ev", "sem_top_ev",
        "evt_exp_fire", "evt_exp_top_ev", "evt_gpd_fire", "evt_gpd_top_ev", "evt_exp_padj", "evt_gpd_padj", "swap_p",
        "sem_z_top_ev", "semel_z", "semel_top_ev", "lexel_z", "lexel_top_ev", "both_z", "both_top_ev", "n_bg"]
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for q in range(len(qids)):
        P = pool[pp[q]:pp[q + 1]]
        Ev = evid[ep[q]:ep[q + 1]]
        if not len(Ev) or str(qids[q]).endswith("_abs"):
            continue
        L, S, H = lex[q].astype(np.float32), sem[q].astype(np.float32), shared[q]
        history = {"pool": P, "evid": np.concatenate([P, Ev]), "full": None}[BG_MODE]
        if history is None:
            history = full_history(q, P, Ev)
        for N in Ns:
            if len(P) < N:
                continue
            thr = max(2.0, math.sqrt(2 * math.log(N * 2))) + 0.5
            for r in range(reps):
                for kind in KINDS:
                    if kind == "null":
                        store = rng.choice(P, size=N, replace=False)
                    else:
                        e = rng.choice(Ev, size=1)
                        rel = {"one": e, "all": Ev, "dup3": np.repeat(e, 3),
                               "session": np.array(by_session[str(sess[e[0]])], dtype=P.dtype)}[POS_MODE]
                        if len(rel) >= N:
                            continue
                        store = np.concatenate([rng.choice(P, size=N - len(rel), replace=False), rel])
                    is_ev = np.isin(store, rel if HIT_REL and kind == "pos" else Ev)
                    ls, ss = L[store], S[store]
                    # background: up to 300 history items outside the store, as in DSH where raw records and cards are distinct
                    avail = np.setdiff1d(history, store)
                    b = rng.choice(avail, size=min(BG, len(avail)), replace=False)
                    lnull, snull = np.concatenate([ls, L[b]]), np.concatenate([ss, S[b]])
                    lc, lsd = null(lnull, 0.04)
                    sc, ssd = null(snull, 0.03)
                    fe, ie, pe = evt_gate(ls, ss, lnull, snull, N, False)
                    fg, ig, pg = evt_gate(ls, ss, lnull, snull, N, True)
                    lz, sz = (ls - lc) / lsd, (ss - sc) / ssd
                    # recall-v1 eligibility: a channel only counts past its similarity floor
                    lz_ok = np.where((ls >= LEX_FLOOR) & (H[store] >= 2), lz, -np.inf)
                    sz_ok = np.where(ss >= SEM_FLOOR, sz, -np.inf)
                    best = np.maximum(lz_ok, sz_ok)
                    i = int(np.argmax(best))
                    both = np.maximum(lz, sz)
                    abl = [int(is_ev[int(np.argmax(sz))])]
                    for arr in (sz_ok, lz_ok, both):
                        j = int(np.argmax(arr))
                        abl += [f"{arr[j]:.4f}", int(is_ev[j])]
                    abl.append(len(b))
                    swap = cue_swap_p(q, store, b, best[i]) if np.isfinite(best[i]) else 1.0
                    w.writerow([qids[q], qtypes[q], N, kind, r,
                                f"{best[i]:.4f}", int(is_ev[i]), f"{thr:.4f}",
                                f"{lz.max():.4f}", f"{sz.max():.4f}", f"{ls.max():.4f}", f"{ss.max():.4f}",
                                int(is_ev[int(np.argmax(ls))]), int(is_ev[int(np.argmax(ss))]),
                                int(fe), int(is_ev[ie]), int(fg), int(is_ev[ig]), f"{pe:.4g}", f"{pg:.4g}", f"{swap:.4f}"] + abl)
print("done", out)
