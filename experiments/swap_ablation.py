"""Cue-swap gate under different numbers of reference cues K and nominal levels alpha.

Same trial design as trials.py with BG_MODE=full: the background is up to 300 items of the user's whole history
outside the store. Evaluation half only.
The best z-score of 399 reference cues is computed once per trial and the first K are used for each K.
Usage: python swap_ablation.py scores.npz out.csv 8,64,512,2048 reps
"""
import sys, csv, zlib
import numpy as np

src, out = sys.argv[1], sys.argv[2]
Ns = [int(x) for x in sys.argv[3].split(",")]
reps = int(sys.argv[4])
Ks, alphas, KMAX, BG = [19, 49, 99, 199, 399], [0.01, 0.05, 0.10], 399, 300
z = np.load(src)
lex, sem, shared = z["lex"], z["sem"], z["shared"]
pool, pp, evid, ep, qids = z["pool"], z["pool_ptr"], z["evid"], z["evid_ptr"], z["qids"]
KMAX = min(KMAX, len(qids) - 1)
sess = z["sess"]
by_session = {}
for i, sid in enumerate(sess):
    by_session.setdefault(str(sid), []).append(i)

def full_history(P, Ev):
    extra = [i for s_ in {str(sess[e]) for e in Ev} for i in by_session[s_]]
    return np.unique(np.concatenate([P, np.array(extra, dtype=P.dtype)]))
Ks = [k for k in Ks if k <= KMAX]
rng = np.random.default_rng(1026)

def v1_best(Lr, Sr, Hr, n):
    """Best eligible recall-v1 z over the first n columns (store); the rest are background."""
    lc, sc = np.median(Lr, 1, keepdims=True), np.median(Sr, 1, keepdims=True)
    lsd = np.maximum(0.04, 1.4826 * np.median(np.abs(Lr - lc), 1, keepdims=True))
    ssd = np.maximum(0.03, 1.4826 * np.median(np.abs(Sr - sc), 1, keepdims=True))
    ls, ss = Lr[:, :n], Sr[:, :n]
    lz = np.where((ls >= 0.1) & (Hr[:, :n] >= 2), (ls - lc) / lsd, -np.inf)
    sz = np.where(ss >= 0.55, (ss - sc) / ssd, -np.inf)
    b = np.maximum(lz, sz)
    return b.max(1), b.argmax(1)

with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["qid", "N", "kind", "rep", "K", "alpha", "fire", "top_ev"])
    for q in range(len(qids)):
        qid = str(qids[q])
        P, Ev = pool[pp[q]:pp[q + 1]], evid[ep[q]:ep[q + 1]]
        if not len(Ev) or qid.endswith("_abs") or zlib.crc32(qid.encode()) % 2 != 1:
            continue
        others_all = np.delete(np.arange(len(qids)), q)
        history = full_history(P, Ev)
        for N in Ns:
            if len(P) < N:
                continue
            for r in range(reps):
                for kind in ("null", "pos"):
                    store = rng.choice(P, N, replace=False) if kind == "null" else \
                        np.concatenate([rng.choice(P, N - 1, replace=False), rng.choice(Ev, 1)])
                    avail = np.setdiff1d(history, store)
                    bg = rng.choice(avail, min(BG, len(avail)), replace=False)
                    cols = np.concatenate([store, bg])
                    rows = np.concatenate([[q], rng.choice(others_all, KMAX, replace=False)])
                    sub = lambda m: m[np.ix_(rows, cols)]
                    t, arg = v1_best(sub(lex).astype(np.float32), sub(sem).astype(np.float32), sub(shared), N)
                    t0, tk = t[0], t[1:]
                    top_ev = int(np.isin(store[arg[0]], Ev))
                    for K in Ks:
                        p = (1 + np.sum(tk[:K] >= t0)) / (K + 1) if np.isfinite(t0) else 1.0
                        for a in alphas:
                            w.writerow([qid, N, kind, r, K, a, int(p <= a), top_ev])
print("done", out)
