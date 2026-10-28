"""Chronological evaluation on LongMemEval-M: the store grows as sessions arrive and nothing reads a label.

Sessions are ordered by date. At time t the store holds every user turn of earlier sessions, the null sample for
standardisation is the store itself, and the cue-swap gate takes its K reference cues from the user's own earlier
turns, each compared with the store minus its own session.
  turn     the opening user turn of a session is the cue, at the first three sessions whose store reaches each
           target size; every firing is an interruption, relevant or not
  q_end    the question (or its implicit rewrite) after the last session
  q_after  the same cue right after the last evidence session
Evaluation half only. Usage: python temporal.py data.json items_prefix scores.npz trials_full.csv out.csv shard nshards
"""
import sys, json, csv, zlib, math
from datetime import datetime
import numpy as np, pandas as pd
import scipy.sparse as sp

src, pre, scores, cal_csv, out, shard, nsh = sys.argv[1:6] + [int(sys.argv[6]), int(sys.argv[7])]
ALPHA, K_MAX, LEX_FLOOR, SEM_FLOOR = 0.05, 199, 0.1, 0.55
TARGETS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
data = json.load(open(src))
emb = np.load(pre + "_emb.npz")
E, C, CI, has_icue = emb["E"], emb["C"], emb["CI"], emb["has_icue"]
X, XC, XI = (sp.load_npz(pre + f"_lex_{k}.npz").tocsr() for k in ("items", "cues", "icues"))
qids = [str(q) for q in np.load(scores)["qids"]]
qpos = {q: i for i, q in enumerate(qids)}

cal = pd.read_csv(cal_csv, keep_default_na=False)
cal = cal[(cal.qid.map(lambda q: zlib.crc32(str(q).encode()) % 2) == 0) & (cal.kind == "null")]
for c in ("max_sem_sim", "max_lex_sim", "v1_z"):
    cal[c] = pd.to_numeric(cal[c], errors="coerce")
TAU_COS = float(np.quantile(cal[cal.N == 8].max_sem_sim, 1 - ALPHA))
TAU_LEX = float(np.quantile(cal[cal.N == 8].max_lex_sim, 1 - ALPHA))
TAU_Z = {N: float(np.quantile(cal[cal.N == N].v1_z, 1 - ALPHA)) for N in TARGETS}
rng = np.random.default_rng(1027 + shard)

key2idx = {}
for q in data:
    for sid, ss in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        for ti, t in enumerate(ss):
            if t["role"] == "user" and (sid, ti) not in key2idx:
                key2idx[(sid, ti)] = len(key2idx)

def scores_of(ce, cx, items):
    """Semantic, lexical and shared-term scores of cue rows against items."""
    s = (ce.astype(np.float32) @ E[items].astype(np.float32).T)
    xs = X[items]
    l = (cx @ xs.T).toarray()
    cb, xb = cx.copy(), xs.copy()
    cb.data[:] = 1; xb.data[:] = 1
    return l, s, (cb @ xb.T).toarray()

def v1_best(L, S, H, mask):
    """Best eligible z per row; mask marks excluded columns (the row's own session)."""
    Ln, Sn = np.where(mask, np.nan, L), np.where(mask, np.nan, S)
    lc, sc = np.nanmedian(Ln, 1, keepdims=True), np.nanmedian(Sn, 1, keepdims=True)
    lsd = np.maximum(0.04, 1.4826 * np.nanmedian(np.abs(Ln - lc), 1, keepdims=True))
    ssd = np.maximum(0.03, 1.4826 * np.nanmedian(np.abs(Sn - sc), 1, keepdims=True))
    lz = np.where((L >= LEX_FLOOR) & (H >= 2) & ~mask, (L - lc) / lsd, -np.inf)
    sz = np.where((S >= SEM_FLOOR) & ~mask, (S - sc) / ssd, -np.inf)
    b = np.maximum(lz, sz)
    return b.max(1), b.argmax(1)

def evt_fire(l, s, N):
    """Exponential-tail gate: whether it fires and the column with the smallest tail p-value over both channels."""
    if len(l) < 11:
        return 0, 0
    def p(x):
        u = np.quantile(x, 0.9)
        if (x > u).sum() < 10:
            u = np.sort(x)[-11]
        y = x[x > u] - u
        return (len(y) / len(x)) * np.exp(-np.maximum(x - u, 0) / max(y.mean(), 1e-6))
    pm = np.fmin(p(l), p(s))  # a channel whose tail is undefined (all scores tied) gives NaN and is skipped
    if np.isnan(pm).all():
        return 0, 0
    i = int(np.nanargmin(pm))
    return int(pm[i] <= ALPHA / (2 * N)), i

def decide(ce, cx, store, store_sess, refs, ref_sess):
    """All gates for one cue against one store; refs are earlier user turns used as reference cues."""
    N = len(store)
    L, S, H = scores_of(ce[None], cx, store)
    t, a = v1_best(L, S, H, np.zeros_like(L, bool))
    t, a = float(t[0]), int(a[0])
    swap_p, K = 1.0, len(refs)
    if K and np.isfinite(t):
        Lr, Sr, Hr = scores_of(E[refs], X[refs], store)
        tr, _ = v1_best(Lr, Sr, Hr, ref_sess[:, None] == store_sess[None, :])
        swap_p = (1 + np.sum(tr >= t)) / (K + 1)
    ef, ei = evt_fire(L[0], S[0], N)
    tn = min(TARGETS, key=lambda x: abs(math.log(max(N, 1) / x)))
    return {"N": N, "v1_z": t, "v1_thr": max(2.0, math.sqrt(2 * math.log(2 * N))) + 0.5, "tau_z": TAU_Z[tn],
            "max_sem": float(S.max()), "max_lex": float(L.max()), "sem_top": int(store[S[0].argmax()]),
            "lex_top": int(store[L[0].argmax()]), "v1_top": int(store[a]) if np.isfinite(t) else -1,
            "evt": ef, "evt_top": int(store[ei]), "swap_p": swap_p, "K": K}

rows = []
for q in data:
    qid = q["question_id"]
    if qid.endswith("_abs") or zlib.crc32(qid.encode()) % 2 != 1 or qid not in qpos:
        continue
    if zlib.crc32(qid.encode()) // 2 % nsh != shard:
        continue
    order = sorted(range(len(q["haystack_dates"])), key=lambda k: (datetime.strptime(q["haystack_dates"][k][:10], "%Y/%m/%d"),
                                                                   q["haystack_dates"][k][-5:], k))
    seq, ssn, ev, ev_pos = [], [], set(), []
    for pos, k in enumerate(order):
        sid, ss = q["haystack_session_ids"][k], q["haystack_sessions"][k]
        for ti, t in enumerate(ss):
            if t["role"] == "user":
                seq.append(key2idx[(sid, ti)]); ssn.append(pos)
                if sid in q["answer_session_ids"] and t.get("has_answer"):
                    ev.add(key2idx[(sid, ti)]); ev_pos.append(pos)
    if not ev:
        continue
    seq, ssn = np.array(seq), np.array(ssn)
    starts = {p: int(np.argmax(ssn == p)) for p in np.unique(ssn)}  # first user turn of each session
    ans_pos = {pos for pos, k in enumerate(order) if q["haystack_session_ids"][k] in q["answer_session_ids"]}

    def refs_before(n):
        k = min(K_MAX, n)
        r = rng.choice(n, size=k, replace=False) if k else np.array([], int)
        return seq[r], ssn[r]

    for tN in TARGETS:
        cands = [p for p in sorted(starts) if starts[p] >= tN][:3]
        for p in cands:
            n = starts[p]
            ri, rs = refs_before(n)
            ci = seq[n]
            d = decide(E[ci], X[ci], seq[:n], ssn[:n], ri, rs)
            rows.append({"qid": qid, "kind": "turn", "target": tN, "cue_answer_session": int(p in ans_pos),
                         "cue_item": int(ci), **d})
    cut_after = int(np.max(np.nonzero(ssn <= max(ev_pos))[0])) + 1
    for kind, n in (("q_end", len(seq)), ("q_after", cut_after)):
        ri, rs = refs_before(n)
        for ctype, ce, cx in (("explicit", C[qpos[qid]], XC[qpos[qid]]), ("implicit", CI[qpos[qid]], XI[qpos[qid]])):
            if ctype == "implicit" and not has_icue[qpos[qid]]:
                continue
            d = decide(ce, cx, seq[:n], ssn[:n], ri, rs)
            d.update({k + "_ev": int(d[k] in ev) for k in ("sem_top", "lex_top", "v1_top", "evt_top")})
            rows.append({"qid": qid, "kind": kind, "cue_type": ctype, **d})
    print(qid, len(seq), flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("done", out)
