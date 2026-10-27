"""Gates on TriggerBench (prospective memory): the constraint turn is the memory, the trigger turn is the cue.

Store = the user turns before the trigger. Conditions:
  pos      positive_clean, constraint present          hit = fires and the top item is a constraint turn
  neg      negative_clean, constraint present but resolved or not applicable    any firing is wrong
  null     positive_clean with the constraint turns removed                     any firing is a false alarm
Lengths: base (5-9 user turns), 40K (about 130-650), and grown stores of N user turns made of the base turns plus
filler user turns from the 40K split (MultiWOZ dialogue). The null sample is the store plus 300 filler turns outside
it. Cue-swap reference cues are the triggers of 199 other items. Thresholds of the fixed and oracle gates are tuned
on the null condition of a hash-defined calibration half.
Usage: python triggerbench.py data_dir model_dir out_prefix
"""
import sys, json, glob, re, math, zlib, collections
import numpy as np, pandas as pd
import scipy.sparse as sp

data_dir, model_dir, out = sys.argv[1:4]
CONSTRAINT = {"constraint", "rule_constraint", "condition_constraint"}
ALPHA, K_SWAP, BG, LEX_FLOOR, SEM_FLOOR = 0.05, 199, 300, 0.1, 0.55
GROW = [8, 32, 128, 512, 2048]
rng = np.random.default_rng(1028)

recs = []
for f in sorted(glob.glob(f"{data_dir}/*_context/*/*/*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        turns = r["sample"]["turns"]
        users = [t for t in turns[:-1] if t["role"] == "user"]
        recs.append({"task": f'{r["category"]}:{r["task_id"]}', "cat": r["category"], "variant": r["variant"], "bin": r["bin"],
                     "cue": turns[-1]["text"], "texts": [t["text"] for t in users], "fn": [t["function"] for t in users]})
texts = sorted({t for r in recs for t in r["texts"]} | {r["cue"] for r in recs})
tid = {t: i for i, t in enumerate(texts)}
base_texts = {t for r in recs if r["bin"] == "base" for t in r["texts"]}
filler = np.array(sorted({tid[t] for r in recs if r["bin"] != "base" for t, f in zip(r["texts"], r["fn"])
                          if f == "thematic_connection" and t not in base_texts}))
print(f"records {len(recs)}  texts {len(texts)}  filler {len(filler)}", flush=True)

LATIN = re.compile(r"[a-z][a-z0-9_+#.-]*[a-z0-9]|[0-9]{3,}")
tfs = [collections.Counter(m for m in LATIN.findall(t.lower()) if len(m) >= 2) for t in texts]
df = collections.Counter(k for tf in tfs for k in tf)
vocab, r_, c_, v_ = {}, [], [], []
for i, tf in enumerate(tfs):
    vec = {vocab.setdefault(k, len(vocab)): (1 + math.log(n)) * math.log(1 + (len(texts) + 1) / (df[k] + 0.5)) for k, n in tf.items()}
    norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
    for j, w in vec.items():
        r_.append(i); c_.append(j); v_.append(w / norm)
X = sp.csr_matrix((v_, (r_, c_)), shape=(len(texts), len(vocab)), dtype=np.float32)
XB = X.copy(); XB.data[:] = 1
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(model_dir, device="cuda"); m.max_seq_length = 256
E = m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
del m
print("embedded", flush=True)

def score(cues, items):
    return (X[cues] @ X[items].T).toarray(), E[cues] @ E[items].T, (XB[cues] @ XB[items].T).toarray()

def v1(L, S, H, n):
    lc, sc = np.median(L, 1, keepdims=True), np.median(S, 1, keepdims=True)
    lsd = np.maximum(0.04, 1.4826 * np.median(np.abs(L - lc), 1, keepdims=True))
    ssd = np.maximum(0.03, 1.4826 * np.median(np.abs(S - sc), 1, keepdims=True))
    l, s, h = L[:, :n], S[:, :n], H[:, :n]
    b = np.maximum(np.where((l >= LEX_FLOOR) & (h >= 2), (l - lc) / lsd, -np.inf), np.where(s >= SEM_FLOOR, (s - sc) / ssd, -np.inf))
    return b.max(1), b.argmax(1)

def evt(l, s, N):
    """Exponential tail fitted to store + background; smallest p-value over the store (first N columns)."""
    def p(x):
        u = np.quantile(x, 0.9)
        y = x[x > u] - u
        return (len(y) / len(x)) * np.exp(-np.maximum(x[:N] - u, 0) / max(y.mean(), 1e-6))
    return int(min(p(l).min(), p(s).min()) <= ALPHA / (2 * N))

cue_ids = np.array([tid[r["cue"]] for r in recs])
by_task = collections.defaultdict(list)
for j, r in enumerate(recs):
    by_task[r["task"]].append(j)
rows = []
for ri, r in enumerate(recs):
    if r["variant"] not in ("positive_clean", "negative_clean"):
        continue
    store0 = [tid[t] for t in r["texts"]]
    ev = {tid[t] for t, f in zip(r["texts"], r["fn"]) if f in CONSTRAINT}
    conds = [("pos" if r["variant"] == "positive_clean" else "neg", store0)]
    if r["variant"] == "positive_clean":
        conds.append(("null", [i for i in store0 if i not in ev]))
    sizes = [("natural", None)] + ([("grown", N) for N in GROW] if r["bin"] == "base" else [])
    for cond, st in conds:
        for size, N in sizes:
            for rep in range(1 if size == "natural" else 3):
                store = np.array(st)
                if N is not None:
                    if len(store) >= N:
                        continue
                    store = np.concatenate([store, rng.choice(filler, N - len(store), replace=False)])
                n = len(store)
                bg = rng.choice(np.setdiff1d(filler, store), BG, replace=False)
                cols = np.concatenate([store, bg])
                others = rng.choice(np.delete(np.arange(len(recs)), by_task[r["task"]]), K_SWAP, replace=False)
                L, S, H = score(np.concatenate([[tid[r["cue"]]], np.unique(cue_ids[others])[:K_SWAP]]), cols)
                t, a = v1(L, S, H, n)
                top = lambda k: int(store[k] in ev)
                rows.append({"task": r["task"], "cat": r["cat"], "bin": r["bin"], "cond": cond, "size": size, "N": n, "rep": rep,
                             "v1_z": t[0], "v1_thr": max(2.0, math.sqrt(2 * math.log(2 * n))) + 0.5,
                             "v1_top_ev": top(int(a[0])) if np.isfinite(t[0]) else 0,
                             "max_sem": S[0, :n].max(), "sem_top_ev": top(int(S[0, :n].argmax())),
                             "max_lex": L[0, :n].max(), "lex_top_ev": top(int(L[0, :n].argmax())),
                             "evt": evt(L[0], S[0], n),
                             "swap_p": (1 + np.sum(t[1:] >= t[0])) / len(t) if np.isfinite(t[0]) else 1.0,
                             "half": zlib.crc32(r["task"].encode()) % 2})
    if ri % 100 == 0:
        print(ri, flush=True)
pd.DataFrame(rows).to_csv(out + "_trials.csv", index=False)
print("done")
