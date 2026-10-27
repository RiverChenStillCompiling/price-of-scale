"""Cross-encoder gate: rerank the 20 most similar memories of the store and speak if the best reranker score clears
a threshold. Stores are built as in trials.py (null: N pool turns; positive: N-1 pool turns + 1 evidence turn).
Writes one row per trial with the best reranker score, whether its item is evidence, and the cosine baseline.
DEVICE=cpu runs on the CPU; SHARD/NSHARD split the questions.
Usage: python rerank.py data.json scores.npz reranker_dir out.csv Ns reps [implicit_cues.json]
"""
import sys, os, json, zlib, csv
import numpy as np, torch
from items import build
from sentence_transformers import CrossEncoder

src, scores, model_dir, out = sys.argv[1:5]
Ns, reps = [int(x) for x in sys.argv[5].split(",")], int(sys.argv[6])
TOP = 20
texts, _, _, _, cues, qids, _ = build(src)
if len(sys.argv) > 7:
    alt = json.load(open(sys.argv[7]))
    cues = {q: alt[q] for q in qids if q in alt}
else:
    cues = dict(zip(qids, cues))
z = np.load(scores)
sem, pool, pp, evid, ep, zq = z["sem"], z["pool"], z["pool_ptr"], z["evid"], z["evid_ptr"], [str(q) for q in z["qids"]]
DEVICE = os.environ.get("DEVICE", "cuda")
SHARD, NSHARD = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
ce = CrossEncoder(model_dir, device=DEVICE, max_length=256)
if DEVICE == "cuda":
    ce.model.half()
rng = np.random.default_rng(1029 + SHARD)
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["qid", "N", "kind", "rep", "rr_max", "rr_top_ev", "cos_max", "cos_top_ev", "ev_in_top"])
    for q, qid in enumerate(zq):
        P, Ev = pool[pp[q]:pp[q + 1]], evid[ep[q]:ep[q + 1]]
        if not len(Ev) or qid.endswith("_abs") or zlib.crc32(qid.encode()) % NSHARD != SHARD:
            continue
        S = sem[q].astype(np.float32)
        cache = {}
        trials = []
        for N in Ns:
            if len(P) < N:
                continue
            for r in range(reps):
                for kind in ("null", "pos"):
                    store = rng.choice(P, N, replace=False) if kind == "null" else \
                        np.concatenate([rng.choice(P, N - 1, replace=False), rng.choice(Ev, 1)])
                    top = store[np.argsort(-S[store])[:TOP]]
                    trials.append((N, kind, r, store, top))
        need = sorted({int(i) for *_, top in trials for i in top} - cache.keys())
        with torch.no_grad():
            sc = ce.predict([(cues[qid], texts[i]) for i in need], batch_size=16, show_progress_bar=False)
        cache.update(zip(need, map(float, sc)))
        for N, kind, r, store, top in trials:
            rr = np.array([cache[int(i)] for i in top])
            j, k = int(rr.argmax()), int(np.argmax(S[store]))
            w.writerow([qid, N, kind, r, f"{rr[j]:.4f}", int(top[j] in Ev), f"{S[store][k]:.4f}", int(store[k] in Ev),
                        int(np.isin(top, Ev).any())])
        print(qid, len(need), flush=True)
print("done", out)
