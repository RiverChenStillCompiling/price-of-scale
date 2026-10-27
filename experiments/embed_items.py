"""Embeddings and tf-idf vectors of every item and cue, for evaluations that compare items with each other.

Same channels as embed.py. Saves E (items x dim, float16), C (cues x dim) and, for implicit cues, CI, plus the
tf-idf matrices of items, cues and implicit cues (scipy sparse, row-normalised) in a second file.
Usage: python embed_items.py data.json model_dir out_prefix [implicit_cues.json]
"""
import sys, json, re, math, collections
import numpy as np
import scipy.sparse as sp
from items import build

src, model_dir, out = sys.argv[1:4]
texts, sess_of, q_pool, q_evid, cues, qids, qtypes = build(src)
implicit = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {}
icues = [implicit.get(q, "") for q in qids]

LATIN = re.compile(r"[a-z][a-z0-9_+#.-]*[a-z0-9]|[0-9]{3,}")
terms = lambda t: [m for m in LATIN.findall(str(t).lower()) if len(m) >= 2]
tfs = [collections.Counter(terms(t)) for t in texts]
df = collections.Counter()
for tf in tfs:
    df.update(tf.keys())
D, vocab = len(texts), {}
idf = lambda t: math.log(1 + (D + 1) / (df.get(t, 0) + 0.5))

def csr(counters):
    r, c, v = [], [], []
    for i, tf in enumerate(counters):
        vec = {vocab.setdefault(t, len(vocab)): (1 + math.log(n)) * idf(t) for t, n in tf.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        for j, w in vec.items():
            r.append(i); c.append(j); v.append(w / norm)
    return r, c, v

parts = [csr(tfs), csr([collections.Counter(terms(c)) for c in cues]), csr([collections.Counter(terms(c)) for c in icues])]
mats = [sp.csr_matrix((v, (r, c)), shape=(n, len(vocab)), dtype=np.float32)
        for (r, c, v), n in zip(parts, (len(texts), len(cues), len(icues)))]
sp.save_npz(out + "_lex_items.npz", mats[0]); sp.save_npz(out + "_lex_cues.npz", mats[1]); sp.save_npz(out + "_lex_icues.npz", mats[2])
print("lex done", flush=True)

from sentence_transformers import SentenceTransformer
m = SentenceTransformer(model_dir, device="cuda")
m.max_seq_length = 256
enc = lambda xs: m.encode(xs, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
np.savez(out + "_emb.npz", E=enc(texts), C=enc(cues), CI=enc([c or "." for c in icues]),
         has_icue=np.array([bool(c) for c in icues]), qids=np.array(qids))
print("sem done", flush=True)
