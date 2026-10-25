"""Score every cue against every item on two channels, matching recall-v1:
  lex: tf-idf cosine over latin words (recall.js terms/weigh/cosine), plus shared-term count
  sem: sentence-embedding cosine, no query instruction (DSH embeds both sides the same way)
Optional 4th argument: a JSON {question_id: cue} that replaces the cues and keeps only those questions.
Outputs npz with float16 score matrices [n_cues, n_items] and the metadata needed for trials.
"""
import json, re, sys, math, collections
import numpy as np
import scipy.sparse as sp
from items import build

src, out, model_dir = sys.argv[1], sys.argv[2], sys.argv[3]
texts, sess_of, q_pool, q_evid, cues, qids, qtypes = build(src)
if len(sys.argv) > 4:
    alt = json.load(open(sys.argv[4]))
    keep = [i for i, q in enumerate(qids) if q in alt]
    q_pool, q_evid, qtypes = [q_pool[i] for i in keep], [q_evid[i] for i in keep], [qtypes[i] for i in keep]
    qids = [qids[i] for i in keep]
    cues = [alt[q] for q in qids]

LATIN = re.compile(r"[a-z][a-z0-9_+#.-]*[a-z0-9]|[0-9]{3,}")
def terms(text):
    return [m for m in LATIN.findall(str(text).lower()) if len(m) >= 2]

print(f"items {len(texts)}  cues {len(cues)}", flush=True)

# ---- lexical channel: recall.js weighting, df over the whole item pool ----
vocab = {}
df = collections.Counter()
tfs = []
for text in texts:
    tf = collections.Counter(terms(text))
    tfs.append(tf)
    df.update(tf.keys())
D = len(texts)
idf = lambda t: math.log(1 + (D + 1) / (df.get(t, 0) + 0.5))

def weigh(tf):
    vec = {}
    for t, n in tf.items():
        vec[vocab.setdefault(t, len(vocab))] = (1 + math.log(n)) * idf(t)
    norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
    return {j: w / norm for j, w in vec.items()}

def to_csr(vecs):
    r, c, v = [], [], []
    for i, vec in enumerate(vecs):
        for j, w in vec.items():
            r.append(i); c.append(j); v.append(w)
    return sp.csr_matrix((v, (r, c)), shape=(len(vecs), len(vocab)), dtype=np.float32)

item_vecs = [weigh(tf) for tf in tfs]
cue_vecs = [weigh(collections.Counter(terms(c))) for c in cues]
I = to_csr(item_vecs)
Q = to_csr(cue_vecs)
I.resize(I.shape[0], len(vocab)); Q.resize(Q.shape[0], len(vocab))
lex = (Q @ I.T).toarray().astype(np.float16)
Ib, Qb = I.copy(), Q.copy()
Ib.data[:] = 1; Qb.data[:] = 1
shared = (Qb @ Ib.T).toarray().astype(np.uint8)
print("lex done", flush=True)

# ---- semantic channel ----
from sentence_transformers import SentenceTransformer
import torch
m = SentenceTransformer(model_dir, device="cuda")
m.max_seq_length = 256
E = m.encode(texts, batch_size=512, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False).half()
C = m.encode(cues, batch_size=512, normalize_embeddings=True, convert_to_tensor=True).half()
sem = (C @ E.T).float().cpu().numpy().astype(np.float16)
del E
torch.cuda.empty_cache()
print("sem done", flush=True)

pool_ptr = np.cumsum([0] + [len(p) for p in q_pool])
evid_ptr = np.cumsum([0] + [len(e) for e in q_evid])
np.savez(out, lex=lex, sem=sem, shared=shared,
         pool=np.concatenate(q_pool), pool_ptr=pool_ptr,
         evid=np.concatenate(q_evid) if evid_ptr[-1] else np.zeros(0, np.int32), evid_ptr=evid_ptr,
         sess=np.array(sess_of), qids=np.array(qids), qtypes=np.array(qtypes))
print("saved", out, flush=True)
