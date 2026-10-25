"""Sample memories that a gate brought up in null trials, plus a few real evidence turns as checks,
translate cue and memory into Chinese, and write the annotation set.
Run in .venv-llm:  python sample_fa.py data.json scores.npz model_dir out.json
"""
import sys, json, math
import numpy as np
from items import build

src, scores, model_dir, out = sys.argv[1:5]
texts, _, pool, evid, cues, qids, qtypes = build(src)
z = np.load(scores)
lex, sem, shared = z["lex"], z["sem"], z["shared"]
rng = np.random.default_rng(1111)
PER_N, CHECKS = 80, 30

def best(q, store, bg):
    L, S, H = lex[q].astype(np.float32), sem[q].astype(np.float32), shared[q]
    ls, ss = L[store], S[store]
    lall, sall = np.concatenate([ls, L[bg]]), np.concatenate([ss, S[bg]])
    lc, sc = np.median(lall), np.median(sall)
    lsd = max(0.04, 1.4826 * np.median(np.abs(lall - lc)))
    ssd = max(0.03, 1.4826 * np.median(np.abs(sall - sc)))
    lz = np.where((ls >= 0.1) & (H[store] >= 2), (ls - lc) / lsd, -np.inf)
    sz = np.where(ss >= 0.55, (ss - sc) / ssd, -np.inf)
    b = np.maximum(lz, sz)
    i = int(np.argmax(b))
    return store[i], b[i]

ok = [q for q in range(len(qids)) if len(evid[q]) and not qids[q].endswith("_abs")]
rows, seen = [], set()
for N in (64, 512, 2048):
    got, tries = 0, 0
    while got < PER_N and tries < 20000:
        tries += 1
        q = int(rng.choice(ok))
        if len(pool[q]) < N + 300:
            continue
        perm = rng.permutation(pool[q])
        store, bg = perm[:N], perm[N:N + 300]
        item, zb = best(q, store, bg)
        if zb < max(2.0, math.sqrt(2 * math.log(2 * N))) + 0.5 or (q, item) in seen:
            continue
        seen.add((q, int(item)))
        rows.append({"q": q, "item": int(item), "source": f"recall-v1 false alarm, N={N}"})
        got += 1
for q in rng.choice(ok, CHECKS, replace=False):
    rows.append({"q": int(q), "item": int(rng.choice(evid[q])), "source": "evidence (check)"})
order = rng.permutation(len(rows))
rows = [rows[i] for i in order]

from vllm import LLM, SamplingParams
llm = LLM(model=model_dir, max_model_len=2048, gpu_memory_utilization=0.85)
def zh(strings):
    outs = llm.chat([[{"role": "user", "content": "把下面这段英文翻译成自然的中文，只输出译文：\n\n" + s[:1200]}] for s in strings],
                    SamplingParams(temperature=0, max_tokens=600), use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outs]
cue_zh = zh([cues[r["q"]] for r in rows])
mem_zh = zh([texts[r["item"]] for r in rows])
data = [{"id": i, "qid": qids[r["q"]], "qtype": qtypes[r["q"]], "cue": cues[r["q"]], "cue_zh": cz,
         "memory": texts[r["item"]][:1200], "memory_zh": mz, "source": r["source"]}
        for i, (r, cz, mz) in enumerate(zip(rows, cue_zh, mem_zh))]
json.dump(data, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", len(data), out)
