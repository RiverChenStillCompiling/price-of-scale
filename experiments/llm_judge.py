"""LLM-judge gate: retrieve the top-5 store items by embedding, ask the model whether to bring one up.

Same trial design as trials.py (null store / store with one evidence turn), evaluation half only.
Run in .venv-llm:  python llm_judge.py data.json scores.npz model_dir out.csv 8,64,512,2048 reps
"""
import sys, csv, zlib, re, time
import numpy as np
from items import build

src, scores, model_dir, out = sys.argv[1:5]
Ns = [int(x) for x in sys.argv[5].split(",")]
reps = int(sys.argv[6])
K = 5
cue_file = sys.argv[7] if len(sys.argv) > 7 else None

texts, _, q_pool, q_evid, cues, qids, qtypes = build(src)
z = np.load(scores)
sem, zq = z["sem"], [str(q) for q in z["qids"]]
row_of = {q: i for i, q in enumerate(zq)}
if cue_file:
    import json
    alt = json.load(open(cue_file))
    cues = [alt.get(q, c) for q, c in zip(qids, cues)]
rng = np.random.default_rng(7)

PROMPT = """You are the memory module of a personal assistant. You see what the user just said and a few notes retrieved from their past conversations. Bring up a note only if it is clearly relevant and would genuinely help the assistant respond right now; otherwise stay silent.

The user just said:
{cue}

Notes:
{notes}

Answer with the number of the single note to bring up, or NONE. Reply with only the number or NONE."""

jobs = []
for q, qid in enumerate(qids):
    P, Ev = q_pool[q], q_evid[q]
    if not len(Ev) or qid.endswith("_abs") or zlib.crc32(qid.encode()) % 2 != 1 or qid not in row_of:
        continue
    S = sem[row_of[qid]].astype(np.float32)
    for N in Ns:
        if len(P) < N:
            continue
        for r in range(reps):
            for kind in ("null", "pos"):
                store = rng.choice(P, size=N, replace=False) if kind == "null" else \
                    np.concatenate([rng.choice(P, size=N - 1, replace=False), rng.choice(Ev, size=1)])
                top = store[np.argsort(-S[store])[:K]]
                notes = "\n".join(f"[{j + 1}] {texts[i][:400]}" for j, i in enumerate(top))
                jobs.append(dict(qid=qid, qtype=qtypes[q], N=N, kind=kind, rep=r,
                                 ev_rank=next((j + 1 for j, i in enumerate(top) if i in set(Ev.tolist())), 0),
                                 top=top, prompt=PROMPT.format(cue=cues[q], notes=notes)))
print("prompts", len(jobs), flush=True)

from vllm import LLM, SamplingParams
llm = LLM(model=model_dir, max_model_len=4096, gpu_memory_utilization=0.85, enforce_eager=False)
t0 = time.time()
outs = llm.chat([[{"role": "user", "content": j["prompt"]}] for j in jobs],
                SamplingParams(temperature=0, max_tokens=6), use_tqdm=False)
dt = time.time() - t0
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["qid", "qtype", "N", "kind", "rep", "ev_rank", "choice", "fire", "top_ev", "raw"])
    for j, o in zip(jobs, outs):
        raw = o.outputs[0].text.strip()
        m = re.match(r"\[?(\d)", raw)
        choice = int(m.group(1)) if m and 1 <= int(m.group(1)) <= K else 0
        w.writerow([j["qid"], j["qtype"], j["N"], j["kind"], j["rep"], j["ev_rank"], choice,
                    int(choice > 0), int(choice > 0 and choice == j["ev_rank"]), raw.replace("\n", " ")[:20]])
print(f"done {out}  {len(jobs)} prompts in {dt:.0f}s", flush=True)
