"""Turn each LongMemEval question into an implicit cue: something the user would say while
doing a task where that memory helps, without asking about it or giving the answer away.
Run in .venv-llm:  python rewrite_cues.py data.json model_dir out.json
"""
import sys, json, re
from items import build

src, model_dir, out = sys.argv[1:4]
data = {q["question_id"]: q for q in json.load(open(src))}
_, _, _, q_evid, cues, qids, _ = build(src)
todo = [(q, cues[i]) for i, q in enumerate(qids) if len(q_evid[i]) and not q.endswith("_abs")]

PROMPT = """A personal assistant keeps long-term memory of its user. Below is a question that tests whether the assistant remembers something, with its answer.

Question: {q}
Answer: {a}

Write ONE short message (one or two sentences) that this user might naturally send while doing some everyday task, in which recalling that memory would help the assistant — but the message must NOT ask about the memory, must NOT say "remember" or "did I", and must NOT contain the answer. Write it in the first person, as the user. Output only the message."""

from vllm import LLM, SamplingParams
llm = LLM(model=model_dir, max_model_len=2048, gpu_memory_utilization=0.85)
outs = llm.chat([[{"role": "user", "content": PROMPT.format(q=c, a=str(data[q]["answer"])[:200])}] for q, c in todo],
                SamplingParams(temperature=0.3, max_tokens=80, seed=1), use_tqdm=False)
res, dropped = {}, []
for (q, c), o in zip(todo, outs):
    msg = o.outputs[0].text.strip().strip('"').split("\n")[0]
    ans = str(data[q]["answer"]).lower().strip(" .")
    leak = ans and ans in msg.lower()
    if leak or re.search(r"\bremember\b|\bdid i\b|\bi told you\b", msg, re.I) or len(msg) < 10:
        dropped.append({"qid": q, "question": c, "cue": msg})
        continue
    res[q] = msg
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
json.dump(dropped, open(out.replace(".json", "_dropped.json"), "w"), indent=1, ensure_ascii=False)
print(f"kept {len(res)} dropped {len(dropped)}", flush=True)
