"""Translate each (cue, memory) pair in one request so terms stay consistent; proper nouns stay in English.
Run in .venv-llm:  python retranslate.py annotate.json model_dir out.json
"""
import sys, json
from vllm import LLM, SamplingParams

src, model_dir, out = sys.argv[1:4]
data = json.load(open(src))
PROMPT = """把下面两段英文翻译成自然、准确的中文，用于人工判断"记忆是否和问题相关"。要求：
1. 两段放在一起翻译，同一个词在两段里必须用同一个译法。
2. 书名、作品名、品牌、产品、课程、地名、人名、机构名一律保留英文原文，不要翻译（可以在后面加括号写中文解释）。
3. 忠实原意，不要改变句子的语气和问法（例如 "Where do I take yoga classes?" 是在问"我平时在哪里上瑜伽课"，不是"我可以去哪里上"）。
只输出一个 JSON：{{"cue": "...", "memory": "..."}}

问题：{cue}

记忆：{memory}"""

llm = LLM(model=model_dir, max_model_len=4096, gpu_memory_utilization=0.85)
outs = llm.chat([[{"role": "user", "content": PROMPT.format(cue=x["cue"], memory=x["memory"])}] for x in data],
                SamplingParams(temperature=0, max_tokens=900), use_tqdm=False)
bad = 0
for x, o in zip(data, outs):
    text = o.outputs[0].text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        y = json.loads(text)
        x["cue_zh"], x["memory_zh"] = y["cue"], y["memory"]
    except Exception:
        bad += 1  # keep the old translation
json.dump(data, open(out, "w"), ensure_ascii=False, indent=1)
print("done", len(data), "unparsed", bad)
