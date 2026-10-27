# The Price of Scale in Proactive Memory Recall

Code, data and results for the paper *The Price of Scale in Proactive Memory Recall*
(Yichuan Chen and Xin Wang, Zhejiang Sci-Tech University), [arXiv:2610.34507](https://arxiv.org/abs/2610.34507).

An assistant that brings up a memory without being asked decides at every turn whether anything in its store is
worth an interruption. The paper shows that this is a multiple-testing problem whose difficulty grows with the size
`N` of the store: a gate that keeps its false-alarm rate at `alpha` works at a per-item false-positive rate of about
`-ln(1 - alpha) / N`, so its hit rate falls with `N` as the single-comparison ROC curve dictates. It also gives a cue-swap conformal gate
whose threshold needs no labels and which has a finite-sample false-alarm guarantee under exchangeability.

The results here follow version 2 of the paper, in which the background sample that standardises the scores is drawn
from the user's whole history outside the store. `results/pool_background/` holds the same runs with the background
drawn from the labelled null pool, the protocol of version 1.

## Contents

| path | what |
|---|---|
| `recall_gates/` | stand-alone gates (numpy only): recall-v1 per-cue robust z-scores and threshold, extreme-value gate, cue-swap conformal gate, tf-idf scorer |
| `tests/` | minimal tests of the gates |
| `experiments/` | streaming replays of LongMemEval and LoCoMo, LLM judges, theory checks, ablation, figures |
| `reproduce.sh` | runs every experiment in the paper in order |
| `data/implicit_cues_longmemeval_m.json` | 385 implicit cues: first-person messages written for LongMemEval questions that do not ask about the memory or contain the answer |
| `data/annotation/` | 270 annotated items (240 false alarms, 30 evidence turns) with labels from one human and two LLM annotators, and the guideline |
| `results/` | per setting: gate tables (`gates.csv`), null-maximum growth fits (`summary.json`) and theory checks; `matched_fa/` scorer components at matched false alarms; `intervals/` cluster-bootstrap intervals and quantile growth fits; `theory/` out-of-sample prediction and simulation; `ablation/` cue-swap parameters and latency; `llm/` LLM-judge outputs; `pool_background/` the null-pool background |
| `figures/` | the figures of the paper and the data of Table 2 |

## Using the gates

```python
import numpy as np
from recall_gates import TfidfScorer, cue_swap_gate

# columns: the N memories of the store first, then a background sample of the user's history
# rows: the current cue first, then K reference cues from the same user (for example earlier messages)
history = store + background
cues = [current_message] + past_messages            # K = len(past_messages), at least 1/alpha - 1
lex, shared = TfidfScorer(history).scores(cues, history)
sem = embed(cues) @ embed(history).T                 # cosine of normalised embeddings
speak, index, p = cue_swap_gate(lex, sem, shared, n_store=len(store), alpha=0.05)
if speak:
    bring_up(store[index])
```

`recall_v1_gate` and `evt_gate` take the same score layout for a single cue. Run the tests with
`python -m pytest tests`.

## Reproducing the paper

1. Data: download `longmemeval_s_cleaned.json` and `longmemeval_m_cleaned.json` from
   [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) and `locomo10.json`
   from [snap-research/locomo](https://github.com/snap-research/locomo) into `data/raw/`.
2. Models: `BAAI/bge-small-en-v1.5`, `BAAI/bge-large-en-v1.5`, `Qwen/Qwen2.5-7B-Instruct-AWQ` and
   `Qwen/Qwen2.5-14B-Instruct-AWQ`.
3. Environments: see `requirements.txt`; the LLM steps use a separate environment with vLLM.
4. Run

   ```bash
   SMALL=... LARGE=... Q7=... Q14=... PY_LLM=... ./reproduce.sh
   ```

All runs in the paper used one RTX 3090. Scoring LongMemEval-M takes about 4 minutes with bge-small and 15 with
bge-large; the replays are CPU-bound and take minutes to tens of minutes each; the LLM judges take 7 to 25 minutes
per setting.

## Licences

Code: MIT (see `LICENSE`). The annotation labels and the implicit cues are released under CC BY 4.0. Text in
`data/annotation/items.jsonl` comes from LongMemEval (MIT). LoCoMo text is not redistributed.

## Citation

```bibtex
@misc{chen2026price,
  title         = {The Price of Scale in Proactive Memory Recall},
  author        = {Chen, Yichuan and Wang, Xin},
  year          = {2026},
  eprint        = {2610.34507},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
