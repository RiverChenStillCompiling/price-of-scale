"""Item pool and cues for LongMemEval and LoCoMo.

Item = one user turn (LoCoMo: every turn of the conversation). Cue = the question.
The null pool of a cue never contains a session that holds its evidence.
"""
import json, collections
import numpy as np


def build(src):
    data = json.load(open(src))
    # ---- items: unique (session, turn) user turns ----
    key2idx, texts, sess_of = {}, [], []
    q_pool, q_evid, cues, qids, qtypes = [], [], [], [], []
    LOCOMO = "locomo" in src
    # LoCoMo: items are all turns of one conversation (two speakers, image captions kept);
    # the null pool excludes every session that holds evidence; category 5 is unanswerable.
    for conv in (data if LOCOMO else []):
        c, sid0 = conv["conversation"], conv["sample_id"]
        idx_of, sess_items = {}, collections.defaultdict(list)
        for k, turns in c.items():
            if not k.startswith("session_") or k.endswith("date_time"):
                continue
            for t in turns:
                text = f'{t["speaker"]}: {t["text"]}' + (f' [image: {t["blip_caption"]}]' if t.get("blip_caption") else "")
                idx_of[t["dia_id"]] = len(texts)
                sess_items[k].append(len(texts))
                texts.append(text)
                sess_of.append(f"{sid0}:{k}")
        for qi, q in enumerate(conv["qa"]):
            if q["category"] == 5:
                continue
            ev = [idx_of[d] for d in q["evidence"] if d in idx_of]
            ev_sess = {sess_of[i] for i in ev}
            pool = [i for k, its in sess_items.items() if f"{sid0}:{k}" not in ev_sess for i in its]
            q_pool.append(np.array(pool, dtype=np.int32))
            q_evid.append(np.array(sorted(set(ev)), dtype=np.int32))
            cues.append(q["question"])
            qids.append(f"{sid0}_{qi}")
            qtypes.append(f"cat{q['category']}")
    for q in ([] if LOCOMO else data):
        pool, ev = [], []
        for sid, ss in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            is_ans = sid in q["answer_session_ids"]
            for ti, t in enumerate(ss):
                if t["role"] != "user":
                    continue
                k = (sid, ti)
                if k not in key2idx:
                    key2idx[k] = len(texts)
                    texts.append(t["content"])
                    sess_of.append(sid)
                i = key2idx[k]
                if is_ans and t.get("has_answer"):
                    ev.append(i)
                # answer sessions stay out of the null pool entirely
                if not is_ans:
                    pool.append(i)
        q_pool.append(np.array(sorted(set(pool)), dtype=np.int32))
        q_evid.append(np.array(sorted(set(ev)), dtype=np.int32))
        cues.append(q["question"])
        qids.append(q["question_id"])
        qtypes.append(q["question_type"])
    return texts, sess_of, q_pool, q_evid, cues, qids, qtypes
