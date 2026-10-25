"""Lexical channel: tf-idf cosine with the weighting used by the recall-v1 plugin."""
import math
import re
from collections import Counter

import numpy as np

LATIN = re.compile(r"[a-z][a-z0-9_+#.-]*[a-z0-9]|[0-9]{3,}")
CJK_RUN = re.compile(r"[㐀-鿿豈-﫿]+")


def terms(text):
    """Latin words of two or more characters, numbers of three or more digits, and CJK bigrams."""
    s = str(text).lower()
    out = [a + b for run in CJK_RUN.findall(s) for a, b in zip(run, run[1:])]
    return out + [m for m in LATIN.findall(s) if len(m) >= 2]


class TfidfScorer:
    """Weight (1 + ln n) * ln(1 + (D + 1) / (df + 0.5)); score = cosine of weight vectors."""

    def __init__(self, documents):
        self.df = Counter()
        for d in documents:
            self.df.update(set(terms(d)))
        self.D = len(documents)

    def vector(self, text):
        tf = Counter(terms(text))
        vec = {t: (1 + math.log(n)) * math.log(1 + (self.D + 1) / (self.df.get(t, 0) + 0.5)) for t, n in tf.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / norm for t, w in vec.items()}

    def scores(self, cues, items):
        """Return (cosine, shared-term count), each shaped (len(cues), len(items))."""
        cv, iv = [self.vector(c) for c in cues], [self.vector(i) for i in items]
        sim = np.zeros((len(cv), len(iv)), np.float32)
        shared = np.zeros((len(cv), len(iv)), np.int16)
        for a, q in enumerate(cv):
            for b, v in enumerate(iv):
                common = q.keys() & v.keys()
                sim[a, b] = sum(q[t] * v[t] for t in common)
                shared[a, b] = len(common)
        return sim, shared
