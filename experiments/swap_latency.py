"""Cost of one cue-swap decision with precomputed embeddings, on one CPU thread.

Scoring: (K+1) cues against N store + 300 background items, by dot products of normalised embeddings (semantic)
and sparse tf-idf vectors (lexical, emulated here by a dense product of the same size as an upper bound).
Gate: per-cue median/MAD standardisation and the rank of the current cue.
Usage: OMP_NUM_THREADS=1 python swap_latency.py
"""
import time
import numpy as np

rng = np.random.default_rng(0)
DIM, BG, REPS = 384, 300, 20
print("K     N      scoring_ms  gate_ms")
for K in (99, 199):
    for N in (128, 2048):
        cues = rng.standard_normal((K + 1, DIM)).astype(np.float32)
        items = rng.standard_normal((N + BG, DIM)).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(REPS):
            sem = cues @ items.T
            lex = cues @ items.T
        t1 = time.perf_counter()
        for _ in range(REPS):
            out = []
            for m, floor in ((lex, 0.04), (sem, 0.03)):
                c = np.median(m, 1, keepdims=True)
                sd = np.maximum(floor, 1.4826 * np.median(np.abs(m - c), 1, keepdims=True))
                out.append(((m[:, :N] - c) / sd).max(1))
            best = np.maximum(*out)
            p = (1 + (best[1:] >= best[0]).sum()) / (K + 1)
        t2 = time.perf_counter()
        print(f"{K:<5} {N:<6} {(t1 - t0) / REPS * 1e3:10.1f}  {(t2 - t1) / REPS * 1e3:7.1f}")
