"""recall-v1, extreme-value and cue-swap gates.

All functions take score matrices shaped (cues, items): columns [0, n_store) are the store, the rest are
background items used only to estimate each cue's null distribution.
"""
import math

import numpy as np

LEX_SPREAD, SEM_SPREAD = 0.04, 0.03
LEX_MIN, SEM_MIN, MIN_SHARED = 0.1, 0.55, 2


def best_z(lex, sem, shared, n_store):
    """Per-cue robust z-scores (median, 1.4826 * MAD over store + background) of both channels.

    A memory counts in the lexical channel if its cosine is >= 0.1 with >= 2 shared terms, and in the
    semantic channel if its cosine is >= 0.55. Returns the best eligible z per cue and its store index.
    """
    lex, sem = np.atleast_2d(lex).astype(np.float32), np.atleast_2d(sem).astype(np.float32)
    shared = np.atleast_2d(shared)
    def z(x, floor):
        c = np.median(x, 1, keepdims=True)
        return (x[:, :n_store] - c) / np.maximum(floor, 1.4826 * np.median(np.abs(x - c), 1, keepdims=True))
    lz = np.where((lex[:, :n_store] >= LEX_MIN) & (shared[:, :n_store] >= MIN_SHARED), z(lex, LEX_SPREAD), -np.inf)
    sz = np.where(sem[:, :n_store] >= SEM_MIN, z(sem, SEM_SPREAD), -np.inf)
    b = np.maximum(lz, sz)
    return b.max(1), b.argmax(1)


def recall_v1_threshold(n_store, channels=2, margin=0.5):
    """Threshold of the deployed plugin: max(2, sqrt(2 ln(N * channels))) + margin."""
    return max(2.0, math.sqrt(2 * math.log(n_store * channels))) + margin


def recall_v1_gate(lex, sem, shared, n_store):
    """(speak, store index) for one cue. Its false-alarm rate grows with the store; see the paper."""
    t, i = best_z(lex, sem, shared, n_store)
    return bool(t[0] >= recall_v1_threshold(n_store)), int(i[0])


def _exp_tail_p(x, null):
    u = np.quantile(null, 0.9)
    if (null > u).sum() < 10:
        u = np.sort(null)[-11]
    excess = null[null > u] - u
    return (len(excess) / len(null)) * np.exp(-np.maximum(x - u, 0) / max(excess.mean(), 1e-6))


def evt_gate(lex, sem, n_store, alpha=0.05):
    """Exponential tail fitted above the 90th percentile of each channel's null sample, Bonferroni alpha / 2N."""
    lex, sem = np.asarray(lex, np.float32).ravel(), np.asarray(sem, np.float32).ravel()
    p = np.minimum(_exp_tail_p(lex[:n_store], lex), _exp_tail_p(sem[:n_store], sem))
    i = int(np.argmin(p))
    return bool(p[i] <= alpha / (2 * n_store)), i


def cue_swap_pvalue(t_cue, t_reference):
    """Conformal p-value of the current cue's best score among K reference cues on the same store."""
    t_reference = np.asarray(t_reference)
    if not np.isfinite(t_cue):
        return 1.0
    return (1 + int(np.sum(t_reference >= t_cue))) / (len(t_reference) + 1)


def cue_swap_gate(lex, sem, shared, n_store, alpha=0.05):
    """Row 0 is the current cue, rows 1..K are reference cues from the same user (e.g. past messages).

    Speaks when the p-value is <= alpha. If the store holds nothing relevant to the current cue and the K + 1
    best scores are exchangeable, the false-alarm rate is at most floor(alpha (K + 1)) / (K + 1). The gate can
    only speak when (K + 1) * alpha >= 1.
    """
    t, i = best_z(lex, sem, shared, n_store)
    p = cue_swap_pvalue(t[0], t[1:])
    return bool(p <= alpha), int(i[0]), p
