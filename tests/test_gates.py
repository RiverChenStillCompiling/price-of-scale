import numpy as np

from recall_gates import TfidfScorer, best_z, cue_swap_gate, evt_gate, recall_v1_threshold


def test_tfidf_scores():
    s = TfidfScorer(["my dog max likes the park", "tax forms are due in april", "max needs a new bowl"])
    sim, shared = s.scores(["what bowl did I buy for max"], ["max needs a new bowl", "tax forms are due in april"])
    assert sim[0, 0] > sim[0, 1] and shared[0, 0] >= 2


def test_threshold_grows():
    assert recall_v1_threshold(8) < recall_v1_threshold(2048)


def test_cue_swap_false_alarms_near_alpha():
    """Exchangeable null cues: the false-alarm rate stays at about alpha whatever the store size."""
    rng = np.random.default_rng(0)
    K, alpha, trials = 99, 0.05, 1000
    for n_store in (8, 256):
        fired = 0
        for _ in range(trials):
            m = n_store + 300
            sem = rng.normal(0.6, 0.05, (K + 1, m))
            lex, shared = np.zeros_like(sem), np.zeros_like(sem, dtype=int)
            fired += cue_swap_gate(lex, sem, shared, n_store, alpha)[0]
        assert fired / trials < alpha + 0.025


def test_cue_swap_finds_planted_memory():
    rng = np.random.default_rng(1)
    sem = rng.normal(0.6, 0.05, (100, 308))
    sem[0, 3] = 0.95
    speak, idx, p = cue_swap_gate(np.zeros_like(sem), sem, np.zeros_like(sem, dtype=int), 8)
    assert speak and idx == 3 and p <= 0.05
    assert best_z(np.zeros((1, 10)), np.full((1, 10), 0.1), np.zeros((1, 10)), 5)[0][0] == -np.inf


def test_evt_tied_channel_does_not_silence_the_other():
    """An all-zero lexical channel has no tail; the semantic channel must still decide."""
    sem = np.linspace(0.2, 0.4, 308)
    sem[3] = 0.95
    speak, idx = evt_gate(np.zeros(308), sem, 8)
    assert speak and idx == 3
    assert evt_gate(np.zeros(8), np.linspace(0.2, 0.4, 8), 8) == (False, 0)
