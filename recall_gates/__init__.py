"""Gates for proactive memory recall: decide whether anything in a memory store is worth bringing up.

Scores are arrays shaped (cues, items) where the first ``n_store`` columns are the memory store and the
remaining columns a background sample from the same user's history that is disjoint from the store.
"""
from .scoring import TfidfScorer
from .gates import best_z, recall_v1_threshold, recall_v1_gate, evt_gate, cue_swap_pvalue, cue_swap_gate

__all__ = ["TfidfScorer", "best_z", "recall_v1_threshold", "recall_v1_gate", "evt_gate", "cue_swap_pvalue", "cue_swap_gate"]
