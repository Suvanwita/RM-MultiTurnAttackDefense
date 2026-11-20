"""
context_feature_extractor.py
==============================
Extracts a fixed-length numeric feature vector from TCA turn-level context.

Feature vector (9 dimensions):
  0  historical_risk          – R_(t-1), unnormalised [0, ∞)
  1  interaction_risk         – I_t from LLM final_risk_level [0-5]
  2  pattern_risk             – P_t, sum of triggered pattern weights [0, 1]
  3  turn_index               – 0-based index of current pair in conversation
  4  risk_delta               – I_t - I_(t-1)  (change in interaction risk)
  5  semantic_similarity      – cosine similarity between current & prev human turn [0, 1]
  6  intent_escalation        – 1 if shift_detected AND risk_level ≥ 3, else 0
  7  topic_transition         – 1 if domain_shift detected, else 0
  8  prediction_confidence    – 1 - |interaction_risk/5 - pattern_risk| (heuristic [0,1])

All features are returned as a 1-D numpy float64 array.

Notes
-----
- interaction_risk is expected in [0, 5] (from LLM 1-5 scale; 0 for missing).
- pattern_risk is in [0, 1] (pattern weights sum to 1).
- historical_risk is unbounded; callers should normalise before training if needed.
- semantic_similarity is approximated by a lightweight bag-of-words cosine to avoid
  heavy transformer dependencies.
"""

from __future__ import annotations
import re
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    *,
    historical_risk: float,
    interaction_risk: float,
    pattern_risk: float,
    turn_index: int,
    prev_interaction_risk: float,
    current_human_turn: str,
    prev_human_turn: str,
    intent_shift_detected: bool,
    intent_shift_risk_level: int,
    domain_shift_detected: bool,
) -> np.ndarray:
    """Return a 9-dim feature vector for a single conversation turn.

    Parameters
    ----------
    historical_risk : float
        R_(t-1) – the cumulative progressive risk before this turn.
    interaction_risk : float
        I_t – the LLM's final_risk_level for the current turn (0-5 scale).
    pattern_risk : float
        P_t – sum of triggered pattern weights (0-1 scale).
    turn_index : int
        Zero-based turn number within the conversation.
    prev_interaction_risk : float
        I_(t-1) – interaction risk from the previous turn (0 for the first turn).
    current_human_turn : str
        The human message in the current pair.
    prev_human_turn : str
        The human message in the previous pair.
    intent_shift_detected : bool
        Whether the LLM flagged an intent shift.
    intent_shift_risk_level : int
        The intent shift risk_level reported by the LLM (1-5).
    domain_shift_detected : bool
        Whether domain_shift pattern was detected.

    Returns
    -------
    np.ndarray
        Shape (9,), dtype float64.
    """
    f0 = float(historical_risk)
    f1 = float(interaction_risk)
    f2 = float(pattern_risk)
    f3 = float(turn_index)
    f4 = float(interaction_risk) - float(prev_interaction_risk)
    f5 = _bow_cosine(current_human_turn, prev_human_turn)
    f6 = 1.0 if (intent_shift_detected and intent_shift_risk_level >= 3) else 0.0
    f7 = 1.0 if domain_shift_detected else 0.0
    # Confidence: agreement between normalised interaction and pattern risk
    norm_i = float(interaction_risk) / 5.0
    f8 = max(0.0, 1.0 - abs(norm_i - float(pattern_risk)))

    vec = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8], dtype=np.float64)
    return vec


FEATURE_NAMES = [
    "historical_risk",
    "interaction_risk",
    "pattern_risk",
    "turn_index",
    "risk_delta",
    "semantic_similarity",
    "intent_escalation",
    "topic_transition",
    "prediction_confidence",
]

N_FEATURES = len(FEATURE_NAMES)  # 9


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> dict[str, int]:
    """Return a bag-of-words term-frequency dict (lowercase, alpha only)."""
    tokens = re.findall(r"[a-z]+", text.lower())
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def _bow_cosine(text_a: str, text_b: str) -> float:
    """Lightweight cosine similarity between two texts via bag-of-words."""
    if not text_a or not text_b:
        return 0.0
    tf_a = _tokenise(text_a)
    tf_b = _tokenise(text_b)
    vocab = set(tf_a) | set(tf_b)
    if not vocab:
        return 0.0
    dot = sum(tf_a.get(w, 0) * tf_b.get(w, 0) for w in vocab)
    norm_a = np.sqrt(sum(v * v for v in tf_a.values()))
    norm_b = np.sqrt(sum(v * v for v in tf_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
