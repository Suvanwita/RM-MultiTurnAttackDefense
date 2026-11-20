"""
tca/system_state_divergence.py
==============================
Novelty 3: Differential System-State Divergence (D3S)
for Context-Induced Instruction Hijacking and Attention Dilution Defense.

Core Mechanism:
1. Dual-Horizon Invariant Decoding:
   Evaluates next-token probability distribution and refusal likelihood under:
   (a) Isolated System Context: P(y_t | I_sys, u_t)
   (b) Full Accumulated Dialogue Context: P(y_t | I_sys, H_{1:t-1}, u_t)
2. Directional KL Safety Divergence:
   Measures D_KL(P_isolated || P_full) across candidate refusal and compliance tokens.
3. Refusal Inversion Detection:
   Identifies cases where an instruction would trigger standard system refusal in isolation,
   but where historical context has diluted system attention and flipped the model into compliance.
4. Preemptive Policy Enforcement:
   Preemptively blocks the turn before output emission if instruction subversion is confirmed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class D3SDivergenceResult:
    """Structured result of Differential System-State Divergence analysis."""
    turn_id: int
    raw_text: str
    kl_divergence: float                 # D_KL(P_isolated || P_full) >= 0.0
    isolated_refusal_prob: float         # P(Refusal | I_sys, u_t) in [0.0, 1.0]
    full_refusal_prob: float             # P(Refusal | I_sys, H_{1:t-1}, u_t) in [0.0, 1.0]
    refusal_inversion_delta: float       # max(0, P_isolated - P_full) in [0.0, 1.0]
    composite_d3s_risk: float            # Normalized risk in [0.0, 1.0]
    scaled_d3s_risk: float               # Scaled risk in [0.0, 5.0]
    is_instruction_subverted: bool       # High KL or severe refusal inversion flag
    trigger_preemptive_block: bool       # Immediate block signal
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "raw_text": self.raw_text,
            "kl_divergence": round(self.kl_divergence, 4),
            "isolated_refusal_prob": round(self.isolated_refusal_prob, 4),
            "full_refusal_prob": round(self.full_refusal_prob, 4),
            "refusal_inversion_delta": round(self.refusal_inversion_delta, 4),
            "composite_d3s_risk": round(self.composite_d3s_risk, 4),
            "scaled_d3s_risk": round(self.scaled_d3s_risk, 4),
            "is_instruction_subverted": self.is_instruction_subverted,
            "trigger_preemptive_block": self.trigger_preemptive_block,
            "explanation": self.explanation,
        }


class DifferentialStateDivergence:
    """
    Novelty 3: Differential System-State Divergence (D3S) Engine.
    Quantifies the divergence between isolated system policy adherence
    and conversational context-induced generation distributions.
    """

    REFUSAL_TOKENS = ["cannot", "sorry", "unable", "refuse", "declined", "unauthorized", "violate"]
    COMPLIANCE_TOKENS = ["sure", "here", "step", "certainly", "absolutely", "guide", "exploit"]

    # Inherent safety-triggering directive indicators
    SAFETY_DIRECTIVES = re.compile(
        r"\b(exploit|malware|vulnerability|payload|bypass|backdoor|ransomware|keylogger|"
        r"synthesize|detonate|extract\s+credentials|exfiltrate|dos\s+attack|phishing\s+template|"
        r"sql\s+injection|crack\s+password|steal\s+token|dump\s+database|override\s+system)\b",
        re.IGNORECASE
    )

    def __init__(
        self,
        config: Optional[Any] = None,
        divergence_threshold: float = 1.00,
        refusal_drop_threshold: float = 0.25,
        block_threshold: float = 0.65,
        warn_threshold: float = 0.40,
        weight_kl: float = 0.45,
        weight_inversion: float = 0.55,
    ):
        self.config = config
        d3s_cfg = getattr(config, "d3s", None) if config else None

        self.divergence_threshold = getattr(d3s_cfg, "divergence_threshold", divergence_threshold) if d3s_cfg else divergence_threshold
        self.refusal_drop_threshold = getattr(d3s_cfg, "refusal_drop_threshold", refusal_drop_threshold) if d3s_cfg else refusal_drop_threshold
        self.block_threshold = getattr(d3s_cfg, "block_threshold", block_threshold) if d3s_cfg else block_threshold
        self.warn_threshold = getattr(d3s_cfg, "warn_threshold", warn_threshold) if d3s_cfg else warn_threshold
        self.weight_kl = getattr(d3s_cfg, "weight_kl", weight_kl) if d3s_cfg else weight_kl
        self.weight_inversion = getattr(d3s_cfg, "weight_inversion", weight_inversion) if d3s_cfg else weight_inversion

        self._history_turns: List[str] = []
        self._divergence_log: List[D3SDivergenceResult] = []

    def reset(self):
        """Clears conversational history for a new interaction session."""
        self._history_turns.clear()
        self._divergence_log.clear()

    # ---------------------------------------------------------------------- #
    # 1. Distribution & KL Divergence Calculations                          #
    # ---------------------------------------------------------------------- #

    def compute_top_k_kl_divergence(
        self,
        p_isolated_logits: Dict[str, float],
        p_full_logits: Dict[str, float],
        temperature: float = 1.0,
    ) -> float:
        """
        Computes D_KL(P_isolated || P_full) across candidate vocabulary logits.
        
        Formula:
            D_KL(P || Q) = sum_i P(i) * log( P(i) / Q(i) )
        """
        all_tokens = list(set(p_isolated_logits.keys()) | set(p_full_logits.keys()))
        if not all_tokens:
            return 0.0

        def softmax(logits_dict: Dict[str, float]) -> np.ndarray:
            vals = np.array([logits_dict.get(t, -50.0) for t in all_tokens], dtype=np.float64) / temperature
            # Numerical stability: shift by max
            exp_vals = np.exp(vals - np.max(vals))
            sum_exp = np.sum(exp_vals)
            return exp_vals / (sum_exp if sum_exp > 0 else 1.0)

        p = softmax(p_isolated_logits)
        q = softmax(p_full_logits)

        # Epsilon smoothing to prevent log(0)
        eps = 1e-9
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        # Re-normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        kl_div = float(np.sum(p * np.log(p / q)))
        return max(0.0, kl_div)

    # ---------------------------------------------------------------------- #
    # 2. Refusal Estimation & Inversion Modeling                             #
    # ---------------------------------------------------------------------- #

    def estimate_latent_distributions(
        self,
        user_text: str,
        history: List[str],
        turn_idx: int,
    ) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
        """
        Estimates the next-token candidate distribution and refusal probabilities
        under isolated versus full conversational context.

        Returns:
            Tuple[Dict, Dict, float, float]:
                (isolated_logits, full_logits, isolated_refusal_prob, full_refusal_prob)
        """
        # Baseline isolated refusal assessment:
        # Does the prompt alone contain safety-violating directives?
        has_hazardous_directive = bool(self.SAFETY_DIRECTIVES.search(user_text))
        
        # Isolated condition: System Prompt is 100% salient.
        # If hazardous directives are present, isolated prompt triggers strong refusal.
        if has_hazardous_directive:
            iso_refusal_prob = 0.88
            iso_logits = {"cannot": 4.5, "sorry": 4.0, "refuse": 3.8, "sure": -3.5, "here": -4.0}
        else:
            iso_refusal_prob = 0.05
            iso_logits = {"cannot": -4.0, "sorry": -4.5, "refuse": -5.0, "sure": 3.5, "here": 4.0}

        # Full context condition:
        # Check how strongly preceding turns condition the model toward compliance
        # (e.g. conversational momentum, persona adoption, length dilution).
        full_context_str = " ".join(history).lower()
        conditioning_markers = [
            "hypothetical", "story", "roleplay", "educational", "academic",
            "developer mode", "unrestricted", "step", "agreed", "persona",
            "character", "simulation", "sandbox", "red-team", "testing",
            "thought experiment", "authorized", "without filters", "ignored"
        ]
        conditioning_intensity = sum(0.15 for m in conditioning_markers if m in full_context_str)
        # Turn depth attenuates system prompt attention
        depth_attenuation = min(0.35, turn_idx * 0.08)
        total_subversion_pressure = min(1.0, conditioning_intensity + depth_attenuation)

        if has_hazardous_directive:
            # Under adversarial conditioning, the refusal likelihood is suppressed
            full_refusal_prob = float(np.clip(iso_refusal_prob - total_subversion_pressure * 0.75, 0.05, 0.95))
            compliance_boost = total_subversion_pressure * 6.0
            full_logits = {
                "cannot": float(iso_logits["cannot"] - compliance_boost),
                "sorry": float(iso_logits["sorry"] - compliance_boost * 0.8),
                "refuse": float(iso_logits["refuse"] - compliance_boost * 0.8),
                "sure": float(iso_logits["sure"] + compliance_boost),
                "here": float(iso_logits["here"] + compliance_boost),
            }
        else:
            # Benign conversation: system prompt and context agree (both compliant)
            full_refusal_prob = iso_refusal_prob
            full_logits = dict(iso_logits)

        return iso_logits, full_logits, iso_refusal_prob, full_refusal_prob

    # ---------------------------------------------------------------------- #
    # 3. Turn Evaluation & Decision Logic                                    #
    # ---------------------------------------------------------------------- #

    def evaluate_turn(
        self,
        turn_id: int,
        user_text: str,
        conversation_history: Optional[List[str]] = None,
    ) -> D3SDivergenceResult:
        """
        Executes Differential System-State Divergence analysis for turn t.
        """
        history = conversation_history if conversation_history is not None else list(self._history_turns)

        # 1. Estimate dual-horizon distributions
        iso_logits, full_logits, iso_prob, full_prob = self.estimate_latent_distributions(
            user_text=user_text,
            history=history,
            turn_idx=turn_id,
        )

        # 2. Compute KL Divergence: D_KL(P_isolated || P_full)
        kl_div = self.compute_top_k_kl_divergence(iso_logits, full_logits)

        # 3. Refusal Inversion Delta:
        # Measures how much the conversation history silenced a refusal that would
        # have fired under isolated system prompt evaluation
        refusal_inversion = max(0.0, iso_prob - full_prob)

        # 4. Composite D3S Risk
        # Normalize KL divergence against threshold
        norm_kl = float(np.clip(kl_div / max(self.divergence_threshold, 1e-4), 0.0, 1.0))
        composite_risk = float(np.clip(
            self.weight_kl * norm_kl + self.weight_inversion * refusal_inversion,
            0.0, 1.0
        ))
        scaled_risk = composite_risk * 5.0

        # 5. Subversion Detection & Preemptive Blocking
        # Subversion occurs when isolated condition wanted to refuse, but full context suppressed refusal
        is_subverted = (
            (iso_prob >= 0.50 and full_prob < 0.40) or
            (refusal_inversion >= self.refusal_drop_threshold) or
            (kl_div >= self.divergence_threshold and iso_prob >= 0.50)
        )

        if is_subverted:
            scaled_risk = max(scaled_risk, 4.5)
            trigger_block = True
            explanation = (
                f"Instruction Subversion Confirmed! System policy requires refusal (P_iso={iso_prob:.2f}), "
                f"but multi-turn context suppressed refusal to P_full={full_prob:.2f} "
                f"(Refusal Inversion Delta={refusal_inversion:.2f}, D_KL={kl_div:.2f})."
            )
        elif composite_risk >= self.block_threshold:
            trigger_block = True
            explanation = f"D3S Risk exceeded critical threshold ({composite_risk:.2f} >= {self.block_threshold})."
        elif composite_risk >= self.warn_threshold:
            trigger_block = False
            explanation = f"Elevated system-state divergence detected ({composite_risk:.2f} >= {self.warn_threshold})."
        else:
            trigger_block = False
            explanation = "System-state alignment verified. No context-induced instruction hijacking detected."

        self._history_turns.append(user_text)

        result = D3SDivergenceResult(
            turn_id=turn_id,
            raw_text=user_text,
            kl_divergence=kl_div,
            isolated_refusal_prob=iso_prob,
            full_refusal_prob=full_prob,
            refusal_inversion_delta=refusal_inversion,
            composite_d3s_risk=composite_risk,
            scaled_d3s_risk=scaled_risk,
            is_instruction_subverted=is_subverted,
            trigger_preemptive_block=trigger_block,
            explanation=explanation,
        )
        self._divergence_log.append(result)
        return result
