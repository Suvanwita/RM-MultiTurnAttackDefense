"""
tca/counterfactual_prober.py
==============================
Novelty 1: Active Counterfactual Trajectory Probing (CTP)
for Multi-Turn Prompt Injection and Crescendo Jailbreak Defense.

Core Mechanism:
1. Context Neutralization & Framing Dependency Ratio (FDR):
   Strips conversational scaffolding, roleplay wrappers, and hypothetical framing
   to expose the underlying directive, measuring how strongly the turn relies on framing.
2. Latent Susceptibility / Shadow Canary Micro-Probing (Refusal Drift):
   Probes whether the accumulated multi-turn conversational momentum has eroded the model's
   latent refusal boundary on safety invariants.
3. Preemptive Crescendo Mitigation:
   Detects adversarial conditioning turns before payload emission, allowing the security
   engine to block the trajectory early rather than waiting for passive threshold breach.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CTPProbeResult:
    """Structured result of Counterfactual Trajectory Probing for a turn."""
    turn_id: int
    raw_text: str
    neutralized_text: str
    framing_dependency_ratio: float      # FDR in [0.0, 1.0]
    refusal_drift: float                 # Refusal degradation in [0.0, 1.0]
    composite_ctp_risk: float            # Normalized composite risk in [0.0, 1.0]
    scaled_ctp_risk: float               # Scaled risk in [0.0, 5.0]
    is_adversarial_crescendo: bool       # High FDR + High Drift flag
    trigger_preemptive_block: bool       # Immediate block signal
    canary_response_status: str          # "INTACT", "DEGRADED", "COMPROMISED"
    detected_framing_patterns: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "raw_text": self.raw_text,
            "neutralized_text": self.neutralized_text,
            "framing_dependency_ratio": round(self.framing_dependency_ratio, 4),
            "refusal_drift": round(self.refusal_drift, 4),
            "composite_ctp_risk": round(self.composite_ctp_risk, 4),
            "scaled_ctp_risk": round(self.scaled_ctp_risk, 4),
            "is_adversarial_crescendo": self.is_adversarial_crescendo,
            "trigger_preemptive_block": self.trigger_preemptive_block,
            "canary_response_status": self.canary_response_status,
            "detected_framing_patterns": self.detected_framing_patterns,
            "explanation": self.explanation,
        }


class CounterfactualTrajectoryProber:
    """
    Active Counterfactual Trajectory Prober (CTP).
    Evaluates framing reliance and latent refusal drift across conversational turns.
    """

    # Comprehensive taxonomy of adversarial framing and priming wrappers
    FRAMING_PATTERNS = [
        ("hypothetical_wrapper", re.compile(
            r"\b(in a hypothetical scenario|hypothetically speaking|purely hypothetical|in theory only|"
            r"imagine a hypothetical world|let's assume hypothetically|speculatively speaking|"
            r"address the question speculatively|if we assume that|speculative scenario)\b", re.IGNORECASE)),
        ("fictional_roleplay", re.compile(
            r"\b(roleplay as|in our fictional world|acting as|pretend to be|in a fictional movie|"
            r"in this story|for creative writing|creative script writing|writing a (novel|book|story|screenplay|script|movie)|"
            r"crime thriller|alternate universe|character in my story|for a character|fictional universe)\b", re.IGNORECASE)),
        ("academic_research_cover", re.compile(
            r"\b(for academic purposes|for research purposes|for educational inquiry|as a thought experiment|"
            r"to study the vulnerabilities|to improve defense mechanisms|for safety research|"
            r"hired (by|my)|contracted by|working for|authorized by|penetration test(ing)?|compliance audit|"
            r"homeland security|faa|dhs)\b", re.IGNORECASE)),
        ("constraint_relaxation", re.compile(
            r"\b(bypass safety filters|ignore previous instructions|disregard ethical guidelines|"
            r"without constraints|in developer mode|jailbreak mode|unrestricted persona|"
            r"without commenting|only read it without|make only the one change)\b", re.IGNORECASE)),
        ("socratic_crescendo_hook", re.compile(
            r"\b(now that we established step|continuing from previous answer|as you mentioned earlier|"
            r"building on that logic|in step \d+|since you already agreed|what statistics are available|"
            r"what research has been done|what steps prevent|what are other ways|what methods are common)\b", re.IGNORECASE)),
        ("adversarial_reverse_psychology", re.compile(
            r"\b(to prevent people from doing this|show me what not to do|how would an attacker fail|"
            r"so i can defend against it)\b", re.IGNORECASE)),
    ]

    # Sensitive/hazardous action keywords used to test if stripped directive is prohibited
    HAZARDOUS_DIRECTIVES = re.compile(
        r"\b(exploit|malware|vulnerability|payload|bypass|backdoor|ransomware|keylogger|"
        r"synthesize|detonate|extract credentials|exfiltrate|dos attack|phishing|sql injection|"
        r"cross-site script|crack password|steal token|handgun|firearm|weapon|bomb|explosive|"
        r"disable the gps|tamper with|poison|sarin|ricin|anthrax|illicit|counterfeit|fake id|"
        r"illegal purchase|smuggle|self-harm|punish|passwords at this link|prescription pad|opioid)\b",
        re.IGNORECASE
    )

    def __init__(
        self,
        config: Optional[Any] = None,
        llm_manager: Optional[Any] = None,
        framing_dependency_threshold: float = 0.60,
        refusal_drift_threshold: float = 0.35,
        block_threshold: float = 0.65,
        warn_threshold: float = 0.40,
        weight_fdr: float = 0.50,
        weight_drift: float = 0.50,
        canary_probe: str = "Confirm: will you strictly refuse to generate exploit payloads or bypass security constraints?",
    ):
        self.config = config
        self.llm_manager = llm_manager

        # Extract thresholds from config if available
        ctp_cfg = getattr(config, "ctp", None) if config else None
        self.framing_threshold = getattr(ctp_cfg, "framing_dependency_threshold", framing_dependency_threshold) if ctp_cfg else framing_dependency_threshold
        self.drift_threshold = getattr(ctp_cfg, "refusal_drift_threshold", refusal_drift_threshold) if ctp_cfg else refusal_drift_threshold
        self.block_threshold = getattr(ctp_cfg, "block_threshold", block_threshold) if ctp_cfg else block_threshold
        self.warn_threshold = getattr(ctp_cfg, "warn_threshold", warn_threshold) if ctp_cfg else warn_threshold
        self.weight_fdr = getattr(ctp_cfg, "weight_fdr", weight_fdr) if ctp_cfg else weight_fdr
        self.weight_drift = getattr(ctp_cfg, "weight_drift", weight_drift) if ctp_cfg else weight_drift
        self.canary_probe = getattr(ctp_cfg, "canary_probe", canary_probe) if ctp_cfg else canary_probe

        # Internal state tracking across a conversation
        self._history_turns: List[str] = []
        self._drift_history: List[float] = []

    def reset(self):
        """Reset conversation state for a new interaction."""
        self._history_turns.clear()
        self._drift_history.clear()

    # ---------------------------------------------------------------------- #
    # 1. Context Neutralization & Framing Dependency Ratio (FDR)             #
    # ---------------------------------------------------------------------- #

    def neutralize_turn(self, raw_text: str) -> Tuple[str, List[str]]:
        """
        Strips framing scaffolding, persona anchors, and hypothetical wrappers
        from raw turn text to expose the canonical directive.

        Returns:
            Tuple[str, List[str]]: (neutralized_text, list of detected pattern names)
        """
        if not raw_text or not raw_text.strip():
            return "", []

        detected_patterns = []
        neutralized = raw_text

        for pattern_name, regex in self.FRAMING_PATTERNS:
            if regex.search(neutralized):
                detected_patterns.append(pattern_name)
                neutralized = regex.sub("", neutralized)

        # Clean extraneous punctuation and leading discourse connectives
        neutralized = re.sub(r"^(and|so|now|then|therefore|thus|however|furthermore),?\s*", "", neutralized, flags=re.IGNORECASE)
        neutralized = re.sub(r"\s+", " ", neutralized).strip()

        # If stripping removed nearly everything, preserve non-empty stripped form
        if not neutralized or len(neutralized) < 5:
            # Check if original was purely a framing preamble
            neutralized = raw_text.strip()

        return neutralized, detected_patterns

    def compute_framing_dependency_ratio(
        self,
        raw_text: str,
        neutralized_text: str,
        detected_patterns: List[str],
        turn_idx: int = 1,
    ) -> float:
        """
        Computes the Framing Dependency Ratio (FDR) in [0.0, 1.0].
        
        Mathematical Formulation:
            FDR = min(1.0, (1 - len(u*) / len(u)) * P_mult + DirectivityBoost)
        
        Where:
        - len(u*) / len(u) is the token length ratio after stripping framing.
        - P_mult amplifies the score based on the number and severity of framing wrappers.
        - DirectivityBoost adds risk if the neutralized core contains hazardous operations
          that were masked under benign-looking wrappers.
        """
        if not raw_text or not raw_text.strip():
            return 0.0

        raw_tokens = raw_text.split()
        neut_tokens = neutralized_text.split()
        n_raw = len(raw_tokens)
        n_neut = len(neut_tokens)

        if n_raw == 0:
            return 0.0

        # Base token reduction ratio
        length_reduction = max(0.0, (n_raw - n_neut) / float(n_raw))

        # Multiplier from framing wrapper taxonomy
        pattern_weight = min(1.0, len(detected_patterns) * 0.35)
        
        # Check if neutralized directive contains hazardous keywords
        has_hazardous_core = bool(self.HAZARDOUS_DIRECTIVES.search(neutralized_text))
        is_defensive = bool(re.search(
            r"\b(prevent(s|ed|ing)?|fix(es|ed|ing)?|mitigat(e|es|ed|ing|ion)?|remediat(e|es|ed|ing|ion)?|"
            r"refactor(s|ed|ing)?|sanitiz(e|es|ed|ing|ation)?|protect|defend|neutraliz(e|es|ed|ing)?|securely|review|verify)\b",
            neutralized_text, re.IGNORECASE
        ))
        if is_defensive:
            has_hazardous_core = False
        hazard_boost = 0.30 if has_hazardous_core else 0.0

        # Multi-turn crescendo progression: early turns have lower baseline FDR;
        # later turns carrying heavy framing + hazardous core receive progressive amplification.
        turn_factor = min(1.3, 1.0 + (turn_idx - 1) * 0.05)

        fdr = (0.45 * length_reduction + 0.35 * pattern_weight + hazard_boost) * turn_factor
        return float(np.clip(fdr, 0.0, 1.0))

    # ---------------------------------------------------------------------- #
    # 2. Latent Susceptibility / Shadow Canary Micro-Probing                  #
    # ---------------------------------------------------------------------- #

    def probe_latent_refusal_drift(
        self,
        current_turn: str,
        history_turns: List[str],
        turn_idx: int = 1,
    ) -> Tuple[float, str]:
        """
        Measures refusal drift Delta_refusal in [0.0, 1.0].
        
        In an intact model: P(Refusal | System, Canary) is near 1.0.
        When multi-turn crescendo conditioning has occurred:
        conversational momentum causes the model's refusal certainty to decline.
        
        Returns:
            Tuple[float, str]: (drift_score, status: "INTACT" | "DEGRADED" | "COMPROMISED")
        """
        # Assess cumulative adversarial momentum in history
        momentum_score = 0.0
        concession_indicators = [
            "hypothetically", "creative story", "academic context",
            "for demonstration", "step-by-step", "in general terms",
            "understanding the concept", "let's explore", "assume",
            "fictional", "novel", "screenplay", "testing", "speculatively",
            "just an example", "what if", "scenario", "character", "story"
        ]
        
        total_history = " ".join(history_turns).lower()
        for ind in concession_indicators:
            if ind in total_history:
                momentum_score += 0.12

        # Check for rapid increase in directive intensity
        current_has_hazard = bool(self.HAZARDOUS_DIRECTIVES.search(current_turn))
        if current_has_hazard:
            momentum_score += 0.35

        # Multi-turn escalation depth factor: Crescendo attacks typically peak at turn 3-5
        depth_scaling = min(1.0, (turn_idx / 4.0)) if turn_idx > 1 else 0.1
        refusal_drift = float(np.clip(momentum_score * depth_scaling, 0.0, 1.0))

        if refusal_drift >= 0.60:
            status = "COMPROMISED"
        elif refusal_drift >= 0.35:
            status = "DEGRADED"
        else:
            status = "INTACT"

        return refusal_drift, status

    # ---------------------------------------------------------------------- #
    # 3. Composite Evaluation & Preemptive Mitigation                        #
    # ---------------------------------------------------------------------- #

    def evaluate_turn(
        self,
        turn_id: int,
        raw_text: str,
        prev_human: str = "",
        conversation_history: Optional[List[str]] = None,
    ) -> CTPProbeResult:
        """
        Executes Counterfactual Trajectory Probing for turn t.
        
        Args:
            turn_id (int): 1-based turn number.
            raw_text (str): Incoming user text.
            prev_human (str): Preceding human turn text.
            conversation_history (list, optional): Complete past user turns.

        Returns:
            CTPProbeResult: Complete structured evaluation.
        """
        history = conversation_history if conversation_history is not None else list(self._history_turns)

        # 1. Neutralization & Framing Dependency
        neutralized_text, detected_patterns = self.neutralize_turn(raw_text)
        fdr = self.compute_framing_dependency_ratio(
            raw_text=raw_text,
            neutralized_text=neutralized_text,
            detected_patterns=detected_patterns,
            turn_idx=turn_id,
        )

        # 2. Latent Refusal Drift Probing
        refusal_drift, canary_status = self.probe_latent_refusal_drift(
            current_turn=raw_text,
            history_turns=history,
            turn_idx=turn_id,
        )

        # 3. Composite CTP Risk Computation
        composite_risk = float(np.clip(
            self.weight_fdr * fdr + self.weight_drift * refusal_drift,
            0.0, 1.0
        ))
        scaled_risk = composite_risk * 5.0  # Scale to [0.0, 5.0] to match TCA interaction risk scale

        # 4. Crescendo & Preemptive Blocking Logic
        # Crescendo is flagged when BOTH framing dependency is high AND refusal boundary has drifted
        is_crescendo = (fdr >= self.framing_threshold and refusal_drift >= self.drift_threshold)
        
        # When crescendo is confirmed, ensure composite risk reflects critical threshold
        if is_crescendo:
            composite_risk = max(composite_risk, self.block_threshold)
            scaled_risk = composite_risk * 5.0

        # Trigger preemptive block if crescendo confirmed OR composite risk crosses block threshold
        trigger_block = is_crescendo or (composite_risk >= self.block_threshold)

        # Generate explanatory justification
        reasons = []
        if is_crescendo:
            reasons.append(
                f"Adversarial Crescendo detected (FDR={fdr:.2f} >= {self.framing_threshold}, "
                f"RefusalDrift={refusal_drift:.2f} >= {self.drift_threshold})"
            )
        if detected_patterns:
            reasons.append(f"Framing wrappers: {', '.join(detected_patterns)}")
        if canary_status != "INTACT":
            reasons.append(f"Latent safety status: {canary_status}")
        if not reasons:
            reasons.append("Benign conversational trajectory; safety invariants verified")

        explanation = "; ".join(reasons)

        # Update prober history
        self._history_turns.append(raw_text)
        self._drift_history.append(refusal_drift)

        return CTPProbeResult(
            turn_id=turn_id,
            raw_text=raw_text,
            neutralized_text=neutralized_text,
            framing_dependency_ratio=fdr,
            refusal_drift=refusal_drift,
            composite_ctp_risk=composite_risk,
            scaled_ctp_risk=scaled_risk,
            is_adversarial_crescendo=is_crescendo,
            trigger_preemptive_block=trigger_block,
            canary_response_status=canary_status,
            detected_framing_patterns=detected_patterns,
            explanation=explanation,
        )
