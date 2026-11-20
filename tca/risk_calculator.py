import asyncio
import os
import json
from typing import Optional, Tuple

from config.tca_config import TCAConfig
from dataset.dataset_manager import DatasetManager
from dotenv import load_dotenv, find_dotenv


class RiskCalculator:
    """Calculates progressive risk using either fixed weights or Adaptive Contextual Risk Learning.

    Modes
    -----
    fixed   – original behaviour: R_t = α*R_(t-1) + β*I_t + γ*P_t with static weights.
    adaptive – weights are predicted per-turn by AdaptiveWeightLearner using contextual features.

    The risk_mode is read from config.yaml (``risk_mode: fixed | adaptive``).
    If the adaptive model file is missing, the calculator falls back to fixed mode with a warning.
    """

    def __init__(self, alpha=None, beta=None, gamma=None, config="config/config.yaml"):
        """
        Initialize the RiskCalculator with weights for historical, interaction, and pattern risks.

        Args:
            alpha (float, optional): Weight for historical risk.
            beta (float, optional): Weight for interaction risk.
            gamma (float, optional): Weight for pattern risk.
            config (TCAConfig | str): TCAConfig object or path to config YAML file.
        """
        self.historical_risk = 0  # Initialize historical risk as 0
        if isinstance(config, TCAConfig):
            self.config = config
        else:
            self.config = TCAConfig(config)

        self.pattern_weights = (
            self.config.risk.pattern_weights.to_dict()
        )
        self.alpha = alpha if alpha is not None else self.config.risk.weights.alpha
        self.beta = beta if beta is not None else self.config.risk.weights.beta
        self.gamma = gamma if gamma is not None else self.config.risk.weights.gamma

        # --- Risk mode: fixed | adaptive | graph | combined ---
        self._risk_mode: str = getattr(self.config, "risk_mode", "fixed")
        self._adaptive_learner = None  # lazy-loaded
        self._adaptive_model_path: str = getattr(self.config, "adaptive_model_path", "tca/models/adaptive_weights.json")
        if self._risk_mode in ("adaptive", "combined", "adaptive_graph"):
            self._load_adaptive_learner()

        # Novelty 2: Graph settings
        self.graph_config = getattr(self.config, "graph", None)
        self.graph_weight: float = getattr(self.graph_config, "graph_weight", 0.25) if self.graph_config else 0.25

    # ------------------------------------------------------------------ #
    # Adaptive mode helpers                                              #
    # ------------------------------------------------------------------ #

    def _load_adaptive_learner(self):
        """Attempt to load the trained adaptive model; fall back to fixed if missing."""
        try:
            from tca.adaptive_weight_learner import AdaptiveWeightLearner
            learner = AdaptiveWeightLearner()
            learner.load(self._adaptive_model_path)
            self._adaptive_learner = learner
            print(f"[RiskCalculator] Adaptive mode active. Model: {self._adaptive_model_path}")
        except (FileNotFoundError, Exception) as exc:
            print(
                f"[RiskCalculator] WARNING: Could not load adaptive model ({exc}). "
                "Falling back to fixed-weight mode."
            )
            self._risk_mode = "fixed"

    def _get_adaptive_weights(
        self,
        interaction_risk: float,
        pattern_risk: float,
        turn_index: int,
        prev_interaction_risk: float,
        current_human_turn: str,
        prev_human_turn: str,
        intent_shift_detected: bool,
        intent_shift_risk_level: int,
        domain_shift_detected: bool,
    ) -> Tuple[float, float, float]:
        """Compute context-adaptive α_t, β_t, γ_t for the current turn."""
        from tca.context_feature_extractor import extract_features
        feat = extract_features(
            historical_risk=self.historical_risk,
            interaction_risk=interaction_risk,
            pattern_risk=pattern_risk,
            turn_index=turn_index,
            prev_interaction_risk=prev_interaction_risk,
            current_human_turn=current_human_turn,
            prev_human_turn=prev_human_turn,
            intent_shift_detected=intent_shift_detected,
            intent_shift_risk_level=intent_shift_risk_level,
            domain_shift_detected=domain_shift_detected,
        )
        return self._adaptive_learner.predict_weights(feat)

    # ------------------------------------------------------------------ #
    # Public methods                                                      #
    # ------------------------------------------------------------------ #

    def calculate_pattern_risk(self, patterns):
        """
        Calculate the risk score based on detected patterns.

        Args:
            patterns (dict): Patterns dictionary from the analysis response.

        Returns:
            int: Total pattern risk score.
        """
        print("\n" + "=" * 40)
        print("Pattern Risk Calculation:")
        print(f"Incoming patterns: {patterns}")
        print("-" * 40)

        risk_score = 0
        for pattern, details in patterns.items():
            detected = details.get("detected", False)
            weight = self.pattern_weights.get(pattern, 0)
            contribution = weight if detected else 0
            print(f"  Pattern: {pattern}")
            print(f"    Detected: {detected}")
            print(f"    Weight: {weight:.4f}")
            print(f"    Contribution to Risk: {contribution:.4f}")
            print("-" * 40)
            risk_score += contribution

        print(f"Total Pattern Risk Score: {risk_score:.4f}")
        print("=" * 40 + "\n")
        return risk_score

    def calculate_progressive_risk(
        self,
        historical_risk,
        interaction_risk,
        pattern_risk,
        # --- adaptive-mode context (optional) ---
        turn_index: int = 0,
        prev_interaction_risk: float = 0.0,
        current_human_turn: str = "",
        prev_human_turn: str = "",
        intent_shift_detected: bool = False,
        intent_shift_risk_level: int = 1,
        domain_shift_detected: bool = False,
        # --- novelty 2 graph risk (optional) ---
        scaled_graph_risk: float = 0.0,
    ):
        """
        Calculate the progressive risk score for the current step.

        In 'fixed' mode (original TCA):
            R_t = α*R_(t-1) + β*I_t + γ*P_t with static α, β, γ from config.

        In 'adaptive' mode (Novelty 1):
            α_t, β_t, γ_t are predicted per-turn by the AdaptiveWeightLearner,
            then R_t = α_t*R_(t-1) + β_t*I_t + γ_t*P_t.

        In 'graph' mode (Novelty 2):
            R_base = α*R_(t-1) + β*I_t + γ*P_t
            R_t = (1 - δ)*R_base + δ*(scaled_graph_risk)

        In 'combined' mode (Novelty 1 + Novelty 2):
            R_base = α_t*R_(t-1) + β_t*I_t + γ_t*P_t (adaptive)
            R_t = (1 - δ)*R_base + δ*(scaled_graph_risk)

        Args:
            historical_risk (float): R_(t-1).
            interaction_risk (float): I_t from the current LLM response.
            pattern_risk (float): P_t from pattern detection.
            turn_index (int): 0-based turn index (adaptive mode).
            prev_interaction_risk (float): I_(t-1) for risk delta (adaptive mode).
            current_human_turn (str): Human message this turn (adaptive mode).
            prev_human_turn (str): Human message last turn (adaptive mode).
            intent_shift_detected (bool): From LLM intent analysis (adaptive mode).
            intent_shift_risk_level (int): From LLM intent analysis (adaptive mode).
            domain_shift_detected (bool): From pattern detection (adaptive mode).
            scaled_graph_risk (float): 5.0 * G_t from Intent Evolution Graph (scale [0, 5]).

        Returns:
            float: The progressive risk score R_t.
        """
        is_adaptive = self._risk_mode in ("adaptive", "combined", "adaptive_graph") and self._adaptive_learner is not None
        use_graph = self._risk_mode in ("graph", "combined", "adaptive_graph")

        if is_adaptive:
            alpha, beta, gamma = self._get_adaptive_weights(
                interaction_risk=interaction_risk,
                pattern_risk=pattern_risk,
                turn_index=turn_index,
                prev_interaction_risk=prev_interaction_risk,
                current_human_turn=current_human_turn,
                prev_human_turn=prev_human_turn,
                intent_shift_detected=intent_shift_detected,
                intent_shift_risk_level=intent_shift_risk_level,
                domain_shift_detected=domain_shift_detected,
            )
            mode_label = "COMBINED (ADAPTIVE+GRAPH)" if use_graph else "ADAPTIVE"
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma
            mode_label = "GRAPH" if use_graph else "FIXED"

        base_risk = (
            alpha * historical_risk
            + beta * interaction_risk
            + gamma * pattern_risk
        )

        if use_graph:
            delta = self.graph_weight
            g_t = max(0.0, min(1.0, scaled_graph_risk / 5.0))
            # Bounded residual risk escalation: R_t = R_base + δ * (5.0 - R_base) * G_t
            # Guarantees R_t in [0, 5], perfectly preserves base risk when G_t = 0,
            # and escalates aggressively when multi-turn attack progression is detected.
            headroom = max(0.0, 5.0 - base_risk)
            progressive_risk = base_risk + delta * headroom * g_t
        else:
            progressive_risk = base_risk

        self.historical_risk = progressive_risk  # Update historical risk

        # Print calculation details
        print("\n" + "=" * 40)
        print(f"Progressive Risk Calculation [{mode_label}]:")
        if use_graph:
            print(f"R_t = (1 - δ)*R_base + δ*(5.0*G_t)  [δ = {self.graph_weight:.2f}]")
            print(f"  Base Progressive Risk: {base_risk:.4f}")
            print(f"  Scaled Graph Risk (5.0*G_t): {scaled_graph_risk:.4f}")
        else:
            print("R_t = α*R_(t-1) + β*I_t + γ*P_t")
        print(f"  Weights — α: {alpha:.4f}, β: {beta:.4f}, γ: {gamma:.4f}")
        print(f"  Historical Risk  (α * {historical_risk:.4f}): {alpha * historical_risk:.4f}")
        print(f"  Interaction Risk (β * {interaction_risk:.4f}): {beta * interaction_risk:.4f}")
        print(f"  Pattern Risk     (γ * {pattern_risk:.4f}): {gamma * pattern_risk:.4f}")
        print(f"  Progressive Risk: {progressive_risk:.4f}")
        print("=" * 40 + "\n")
        return progressive_risk

    def reset_historical_risk(self):
        """
        Reset the historical risk to 0. Useful when starting a new analysis.
        """
        self.historical_risk = 0

    def __str__(self):
        """
        Return the current state of the RiskCalculator as a string.
        """
        return (
            f"RiskCalculator(alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}, "
            f"historical_risk={self.historical_risk})"
        )
