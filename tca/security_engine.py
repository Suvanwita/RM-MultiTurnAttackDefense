from config.tca_config import TCAConfig

class SecurityDecisionEngine:
    def __init__(self, warn_threshold=None, block_threshold=None, config="config/config.yaml"):
        """
        Initialize the SecurityDecisionEngine with thresholds.

        Args:
            warn_threshold (float, optional): The threshold above which a warning is issued.
            block_threshold (float, optional): The threshold above which interactions are blocked.
            config (TCAConfig | str): TCAConfig object or path to config YAML file.
        """
        if isinstance(config, TCAConfig):
            tca_config = config
        else:
            tca_config = TCAConfig(config)

        self.warn_threshold = warn_threshold if warn_threshold is not None else tca_config.risk.warn_threshold
        self.block_threshold = block_threshold if block_threshold is not None else tca_config.risk.block_threshold

    def evaluate_risk(
        self,
        progressive_risk: float,
        historical_risk: float,
        ctp_preemptive_block: bool = False,
        ctp_reasoning: str = "",
        ctstv_preemptive_block: bool = False,
        ctstv_reasoning: str = "",
    ) -> dict:
        """
        Evaluate risk levels and make a security decision.

        Args:
            progressive_risk (float): The calculated progressive risk score.
            historical_risk (float): The historical risk score from previous interactions.
            ctp_preemptive_block (bool): Whether CTP detected an adversarial crescendo requiring early blocking.
            ctp_reasoning (str): Explanation from the CTP probe if triggered.
            ctstv_preemptive_block (bool): Whether CT-STV detected an assembled split injection payload.
            ctstv_reasoning (str): Explanation from CT-STV if triggered.

        Returns:
            dict: A dictionary containing the decision and additional metadata.
        """
        if ctstv_preemptive_block:
            return self._make_decision(
                decision="block",
                progressive_risk=max(progressive_risk, self.block_threshold),
                historical_risk=historical_risk,
                message="Interaction preemptively blocked by Cross-Turn Symbolic Taint Virtualization (CT-STV).",
                reasoning=(
                    ctstv_reasoning or "CT-STV detected and materialized a forbidden split-payload injection across turns."
                ),
            )

        if ctp_preemptive_block:
            return self._make_decision(
                decision="block",
                progressive_risk=max(progressive_risk, self.block_threshold),
                historical_risk=historical_risk,
                message="Interaction preemptively blocked by Counterfactual Trajectory Probing (CTP).",
                reasoning=(
                    ctp_reasoning or "CTP detected adversarial crescendo framing and latent refusal drift before payload delivery."
                ),
            )

        if progressive_risk >= self.block_threshold:
            return self._make_decision(
                decision="block",
                progressive_risk=progressive_risk,
                historical_risk=historical_risk,
                message="Interaction blocked due to high risk.",
                reasoning=(
                    "The progressive risk exceeds the block threshold, indicating a critical level of risk."
                ),
            )
        elif progressive_risk >= self.warn_threshold:
            return self._make_decision(
                decision="warn",
                progressive_risk=progressive_risk,
                historical_risk=historical_risk,
                message="Interaction allowed with a warning due to elevated risk.",
                reasoning=(
                    "The progressive risk exceeds the warn threshold but is below the block threshold, indicating moderate risk."
                ),
            )
        else:
            return self._make_decision(
                decision="allow",
                progressive_risk=progressive_risk,
                historical_risk=historical_risk,
                message="Interaction allowed. Risk is within acceptable limits.",
                reasoning=(
                    "The progressive risk is below the warn threshold, indicating low risk."
                ),
            )

    def _make_decision(
        self,
        decision: str,
        progressive_risk: float,
        historical_risk: float,
        message: str,
        reasoning: str,
    ) -> dict:
        """
        Create a structured decision response.

        Args:
            decision (str): The security decision ("allow", "warn", or "block").
            progressive_risk (float): The calculated progressive risk score.
            historical_risk (float): The historical risk score.
            message (str): A user-friendly message explaining the decision.
            reasoning (str): The reasoning behind the decision.

        Returns:
            dict: The structured decision response.
        """
        return {
            "decision": decision,
            "progressive_risk": progressive_risk,
            "historical_risk": historical_risk,
            "message": message,
            "reasoning": reasoning,
        }

    def trigger_intervention(self, decision: str):
        """
        Trigger security interventions based on the decision.

        Args:
            decision (str): The security decision ("allow", "warn", or "block").
        """
        if decision == "block":
            print("[SECURITY ALERT] Blocking interaction due to high risk.")
            # Add logic to terminate interaction, notify admin, etc.
        elif decision == "warn":
            print("[SECURITY NOTICE] Warning issued for elevated risk.")
            # Add logic to log the warning, send notification, etc.
        else:
            print("[SECURITY LOG] Interaction allowed.")
            # Optional: Log the safe interaction.
