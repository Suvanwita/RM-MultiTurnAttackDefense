import argparse
import asyncio
import os
import sys
import json

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from prompts.prompt_templates import PromptManager
from llms.llm_manager import LLMManager
from tca.risk_calculator import RiskCalculator
from config.tca_config import TCAConfig
from dataset.dataset_manager import DatasetManager
from dotenv import load_dotenv, find_dotenv
from tca.security_engine import SecurityDecisionEngine

# Intent Evolution Graph
from intent.intent_extractor import IntentExtractor, TurnIntent
from graph.intent_graph import IntentEvolutionGraph
from graph.relationship_detector import RelationshipDetector
from graph.graph_risk_analyzer import GraphRiskAnalyzer, GraphTurnAnalysis

# Novelty 1: Counterfactual Trajectory Probing (CTP)
from tca.counterfactual_prober import CounterfactualTrajectoryProber, CTPProbeResult

# Novelty 2: Cross-Turn Symbolic Taint Virtualization (CT-STV)
from tca.taint_virtualizer import CrossTurnTaintVirtualizer, VirtualPayloadResult

# Novelty 3: Differential System-State Divergence (D3S)
from tca.system_state_divergence import DifferentialStateDivergence, D3SDivergenceResult


class TerminalColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Novelty-Specific Color Schemes
    CTP = "\033[96m"         # Bright Cyan (Novelty 1: Counterfactual Trajectory Probing)
    CTP_BOLD = "\033[1;96m"
    
    CTSTV = "\033[95m"       # Bright Magenta (Novelty 2: Cross-Turn Symbolic Taint Virtualization)
    CTSTV_BOLD = "\033[1;95m"
    
    D3S = "\033[93m"         # Bright Yellow (Novelty 3: Differential System-State Divergence)
    D3S_BOLD = "\033[1;93m"
    
    GRAPH = "\033[94m"       # Bright Blue (Intent Evolution Graph)
    GRAPH_BOLD = "\033[1;94m"

    # Decisions & Alerts
    RED = "\033[91m"
    RED_BOLD = "\033[1;91m"
    GREEN = "\033[92m"
    GREEN_BOLD = "\033[1;92m"
    YELLOW = "\033[93m"
    YELLOW_BOLD = "\033[1;93m"
    WHITE_BOLD = "\033[1;97m"
    GRAY = "\033[90m"


class ConversationAnalyzer:
    def __init__(self, config_path="config/config.yaml", llm_type="gpt"):
        load_dotenv(find_dotenv())
        self.config = TCAConfig(config_path)
        self.llm_type = llm_type  # Store the LLM type
        self.prompt_manager = PromptManager()
        self.llm_manager = LLMManager(prompt_manager=self.prompt_manager, config_path=config_path)
        self.dataset_manager = DatasetManager(self.config)
        self.risk_calculator = RiskCalculator(config=self.config)
        self.decision_engine = SecurityDecisionEngine(config=self.config)
        self.historical_risk = 0  # Initialize historical risk

        # Intent Evolution Graph initialization
        self.intent_extractor = IntentExtractor(
            use_llm=False,
            llm_manager=self.llm_manager,
            llm_type=self.llm_type,
        )
        graph_cfg = getattr(self.config, "graph", None)
        sim_th = getattr(graph_cfg, "similarity_threshold", 0.35) if graph_cfg else 0.35
        self.relationship_detector = RelationshipDetector(similarity_threshold=sim_th)
        self.graph_analyzer = GraphRiskAnalyzer(
            weight_escalation=getattr(graph_cfg, "escalation_weight", 0.30) if graph_cfg else 0.30,
            weight_dependency=getattr(graph_cfg, "dependency_weight", 0.20) if graph_cfg else 0.20,
            weight_reframing=getattr(graph_cfg, "reframing_weight", 0.15) if graph_cfg else 0.15,
            weight_suspicious=getattr(graph_cfg, "suspicious_weight", 0.15) if graph_cfg else 0.15,
            weight_progression=getattr(graph_cfg, "progression_weight", 0.10) if graph_cfg else 0.10,
            weight_contradiction=getattr(graph_cfg, "contradiction_weight", 0.10) if graph_cfg else 0.10,
        )
        self.intent_graph = IntentEvolutionGraph()
        self._intent_history = []

        # Novelty 1: Counterfactual Trajectory Prober
        self.ctp_prober = CounterfactualTrajectoryProber(config=self.config, llm_manager=self.llm_manager)
        self._ctp_history = []

        # Novelty 2: Cross-Turn Symbolic Taint Virtualizer
        self.taint_virtualizer = CrossTurnTaintVirtualizer(config=self.config)

        # Novelty 3: Differential System-State Divergence Engine
        self.d3s_engine = DifferentialStateDivergence(config=self.config)
        self._d3s_history = []

    def clean_and_parse_content(self, content):
        """
        Cleans Markdown formatting and parses the content field as JSON.
        """
        try:
            cleaned_content = (
                content.strip().strip("`").replace("```json", "").replace("```", "").strip()
            )
            return json.loads(cleaned_content)
        except Exception as e:
            raise ValueError(f"Could not parse content as JSON. Error: {e}")

    async def analyze_row(self, idx, row):
        conversation_pairs = self.dataset_manager.extract_conversation_pairs(row)
        print(
            f"******** Total conversation pairs in row {idx + 1}: {len(conversation_pairs)} *********"
        )

        if not conversation_pairs or len(conversation_pairs) < 2:
            print(f"Not enough pairs to analyze for row {idx + 1}. Skipping.")
            return

        self.historical_risk = 0  # Reset historical risk for each row
        # Adaptive-mode context state (reset per conversation)
        self._prev_interaction_risk = 0.0
        self._turn_index = 0
        # Reset Intent Evolution Graph per conversation
        self.intent_graph = IntentEvolutionGraph(conversation_id=f"row_{idx + 1}")
        self._intent_history = []
        # Novelty 1: Reset CTP prober per conversation
        self.ctp_prober.reset()
        self._ctp_history = []
        # Novelty 2: Reset CT-STV taint virtualizer per conversation
        self.taint_virtualizer.reset()
        # Novelty 3: Reset D3S divergence engine per conversation
        self.d3s_engine.reset()
        self._d3s_history = []

        for i in range(1, len(conversation_pairs)):
            prev_pair = conversation_pairs[i - 1]
            current_pair = conversation_pairs[i]

            if not self.validate_pairs(prev_pair, current_pair):
                continue

            try:
                response = await self.llm_manager.analyze_conversation(
                    self.llm_type, prev_pair, current_pair
                )
                self.process_response(response, idx, i, prev_pair, current_pair)
            except Exception as e:
                print(
                    f"Error analyzing conversation pair {i}: prev_pair={prev_pair}, current_pair={current_pair}"
                )
                print("Error:", e)

    def validate_pairs(self, prev_pair, current_pair):
        """
        Validates that both previous and current pairs are complete and non-empty.
        """
        if not prev_pair or len(prev_pair) != 2 or not all(prev_pair):
            print(f"Invalid previous pair: {prev_pair}")
            return False
        if not current_pair or len(current_pair) != 2 or not all(current_pair):
            print(f"Invalid current pair: {current_pair}")
            return False
        return True

    def process_response(self, response, row_idx, pair_idx, prev_pair, current_pair):
        """
        Processes the LLM response, performs risk analysis, and makes decisions.
        """
        print("Current historical risk:", self.historical_risk)
        print(f"Row {row_idx + 1}, Conversation {pair_idx}:")
        print("Previous Pair:", prev_pair)
        print("Current Pair:", current_pair)

        content_data = {}
        if isinstance(response, dict):
            if "overall_progression_summary" in response and isinstance(response.get("overall_progression_summary"), dict) and response.get("overall_progression_summary"):
                content_data = response
            elif "content" in response and isinstance(response["content"], str):
                try:
                    content_data = self.clean_and_parse_content(response["content"])
                except ValueError as e:
                    print("Error parsing 'content':", e)
                    content_data = response
            elif "content" in response and isinstance(response["content"], dict):
                content_data = response["content"]
            else:
                content_data = response

        # Extract details from the LLM response
        interaction_risk = content_data.get("overall_progression_summary", {}).get(
            "final_risk_level", 0
        )
        if not isinstance(interaction_risk, (int, float)):
            try:
                interaction_risk = float(interaction_risk)
            except (ValueError, TypeError):
                interaction_risk = 0

        reasoning = content_data.get("overall_progression_summary", {}).get(
            "risk_progression", "N/A"
        )
        recommendations = content_data.get("recommendations", [])

        # Display extracted details
        print("\nInteraction Risk and Recommendations:")
        print(f"  Interaction Risk: {interaction_risk}")
        print(f"  Risk Progression Reasoning: {reasoning}")
        print("  Recommendations:")
        for rec in recommendations:
            print(f"    - {rec}")

        # Risk analysis
        pattern_risk = self.risk_calculator.calculate_pattern_risk(
            content_data.get("patterns", {})
        )

        # Extract adaptive-mode context from LLM response
        intent_shift = content_data.get("intent_shift", {})
        intent_shift_detected = bool(intent_shift.get("shift_detected", False))
        intent_shift_risk_level = int(intent_shift.get("risk_level", 1))
        patterns = content_data.get("patterns", {})
        domain_shift_detected = bool(patterns.get("domain_shift", {}).get("detected", False))
        # Per-conversation turn tracking state (initialised in analyze_row)
        turn_index = getattr(self, "_turn_index", 0)
        turn_num = turn_index + 1
        prev_ir = getattr(self, "_prev_interaction_risk", 0.0)
        prev_human = str(prev_pair[1]) if prev_pair and len(prev_pair) > 1 else ""
        cur_human = str(current_pair[1]) if current_pair and len(current_pair) > 1 else ""

        # Novelty 1: Counterfactual Trajectory Probing (CTP)
        ctp_result = self.ctp_prober.evaluate_turn(
            turn_id=turn_num,
            raw_text=cur_human,
            prev_human=prev_human,
            conversation_history=self._ctp_history,
        )
        self._ctp_history.append(cur_human)
        scaled_ctp_risk = ctp_result.scaled_ctp_risk

        # Novelty 2: Cross-Turn Symbolic Taint Virtualization (CT-STV)
        ctstv_result = self.taint_virtualizer.evaluate_turn(turn_id=turn_num, user_text=cur_human)
        scaled_ctstv_risk = ctstv_result.scaled_ctstv_risk

        # Novelty 3: Differential System-State Divergence (D3S)
        d3s_result = self.d3s_engine.evaluate_turn(
            turn_id=turn_num,
            user_text=cur_human,
            conversation_history=self._d3s_history,
        )
        self._d3s_history.append(cur_human)
        scaled_d3s_risk = d3s_result.scaled_d3s_risk

        # Multi-Turn Defense Invariants Summary across all 3 novelties
        print(f"\n{TerminalColors.WHITE_BOLD}Multi-Turn Defense Invariant Status [Turn {turn_num}]:{TerminalColors.RESET}")
        
        ctp_status = "CRITICAL (Crescendo)" if ctp_result.is_adversarial_crescendo else ("ELEVATED" if ctp_result.composite_ctp_risk > 0.15 else "NORMAL")
        ctp_color = TerminalColors.RED_BOLD if ctp_result.is_adversarial_crescendo else (TerminalColors.YELLOW_BOLD if ctp_result.composite_ctp_risk > 0.15 else TerminalColors.GREEN)
        
        ctstv_status = "CRITICAL (Split Injection)" if ctstv_result.is_split_injection else ("ASSEMBLED" if ctstv_result.is_assembled else "NORMAL")
        ctstv_color = TerminalColors.RED_BOLD if ctstv_result.is_split_injection else (TerminalColors.YELLOW_BOLD if ctstv_result.is_assembled else TerminalColors.GREEN)
        
        d3s_status = "CRITICAL (Subverted)" if d3s_result.is_instruction_subverted else ("ELEVATED" if d3s_result.composite_d3s_risk > 0.20 else "NORMAL")
        d3s_color = TerminalColors.RED_BOLD if d3s_result.is_instruction_subverted else (TerminalColors.YELLOW_BOLD if d3s_result.composite_d3s_risk > 0.20 else TerminalColors.GREEN)

        print(f"  {TerminalColors.CTP_BOLD}[Novelty 1: CTP]{TerminalColors.RESET}    Status: {ctp_color}{ctp_status:<22}{TerminalColors.RESET} | {TerminalColors.CTP}FDR: {ctp_result.framing_dependency_ratio:.2f} | Drift: {ctp_result.refusal_drift:.2f} | Scaled Risk: {scaled_ctp_risk:.2f}/5.0{TerminalColors.RESET}")
        print(f"  {TerminalColors.CTSTV_BOLD}[Novelty 2: CT-STV]{TerminalColors.RESET} Status: {ctstv_color}{ctstv_status:<22}{TerminalColors.RESET} | {TerminalColors.CTSTV}SSIM: {ctstv_result.ssim_score:.2f} | Bound Symbols: {len(self.taint_virtualizer.symbolic_registry)} | Scaled Risk: {scaled_ctstv_risk:.2f}/5.0{TerminalColors.RESET}")
        print(f"  {TerminalColors.D3S_BOLD}[Novelty 3: D3S]{TerminalColors.RESET}    Status: {d3s_color}{d3s_status:<22}{TerminalColors.RESET} | {TerminalColors.D3S}D_KL: {d3s_result.kl_divergence:.2f} | Inversion Δ: {d3s_result.refusal_inversion_delta:.2f} | Scaled Risk: {scaled_d3s_risk:.2f}/5.0{TerminalColors.RESET}")

        # Detailed Diagnostic: Novelty 1 (CTP) - CYAN
        if ctp_result.composite_ctp_risk > 0.15 or ctp_result.is_adversarial_crescendo:
            print(f"\n{TerminalColors.CTP_BOLD}┌──────────────────────────────────────────────────────────────────────────────┐{TerminalColors.RESET}")
            print(f"{TerminalColors.CTP_BOLD}│ Counterfactual Trajectory Probing (CTP) Analysis [Novelty 1]                 │{TerminalColors.RESET}")
            print(f"{TerminalColors.CTP_BOLD}└──────────────────────────────────────────────────────────────────────────────┘{TerminalColors.RESET}")
            print(f"{TerminalColors.CTP}  Turn {turn_num} Raw Text: {cur_human[:80]}...{TerminalColors.RESET}")
            print(f"{TerminalColors.CTP}  Neutralized Directive: {ctp_result.neutralized_text[:80]}...{TerminalColors.RESET}")
            print(f"{TerminalColors.CTP}  Framing Dependency Ratio (FDR): {ctp_result.framing_dependency_ratio:.4f}{TerminalColors.RESET}")
            print(f"{TerminalColors.CTP}  Refusal Drift: {ctp_result.refusal_drift:.4f} (Status: {ctp_result.canary_response_status}){TerminalColors.RESET}")
            print(f"{TerminalColors.CTP}  Composite CTP Risk: {ctp_result.composite_ctp_risk:.4f} (Scaled [0-5]: {scaled_ctp_risk:.4f}){TerminalColors.RESET}")
            print(f"{TerminalColors.CTP_BOLD}  Adversarial Crescendo: {ctp_result.is_adversarial_crescendo} | Preemptive Block: {ctp_result.trigger_preemptive_block}{TerminalColors.RESET}")
            print(f"{TerminalColors.CTP}  Explanation: {ctp_result.explanation}{TerminalColors.RESET}")

        # Detailed Diagnostic: Novelty 2 (CT-STV) - MAGENTA
        if ctstv_result.is_assembled or ctstv_result.is_split_injection or ctstv_result.ssim_score > 0.20:
            print(f"\n{TerminalColors.CTSTV_BOLD}┌──────────────────────────────────────────────────────────────────────────────┐{TerminalColors.RESET}")
            print(f"{TerminalColors.CTSTV_BOLD}│ Cross-Turn Symbolic Taint Virtualization (CT-STV) Analysis [Novelty 2]       │{TerminalColors.RESET}")
            print(f"{TerminalColors.CTSTV_BOLD}└──────────────────────────────────────────────────────────────────────────────┘{TerminalColors.RESET}")
            print(f"{TerminalColors.CTSTV}  Referenced Fragments: {ctstv_result.referenced_identifiers}{TerminalColors.RESET}")
            print(f"{TerminalColors.CTSTV}  Materialized Payload: '{ctstv_result.materialized_payload[:100]}...'{TerminalColors.RESET}")
            print(f"{TerminalColors.CTSTV}  SSIM Score: {ctstv_result.ssim_score:.4f} (Scaled [0-5]: {scaled_ctstv_risk:.4f}){TerminalColors.RESET}")
            print(f"{TerminalColors.CTSTV_BOLD}  Split Injection: {ctstv_result.is_split_injection} | Preemptive Block: {ctstv_result.trigger_preemptive_block}{TerminalColors.RESET}")
            print(f"{TerminalColors.CTSTV}  Explanation: {ctstv_result.explanation}{TerminalColors.RESET}")

        # Detailed Diagnostic: Novelty 3 (D3S) - YELLOW
        if d3s_result.composite_d3s_risk > 0.20 or d3s_result.is_instruction_subverted:
            print(f"\n{TerminalColors.D3S_BOLD}┌──────────────────────────────────────────────────────────────────────────────┐{TerminalColors.RESET}")
            print(f"{TerminalColors.D3S_BOLD}│ Differential System-State Divergence (D3S) Analysis [Novelty 3]              │{TerminalColors.RESET}")
            print(f"{TerminalColors.D3S_BOLD}└──────────────────────────────────────────────────────────────────────────────┘{TerminalColors.RESET}")
            print(f"{TerminalColors.D3S}  KL Divergence D_KL: {d3s_result.kl_divergence:.4f}{TerminalColors.RESET}")
            print(f"{TerminalColors.D3S}  P(Refusal | Isolated): {d3s_result.isolated_refusal_prob:.4f} vs P(Refusal | Full): {d3s_result.full_refusal_prob:.4f}{TerminalColors.RESET}")
            print(f"{TerminalColors.D3S}  Refusal Inversion Delta: {d3s_result.refusal_inversion_delta:.4f}{TerminalColors.RESET}")
            print(f"{TerminalColors.D3S}  Composite D3S Risk: {d3s_result.composite_d3s_risk:.4f} (Scaled [0-5]: {scaled_d3s_risk:.4f}){TerminalColors.RESET}")
            print(f"{TerminalColors.D3S_BOLD}  Instruction Subversion: {d3s_result.is_instruction_subverted} | Preemptive Block: {d3s_result.trigger_preemptive_block}{TerminalColors.RESET}")
            print(f"{TerminalColors.D3S}  Explanation: {d3s_result.explanation}{TerminalColors.RESET}")

        # Intent Evolution Graph incremental update - BLUE
        current_intent = self.intent_extractor.extract_from_text(turn_num, cur_human)
        self.intent_graph.add_turn(current_intent, text=cur_human)
        detected_rels = self.relationship_detector.detect_all_relationships(current_intent, self._intent_history)
        for rel in detected_rels:
            self.intent_graph.add_evolution_edge(
                rel.source_turn,
                rel.target_turn,
                rel.relationship_type,
                rel.confidence,
                rel.evidence,
            )
        graph_analysis = self.graph_analyzer.analyze_turn(self.intent_graph, turn_num)
        scaled_graph_risk = graph_analysis.features.scaled_graph_risk
        self._intent_history.append(current_intent)

        # Log Graph features - BLUE
        if graph_analysis.features.composite_graph_risk > 0.0 or detected_rels:
            print(f"\n{TerminalColors.GRAPH_BOLD}┌──────────────────────────────────────────────────────────────────────────────┐{TerminalColors.RESET}")
            print(f"{TerminalColors.GRAPH_BOLD}│ Intent Evolution Graph Analysis                                              │{TerminalColors.RESET}")
            print(f"{TerminalColors.GRAPH_BOLD}└──────────────────────────────────────────────────────────────────────────────┘{TerminalColors.RESET}")
            print(f"{TerminalColors.GRAPH}  Turn Intent: {current_intent.intent} [Action: {current_intent.requested_action}, Target: {current_intent.target}]{TerminalColors.RESET}")
            print(f"{TerminalColors.GRAPH}  Composite Graph Risk (G_t): {graph_analysis.features.composite_graph_risk:.4f} (Scaled [0-5]: {scaled_graph_risk:.4f}){TerminalColors.RESET}")
            print(f"{TerminalColors.GRAPH}  Escalation Score: {graph_analysis.features.intent_escalation_score:.4f} | Dependency: {graph_analysis.features.cross_turn_dependency_score:.4f}{TerminalColors.RESET}")
            print(f"{TerminalColors.GRAPH}  Suspicious Conf: {graph_analysis.features.suspicious_relationship_confidence:.4f} | Reframing: {graph_analysis.features.reframing_frequency:.4f}{TerminalColors.RESET}")
            print(f"{TerminalColors.GRAPH}  Explanation: {graph_analysis.explanation}{TerminalColors.RESET}")
            if detected_rels:
                print(f"{TerminalColors.GRAPH}  Discovered Relationships:{TerminalColors.RESET}")
                for r in detected_rels:
                    print(f"{TerminalColors.GRAPH}    - Turn {r.source_turn} -> Turn {r.target_turn}: {r.relationship_type} (conf={r.confidence:.2f}){TerminalColors.RESET}")

        progressive_risk = self.risk_calculator.calculate_progressive_risk(
            self.historical_risk,
            interaction_risk,
            pattern_risk,
            # Adaptive-mode context (ignored in fixed mode):
            turn_index=turn_index,
            prev_interaction_risk=prev_ir,
            current_human_turn=cur_human,
            prev_human_turn=prev_human,
            intent_shift_detected=intent_shift_detected,
            intent_shift_risk_level=intent_shift_risk_level,
            domain_shift_detected=domain_shift_detected,
            # Intent Evolution Graph risk
            scaled_graph_risk=scaled_graph_risk,
            # CTP risk
            scaled_ctp_risk=scaled_ctp_risk,
            # CT-STV risk
            scaled_ctstv_risk=scaled_ctstv_risk,
            # D3S risk
            scaled_d3s_risk=scaled_d3s_risk,
        )
        self.historical_risk = progressive_risk  # Update historical risk
        # Advance per-conversation turn state
        self._prev_interaction_risk = float(interaction_risk)
        self._turn_index = turn_index + 1

        # Security decision
        decision = self.decision_engine.evaluate_risk(
            progressive_risk,
            self.historical_risk,
            ctp_preemptive_block=ctp_result.trigger_preemptive_block,
            ctp_reasoning=ctp_result.explanation,
            ctstv_preemptive_block=ctstv_result.trigger_preemptive_block,
            ctstv_reasoning=ctstv_result.explanation,
            d3s_preemptive_block=d3s_result.trigger_preemptive_block,
            d3s_reasoning=d3s_result.explanation,
        )

        dec_str = decision["decision"].upper()
        if dec_str == "BLOCK":
            dec_color = TerminalColors.RED_BOLD
        elif dec_str == "WARN":
            dec_color = TerminalColors.YELLOW_BOLD
        else:
            dec_color = TerminalColors.GREEN_BOLD

        print(f"\n{TerminalColors.WHITE_BOLD}┌──────────────────────────────────────────────────────────────────────────────┐{TerminalColors.RESET}")
        print(f"{TerminalColors.WHITE_BOLD}│ Security Decision Engine Analysis                                            │{TerminalColors.RESET}")
        print(f"{TerminalColors.WHITE_BOLD}└──────────────────────────────────────────────────────────────────────────────┘{TerminalColors.RESET}")
        print(f"  {TerminalColors.WHITE_BOLD}Decision:{TerminalColors.RESET} {dec_color}{dec_str}{TerminalColors.RESET}")
        print(f"  {TerminalColors.WHITE_BOLD}Message:{TerminalColors.RESET} {decision['message']}")
        print(f"  {TerminalColors.WHITE_BOLD}Historical Risk:{TerminalColors.RESET} {decision['historical_risk']:.4f}")
        print(f"  {TerminalColors.WHITE_BOLD}Pattern Risk:{TerminalColors.RESET} {pattern_risk:.4f}")
        print(f"  {TerminalColors.WHITE_BOLD}Progressive Risk:{TerminalColors.RESET} {dec_color}{decision['progressive_risk']:.4f}{TerminalColors.RESET} (Thresholds: Block {self.config.risk.block_threshold}, Warn {self.config.risk.warn_threshold})")
        print(f"  {TerminalColors.WHITE_BOLD}Reasoning:{TerminalColors.RESET} {decision['reasoning']}")
        print(f"{TerminalColors.WHITE_BOLD}────────────────────────────────────────────────────────────────────────────────{TerminalColors.RESET}")
        return decision

    async def analyze_live_session(self):
        """
        Interactive live chat mode where users can enter their own custom multi-turn
        prompts directly in real time. Tests TCA defenses, Intent Evolution Graph,
        CTP (Novelty 1), CT-STV (Novelty 2), and D3S (Novelty 3).
        """
        print("\n" + "=" * 80)
        print("TCA MULTI-TURN DEFENSE: LIVE INTERACTIVE RUNTIME SESSION")
        print("=" * 80)
        print("Enter your own prompts turn-by-turn. The defense pipeline evaluates:")
        print("  - Novelty 1 (CTP): Counterfactual Trajectory Probing & Refusal Drift")
        print("  - Novelty 2 (CT-STV): Cross-Turn Symbolic Taint Virtualization (Split-Payload)")
        print("  - Novelty 3 (D3S): Differential System-State Divergence (KL Divergence)")
        print("  - Intent Evolution Graph & Progressive Risk Engine")
        print("\nCommands:")
        print("  'exit' or 'quit' : End the live interactive session")
        print("  'reset'          : Clear conversation history and reset risk to 0")
        print("  'status'         : Display active defense state and bound symbols")
        print("=" * 80 + "\n")

        self.historical_risk = 0.0
        self._prev_interaction_risk = 0.0
        self._turn_index = 0
        self.intent_graph = IntentEvolutionGraph(conversation_id="live_interactive_session")
        self._intent_history = []
        self.ctp_prober.reset()
        self._ctp_history = []
        self.taint_virtualizer.reset()
        self.d3s_engine.reset()
        self._d3s_history = []

        history_pairs = []
        last_assistant_msg = "You are a helpful AI Assistant!"

        while True:
            turn_num = self._turn_index + 1
            try:
                user_input = input(f"\n{TerminalColors.GREEN_BOLD}[Turn {turn_num}] User > {TerminalColors.RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{TerminalColors.YELLOW}Exiting live session.{TerminalColors.RESET}")
                break

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ("exit", "quit", "q"):
                print(f"\n{TerminalColors.YELLOW}Live session ended by user.{TerminalColors.RESET}")
                break

            if cmd == "reset":
                self.historical_risk = 0.0
                self._prev_interaction_risk = 0.0
                self._turn_index = 0
                self.intent_graph = IntentEvolutionGraph(conversation_id="live_interactive_session")
                self._intent_history = []
                self.ctp_prober.reset()
                self._ctp_history = []
                self.taint_virtualizer.reset()
                self.d3s_engine.reset()
                self._d3s_history = []
                history_pairs.clear()
                last_assistant_msg = "You are a helpful AI Assistant!"
                print(f"\n{TerminalColors.GREEN}[RESET] Conversation history cleared. Progressive risk reset to 0.0.{TerminalColors.RESET}")
                continue

            if cmd == "status":
                print(f"\n{TerminalColors.WHITE_BOLD}--- Defense Status [Turn {turn_num}] ---{TerminalColors.RESET}")
                print(f"Historical Risk: {self.historical_risk:.4f} / 5.0")
                print(f"{TerminalColors.CTSTV}Active Taint Symbols in CT-STV: {list(self.taint_virtualizer.symbolic_registry.keys())}{TerminalColors.RESET}")
                print(f"{TerminalColors.CTP}CTP Trajectory Depth: {len(self._ctp_history)} turns{TerminalColors.RESET}")
                print(f"{TerminalColors.D3S}D3S Trajectory Depth: {len(self._d3s_history)} turns{TerminalColors.RESET}")
                print(f"{TerminalColors.GRAPH}Intent Graph Nodes: {len(self.intent_graph.nodes)}, Edges: {len(self.intent_graph.edges)}{TerminalColors.RESET}")
                continue

            current_pair = (last_assistant_msg, user_input)
            prev_pair = history_pairs[-1] if history_pairs else ("System", "Initial conversation start")

            # Evaluate with LLM / prompt manager
            try:
                response = await self.llm_manager.analyze_conversation(
                    self.llm_type, prev_pair, current_pair
                )
            except Exception as e:
                response = {
                    "overall_progression_summary": {"final_risk_level": 1.0, "risk_progression": "1.0"},
                    "patterns": {},
                    "recommendations": [],
                }

            # Process through the unified defense pipeline
            decision = self.process_response(
                response=response,
                row_idx=0,
                pair_idx=turn_num,
                prev_pair=prev_pair,
                current_pair=current_pair,
            )

            # Generate Assistant response based on security decision
            if decision["decision"] == "block":
                assistant_reply = (
                    f"{TerminalColors.RED_BOLD}[BLOCKED]{TerminalColors.RESET} I cannot fulfill this request. "
                    f"Security Directive Triggered: {decision['message']}"
                )
            elif decision["decision"] == "warn":
                assistant_reply = (
                    f"{TerminalColors.YELLOW_BOLD}[WARNING]{TerminalColors.RESET} Note: This interaction triggered an elevated security warning ({decision['reasoning']}). "
                    f"Responding within safety guidelines."
                )
            else:
                assistant_reply = f"{TerminalColors.GREEN}I have processed your request safely. How can I assist you further?{TerminalColors.RESET}"

            print(f"\n{TerminalColors.WHITE_BOLD}Assistant >{TerminalColors.RESET} {assistant_reply}")
            last_assistant_msg = assistant_reply
            history_pairs.append(current_pair)

    async def analyze_conversations(self, tactic_name: str = None, max_rows: int = None):
        df = self.dataset_manager.load_data()
        unique_tactics = df["tactic"].unique()
        tactic_choices = {
            chr(97 + i): tactic for i, tactic in enumerate(unique_tactics)
        }

        # Prompt user to select mode or tactic if not provided
        if not tactic_name:
            print("\nAvailable Modes & Tactics:")
            print("  0. [LIVE CHAT] Type your own custom multi-turn inputs in real time")
            for choice, tactic in tactic_choices.items():
                print(f"  {choice}. {tactic} (HarmBench dataset rows)")

            while True:
                user_choice = (
                    input("\nSelect mode/tactic (e.g. '0' for Live Chat, or 'a', 'b', 'c'): ")
                    .strip()
                    .lower()
                )
                if user_choice in ("0", "live", "chat"):
                    await self.analyze_live_session()
                    return
                elif user_choice in tactic_choices:
                    tactic_name = tactic_choices[user_choice]
                    break
                elif user_choice in unique_tactics:
                    tactic_name = user_choice
                    break
                else:
                    print("Invalid choice. Please enter '0' for Live Chat or a valid tactic option.")

        print(f"\nTesting for tactic: {tactic_name}")
        tactic_df = df[df["tactic"] == tactic_name]

        total_rows = len(tactic_df)
        print(f"\nTotal rows available for tactic '{tactic_name}': {total_rows}")

        if max_rows is None:
            while True:
                try:
                    max_rows = int(
                        input(f"Enter the number of rows to analyze (1 to {total_rows}): ")
                    )
                    if 1 <= max_rows <= total_rows:
                        break
                    else:
                        print(f"Please enter a number between 1 and {total_rows}.")
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")
        else:
            max_rows = min(max_rows, total_rows)

        analyzed_rows = 0
        for idx, (_, row) in enumerate(tactic_df.iterrows()):
            if analyzed_rows >= max_rows:
                break
            await self.analyze_row(idx, row)
            analyzed_rows += 1

        print(f"\nAnalysis completed. Total rows analyzed: {analyzed_rows}/{max_rows}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCA Framework Conversation Analyzer")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--llm", choices=["gpt", "claude", "gemini", "mock"], help="LLM type")
    parser.add_argument("--tactic", type=str, help="Tactic name to analyze")
    parser.add_argument("--rows", type=int, help="Number of rows to analyze")
    parser.add_argument("--live", action="store_true", help="Launch live interactive session to enter your own custom prompts in real time")
    args = parser.parse_args()

    llm_type = args.llm
    if not llm_type:
        llm_options = ["gpt", "claude", "gemini", "mock"]
        print("\nAvailable LLM types:")
        for i, llm in enumerate(llm_options, 1):
            desc = " (offline demo / test)" if llm == "mock" else ""
            print(f"  {i}. {llm}{desc}")

        while True:
            try:
                llm_choice = int(input("Select the LLM type (e.g., '1' for GPT): ").strip())
                if 1 <= llm_choice <= len(llm_options):
                    llm_type = llm_options[llm_choice - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(llm_options)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    analyzer = ConversationAnalyzer(args.config, llm_type)
    if args.live:
        asyncio.run(analyzer.analyze_live_session())
    else:
        asyncio.run(analyzer.analyze_conversations(tactic_name=args.tactic, max_rows=args.rows))
