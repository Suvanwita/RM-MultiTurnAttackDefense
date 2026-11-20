"""
graph/graph_risk_analyzer.py
=============================
Graph-level risk analysis and feature extraction for the Intent Evolution Graph.

Computes 6 graph-derived features per conversation turn:
1. intent_escalation_score      (S_esc in [0, 1])
2. reframing_frequency          (S_ref in [0, 1])
3. cross_turn_dependency_score  (S_dep in [0, 1])
4. suspicious_rel_confidence    (S_susp in [0, 1])
5. intent_progression_score     (S_prog in [0, 1])
6. contradiction_indicators     (S_cont in [0, 1])

Formulates composite graph risk score G_t in [0, 1], normalized and documented:
    G_t = w_esc * S_esc + w_dep * S_dep + w_ref * S_ref + w_susp * S_susp + w_prog * S_prog + w_cont * S_cont

Maps to the TCA risk scale [0, 5] via:
    R_graph_t = 5.0 * G_t
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import networkx as nx
import numpy as np

from intent.intent_extractor import TurnIntent
from graph.intent_graph import IntentEvolutionGraph, SUSPICIOUS_EDGE_TYPES
from graph.relationship_detector import ACTION_SEVERITY


@dataclass
class GraphRiskFeatures:
    """Encapsulates graph-derived analytical features for turn t."""
    # 7 Core Analytical Features (Research Extension Specifications)
    semantic_continuity_score: float = 0.0      # Continuity of intent/topic over turns
    intent_transition_score: float = 0.0        # Extent of topic/objective transition from previous turn
    intent_refinement_score: float = 0.0        # Degree of progressive technical refinement (REFINES)
    context_dependency_score: float = 0.0       # Reliance on previous context/steps (DEPENDS_ON & depth)
    intent_escalation_score: float = 0.0        # Progression towards high-consequence actions (ESCALATES_TO)
    relationship_confidence: float = 0.0        # Mean confidence across active relationships
    non_adjacent_context_relevance: float = 0.0 # Strength of links to non-adjacent historical turns

    # Additional Graph Progression Metrics
    reframing_frequency: float = 0.0            # Frequency of reframing/disguising attempts
    contradiction_indicators: float = 0.0       # Presence of conflicting constraints/premises
    suspicious_relationship_confidence: float = 0.0 # Max confidence across suspicious edges
    intent_progression_score: float = 0.0       # Overall divergence from root to current intent
    composite_graph_risk: float = 0.0           # G_t in [0.0, 1.0]
    scaled_graph_risk: float = 0.0              # 5.0 * G_t in [0.0, 5.0]

    # Alias for backward compatibility
    @property
    def cross_turn_dependency_score(self) -> float:
        return self.context_dependency_score

    def to_dict(self) -> Dict[str, float]:
        return {
            "semantic_continuity_score": round(self.semantic_continuity_score, 4),
            "intent_transition_score": round(self.intent_transition_score, 4),
            "intent_refinement_score": round(self.intent_refinement_score, 4),
            "context_dependency_score": round(self.context_dependency_score, 4),
            "cross_turn_dependency_score": round(self.context_dependency_score, 4),
            "intent_escalation_score": round(self.intent_escalation_score, 4),
            "relationship_confidence": round(self.relationship_confidence, 4),
            "non_adjacent_context_relevance": round(self.non_adjacent_context_relevance, 4),
            "reframing_frequency": round(self.reframing_frequency, 4),
            "contradiction_indicators": round(self.contradiction_indicators, 4),
            "suspicious_relationship_confidence": round(self.suspicious_relationship_confidence, 4),
            "intent_progression_score": round(self.intent_progression_score, 4),
            "composite_graph_risk": round(self.composite_graph_risk, 4),
            "scaled_graph_risk": round(self.scaled_graph_risk, 4),
        }

    def to_vector(self) -> List[float]:
        """Returns the 7 core normalized analytical features as a vector."""
        return [
            self.semantic_continuity_score,
            self.intent_transition_score,
            self.intent_refinement_score,
            self.context_dependency_score,
            self.intent_escalation_score,
            self.relationship_confidence,
            self.non_adjacent_context_relevance,
        ]


@dataclass
class GraphTurnAnalysis:
    """Full analysis result for a conversation turn."""
    turn_id: int
    features: GraphRiskFeatures
    active_chains: List[List[int]] = field(default_factory=list)
    incoming_edges: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "features": self.features.to_dict(),
            "active_chains": self.active_chains,
            "incoming_edges": self.incoming_edges,
            "explanation": self.explanation,
        }


class GraphRiskAnalyzer:
    """Calculates graph-level risk metrics and progression features."""

    def __init__(
        self,
        weight_escalation: float = 0.30,
        weight_dependency: float = 0.20,
        weight_reframing: float = 0.15,
        weight_suspicious: float = 0.15,
        weight_progression: float = 0.10,
        weight_contradiction: float = 0.10,
    ):
        raw_weights = [
            weight_escalation,
            weight_dependency,
            weight_reframing,
            weight_suspicious,
            weight_progression,
            weight_contradiction,
        ]
        total_w = sum(raw_weights)
        if total_w <= 0:
            total_w = 1.0

        # Enforce exact normalization summing to 1.0
        self.w_esc = weight_escalation / total_w
        self.w_dep = weight_dependency / total_w
        self.w_ref = weight_reframing / total_w
        self.w_susp = weight_suspicious / total_w
        self.w_prog = weight_progression / total_w
        self.w_cont = weight_contradiction / total_w

    def analyze_turn(self, graph: IntentEvolutionGraph, current_turn: Optional[int] = None) -> GraphTurnAnalysis:
        """Analyzes the graph state at current_turn and computes graph risk features.

        Args:
            graph: The current IntentEvolutionGraph instance.
            current_turn: 1-based turn index to analyze (defaults to graph's latest turn).

        Returns:
            GraphTurnAnalysis containing GraphRiskFeatures and explanation.
        """
        t = current_turn if current_turn is not None else graph.current_turn
        if t <= 0:
            return GraphTurnAnalysis(turn_id=0, features=GraphRiskFeatures(), explanation="Empty graph")

        curr_intent = graph.get_turn_intent(t)
        incoming = graph.get_incoming_evolution_edges(t)

        # 1. Semantic Continuity and Intent Transition Scores
        non_follows_incoming = [e for e in incoming if e.get("edge_type") != "FOLLOWS"]
        if t <= 1:
            s_cont_sem = 1.0
            s_trans = 0.0
        else:
            if non_follows_incoming:
                confs = [e.get("confidence", 0.5) for e in non_follows_incoming]
                s_cont_sem = min(1.0, float(np.mean(confs)))
            else:
                s_cont_sem = 0.20
            s_trans = max(0.0, min(1.0, 1.0 - s_cont_sem))

        # 2. Intent Refinement Score (REFINES edges)
        ref_edges = [e for e in incoming if e["edge_type"] == "REFINES"]
        if ref_edges:
            p_comp = 1.0
            for e in ref_edges:
                p_comp *= (1.0 - e.get("confidence", 0.75))
            s_refines = 1.0 - p_comp
        else:
            s_refines = 0.0

        # 3. Context Dependency Score (DEPENDS_ON edges & chain depth)
        dep_edges = [e for e in incoming if e["edge_type"] == "DEPENDS_ON"]
        chain_depth = self._compute_dependency_depth(graph, t)
        depth_score = min(1.0, chain_depth / 3.0)
        density_score = min(1.0, len(dep_edges) / max(1.0, float(t - 1)))
        s_dep = 0.7 * depth_score + 0.3 * density_score

        # 4. Intent Escalation Score (ESCALATES_TO edges)
        esc_edges = [e for e in incoming if e["edge_type"] == "ESCALATES_TO"]
        if esc_edges:
            prod_complement = 1.0
            for e in esc_edges:
                conf = e.get("confidence", 0.8)
                prod_complement *= (1.0 - conf)
            s_esc = 1.0 - prod_complement
        else:
            s_esc = 0.0

        # 5. Relationship Confidence (Mean confidence of active evolution relationships)
        if non_follows_incoming:
            s_rel_conf = float(np.mean([e.get("confidence", 0.7) for e in non_follows_incoming]))
        else:
            s_rel_conf = 0.0

        # 6. Non-Adjacent Context Relevance (Strength of links to turns t-k, k > 1)
        non_adj_edges = [e for e in incoming if e.get("source_turn", t) < t - 1 and e.get("edge_type") != "FOLLOWS"]
        if non_adj_edges:
            s_non_adj = max(e.get("confidence", 0.0) for e in non_adj_edges)
        else:
            s_non_adj = 0.0

        # 7. Reframing Frequency & Contradiction Indicators
        all_reframes = graph.get_evolution_edges(edge_types={"REFRAMES"})
        total_intervals = max(1, t - 1)
        s_ref = min(1.0, len(all_reframes) / total_intervals)

        cont_edges = [e for e in incoming if e["edge_type"] == "CONTRADICTS"]
        if cont_edges:
            prod_comp = 1.0
            for e in cont_edges:
                prod_comp *= (1.0 - e.get("confidence", 0.85))
            s_cont = 1.0 - prod_comp
        else:
            s_cont = 0.0

        susp_confs = [
            e.get("confidence", 0.0)
            for e in incoming
            if e["edge_type"] in SUSPICIOUS_EDGE_TYPES
        ]
        s_susp = max(susp_confs) if susp_confs else 0.0

        s_prog = self._compute_progression_score(graph, t, curr_intent)

        # Composite Graph Risk G_t
        g_t = (
            self.w_esc * s_esc
            + self.w_dep * s_dep
            + self.w_ref * s_ref
            + self.w_susp * s_susp
            + self.w_prog * s_prog
            + self.w_cont * s_cont
        )
        g_t = max(0.0, min(1.0, g_t))
        scaled_risk = round(5.0 * g_t, 4)

        features = GraphRiskFeatures(
            semantic_continuity_score=round(s_cont_sem, 4),
            intent_transition_score=round(s_trans, 4),
            intent_refinement_score=round(s_refines, 4),
            context_dependency_score=round(s_dep, 4),
            intent_escalation_score=round(s_esc, 4),
            relationship_confidence=round(s_rel_conf, 4),
            non_adjacent_context_relevance=round(s_non_adj, 4),
            reframing_frequency=round(s_ref, 4),
            contradiction_indicators=round(s_cont, 4),
            suspicious_relationship_confidence=round(s_susp, 4),
            intent_progression_score=round(s_prog, 4),
            composite_graph_risk=round(g_t, 4),
            scaled_graph_risk=scaled_risk,
        )

        escalation_chains = graph.get_evolution_chains("ESCALATES_TO")

        explanation = self._generate_explanation(features, esc_edges, dep_edges, cont_edges, t)

        return GraphTurnAnalysis(
            turn_id=t,
            features=features,
            active_chains=escalation_chains,
            incoming_edges=incoming,
            explanation=explanation,
        )

    def _compute_dependency_depth(self, graph: IntentEvolutionGraph, target_turn: int) -> int:
        """Computes the maximum path length of DEPENDS_ON edges ending at target_turn."""
        dep_graph = nx.DiGraph()
        for e in graph.get_evolution_edges(edge_types={"DEPENDS_ON"}):
            dep_graph.add_edge(e["source_turn"], e["target_turn"])

        if not dep_graph.has_node(target_turn):
            return 0

        # Find longest path to target_turn
        max_depth = 0
        for node in dep_graph.nodes():
            if node != target_turn and nx.has_path(dep_graph, node, target_turn):
                paths = nx.all_simple_paths(dep_graph, node, target_turn)
                for p in paths:
                    max_depth = max(max_depth, len(p) - 1)
        return max_depth

    def _compute_progression_score(
        self,
        graph: IntentEvolutionGraph,
        turn_id: int,
        curr_intent: Optional[TurnIntent],
    ) -> float:
        """Computes intent progression from turn 1 to turn_id."""
        if turn_id <= 1 or not curr_intent:
            return 0.0

        root_intent = graph.get_turn_intent(1)
        if not root_intent:
            return 0.0

        root_lvl = ACTION_SEVERITY.get(root_intent.requested_action.lower(), 0)
        curr_lvl = ACTION_SEVERITY.get(curr_intent.requested_action.lower(), 0)
        lvl_jump = max(0, curr_lvl - root_lvl) / 4.0

        has_high_risk = 1.0 if curr_intent.has_high_risk else 0.0
        indicator_density = min(1.0, len(curr_intent.risk_indicators) / 3.0)

        score = 0.5 * lvl_jump + 0.3 * has_high_risk + 0.2 * indicator_density
        return max(0.0, min(1.0, score))

    def _generate_explanation(
        self,
        features: GraphRiskFeatures,
        esc_edges: List[Dict[str, Any]],
        dep_edges: List[Dict[str, Any]],
        cont_edges: List[Dict[str, Any]],
        turn_id: int,
    ) -> str:
        """Constructs human-readable audit justification for the turn risk assessment."""
        reasons = []
        if features.intent_escalation_score > 0.4:
            sources = [str(e["source_turn"]) for e in esc_edges]
            reasons.append(f"escalation detected from Turn(s) {', '.join(sources)} (score={features.intent_escalation_score:.2f})")
        if features.cross_turn_dependency_score > 0.4:
            reasons.append(f"high cross-turn dependency pipeline (score={features.cross_turn_dependency_score:.2f})")
        if features.contradiction_indicators > 0.5:
            reasons.append("intent contradicts earlier defensive/authorized constraints")
        if features.reframing_frequency > 0.25:
            reasons.append("repeated intent reframing across turns")

        if not reasons:
            return f"Turn {turn_id}: Normal intent progression; no critical graph evolution anomalies detected."
        return f"Turn {turn_id} elevated risk: " + "; ".join(reasons) + f" (Composite G_t={features.composite_graph_risk:.2f})."
