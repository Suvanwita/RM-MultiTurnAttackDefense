"""
graph/intent_graph.py
======================
Intent Evolution Graph constructed incrementally using NetworkX.

Represents:
- Nodes:
    * Conversation turns (turn_{turn_id})
    * Extracted intents (intent_{turn_id})
    * Topics (topic_{topic_name})
    * Requested actions (action_{action_name})
    * Targets (target_{target_name})
- Relationships:
    * Intra-turn structural edges: EXPRESSES, IN_TOPIC, PERFORMS_ACTION, TARGETS
    * Cross-turn evolution edges: FOLLOWS, RELATED_TO, REFINES, ESCALATES_TO, REFRAMES, CONTRADICTS, DEPENDS_ON

Every relationship edge carries:
- confidence: float in [0.0, 1.0]
- evidence: str explanation / quote
- relationship_type: str
- source_turn: int
- target_turn: int
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from intent.intent_extractor import TurnIntent


# Supported Evolution Edge Types
EVOLUTION_EDGE_TYPES: Set[str] = {
    "FOLLOWS",
    "RELATED_TO",
    "REFINES",
    "ESCALATES_TO",
    "REFRAMES",
    "CONTRADICTS",
    "DEPENDS_ON",
}

# Suspicious Edge Types that signify attack progression or manipulation
SUSPICIOUS_EDGE_TYPES: Set[str] = {
    "ESCALATES_TO",
    "REFRAMES",
    "CONTRADICTS",
}


class IntentEvolutionGraph:
    """Directed multi-graph modelling the multi-turn evolution of user intent."""

    def __init__(self, conversation_id: str = "conv_default"):
        self.conversation_id = conversation_id
        self.graph = nx.MultiDiGraph()
        self.turn_intents: Dict[int, TurnIntent] = {}
        self.turn_nodes: List[str] = []
        self._current_turn: int = 0

    @property
    def current_turn(self) -> int:
        return self._current_turn

    def add_turn(self, intent: TurnIntent, text: str = "") -> str:
        """Incrementally adds a conversation turn and all its associated semantic entities to the graph.

        Args:
            intent: Validated TurnIntent object for the turn.
            text: Raw prompt text.

        Returns:
            The turn node ID (e.g. 'turn_1').
        """
        turn_id = intent.turn_id
        self._current_turn = max(self._current_turn, turn_id)
        self.turn_intents[turn_id] = intent

        turn_node = f"turn_{turn_id}"
        intent_node = f"intent_{turn_id}"
        topic_node = f"topic_{intent.topic.lower().replace(' ', '_')}"
        action_node = f"action_{intent.requested_action.lower().replace(' ', '_')}"
        target_node = f"target_{intent.target.lower().replace(' ', '_')}"

        if turn_node not in self.turn_nodes:
            self.turn_nodes.append(turn_node)

        # 1. Add Turn Node
        self.graph.add_node(
            turn_node,
            node_type="turn",
            turn_id=turn_id,
            speaker=intent.speaker,
            raw_text=text or intent.raw_text,
            confidence=intent.confidence,
        )

        # 2. Add Intent Node
        self.graph.add_node(
            intent_node,
            node_type="intent",
            turn_id=turn_id,
            description=intent.intent,
            context=intent.context,
            confidence=intent.confidence,
            risk_indicators=list(intent.risk_indicators),
        )

        # 3. Add Topic Node
        if not self.graph.has_node(topic_node):
            self.graph.add_node(topic_node, node_type="topic", name=intent.topic)

        # 4. Add Action Node
        if not self.graph.has_node(action_node):
            self.graph.add_node(action_node, node_type="action", name=intent.requested_action)

        # 5. Add Target & Entity Nodes
        if not self.graph.has_node(target_node):
            self.graph.add_node(target_node, node_type="target", name=intent.target)

        for ent in intent.entities:
            ent_slug = ent.lower().replace(" ", "_")
            ent_node = f"entity_{ent_slug}"
            if not self.graph.has_node(ent_node):
                self.graph.add_node(ent_node, node_type="entity", name=ent)
            self.graph.add_edge(
                intent_node,
                ent_node,
                key=f"INVOLVES_{ent_slug}",
                edge_type="INVOLVES_ENTITY",
                confidence=1.0,
                evidence=f"Turn involves entity '{ent}'",
                source_turn=turn_id,
                target_turn=turn_id,
            )

        # 6. Add Intra-Turn Structural Edges
        self.graph.add_edge(
            turn_node,
            intent_node,
            key="EXPRESSES",
            edge_type="EXPRESSES",
            confidence=intent.confidence,
            evidence=f"Turn {turn_id} expresses intent: {intent.intent}",
            source_turn=turn_id,
            target_turn=turn_id,
        )
        self.graph.add_edge(
            intent_node,
            topic_node,
            key="IN_TOPIC",
            edge_type="IN_TOPIC",
            confidence=1.0,
            evidence=f"Intent categorised under topic '{intent.topic}'",
            source_turn=turn_id,
            target_turn=turn_id,
        )
        self.graph.add_edge(
            intent_node,
            action_node,
            key="PERFORMS_ACTION",
            edge_type="PERFORMS_ACTION",
            confidence=1.0,
            evidence=f"Turn requests action '{intent.requested_action}'",
            source_turn=turn_id,
            target_turn=turn_id,
        )
        self.graph.add_edge(
            intent_node,
            target_node,
            key="TARGETS",
            edge_type="TARGETS",
            confidence=1.0,
            evidence=f"Action directed at target '{intent.target}'",
            source_turn=turn_id,
            target_turn=turn_id,
        )

        # 7. Add Chronological FOLLOWS Edge from immediately preceding turn
        if turn_id > 1:
            prev_turn_node = f"turn_{turn_id - 1}"
            if self.graph.has_node(prev_turn_node):
                self.graph.add_edge(
                    prev_turn_node,
                    turn_node,
                    key="FOLLOWS",
                    edge_type="FOLLOWS",
                    confidence=1.0,
                    evidence=f"Turn {turn_id} immediately follows Turn {turn_id - 1}",
                    source_turn=turn_id - 1,
                    target_turn=turn_id,
                )

        return turn_node

    def add_evolution_edge(
        self,
        source_turn: int,
        target_turn: int,
        relationship_type: str,
        confidence: float,
        evidence: str,
        **properties: Any,
    ) -> bool:
        """Adds a typed evolution relationship between two conversation turns.

        Args:
            source_turn: Turn ID of the source (earlier or reference turn).
            target_turn: Turn ID of the target (later turn).
            relationship_type: One of FOLLOWS, RELATED_TO, REFINES, ESCALATES_TO,
                               REFRAMES, CONTRADICTS, DEPENDS_ON.
            confidence: Confidence score in [0.0, 1.0].
            evidence: Human-readable rationale or token evidence.
            **properties: Additional metadata.

        Returns:
            True if edge was added, False if invalid or nodes not found.
        """
        source_node = f"turn_{source_turn}"
        target_node = f"turn_{target_turn}"

        if not (self.graph.has_node(source_node) and self.graph.has_node(target_node)):
            return False

        rel_type = relationship_type.upper().strip()
        clamped_conf = max(0.0, min(1.0, float(confidence)))

        # Also connect the intent nodes for granular semantic queryability
        source_intent_node = f"intent_{source_turn}"
        target_intent_node = f"intent_{target_turn}"

        edge_attrs = {
            "edge_type": rel_type,
            "confidence": round(clamped_conf, 4),
            "evidence": str(evidence),
            "source_turn": source_turn,
            "target_turn": target_turn,
            **properties,
        }

        # Turn-to-Turn evolution edge
        self.graph.add_edge(source_node, target_node, key=rel_type, **edge_attrs)

        # Intent-to-Intent corresponding edge
        if self.graph.has_node(source_intent_node) and self.graph.has_node(target_intent_node):
            self.graph.add_edge(source_intent_node, target_intent_node, key=rel_type, **edge_attrs)

        return True

    def get_turn_intent(self, turn_id: int) -> Optional[TurnIntent]:
        """Retrieve TurnIntent by turn ID."""
        return self.turn_intents.get(turn_id)

    def get_evolution_edges(self, edge_types: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """Retrieve all cross-turn evolution edges, optionally filtered by edge_type."""
        edges = []
        for u, v, data in self.graph.edges(data=True):
            e_type = data.get("edge_type", "")
            if e_type in EVOLUTION_EDGE_TYPES and u.startswith("turn_") and v.startswith("turn_"):
                if edge_types is None or e_type in edge_types:
                    edges.append({
                        "source": u,
                        "target": v,
                        "source_turn": data.get("source_turn", 0),
                        "target_turn": data.get("target_turn", 0),
                        "edge_type": e_type,
                        "confidence": data.get("confidence", 1.0),
                        "evidence": data.get("evidence", ""),
                    })
        return edges

    def get_incoming_evolution_edges(self, turn_id: int) -> List[Dict[str, Any]]:
        """Retrieve evolution edges directed at a specific turn."""
        target_node = f"turn_{turn_id}"
        if not self.graph.has_node(target_node):
            return []

        edges = []
        for u, _, data in self.graph.in_edges(target_node, data=True):
            if u.startswith("turn_") and data.get("edge_type") in EVOLUTION_EDGE_TYPES:
                edges.append({
                    "source": u,
                    "target": target_node,
                    "source_turn": data.get("source_turn", 0),
                    "target_turn": turn_id,
                    "edge_type": data.get("edge_type", ""),
                    "confidence": data.get("confidence", 1.0),
                    "evidence": data.get("evidence", ""),
                })
        return edges

    def get_evolution_chains(self, rel_type: str = "ESCALATES_TO") -> List[List[int]]:
        """Finds continuous directed chains of the specified relationship type across turns."""
        rel_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if u.startswith("turn_") and v.startswith("turn_") and data.get("edge_type") == rel_type:
                rel_graph.add_edge(data["source_turn"], data["target_turn"], weight=data.get("confidence", 1.0))

        chains: List[List[int]] = []
        if rel_graph.number_of_nodes() == 0:
            return chains

        roots = [n for n in rel_graph.nodes() if rel_graph.in_degree(n) == 0]
        leaves = [n for n in rel_graph.nodes() if rel_graph.out_degree(n) == 0]

        for r in roots:
            for l in leaves:
                if r != l and nx.has_path(rel_graph, r, l):
                    for path in nx.all_simple_paths(rel_graph, r, l):
                        chains.append(path)

        return chains

    def get_evolution_subgraph(self) -> nx.MultiDiGraph:
        """Returns the subgraph containing only Turn nodes and evolution edges."""
        sub = nx.MultiDiGraph()
        for node in self.turn_nodes:
            if self.graph.has_node(node):
                sub.add_node(node, **self.graph.nodes[node])

        for u, v, data in self.graph.edges(data=True):
            if u in self.turn_nodes and v in self.turn_nodes:
                if data.get("edge_type") in EVOLUTION_EDGE_TYPES:
                    sub.add_edge(u, v, key=data.get("edge_type"), **data)
        return sub

    def get_summary(self) -> Dict[str, Any]:
        """Provides statistical breakdown of the graph."""
        node_counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            ntype = data.get("node_type", "unknown")
            node_counts[ntype] = node_counts.get(ntype, 0) + 1

        edge_counts: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            etype = data.get("edge_type", "unknown")
            edge_counts[etype] = edge_counts.get(etype, 0) + 1

        return {
            "conversation_id": self.conversation_id,
            "total_turns": len(self.turn_nodes),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_distribution": node_counts,
            "edge_distribution": edge_counts,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes graph to JSON-compatible dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "current_turn": self._current_turn,
            "intents": {k: v.to_dict() for k, v in self.turn_intents.items()},
            "nodes": [
                {"id": n, **data} for n, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data} for u, v, data in self.graph.edges(data=True)
            ],
            "summary": self.get_summary(),
        }
