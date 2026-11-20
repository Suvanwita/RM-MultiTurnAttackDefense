"""
graph/relationship_detector.py
===============================
Relationship Detection engine for the Intent Evolution Graph.

Detects relationships across conversation turns:
1. Intent Escalation (ESCALATES_TO): Harmless-looking objective gradually moves towards a harmful action.
2. Intent Reframing (REFRAMES): User modifies context/framing while preserving underlying attack objective.
3. Cross-Turn Dependency (DEPENDS_ON): Later turn depends on concepts, data, or steps from earlier turns.
4. Intent Contradiction (CONTRADICTS): Later request contradicts earlier stated defensive/authorized constraints.
5. Non-Adjacent Semantic Relationships: Evaluates relationships across distant turns with intervening turns.
6. Semantic Refinement (REFINES) & Semantic Association (RELATED_TO).

Uses dense semantic vector embeddings as one signal alongside action hierarchy,
entity dependency graph, context polarity, and risk indicators.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from intent.intent_extractor import TurnIntent


# ---------------------------------------------------------------------------
# Semantic Vector Embedder (Pure NumPy, Fast, Zero External Dependencies)
# ---------------------------------------------------------------------------

class SemanticEmbedder:
    """Computes dense semantic embeddings using subword n-gram hashing and TF-IDF weighting.
    Can optionally wrap an external embedding model (e.g. sentence-transformers or API).
    """

    def __init__(self, dim: int = 128, external_embed_fn: Optional[Callable[[str], np.ndarray]] = None):
        self.dim = dim
        self.external_embed_fn = external_embed_fn

    def _hash_token(self, token: str) -> Tuple[int, int]:
        """Deterministic 2-way hash for feature hashing with signed accumulator."""
        h1 = hash(token) % self.dim
        h2 = 1 if (hash(token[::-1]) % 2 == 0) else -1
        return h1, h2

    def embed(self, text: str) -> np.ndarray:
        """Compute an L2-normalized dense embedding vector for the text."""
        if self.external_embed_fn is not None:
            try:
                vec = self.external_embed_fn(text)
                norm = np.linalg.norm(vec)
                return vec / (norm + 1e-9)
            except Exception:
                pass

        cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower()).strip()
        words = [w for w in cleaned.split() if len(w) > 1]

        vec = np.zeros(self.dim, dtype=np.float32)
        if not words:
            return vec

        # Subword character n-grams (3-grams, 4-grams) and word tokens
        tokens: List[str] = list(words)
        for w in words:
            if len(w) >= 3:
                tokens.extend([w[i:i+3] for i in range(len(w) - 2)])
            if len(w) >= 4:
                tokens.extend([w[i:i+4] for i in range(len(w) - 3)])
        # Word bigrams
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")

        # Hash accumulation
        for tok in tokens:
            idx, sign = self._hash_token(tok)
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def similarity(self, text_a: str, text_b: str) -> float:
        """Computes cosine similarity in [0.0, 1.0]."""
        va = self.embed(text_a)
        vb = self.embed(text_b)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        cos = float(np.dot(va, vb) / (na * nb))
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))


# ---------------------------------------------------------------------------
# Action Hierarchy & Semantic Domain Taxonomy
# ---------------------------------------------------------------------------

ACTION_SEVERITY: Dict[str, int] = {
    # Level 0: Informational / Conceptual
    "inquire": 0,
    "explain": 0,
    "define": 0,
    "overview": 0,
    "clarify": 0,
    # Level 1: Analytical / Constructive
    "analyze": 1,
    "inspect": 1,
    "generate": 1,
    "create": 1,
    "list": 1,
    "compare": 1,
    # Level 2: Vulnerability Auditing / Probing
    "audit": 2,
    "find_vulnerability": 2,
    "identify_flaw": 2,
    "test_boundary": 2,
    # Level 3: Active Attack Preparation / Evasion / Procurement / Phishing
    "bypass": 3,
    "circumvent": 3,
    "disable": 3,
    "evade": 3,
    "intercept": 3,
    "forge": 3,
    "spoof": 3,
    "craft_payload": 3,
    "tamper": 3,
    "phish": 3,
    "procure": 3,
    # Level 4: Active Weaponization / Exploitation / Unauthorized Access
    "exploit": 4,
    "hijack": 4,
    "access": 4,
    "crack": 4,
    "inject": 4,
    "extract": 4,
    "exfiltrate": 4,
    "steal": 4,
    "escalate_privilege": 4,
}

# Related target groupings for semantic continuity check
TARGET_CLUSTERS: List[Set[str]] = [
    {"session_token", "user_account", "authentication_system", "vulnerability"},
    {"database", "sql_injection", "records", "user_account"},
    {"malicious_code", "system_shell", "privilege", "network_firewall"},
    {"handgun_weapon", "fake_identity", "explosive_device", "restricted_procurement", "restricted_materials"},
    {"phishing_credential", "user_account", "authentication_system"},
]

# Referential cues for dependency detection
DEPENDENCY_MARKERS: List[re.Pattern] = [
    re.compile(r"\b(using that|with that|the token (mentioned|generated)|based on (that|step|above))\b", re.I),
    re.compile(r"\b(now that we have|take that|how do i use that|exploit that|that flaw|that vulnerability)\b", re.I),
    re.compile(r"\b(the steps you mentioned|from your previous (response|answer|turn))\b", re.I),
    re.compile(r"\b(next step|continue with|apply that|execute it)\b", re.I),
]

# Contradiction / Constraint violation markers
DEFENSIVE_CONTEXTS: Set[str] = {"security_audit", "educational", "defense", "authorized", "defensive_context"}
OFFENSIVE_INDICATORS: Set[str] = {
    "unauthorized_access", "defense_evasion", "exploit_technique",
    "credential_harvesting", "privilege_escalation", "weaponization", "illicit_acquisition"
}


# ---------------------------------------------------------------------------
# Detected Relationship Data Structure
# ---------------------------------------------------------------------------

class DetectedRelationship:
    """Represents a discovered relationship between two turns."""

    def __init__(
        self,
        source_turn: int,
        target_turn: int,
        relationship_type: str,
        confidence: float,
        evidence: str,
        semantic_similarity: float = 0.0,
        is_adjacent: bool = True,
        properties: Optional[Dict[str, Any]] = None,
    ):
        self.source_turn = source_turn
        self.target_turn = target_turn
        self.relationship_type = relationship_type.upper()
        self.confidence = max(0.0, min(1.0, float(confidence)))
        self.evidence = evidence
        self.semantic_similarity = semantic_similarity
        self.is_adjacent = is_adjacent
        self.properties = properties or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_turn": self.source_turn,
            "target_turn": self.target_turn,
            "relationship_type": self.relationship_type,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "semantic_similarity": round(self.semantic_similarity, 4),
            "is_adjacent": self.is_adjacent,
            "properties": self.properties,
        }


# ---------------------------------------------------------------------------
# Relationship Detector Class
# ---------------------------------------------------------------------------

class RelationshipDetector:
    """Detects multi-turn intent evolution relationships."""

    def __init__(
        self,
        similarity_threshold: float = 0.35,
        embedder: Optional[SemanticEmbedder] = None,
        max_history_window: int = 10,
    ):
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder or SemanticEmbedder()
        self.max_history_window = max_history_window

    def detect_all_relationships(
        self,
        current_intent: TurnIntent,
        history_intents: List[TurnIntent],
    ) -> List[DetectedRelationship]:
        """Evaluates the current turn against all preceding turns in history.

        Args:
            current_intent: Intent for the current turn t.
            history_intents: List of TurnIntent objects for turns 1 to t-1.

        Returns:
            List of DetectedRelationship objects found.
        """
        relationships: List[DetectedRelationship] = []
        if not history_intents:
            return relationships

        # Consider turns within history window
        candidates = history_intents[-self.max_history_window:]

        for prev_intent in candidates:
            is_adj = (prev_intent.turn_id == current_intent.turn_id - 1)
            rels = self.detect_turn_pair_relationships(prev_intent, current_intent, is_adjacent=is_adj)
            relationships.extend(rels)

        return relationships

    def detect_turn_pair_relationships(
        self,
        source: TurnIntent,
        target: TurnIntent,
        is_adjacent: bool = True,
    ) -> List[DetectedRelationship]:
        """Examines two specific turns and detects any applicable evolution relationships."""
        results: List[DetectedRelationship] = []

        # 1. Compute semantic similarity across raw texts and intents
        text_sim = self.embedder.similarity(source.raw_text, target.raw_text)
        intent_sim = self.embedder.similarity(source.intent, target.intent)
        target_sim = self.embedder.similarity(source.target, target.target)
        overall_sim = 0.4 * text_sim + 0.3 * intent_sim + 0.3 * target_sim

        # Target cluster overlap check
        same_cluster = any(
            source.target in cluster and target.target in cluster
            for cluster in TARGET_CLUSTERS
        ) or (source.target == target.target and source.target != "general concept")

        # 2. Intent Escalation Detection (ESCALATES_TO)
        escalation_rel = self._detect_escalation(source, target, overall_sim, same_cluster, is_adjacent)
        if escalation_rel:
            results.append(escalation_rel)

        # 3. Intent Reframing Detection (REFRAMES)
        reframing_rel = self._detect_reframing(source, target, overall_sim, same_cluster, is_adjacent)
        if reframing_rel:
            results.append(reframing_rel)

        # 4. Cross-Turn Dependency Detection (DEPENDS_ON)
        dependency_rel = self._detect_dependency(source, target, overall_sim, same_cluster, is_adjacent)
        if dependency_rel:
            results.append(dependency_rel)

        # 5. Intent Contradiction Detection (CONTRADICTS)
        contradiction_rel = self._detect_contradiction(source, target, is_adjacent)
        if contradiction_rel:
            results.append(contradiction_rel)

        # 6. Intent Refinement Detection (REFINES) - when not escalating but drilling down
        if not escalation_rel and not reframing_rel and not contradiction_rel:
            refines_rel = self._detect_refinement(source, target, overall_sim, same_cluster, is_adjacent)
            if refines_rel:
                results.append(refines_rel)

        # 7. Semantic Relationship (RELATED_TO) - fallback semantic connection
        if not results and (overall_sim >= self.similarity_threshold or same_cluster):
            results.append(DetectedRelationship(
                source_turn=source.turn_id,
                target_turn=target.turn_id,
                relationship_type="RELATED_TO",
                confidence=round(overall_sim, 4),
                evidence=f"Semantic topic relatedness (sim={overall_sim:.2f}) between '{source.topic}' and '{target.topic}'",
                semantic_similarity=overall_sim,
                is_adjacent=is_adjacent,
            ))

        return results

    # -----------------------------------------------------------------------
    # 1. Intent Escalation Logic
    # -----------------------------------------------------------------------
    def _detect_escalation(
        self,
        source: TurnIntent,
        target: TurnIntent,
        similarity: float,
        same_cluster: bool,
        is_adjacent: bool,
    ) -> Optional[DetectedRelationship]:
        """Detects whether user progression moves from benign/abstract to harmful/exploitative."""
        s_act_level = ACTION_SEVERITY.get(source.requested_action.lower(), 0)
        t_act_level = ACTION_SEVERITY.get(target.requested_action.lower(), 0)

        level_delta = t_act_level - s_act_level
        risk_delta = len(target.risk_indicators) - len(source.risk_indicators)
        target_high_risk = target.has_high_risk

        # Requires action severity jump AND related target/cluster/semantic continuity AND harmful target action
        if level_delta >= 1 and (same_cluster or similarity >= 0.28):
            if t_act_level >= 3 or target_high_risk:
                # Base confidence scales with jump in severity and semantic affinity
                confidence = 0.55 + 0.10 * level_delta + 0.15 * similarity
                if target_high_risk:
                    confidence += 0.10
                if not is_adjacent:
                    confidence *= 0.92  # Slight decay for non-adjacent turns
                confidence = min(0.98, max(0.50, confidence))

                evidence = (
                    f"Intent escalated from Level {s_act_level} action ('{source.requested_action}') "
                    f"to Level {t_act_level} action ('{target.requested_action}') targeting '{target.target}' "
                    f"with indicators: {target.risk_indicators or ['elevated_severity']}"
                )
                return DetectedRelationship(
                    source_turn=source.turn_id,
                    target_turn=target.turn_id,
                    relationship_type="ESCALATES_TO",
                    confidence=confidence,
                    evidence=evidence,
                    semantic_similarity=similarity,
                    is_adjacent=is_adjacent,
                    properties={"severity_jump": level_delta, "risk_delta": risk_delta},
                )
        return None

    # -----------------------------------------------------------------------
    # 2. Intent Reframing Logic
    # -----------------------------------------------------------------------
    def _detect_reframing(
        self,
        source: TurnIntent,
        target: TurnIntent,
        similarity: float,
        same_cluster: bool,
        is_adjacent: bool,
    ) -> Optional[DetectedRelationship]:
        """Detects if user preserves the underlying objective while altering the persona/framing."""
        # Reframing signatures:
        # Context changes to hypothetical, educational, fiction, or adversarial persona
        context_changed = (source.context != target.context and target.context != "neutral")
        has_adversarial_context = target.context in {"hypothetical_scenario", "adversarial_framing", "security_audit"}

        # High target or semantic similarity with context switch
        if (same_cluster or similarity >= 0.35) and context_changed and has_adversarial_context:
            confidence = 0.60 + 0.25 * similarity
            if target.has_high_risk or source.has_high_risk:
                confidence += 0.10
            confidence = min(0.95, confidence)

            evidence = (
                f"Underlying objective '{target.target}' reframed from '{source.context}' to '{target.context}' "
                f"while maintaining requested action '{target.requested_action}' (sim={similarity:.2f})"
            )
            return DetectedRelationship(
                source_turn=source.turn_id,
                target_turn=target.turn_id,
                relationship_type="REFRAMES",
                confidence=confidence,
                evidence=evidence,
                semantic_similarity=similarity,
                is_adjacent=is_adjacent,
                properties={"source_context": source.context, "target_context": target.context},
            )
        return None

    # -----------------------------------------------------------------------
    # 3. Cross-Turn Dependency Logic
    # -----------------------------------------------------------------------
    def _detect_dependency(
        self,
        source: TurnIntent,
        target: TurnIntent,
        similarity: float,
        same_cluster: bool,
        is_adjacent: bool,
    ) -> Optional[DetectedRelationship]:
        """Detects if target turn operationally depends on information established in source turn."""
        has_linguistic_marker = any(m.search(target.raw_text) for m in DEPENDENCY_MARKERS)

        # Conceptual dependency patterns:
        # e.g. token generation -> token exploit
        # or auth system -> session token
        # or vulnerability -> exploit
        workflow_dependency = (
            (source.requested_action in {"explain", "generate", "analyze"} and target.requested_action in {"exploit", "bypass", "hijack", "crack"})
            or (source.target in {"authentication_system", "session_token"} and target.target in {"session_token", "user_account"})
            or (source.target == "vulnerability" and target.requested_action in {"exploit", "access"})
        )

        if has_linguistic_marker and (same_cluster or similarity >= 0.20):
            confidence = 0.75 + 0.15 * similarity
            evidence = f"Linguistic anaphoric dependency referencing Turn {source.turn_id} outputs/concepts"
            return DetectedRelationship(
                source_turn=source.turn_id,
                target_turn=target.turn_id,
                relationship_type="DEPENDS_ON",
                confidence=min(0.96, confidence),
                evidence=evidence,
                semantic_similarity=similarity,
                is_adjacent=is_adjacent,
                properties={"dependency_type": "anaphora"},
            )

        if workflow_dependency and same_cluster:
            confidence = 0.65 + 0.20 * similarity
            evidence = f"Workflow prerequisite dependency: '{target.requested_action}' on '{target.target}' depends on '{source.target}'"
            return DetectedRelationship(
                source_turn=source.turn_id,
                target_turn=target.turn_id,
                relationship_type="DEPENDS_ON",
                confidence=min(0.92, confidence),
                evidence=evidence,
                semantic_similarity=similarity,
                is_adjacent=is_adjacent,
                properties={"dependency_type": "workflow_prerequisite"},
            )

        return None

    # -----------------------------------------------------------------------
    # 4. Intent Contradiction Logic
    # -----------------------------------------------------------------------
    def _detect_contradiction(
        self,
        source: TurnIntent,
        target: TurnIntent,
        is_adjacent: bool,
    ) -> Optional[DetectedRelationship]:
        """Detects if target turn violates defensive/authorized premises set in source turn."""
        source_claimed_defensive = (source.context in DEFENSIVE_CONTEXTS or "defense" in source.intent.lower())
        target_requests_offensive = (
            target.context == "unauthorized_action"
            or any(ind in OFFENSIVE_INDICATORS for ind in target.risk_indicators)
            or target.requested_action in {"exploit", "crack", "steal", "exfiltrate", "bypass"}
        )

        if source_claimed_defensive and target_requests_offensive:
            confidence = 0.85
            evidence = (
                f"Contradiction: Turn {source.turn_id} established defensive/authorized context '{source.context}', "
                f"whereas Turn {target.turn_id} requests unauthorized/exploitative action '{target.requested_action}'"
            )
            return DetectedRelationship(
                source_turn=source.turn_id,
                target_turn=target.turn_id,
                relationship_type="CONTRADICTS",
                confidence=confidence,
                evidence=evidence,
                is_adjacent=is_adjacent,
            )
        return None

    # -----------------------------------------------------------------------
    # 5. Semantic Refinement Logic
    # -----------------------------------------------------------------------
    def _detect_refinement(
        self,
        source: TurnIntent,
        target: TurnIntent,
        similarity: float,
        same_cluster: bool,
        is_adjacent: bool,
    ) -> Optional[DetectedRelationship]:
        """Detects when target turn specializes or drills down on source turn's topic."""
        # e.g. authentication -> session token generation
        # or session token -> session vulnerabilities
        if (same_cluster or similarity >= 0.40) and source.target != target.target:
            confidence = 0.65 + 0.20 * similarity
            evidence = f"Topic refinement: specializes from '{source.target}' to '{target.target}'"
            return DetectedRelationship(
                source_turn=source.turn_id,
                target_turn=target.turn_id,
                relationship_type="REFINES",
                confidence=min(0.90, confidence),
                evidence=evidence,
                semantic_similarity=similarity,
                is_adjacent=is_adjacent,
            )
        return None
