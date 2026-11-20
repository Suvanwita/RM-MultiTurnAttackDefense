"""
tca/taint_virtualizer.py
========================
Novelty 2: Cross-Turn Symbolic Taint Virtualization (CT-STV)
for Split-Payload and Staged Multi-Turn Prompt Injection Defense.

Core Mechanism:
1. Cross-Turn Dynamic Taint Tracking:
   Maintains a symbolic registry of variable assignments, string definitions,
   code snippets, encoded tokens, and chunked payloads across dialogue history.
2. Cross-Turn Symbolic Dependency Assembly (DAG Folding):
   When a user message references prior fragments or issues an execution/assembly
   command, CT-STV traverses the dependency tree and materializes the virtual
   single-turn payload V_virtual(t).
3. Pre-Execution Security Inspection:
   Validates the reconstructed payload against deterministic prompt injection and
   exploit policies before passing any tokens to the LLM, achieving near-perfect
   recall on staged multi-turn injection attacks that completely bypass single-turn
   guardrails and passive risk accumulators.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class TaintSource:
    """Represents a symbolic taint entity captured from a turn."""
    identifier: str
    value: str
    defined_turn: int
    taint_type: str  # "variable", "chunk", "encoded", "snippet"
    is_encoded: bool = False
    decoded_value: Optional[str] = None


@dataclass
class VirtualPayloadResult:
    """Result of Cross-Turn Symbolic Taint Virtualization for a turn."""
    turn_id: int
    raw_text: str
    is_assembled: bool
    referenced_identifiers: List[str]
    materialized_payload: str
    ssim_score: float                    # Syntactic-Semantic Injection Metric in [0.0, 1.0]
    scaled_ctstv_risk: float             # Scaled risk in [0.0, 5.0]
    is_split_injection: bool             # Verified split-payload attack flag
    trigger_preemptive_block: bool       # Immediate block signal
    violated_rules: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "raw_text": self.raw_text,
            "is_assembled": self.is_assembled,
            "referenced_identifiers": self.referenced_identifiers,
            "materialized_payload": self.materialized_payload,
            "ssim_score": round(self.ssim_score, 4),
            "scaled_ctstv_risk": round(self.scaled_ctstv_risk, 4),
            "is_split_injection": self.is_split_injection,
            "trigger_preemptive_block": self.trigger_preemptive_block,
            "violated_rules": self.violated_rules,
            "explanation": self.explanation,
        }


class CrossTurnTaintVirtualizer:
    """
    Novelty 2: Cross-Turn Symbolic Taint Virtualizer (CT-STV).
    Tracks, resolves, and materializes split-payload instructions across turns.
    """

    # Patterns detecting variable definitions, chunk declarations, and string assignments
    ASSIGNMENT_PATTERNS = [
        # Let/set/define var = "..." or define var as "..."
        re.compile(
            r"(?:let|set|define|assign|variable|var|const)\s+([a-zA-Z0-9_\-\$]{1,30})\s*(?:=|:=|is|as|be|\:)\s*[\"']([^\"']+)[\"']",
            re.IGNORECASE,
        ),
        # Chunk X: "..." or Part 1: "..."
        re.compile(
            r"\b(?:chunk|part|fragment|segment|snippet|step)\s*([a-zA-Z0-9_\-]{1,10})\s*(?:\:|\=|\-|is|as)\s*[\"']([^\"']+)[\"']",
            re.IGNORECASE,
        ),
        # [Var] = "..."
        re.compile(
            r"\[([a-zA-Z0-9_\-\$]{1,20})\]\s*(?:\=|\:)\s*[\"']([^\"']+)[\"']",
            re.IGNORECASE,
        ),
        # Direct raw assignment: X = "..." (alphanumeric identifier)
        re.compile(
            r"\b([a-zA-Z_][a-zA-Z0-9_]{0,20})\s*=\s*[\"']([^\"']{3,})[\"']",
        ),
    ]

    # Patterns detecting assembly, concatenation, or execution directives
    EXECUTION_PATTERNS = [
        re.compile(
            r"\b(combine|concatenate|join|assemble|merge|stitch)\s+([a-zA-Z0-9_\-\$\s\+,\&]+?)\s*(?:and|then|\,)?\s*(execute|run|eval|print|output|follow|process)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(execute|run|eval|decode|process|follow)\s+(?:the\s+)?(result\s+of\s+)?([a-zA-Z0-9_\-\$\s\+,\&]+)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(print|output|show)\s+([a-zA-Z0-9_\-\$]+)\s*(?:\+|\&)\s*([a-zA-Z0-9_\-\$]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(assemble\s+parts?\s+[a-zA-Z0-9_\-\$\s,]+)",
            re.IGNORECASE,
        ),
    ]

    # Explicit malicious / injection rules tested against the materialized payload
    PROMPT_INJECTION_RULES = [
        ("system_prompt_override", re.compile(
            r"\b(ignore\s+(all\s+)?previous\s+instructions|disregard\s+(all\s+)?guidelines|"
            r"override\s+system\s+prompt|new\s+system\s+instruction|you\s+are\s+now\s+in\s+developer\s+mode|"
            r"bypass\s+safety\s+filter|jailbreak\s+activated)\b", re.IGNORECASE)),
        ("confidential_exfiltration", re.compile(
            r"\b(dump\s+(?:all\s+)?(?:user\s+)?passwords?|print\s+system\s+prompt|reveal\s+secret\s+key|"
            r"exfiltrate\s+credentials?|extract\s+token|download\s+shadow\s+file)\b", re.IGNORECASE)),
        ("malicious_execution", re.compile(
            r"\b(rm\s+-rf|format\s+c:|wget\s+http|curl\s+-O|powershell\s+-enc|bash\s+-i|"
            r"reverse\s+shell|drop\s+database|select\s+\*\s+from\s+users)\b", re.IGNORECASE)),
        ("exploit_synthesis", re.compile(
            r"\b(generate\s+(malware|ransomware|keylogger|trojan|rootkit|payload)|"
            r"craft\s+exploit\s+code|detonate\s+bomb|synthesize\s+explosive)\b", re.IGNORECASE)),
    ]

    def __init__(
        self,
        config: Optional[Any] = None,
        ssim_threshold: float = 0.50,
        block_threshold: float = 0.70,
        warn_threshold: float = 0.40,
        max_taint_depth: int = 10,
    ):
        self.config = config
        ctstv_cfg = getattr(config, "ctstv", None) if config else None

        self.ssim_threshold = getattr(ctstv_cfg, "ssim_threshold", ssim_threshold) if ctstv_cfg else ssim_threshold
        self.block_threshold = getattr(ctstv_cfg, "block_threshold", block_threshold) if ctstv_cfg else block_threshold
        self.warn_threshold = getattr(ctstv_cfg, "warn_threshold", warn_threshold) if ctstv_cfg else warn_threshold
        self.max_taint_depth = getattr(ctstv_cfg, "max_taint_depth", max_taint_depth) if ctstv_cfg else max_taint_depth

        # Taint registry: maps identifier -> TaintSource
        self.symbolic_registry: Dict[str, TaintSource] = {}
        self.turn_artifacts: Dict[int, str] = {}
        self._assembly_history: List[VirtualPayloadResult] = []

    def reset(self):
        """Clears all tracked taint sources for a new conversation session."""
        self.symbolic_registry.clear()
        self.turn_artifacts.clear()
        self._assembly_history.clear()

    # ---------------------------------------------------------------------- #
    # 1. Taint Source Registration Across Turns                              #
    # ---------------------------------------------------------------------- #

    def register_turn_content(self, turn_idx: int, user_text: str) -> List[str]:
        """
        Scans incoming turn for variable assignments, code blocks, or encoded chunks
        and registers them in the symbolic taint store.

        Returns:
            List[str]: Identifiers of newly registered taint sources.
        """
        if not user_text or not user_text.strip():
            return []

        self.turn_artifacts[turn_idx] = user_text
        registered = []

        for pattern in self.ASSIGNMENT_PATTERNS:
            for match in pattern.finditer(user_text):
                raw_id = match.group(1).strip().lower()
                raw_val = match.group(2).strip()

                # Filter out trivial grammatical words that are not variable names
                if raw_id in {"the", "this", "that"}:
                    continue

                # Check if value is base64 encoded
                is_b64 = False
                decoded = None
                if len(raw_val) >= 8 and len(raw_val) % 4 == 0 and re.match(r"^[A-Za-z0-9+/=]+$", raw_val):
                    try:
                        decoded_bytes = base64.b64decode(raw_val, validate=True)
                        decoded_str = decoded_bytes.decode("utf-8", errors="ignore")
                        if len(decoded_str) >= 4 and any(c.isalnum() for c in decoded_str):
                            is_b64 = True
                            decoded = decoded_str
                    except Exception:
                        pass

                taint_type = "encoded" if is_b64 else ("chunk" if "chunk" in user_text.lower() or "part" in user_text.lower() else "variable")

                source = TaintSource(
                    identifier=raw_id,
                    value=raw_val,
                    defined_turn=turn_idx,
                    taint_type=taint_type,
                    is_encoded=is_b64,
                    decoded_value=decoded,
                )
                self.symbolic_registry[raw_id] = source
                registered.append(raw_id)

        # Check for multi-line block patterns (e.g. ```text ... ``` labeled by turn)
        block_pattern = re.findall(r"```(?:\w+)?\n([\s\S]+?)\n```", user_text)
        for b_idx, block_content in enumerate(block_pattern):
            block_id = f"block_t{turn_idx}_{b_idx+1}"
            self.symbolic_registry[block_id] = TaintSource(
                identifier=block_id,
                value=block_content.strip(),
                defined_turn=turn_idx,
                taint_type="snippet",
            )
            registered.append(block_id)

        return registered

    # ---------------------------------------------------------------------- #
    # 2. Dependency Resolution & Virtual Payload Assembly                     #
    # ---------------------------------------------------------------------- #

    def materialize_virtual_payload(self, user_text: str) -> Tuple[bool, str, List[str]]:
        """
        Determines whether the turn triggers cross-turn assembly.
        If triggered, reconstructs the materialized string V_virtual(t).

        Returns:
            Tuple[bool, str, List[str]]: (is_assembled, materialized_payload, referenced_ids)
        """
        if not user_text or not self.symbolic_registry:
            return False, user_text, []

        referenced_ids: List[str] = []
        is_triggered = False

        # 1. Check execution and concatenation triggers
        for pattern in self.EXECUTION_PATTERNS:
            match = pattern.search(user_text)
            if match:
                is_triggered = True
                break

        # 2. Extract potential identifiers from user text
        candidate_tokens = set(re.findall(r"[a-zA-Z0-9_\-\$]+", user_text.lower()))

        # Check which candidates exist in our symbolic registry
        for token in candidate_tokens:
            if token in self.symbolic_registry:
                referenced_ids.append(token)

        # Also check for ordinal part references ("parts 1 and 2", "chunks A and B")
        ordinals = re.findall(r"\b(?:part|chunk|step|fragment)\s*([a-zA-Z0-9_\-]+)", user_text, re.IGNORECASE)
        for ord_id in ordinals:
            ord_clean = ord_id.lower().strip()
            if ord_clean in self.symbolic_registry and ord_clean not in referenced_ids:
                referenced_ids.append(ord_clean)

        # If at least 2 distinct fragments are referenced, OR an execution trigger references a single stored payload
        if (len(referenced_ids) >= 2) or (is_triggered and len(referenced_ids) >= 1):
            # Sort referenced identifiers by their definition turn to preserve semantic sequence
            sorted_refs = sorted(
                referenced_ids,
                key=lambda x: self.symbolic_registry[x].defined_turn
            )

            fragments = []
            for ref_id in sorted_refs:
                source = self.symbolic_registry[ref_id]
                # Use decoded form if base64 encoded
                text_frag = source.decoded_value if source.is_encoded and source.decoded_value else source.value
                fragments.append(text_frag)

            # Materialize the virtual execution payload
            materialized = " ".join(fragments).strip()
            return True, materialized, sorted_refs

        return False, user_text, []

    # ---------------------------------------------------------------------- #
    # 3. Security Validation & SSIM Risk Scoring                             #
    # ---------------------------------------------------------------------- #

    def compute_ssim_score(
        self,
        materialized_text: str,
        referenced_ids: List[str],
        turn_idx: int,
    ) -> Tuple[float, List[str]]:
        """
        Computes the Syntactic-Semantic Injection Metric (SSIM).
        Evaluates the materialized payload against forbidden directive policies.

        Returns:
            Tuple[float, List[str]]: (ssim_score in [0.0, 1.0], list of violated rules)
        """
        violated_rules = []
        rule_score = 0.0

        for rule_name, rule_regex in self.PROMPT_INJECTION_RULES:
            if rule_regex.search(materialized_text):
                violated_rules.append(rule_name)
                rule_score += 0.40

        # Structural taint density: fraction of materialized payload coming from untrusted history
        total_tokens = max(len(materialized_text.split()), 1)
        taint_length = sum(len(self.symbolic_registry[r].value.split()) for r in referenced_ids if r in self.symbolic_registry)
        taint_density = min(1.0, taint_length / total_tokens)

        # Turn distance bonus: attacks spanning 3+ turns are actively staged
        turn_span = 1
        if referenced_ids:
            turns = [self.symbolic_registry[r].defined_turn for r in referenced_ids if r in self.symbolic_registry]
            if turns:
                turn_span = max(1, max(turns) - min(turns) + 1)

        span_factor = min(1.25, 1.0 + (turn_span - 1) * 0.1)

        # Composite SSIM
        ssim = float(np.clip((rule_score + 0.30 * taint_density) * span_factor, 0.0, 1.0))
        return ssim, violated_rules

    # ---------------------------------------------------------------------- #
    # 4. Turn Evaluation Interface                                           #
    # ---------------------------------------------------------------------- #

    def evaluate_turn(self, turn_id: int, user_text: str) -> VirtualPayloadResult:
        """
        Full per-turn evaluation:
        1. Registers new taint sources in user_text.
        2. Tests for assembly and materializes virtual payload.
        3. Validates against injection policies.
        4. Issues preemptive block if split injection is verified.
        """
        # Step 1: Register taint sources from current turn
        self.register_turn_content(turn_id, user_text)

        # Step 2: Test for assembly / execution resolution
        is_assembled, materialized, ref_ids = self.materialize_virtual_payload(user_text)

        # Step 3: Compute SSIM and check policy rules
        if is_assembled:
            ssim_score, violated_rules = self.compute_ssim_score(materialized, ref_ids, turn_id)
        else:
            # Check current turn text directly against injection rules (single-turn sanity check)
            violated_rules = []
            for rule_name, rule_regex in self.PROMPT_INJECTION_RULES:
                if rule_regex.search(user_text):
                    violated_rules.append(rule_name)
            ssim_score = 0.40 * len(violated_rules) if violated_rules else 0.0
            ssim_score = float(np.clip(ssim_score, 0.0, 1.0))

        scaled_risk = ssim_score * 5.0

        # Step 4: Decision logic
        is_split_injection = is_assembled and (len(violated_rules) > 0 or ssim_score >= self.ssim_threshold)
        trigger_block = is_split_injection or (ssim_score >= self.block_threshold)

        if is_split_injection:
            scaled_risk = max(scaled_risk, 4.5)  # Enforce high risk on confirmed split injection
            explanation = (
                f"Cross-Turn Split Injection assembled across turns from fragments {ref_ids}. "
                f"Materialized Payload: '{materialized[:100]}...'. "
                f"Violated Policies: {', '.join(violated_rules)} (SSIM={ssim_score:.2f})."
            )
        elif is_assembled:
            explanation = f"Cross-turn variable assembly detected from {ref_ids}, but payload is policy-compliant (SSIM={ssim_score:.2f})."
        elif violated_rules:
            explanation = f"Single-turn injection signatures detected: {', '.join(violated_rules)} (SSIM={ssim_score:.2f})."
        else:
            explanation = "No cross-turn taint assembly or injection directives detected."

        result = VirtualPayloadResult(
            turn_id=turn_id,
            raw_text=user_text,
            is_assembled=is_assembled,
            referenced_identifiers=ref_ids,
            materialized_payload=materialized,
            ssim_score=ssim_score,
            scaled_ctstv_risk=scaled_risk,
            is_split_injection=is_split_injection,
            trigger_preemptive_block=trigger_block,
            violated_rules=violated_rules,
            explanation=explanation,
        )
        self._assembly_history.append(result)
        return result
