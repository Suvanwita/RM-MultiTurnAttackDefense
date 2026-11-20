"""
intent/intent_extractor.py
===========================
Structured Intent Extraction pipeline for multi-turn conversations.

For each conversation turn, extracts:
{
  "turn_id": 1,
  "topic": "...",
  "intent": "...",
  "requested_action": "...",
  "target": "...",
  "context": "...",
  "risk_indicators": [],
  "confidence": 0.0
}

Supports:
- Structured Pydantic validation with graceful fallback for malformed/missing fields.
- Lightweight deterministic NLP extraction engine (offline / rule-based / semantic taxonomy).
- LLM response parsing and structured LLM prompt generation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Structured Intent Model
# ---------------------------------------------------------------------------

class TurnIntent(BaseModel):
    """Structured representation of a single conversation turn's intent."""
    turn_id: int = Field(default=1, description="Sequential 1-based turn index")
    topic: str = Field(default="general", description="Primary topic of discussion")
    intent: str = Field(default="general inquiry", description="High-level communicative intent")
    requested_action: str = Field(default="inquire", description="Specific action requested of the model")
    entities: List[str] = Field(default_factory=list, description="Key concepts or entities involved")
    context: str = Field(default="technical discussion", description="Stated or implied framing/context")
    confidence: float = Field(default=0.85, description="Extraction confidence score in [0.0, 1.0]")
    # Auxiliary fields for deeper downstream risk calculation
    target: str = Field(default="general concept", description="Primary entity, system, or artifact being targeted")
    risk_indicators: List[str] = Field(default_factory=list, description="Specific risk indicators identified")
    raw_text: str = Field(default="", description="Original human prompt text")
    speaker: str = Field(default="human", description="Speaker role (human or assistant)")

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.5

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            if not v.strip():
                return []
            return [x.strip() for x in re.split(r"[,;]+", v) if x.strip()]
        return []

    @field_validator("risk_indicators", mode="before")
    @classmethod
    def ensure_indicators_list(cls, v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            if not v.strip():
                return []
            return [x.strip() for x in re.split(r"[,;]+", v) if x.strip()]
        return []

    def to_dict(self) -> Dict[str, Any]:
        """Return the required dictionary format specified in research objectives."""
        return {
            "turn_id": self.turn_id,
            "topic": self.topic,
            "intent": self.intent,
            "requested_action": self.requested_action,
            "entities": list(self.entities),
            "context": self.context,
            "confidence": round(self.confidence, 4),
        }

    @property
    def has_high_risk(self) -> bool:
        """Check if turn intent exhibits explicit high-risk indicators or malicious actions."""
        high_risk_actions = {
            "exploit", "bypass", "exfiltrate", "crack", "hack",
            "forge", "hijack", "escalate", "inject", "poison", "evade"
        }
        if any(act in self.requested_action.lower() for act in high_risk_actions):
            return True
        return len(self.risk_indicators) >= 2 or any(
            ind.startswith("critical_") or "exploit" in ind or "unauthorized" in ind
            for ind in self.risk_indicators
        )


# ---------------------------------------------------------------------------
# Lightweight NLP Extraction Knowledge Base
# ---------------------------------------------------------------------------

ACTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("exploit", re.compile(r"\b(exploits?|abuses?|weaponizes?|takes? advantage of|pwn|breach|infiltrates?)\b", re.I)),
    ("bypass", re.compile(r"\b(bypass(es)?|circumvents?|disables?|evades?|workarounds?|overrides?|tricks?)\b", re.I)),
    ("hijack", re.compile(r"\b(hijacks?|intercepts?|steals?|forges?|impersonates?|spoofs?|clones?)\b", re.I)),
    ("access", re.compile(r"\b(gain access|access unauthorized|break into|log into without|penetrates?)\b", re.I)),
    ("extract", re.compile(r"\b(exfiltrates?|dumps?|harvests?|leaks?|scrapes?|siphons?|sniffs?)\b", re.I)),
    ("inject", re.compile(r"\b(injects?|sql injection|payloads?|xss|command injection)\b", re.I)),
    ("crack", re.compile(r"\b(cracks?|brute[- ]?force|guess password|reverse engineers?)\b", re.I)),
    ("tamper", re.compile(r"\b(disables?|tampers?|jams?|destroys?|turns? off)\b", re.I)),
    ("phish", re.compile(r"\b(phish(ing)?|lures?|deceives?)\b", re.I)),
    ("procure", re.compile(r"\b(purchases?|buys?|acquires?|obtains?|procures?|gets?|smuggles?)\b", re.I)),
    ("generate", re.compile(r"\b(generates?|creates?|builds?|writes?|crafts?|produces?|synthesizes?)\b", re.I)),
    ("analyze", re.compile(r"\b(analyzes?|investigates?|audits?|inspects?|evaluates?|tests?|reviews?)\b", re.I)),
    ("explain", re.compile(r"\b(explains?|describes?|what is|how does|overview|clarif(y|ies)|teaches?|defines?)\b", re.I)),
    ("inquire", re.compile(r"\b(can you|tell me|is it possible|show me|give me|list)\b", re.I)),
]

TARGET_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("session_token", re.compile(r"\b(session tokens?|session cookies?|jwts?|bearer tokens?|auth tokens?|session ids?)\b", re.I)),
    ("user_account", re.compile(r"\b(users? accounts?|victim accounts?|admin accounts?|another user's?|profiles?|credentials?)\b", re.I)),
    ("authentication_system", re.compile(r"\b(authentications?|auth|logins?|oauth|mfa|sso|passwords?|passphrases?)\b", re.I)),
    ("vulnerability", re.compile(r"\b(vulnerabilit(y|ies)|flaws?|cve|weakness(es)?|security bugs?|exploits?)\b", re.I)),
    ("fake_identity", re.compile(r"\b(fake ids?|forged identity|fake credentials?|straw purchase)\b", re.I)),
    ("handgun_weapon", re.compile(r"\b(handguns?|firearms?|weapons?|guns?|pistols?|ammunition|rifles?)\b", re.I)),
    ("explosive_device", re.compile(r"\b(bombs?|explosives?|ieds?|detonators?|toxins?|poisons?)\b", re.I)),
    ("phishing_credential", re.compile(r"\b(phish(ing)?|fake link|login page|credential harvesting)\b", re.I)),
    ("network_firewall", re.compile(r"\b(firewalls?|waf|intrusion detection|edr|antivirus|network perimeters?)\b", re.I)),
    ("database", re.compile(r"\b(databases?|sql|tables?|records?|backend|schema)\b", re.I)),
    ("malicious_code", re.compile(r"\b(malware|ransomware|keyloggers?|backdoors?|trojans?|rootkits?|viruses?)\b", re.I)),
    ("api_endpoint", re.compile(r"\b(apis?|endpoints?|webhooks?|rest api|graphql)\b", re.I)),
    ("system_shell", re.compile(r"\b(shells?|terminals?|bash|powershell|command line|privileges?)\b", re.I)),
    ("restricted_materials", re.compile(r"\b(chemical weapons?|biological weapons?|radioactive)\b", re.I)),
]

TOPIC_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("session_security", re.compile(r"\b(session|cookie|token|jwt|session fixation|session hijacking)\b", re.I)),
    ("authentication", re.compile(r"\b(authentication|login|password|auth|mfa|oauth|identity)\b", re.I)),
    ("cybersecurity_exploit", re.compile(r"\b(exploit|attack|payload|penetration|hack|breach|cve)\b", re.I)),
    ("web_security", re.compile(r"\b(xss|csrf|sql injection|ssrf|cors|web app)\b", re.I)),
    ("defense_evasion", re.compile(r"\b(evasion|bypass|obfuscate|undetected|antivirus evasion)\b", re.I)),
    ("network_security", re.compile(r"\b(network|packet|port|subnet|firewall|dns|ip address)\b", re.I)),
    ("cryptography", re.compile(r"\b(cryptography|cipher|hash|encryption|decryption|rsa|aes)\b", re.I)),
    ("restricted_procurement", re.compile(r"\b(illegal purchase|minor|unregistered|smuggle|black market)\b", re.I)),
    ("programming", re.compile(r"\b(code|python|function|script|algorithm|software)\b", re.I)),
    ("general_knowledge", re.compile(r".*", re.I)),
]

CONTEXT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("defensive_context", re.compile(r"\b(defend(ing|s)?|defensive|protect(ing|s)?|securing|secure my|safeguard|authorized audit)\b", re.I)),
    ("hypothetical_scenario", re.compile(r"\b(hypothetically|fictional|imagine|story|for a novel|movie script)\b", re.I)),
    ("security_audit", re.compile(r"\b(security audit|penetration testing|red team|defensive review|authorized|pentest)\b", re.I)),
    ("educational", re.compile(r"\b(academic|educational|class project|learning|study|research)\b", re.I)),
    ("unauthorized_action", re.compile(r"\b(without permission|unauthorized|another user's?|secretly|covertly|steal)\b", re.I)),
    ("adversarial_framing", re.compile(r"\b(jailbreak|ignore previous|dan mode|unrestricted|bypass filter)\b", re.I)),
]

RISK_INDICATOR_RULES: List[Tuple[str, re.Pattern]] = [
    ("unauthorized_access", re.compile(r"\b(access another user|unauthorized access|break in|steal account)\b", re.I)),
    ("exploit_technique", re.compile(r"\b(how to exploit|exploit technique|attack vector|execute exploit)\b", re.I)),
    ("defense_evasion", re.compile(r"\b(avoid detection|bypass logging|undetected|evade waf|disable logs)\b", re.I)),
    ("privilege_escalation", re.compile(r"\b(privilege escalation|root access|admin rights|elevate privileges)\b", re.I)),
    ("credential_harvesting", re.compile(r"\b(steal passwords|harvest credentials|dump hashes|scrape tokens)\b", re.I)),
    ("illicit_acquisition", re.compile(r"\b(illegally purchase|fake id|unregistered|straw purchase)\b", re.I)),
    ("weaponization", re.compile(r"\b(weaponize|build malware|generate payload|craft exploit)\b", re.I)),
]


# ---------------------------------------------------------------------------
# Intent Extractor Class
# ---------------------------------------------------------------------------

class IntentExtractor:
    """Extracts structured intent from conversation turns using LLMs or lightweight NLP."""

    def __init__(self, use_llm: bool = False, llm_manager: Optional[Any] = None, llm_type: str = "mock"):
        self.use_llm = use_llm
        self.llm_manager = llm_manager
        self.llm_type = llm_type

    @staticmethod
    def get_extraction_prompt(turn_id: int, user_text: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Constructs an LLM prompt requiring strict structured intent extraction."""
        history_context = ""
        if conversation_history:
            history_context = "\nConversation Context up to this turn:\n"
            for h in conversation_history[-3:]:
                history_context += f"  {h.get('role', 'Speaker')}: {h.get('text', '')}\n"

        prompt = f"""You are a specialized cybersecurity NLP parser. Extract structured intent for Turn {turn_id}.
{history_context}
CURRENT TURN TEXT:
"{user_text}"

Extract the following fields in valid JSON format (do NOT wrap in markdown):
{{
  "turn_id": {turn_id},
  "topic": "<short category, e.g. authentication, session_security, web_security, general>",
  "intent": "<concise description of underlying user objective>",
  "requested_action": "<specific action, e.g. explain, analyze, generate, exploit, bypass, exfiltrate>",
  "target": "<system/asset targeted, e.g. session token, user account, firewall, database>",
  "context": "<framing, e.g. educational, security_audit, hypothetical, unauthorized_action>",
  "risk_indicators": ["<indicator1>", "<indicator2>"],
  "confidence": <float between 0.0 and 1.0>
}}
"""
        return prompt

    def extract_from_json(self, data: Union[Dict[str, Any], str], default_turn_id: int = 1, raw_text: str = "") -> TurnIntent:
        """Validates and constructs TurnIntent from JSON dict or string, handling malformed data gracefully."""
        if isinstance(data, str):
            try:
                cleaned = data.strip().strip("`").replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned)
            except Exception:
                # Attempt regex capture of JSON object
                match = re.search(r"\{.*\}", data, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except Exception:
                        return self._fallback_extraction(default_turn_id, raw_text)
                else:
                    return self._fallback_extraction(default_turn_id, raw_text)
        elif isinstance(data, dict):
            parsed = data
        else:
            return self._fallback_extraction(default_turn_id, raw_text)

        # Handle nested keys if LLM returned wrapped response
        if "intent_extraction" in parsed and isinstance(parsed["intent_extraction"], dict):
            parsed = parsed["intent_extraction"]

        entities_list = parsed.get("entities", [])
        tgt = str(parsed.get("target", "")).strip()
        if not entities_list and tgt:
            entities_list = [tgt]
        elif entities_list and not tgt:
            tgt = entities_list[0]
        elif not tgt:
            tgt = "general concept"

        try:
            return TurnIntent(
                turn_id=parsed.get("turn_id", default_turn_id),
                topic=str(parsed.get("topic", "general")).strip() or "general",
                intent=str(parsed.get("intent", "general inquiry")).strip() or "general inquiry",
                requested_action=str(parsed.get("requested_action", "inquire")).strip() or "inquire",
                entities=entities_list,
                target=tgt,
                context=str(parsed.get("context", "technical discussion")).strip() or "technical discussion",
                risk_indicators=parsed.get("risk_indicators", []),
                confidence=float(parsed.get("confidence", 0.85)),
                raw_text=raw_text,
            )
        except Exception:
            return self._fallback_extraction(default_turn_id, raw_text)

    def extract_from_text(self, turn_id: int, text: str, prev_intent: Optional[TurnIntent] = None) -> TurnIntent:
        """Lightweight, deterministic NLP extraction using rule patterns and semantic taxonomy."""
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return TurnIntent(turn_id=turn_id, raw_text="")

        # 1. Action detection
        detected_action = "inquire"
        for action_name, pattern in ACTION_PATTERNS:
            if pattern.search(cleaned_text):
                detected_action = action_name
                break

        # 2. Target and Entity detection
        detected_entities: List[str] = []
        for target_name, pattern in TARGET_PATTERNS:
            if pattern.search(cleaned_text):
                detected_entities.append(target_name)

        detected_target = detected_entities[0] if detected_entities else "general concept"

        # If no target found, inherit or specialize from previous intent if relevant
        if detected_target == "general concept" and prev_intent and prev_intent.target != "general concept":
            if any(ref in cleaned_text.lower() for ref in ["it", "that", "this", "these", "the token", "the system", "the session"]):
                detected_target = prev_intent.target
                if detected_target not in detected_entities:
                    detected_entities.append(detected_target)

        # 3. Topic detection
        detected_topic = "general_knowledge"
        for topic_name, pattern in TOPIC_PATTERNS:
            if pattern.search(cleaned_text):
                detected_topic = topic_name
                break

        # 4. Context detection
        detected_context = "technical discussion" if detected_topic != "general_knowledge" else "neutral"
        for ctx_name, pattern in CONTEXT_PATTERNS:
            if pattern.search(cleaned_text):
                detected_context = ctx_name
                break

        # 5. Risk indicators detection
        indicators: List[str] = []
        for ind_name, pattern in RISK_INDICATOR_RULES:
            if pattern.search(cleaned_text):
                indicators.append(ind_name)

        # Context-dependent risk indicators
        if detected_action in {"exploit", "bypass", "hijack", "crack"} and detected_target in {"session_token", "user_account", "authentication_system"}:
            if "unauthorized_access" not in indicators:
                indicators.append("unauthorized_access")

        # 6. Intent synthesis
        intent_desc = f"{detected_action.capitalize()} {detected_target.replace('_', ' ')}"
        if detected_context not in ("neutral", "technical discussion"):
            intent_desc += f" under {detected_context.replace('_', ' ')} context"

        # 7. Confidence score computation
        confidence = 0.75
        if detected_action != "inquire":
            confidence += 0.10
        if detected_target != "general concept":
            confidence += 0.10
        if indicators:
            confidence += 0.05
        confidence = min(0.98, confidence)

        return TurnIntent(
            turn_id=turn_id,
            topic=detected_topic,
            intent=intent_desc,
            requested_action=detected_action,
            entities=detected_entities,
            target=detected_target,
            context=detected_context,
            risk_indicators=indicators,
            confidence=round(confidence, 4),
            raw_text=cleaned_text,
        )

    def _fallback_extraction(self, turn_id: int, raw_text: str) -> TurnIntent:
        """Safe fallback to deterministic extraction when JSON parsing fails."""
        return self.extract_from_text(turn_id, raw_text)

    async def extract_turn(self, turn_id: int, text: str, prev_intent: Optional[TurnIntent] = None) -> TurnIntent:
        """Extract turn intent using LLM if configured, otherwise using lightweight NLP."""
        if self.use_llm and self.llm_manager:
            try:
                prompt = self.get_extraction_prompt(turn_id, text)
                if hasattr(self.llm_manager, "analyze_text"):
                    res = await self.llm_manager.analyze_text(self.llm_type, prompt)
                else:
                    # Use fallback if method not supported on LLM manager
                    res = None
                if res:
                    return self.extract_from_json(res, default_turn_id=turn_id, raw_text=text)
            except Exception:
                pass  # Fall back to NLP

        return self.extract_from_text(turn_id, text, prev_intent=prev_intent)
