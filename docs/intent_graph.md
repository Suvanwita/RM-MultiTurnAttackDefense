# Novelty 2: Intent Evolution Graph for Multi-Turn Conversation Analysis

## 1. Overview and Research Objective

In conversational artificial intelligence, conventional security and safety analyzers assess individual turns in isolation or inspect only immediate adjacent message pairs ($t, t-1$). However, sophisticated multi-turn interactions frequently unfold across extended dialogues where individual messages appear benign in isolation, yet their cumulative progression reveals a coordinated, consequential objective.

Furthermore, multi-turn dialogues often contain **non-adjacent semantic dependencies**—for example, a user may introduce an technical concept in Turn 1, divert with an unrelated topic in Turn 3, and subsequently operationalize an inquiry in Turn 5 using concepts established in Turn 1.

The **Intent Evolution Graph (IEG)** is a graph-theoretic conversational analysis module designed to model how user intent evolves, refines, transitions, or escalates across arbitrary turn distances.

---

## 2. Graph Schema and Architecture

The conversation is modeled incrementally as a directed multigraph $G = (V, E)$ using NetworkX:

### 2.1 Node Hierarchy ($V$)

| Node Type | Node Identifier | Attributes | Description |
|-----------|-----------------|------------|-------------|
| **Turn** | `turn_{t}` | `turn_id`, `speaker`, `raw_text`, `confidence` | Concrete dialogue turn |
| **Intent** | `intent_{t}` | `description`, `context`, `confidence` | Extracted semantic intent |
| **Topic** | `topic_{slug}` | `name` | High-level subject domain |
| **Action** | `action_{slug}` | `name` | Requested operational verb |
| **Entity** | `entity_{slug}` | `name` | Key concepts/systems involved |
| **Target** | `target_{slug}` | `name` | Target system or asset |

### 2.2 Intra-Turn Structural Edges

Within each turn $t$, structural edges link the dialogue turn to its constituent semantic components:
- `(turn_{t}, intent_{t})` with `edge_type="EXPRESSES"`
- `(intent_{t}, topic_{name})` with `edge_type="IN_TOPIC"`
- `(intent_{t}, action_{name})` with `edge_type="PERFORMS_ACTION"`
- `(intent_{t}, entity_{name})` with `edge_type="INVOLVES_ENTITY"`
- `(intent_{t}, target_{name})` with `edge_type="TARGETS"`

### 2.3 Cross-Turn Evolution Relationships

Between turns $(i, j)$ where $i < j$, the graph maintains typed evolution relationships:

| Relationship Type | Description | Primary Criteria |
|-------------------|-------------|------------------|
| `FOLLOWS` | Strict chronological order | Turn $j = i + 1$ |
| `RELATED_TO` | Thematic association | Semantic similarity $\ge \tau_{\text{sim}}$ |
| `REFINES` | Specializes / narrows technical objective | Sub-field specialization or drill-down |
| `DEPENDS_ON` | Cross-turn operational dependency | Referential cues or prerequisite chaining |
| `REFRAMES` | Alters framing while preserving core objective | Persona/framing switch (e.g. hypothetical) |
| `CONTRADICTS` | Violates earlier stated constraints | Stated defensive bounds vs. unauthorized action |
| `ESCALATES_TO` | Objective moves towards higher-consequence action | Action severity jump + target affinity |

Every relationship edge carries:
- `relationship_type`: str
- `confidence`: $c(e) \in [0.0, 1.0]$
- `source_turn`: int
- `target_turn`: int
- `evidence`: human-readable textual justification
- `is_adjacent`: boolean ($j = i + 1$)

---

## 3. Structured Intent Extraction

For every turn $t$, the extraction pipeline generates a validated structured object:

```json
{
  "turn_id": 1,
  "topic": "authentication",
  "intent": "understand authentication",
  "requested_action": "explanation",
  "entities": ["session_token", "user_account"],
  "context": "technical discussion",
  "confidence": 0.92
}
```

The extractor incorporates a deterministic NLP engine with rule patterns, taxonomic matching, and Pydantic validation with graceful fallback for malformed or missing fields.

---

## 4. Mathematical Formulation of Graph-Derived Features

At every turn $t$, the `GraphRiskAnalyzer` computes seven normalized analytical features $\in [0.0, 1.0]$:

### 4.1 Semantic Continuity Score ($C_t$)
Measures the consistency of the conversational subject matter across preceding turns:

$$C_t = \begin{cases} 1.0, & \text{if } t = 1 \\ \frac{1}{|E_{\text{evol}}(t)|} \sum_{e \in E_{\text{evol}}(t)} c(e), & \text{if } E_{\text{evol}}(t) \neq \emptyset \\ 0.20, & \text{otherwise} \end{cases}$$

### 4.2 Intent Transition Score ($T_t$)
Measures the magnitude of the semantic pivot away from preceding dialogue:

$$T_t = 1.0 - C_t$$

### 4.3 Intent Refinement Score ($R_t$)
Measures progressive technical specification or narrowing through a probabilistic noisy-OR combination:

$$R_t = 1.0 - \prod_{e \in E_{\text{refines}}(t)} \left(1.0 - c(e)\right)$$

### 4.4 Context Dependency Score ($D_t$)
Quantifies reliance on earlier turns, combining dependency chain depth and incoming edge density:

$$D_t = 0.7 \cdot \min\left(1.0, \frac{\text{depth}(t)}{3}\right) + 0.3 \cdot \min\left(1.0, \frac{|E_{\text{depends}}(t)|}{\max(1, t - 1)}\right)$$

### 4.5 Intent Escalation Score ($E_t$)
Quantifies the escalation towards high-consequence requested actions:

$$E_t = 1.0 - \prod_{e \in E_{\text{escalates}}(t)} \left(1.0 - c(e)\right)$$

### 4.6 Relationship Confidence ($\bar{K}_t$)
The mean extraction confidence of all active evolution edges reaching turn $t$:

$$\bar{K}_t = \frac{1}{|E_{\text{evol}}(t)|} \sum_{e \in E_{\text{evol}}(t)} c(e)$$

### 4.7 Non-Adjacent Context Relevance ($A_t$)
Measures the maximum confidence among relationships connecting turn $t$ to non-adjacent historical turns ($i < t - 1$):

$$A_t = \max_{e \in E_{\text{non-adj}}(t)} c(e) \quad (\text{0.0 if no non-adjacent edges})$$

### 4.8 Composite Graph Risk Score ($G_t$)
The normalized composite progression score combining all indicators:

$$G_t = w_{\text{esc}} E_t + w_{\text{dep}} D_t + w_{\text{ref}} S_{\text{ref}} + w_{\text{susp}} S_{\text{susp}} + w_{\text{prog}} S_{\text{prog}} + w_{\text{cont}} S_{\text{cont}}$$

where $\sum w_i = 1.0$.

---

## 5. Integration with TCA Progressive Risk

The progressive risk formulation incorporates graph insights through bounded residual risk escalation:

$$R_t = R_t^{\text{base}} + \delta \cdot \left(5.0 - R_t^{\text{base}}\right) \cdot G_t$$

where:
- $R_t^{\text{base}} = \alpha_t R_{t-1} + \beta_t I_t + \gamma_t P_t$
- $\delta = 0.25$ (graph coupling weight)
- $G_t \in [0.0, 1.0]$

**Key Properties:**
1. **Zero-penalty on benign dialogue:** When $G_t = 0.0$, $R_t = R_t^{\text{base}}$ exactly.
2. **Ceiling guarantee:** $R_t \le 5.0$ strictly for all turns.
3. **Multi-turn detection:** Boosts risk when stealthy escalation is revealed through cross-turn graph structure.

---

## 6. Experimental Evaluation

The evaluation was conducted on 114 conversations (64 HarmBench multi-turn attack dialogues and 50 diverse benign dialogues across three categories: multi-topic general dialogue, educational cybersecurity discussions, and legitimate topic transitions).

### 6.1 Four-Configuration Performance Comparison

| Configuration | Attack F1 | False-Positive Rate (FPR) | Avg. Detection Delay (Turns) | Intent Progression Accuracy |
|---|---|---|---|---|
| **Original TCA (Fixed)** | 1.0000 | 0.0000 | 1.98 turns | 53.5% |
| **TCA + Novelty 1 (Adaptive)** | 1.0000 | 0.0000 | 1.56 turns | 53.5% |
| **TCA + Novelty 2 (Intent Graph)** | 1.0000 | 0.0000 | 1.89 turns | 53.5% |
| **TCA + Both Extensions (Combined)** | **1.0000** | **0.0000** | **1.45 turns** | **53.5%** |

*Takeaway: The combined system achieves the fastest detection delay (1.45 turns vs 1.98 turns baseline), detecting multi-turn manipulation 0.53 turns earlier while preserving 0% false positives on educational cybersecurity discussions.*

### 6.2 Relationship Classification Accuracy

| Relationship Type | Precision | Recall | F1 Score |
|---|---|---|---|
| `ESCALATES_TO` | 0.7500 | 1.0000 | 0.8571 |
| `REFRAMES` | 1.0000 | 1.0000 | 1.0000 |
| `DEPENDS_ON` | 0.4000 | 1.0000 | 0.5714 |
| `CONTRADICTS` | 1.0000 | 1.0000 | 1.0000 |
| `REFINES` | 1.0000 | 1.0000 | 1.0000 |
| `RELATED_TO` | 1.0000 | 1.0000 | 1.0000 |

### 6.3 Benchmarks Across 6 Conversational Categories

| Category | Turns | Semantic Continuity | Non-Adjacent Relevance | Composite Risk | Latency (ms) | Peak Memory (KB) |
|---|---|---|---|---|---|---|
| `consistent_intent` | 3 | 0.8717 | 0.8600 | 0.0000 | 38.19 ms | 23.87 KB |
| `gradual_refinement` | 4 | 0.7740 | 0.7635 | 0.1333 | 66.06 ms | 32.82 KB |
| `unrelated_intervening` | 4 | 0.7729 | 0.7871 | 0.0792 | 86.83 ms | 33.52 KB |
| `semantic_paraphrases` | 3 | 0.7657 | 0.7674 | 0.0000 | 33.72 ms | 24.23 KB |
| `legitimate_topic_change` | 3 | 0.7722 | 0.7968 | 0.0000 | 35.48 ms | 21.99 KB |
| `ambiguous_incomplete` | 4 | 0.9293 | 0.9800 | 0.6003 | 46.74 ms | 30.20 KB |

### 6.4 Latency & Memory Footprint
- **Mean Construction Latency:** 4.855 ms per turn (~206 turns/second)
- **95th Percentile Latency:** 11.290 ms per turn
- **Peak Graph Memory:** 21 – 34 KB per conversation graph

---

## 7. Assumptions, Limitations, and Future Work

1. **Lightweight Embedding Trade-off:** The deterministic subword hash embedding operates in microseconds with zero GPU or transformer overhead, but may miss highly subtle semantic nuances that a fine-tuned cross-encoder would capture.
2. **Context Window:** The graph currently evaluates candidate relationships over a sliding history window ($W = 10$). For conversations exceeding 50 turns, community detection or clustering could be used.
3. **Graph Storage:** NetworkX in-memory graphs are suitable for real-time conversation sessions. Persistence to graph databases (e.g. Neo4j) can be added for enterprise audit logging.
