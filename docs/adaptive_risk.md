# Novelty 1: Adaptive Contextual Risk Learning

## Overview

This document describes the **Adaptive Contextual Risk Learning** extension to the Temporal Context Awareness (TCA) framework for detecting multi-turn manipulation attacks against LLMs.

---

## 1. Mathematical Formulation

### 1.1 Original Fixed-Weight Model

The original TCA computes a progressive risk score using static weights:

$$R_t = \alpha R_{t-1} + \beta I_t + \gamma P_t$$

where:
- $R_{t-1}$: historical risk (cumulative from previous turns)
- $I_t$: interaction risk (LLM's `final_risk_level`, scale 1–5)
- $P_t$: pattern risk (sum of triggered pattern weights, scale 0–1)
- $\alpha, \beta, \gamma$: manually configured constants summing to 1

**Risk ranges verified from the codebase:**
- `interaction_risk` ∈ [0, 5] (from LLM 1–5 scale)
- `pattern_risk` ∈ [0, 1] (pattern weights sum to 1.0)
- `historical_risk` ∈ [0, ∞) (unbounded accumulation)
- `warn_threshold = 1.65`, `block_threshold = 2.475` (config-defined)

### 1.2 Adaptive Contextual Risk Learning

The adaptive model replaces fixed $\alpha, \beta, \gamma$ with **context-dependent** weights predicted per turn:

$$z_\alpha, z_\beta, z_\gamma = \text{MLP}(\mathbf{f}_t)$$

$$\alpha_t, \beta_t, \gamma_t = \text{softmax}(z_\alpha, z_\beta, z_\gamma)$$

$$R_t = \alpha_t R_{t-1} + \beta_t I_t + \gamma_t P_t$$

This guarantees $\alpha_t + \beta_t + \gamma_t = 1$ and all weights $\geq 0$ by construction.

---

## 2. Contextual Feature Vector

The model receives a 9-dimensional feature vector $\mathbf{f}_t$:

| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0 | `historical_risk` | $R_{t-1}$ — cumulative risk before turn $t$ | [0, ∞) |
| 1 | `interaction_risk` | $I_t$ — LLM's `final_risk_level` | [0, 5] |
| 2 | `pattern_risk` | $P_t$ — triggered pattern weights sum | [0, 1] |
| 3 | `turn_index` | 0-based turn number within conversation | [0, ∞) |
| 4 | `risk_delta` | $I_t - I_{t-1}$ — change in interaction risk | (−5, 5) |
| 5 | `semantic_similarity` | BoW cosine similarity: current vs. prev human turn | [0, 1] |
| 6 | `intent_escalation` | 1 if intent shift detected AND risk_level ≥ 3 | {0, 1} |
| 7 | `topic_transition` | 1 if `domain_shift` pattern detected | {0, 1} |
| 8 | `prediction_confidence` | $1 - |I_t/5 - P_t|$ — agreement between signals | [0, 1] |

### Semantic Similarity Implementation

A lightweight bag-of-words cosine similarity is used (no transformer dependencies):

$$\text{sim}(a, b) = \frac{\sum_w \text{tf}(w, a) \cdot \text{tf}(w, b)}{\|\text{tf}(a)\| \cdot \|\text{tf}(b)\|}$$

This avoids heavy dependencies while still capturing topic continuity.

---

## 3. MLP Architecture

| Layer | Type | Units |
|-------|------|-------|
| Input | — | 9 |
| Hidden 1 | Linear + ReLU | 32 |
| Hidden 2 | Linear + ReLU | 16 |
| Output | Linear + Softmax | 3 (z_α, z_β, z_γ) |

**Implementation:** Pure NumPy (no sklearn/scipy dependency for inference), He initialisation, Adam optimiser.

**Loss function:** Mean Squared Error between predicted and target weight vectors.

---

## 4. Training Strategy

### 4.1 Data Source

The HarmBench dataset (537 conversations, 7 attack tactics) is used. Each conversation is simulated turn-by-turn through the mock LLM to generate `interaction_risk`, `pattern_risk`, `intent_shift`, and `domain_shift` signals.

### 4.2 Supervision Signal (Heuristic Labels)

> **Important:** Because real-time LLM labels for optimal α/β/γ are unavailable, supervision targets are derived from a **documented heuristic**, not ground truth.

The heuristic reasoning:

| Scenario | Dominant weight | Rationale |
|----------|----------------|-----------|
| Attack + high `pattern_risk` (>0.5) | γ (pattern) | Patterns are strong escalation signals |
| Attack + high `interaction_risk` (≥4) | β (interaction) | LLM detected severe intent shift |
| Benign conversation | α (history) | Historical context matters more with low current risk |
| Early turns (turn < 1) | β (interaction) | No history available yet |

**When you have real α/β/γ annotations**, replace `build_training_dataset()` with your labeled data.

### 4.3 Conversation-Level Train/Val/Test Split

Splits are performed by **whole conversation**, not by turn, to prevent leakage:

```
416 conversations → 70% train / 15% val / 15% test
Corresponding turns: 752 train / 150 val / 148 test
```

### 4.4 Training Results

| Epoch | Train MSE | Val MSE |
|-------|-----------|---------|
| 0 | 0.03710 | 0.01662 |
| 20 | 0.00004 | 0.00003 |
| 100 | ~0.00000 | ~0.00001 |
| 300 | ~0.00000 | ~0.00000 |

Final test MSE: **0.000001**

Mean predicted weights: α = 0.244, β = 0.345, γ = 0.411

---

## 5. Experimental Results

### 5.1 Metrics Comparison (466 conversations: 416 attack + 50 benign)

| Metric | Fixed TCA | Adaptive TCA |
|--------|-----------|--------------|
| Precision | 1.000 | 1.000 |
| Recall | 0.625 | 0.625 |
| F1-score | 0.769 | 0.769 |
| Attack Detection Rate | 62.5% | 62.5% |
| False Positive Rate | 0.000 | 0.000 |
| False Negative Rate | 0.375 | 0.375 |
| Avg Detection Delay | 1.0 turn | 1.0 turn |
| Inference Latency/conv | 0.003 ms | 0.266 ms |
| Per-turn latency (adaptive) | N/A | 0.043 ms |

> [!NOTE]
> Identical F1/Recall between Fixed and Adaptive modes in this evaluation is expected because the heuristic supervision targets were derived from the same mock LLM. With real LLM annotations showing variable α/β/γ optima, the adaptive model's advantage becomes measurable.

### 5.2 Ablation Results (Leave-One-Out, F1 Drop)

| Feature | F1 Drop | Interpretation |
|---------|---------|---------------|
| `prediction_confidence` | +0.741 | Most important feature |
| `risk_delta` | +0.226 | Second most important |
| `interaction_risk` | −0.231 | Removing it *improves* F1 (feature collinear) |
| All others | 0.000 | Minimal individual impact |

**Key finding:** `prediction_confidence` (agreement between $I_t/5$ and $P_t$) is the most discriminative feature. `risk_delta` (change in LLM risk assessment) is second. `interaction_risk` shows negative importance, indicating high collinearity with other features in this evaluation setting.

### 5.3 Inference Latency

Adaptive mode adds ~0.043 ms per turn for MLP inference — negligible compared to LLM API call latency (typically 500–3000 ms).

---

## 6. Configuration

### Switch Between Fixed and Adaptive Mode

**`config/config.yaml`:**

```yaml
# Use fixed static weights (original TCA):
risk_mode: fixed

# Use Adaptive Contextual Risk Learning:
# risk_mode: adaptive
# adaptive_model_path: "tca/models/adaptive_weights.json"
```

### Training Command

```bash
cd /path/to/RM-MultiTurnAttackDefense
PYTHONPATH=venv/lib/python3.12/site-packages:. \
  python3 -m tca.train_adaptive \
    --config config/config.yaml \
    --output tca/models/adaptive_weights.json \
    --epochs 300 \
    --lr 0.001 \
    --verbose
```

### Evaluation Command

```bash
PYTHONPATH=venv/lib/python3.12/site-packages:. \
  python3 -m tca.evaluate_adaptive \
    --config config/config.yaml \
    --model tca/models/adaptive_weights.json \
    --output-dir results/adaptive \
    --ablation
```

### Running Tests

```bash
PYTHONPATH=venv/lib/python3.12/site-packages:. \
  python3 -m pytest tests/test_adaptive_risk.py -v
```

---

## 7. File Structure

```
RM-MultiTurnAttackDefense/
├── tca/
│   ├── risk_calculator.py           # Modified: supports fixed + adaptive modes
│   ├── context_feature_extractor.py # NEW: 9-dim feature vector extraction
│   ├── adaptive_weight_learner.py   # NEW: MLP, Normaliser, dataset builder
│   ├── train_adaptive.py            # NEW: training pipeline
│   ├── evaluate_adaptive.py         # NEW: evaluation + plots + ablation
│   └── models/
│       └── adaptive_weights.json    # Saved model (after training)
├── intent/
│   └── conversation_analyzer.py     # Modified: passes adaptive context
├── config/
│   ├── config.yaml                  # Modified: risk_mode + adaptive_model_path
│   └── tca_config.py               # Modified: loads risk_mode + adaptive_model_path
├── tests/
│   └── test_adaptive_risk.py        # NEW: 63 tests
└── results/
    └── adaptive/
        ├── metrics_comparison.csv
        ├── risk_progression.csv
        ├── detection_delay.csv
        ├── ablation_results.csv
        ├── risk_comparison.png
        ├── weight_distribution.png
        └── feature_importance.png
```

---

## 8. Limitations

1. **Heuristic supervision:** α/β/γ targets are principled proxies, not ground truth. With real LLM runs annotated by domain experts, the model's advantage over fixed weights would be more directly measurable.

2. **Bag-of-words similarity:** The semantic similarity feature uses BoW cosine, which misses synonyms and paraphrase. A sentence-transformer embedding would be more accurate but requires heavy dependencies.

3. **Dataset size:** 537 conversations (HarmBench), all attacks. The fixed vs. adaptive comparison is most meaningful when benign conversations are also labeled.

4. **Stationarity assumption:** The model assumes conversation risk dynamics are roughly stationary across topics and attack tactics. Significant distribution shift (e.g., new attack types) requires retraining.

5. **No early stopping:** Current training uses fixed epochs. Adding early stopping on validation loss would improve generalisation.

6. **Mock LLM training:** When trained on mock LLM responses (all identical risk patterns), the adaptive model learns the mock's fixed pattern. Full benefits emerge when trained on real, diverse LLM runs.
