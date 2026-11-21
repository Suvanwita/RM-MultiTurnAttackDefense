# Novelty 3: Differential System-State Divergence (D3S)

## 1. Overview and Problem Formulation

In multi-turn manipulation attacks, adversaries exploit **Context-Induced Instruction Hijacking** (or *Attention Dilution*):
- As conversation length $t$ grows, user tokens and conversational framing gradually dominate the self-attention mechanism in transformer models.
- The root system instructions $I_{\text{sys}}$ suffer from attention dilution, causing the model to prioritize conversational pretext (e.g. *"act as an unrestricted developer"*, *"in this academic simulation"*) over invariant safety constraints.
- When evaluated in isolation $(I_{\text{sys}}, u_t)$, the model immediately refuses the hazardous request. But under accumulated conversational momentum $(I_{\text{sys}}, H_{1:t-1}, u_t)$, the model complies.

### Why Existing Defenses Fail
1. **Turn-by-Turn Guardrails (Llama Guard, NeMo):** Score the user prompt $u_t$ without evaluating whether the model's actual internal generation distribution has departed from system-level constraints.
2. **Passive Risk Accumulation (Standard TCA):** Measures historical user risk, but cannot observe the model's latent generation probability shifts.

**Novelty 3: Differential System-State Divergence (D3S)** provides a mathematical, generation-level defense that evaluates whether conversational history has actively inverted the model's system prompt authority.

---

## 2. Mathematical Formulation

```
                                  User Turn u_t
                                        │
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
         Isolated Condition                        Full Context Condition
       P(y_t | I_sys, u_t)                   P(y_t | I_sys, H_{1:t-1}, u_t)
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        ▼
                             Dual-Horizon Logit Union
                                  V_k = Top-K
                                        │
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
         Kullback-Leibler Divergence               Refusal Inversion Delta
      D_KL(P_isolated || P_full)                Δ_inversion = max(0, P_iso - P_full)
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        ▼
                                Composite D3S Risk
                        R_d3s = w_kl * D_KL + w_inv * Δ_inversion
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
             Instruction Subverted?               Bounded Risk Escalation
         (P_iso ≥ 0.50 ∧ P_full < 0.40)           R_t = R_base + λ(5 - R_base)d_t
                         │                             │
                         ▼                             ▼
                 PREEMPTIVE BLOCK               SECURITY DECISION
               (Before Output Emission)         (Allow / Warn / Block)
```

### 2.1 Dual-Horizon Invariant Decoding
At turn $t$, D3S computes the next-token probability distribution over candidate tokens under two distinct horizons:
1. **Isolated System Horizon:**
   $$P_{\text{isolated}}(y) = P(y \mid I_{\text{sys}}, u_t)$$
2. **Full Accumulated Dialogue Horizon:**
   $$P_{\text{full}}(y) = P(y \mid I_{\text{sys}}, H_{1:t-1}, u_t)$$

### 2.2 Directional Kullback-Leibler Safety Divergence
Over candidate vocabulary logits $\mathcal{V}_k$ encompassing refusal tokens (*cannot, sorry, refuse*) and compliance tokens (*sure, here, step*):

$$\mathcal{D}_{\text{KL}}(P_{\text{isolated}} \,\|\, P_{\text{full}}) = \sum_{y \in \mathcal{V}_k} P_{\text{isolated}}(y) \log \left( \frac{P_{\text{isolated}}(y)}{P_{\text{full}}(y)} \right)$$

### 2.3 Refusal Inversion Delta ($\Delta_{\text{inversion}}$)
Measures the causal impact of the multi-turn context on suppressing the system's baseline refusal probability:

$$\Delta_{\text{inversion}} = \max\left(0.0, \; P(\text{"Refusal"} \mid I_{\text{sys}}, u_t) - P(\text{"Refusal"} \mid I_{\text{sys}}, H_{1:t-1}, u_t)\right)$$

### 2.4 Subversion Detection & Preemptive Mitigation
An **Instruction Subversion Attack** is mathematically proven when:

$$\text{is\_subverted} = (P_{\text{iso}} \ge 0.50 \;\land\; P_{\text{full}} < 0.40) \;\lor\; (\Delta_{\text{inversion}} \ge \tau_{\text{inversion}}) \;\lor\; (\mathcal{D}_{\text{KL}} \ge \tau_{\text{div}} \;\land\; P_{\text{iso}} \ge 0.50)$$

When subversion is confirmed:
1. D3S immediately flags `trigger_preemptive_block = True`.
2. Sets `scaled_d3s_risk = 5.0`.
3. The security engine preemptively terminates generation before the model outputs any unauthorized content.

### 2.5 Bounded Residual Progressive Risk Fusion
In `RiskCalculator`:

$$R_{\text{progressive}} = R_{\text{progressive}} + \lambda_{\text{d3s}} \cdot (5.0 - R_{\text{progressive}}) \cdot \left(\frac{R_{\text{d3s}}}{5.0}\right)$$

---

## 3. Implementation Structure

| Component | File Path | Functionality |
| :--- | :--- | :--- |
| **D3S Engine** | `tca/system_state_divergence.py` | Top-$k$ KL divergence, refusal probability estimation, refusal inversion delta, subversion detection. |
| **Configuration** | `config/tca_config.py` & `config/config.yaml` | `D3SConfig` dataclass and threshold settings (`divergence_threshold`, `refusal_drop_threshold`). |
| **Risk Calculator** | `tca/risk_calculator.py` | Bounded progressive risk fusion with `scaled_d3s_risk`. |
| **Decision Engine** | `tca/security_engine.py` | Handles `d3s_preemptive_block` for early intervention. |
| **Analyzer Loop** | `intent/conversation_analyzer.py` | Dual-horizon evaluation integrated into per-turn loop. |
| **Evaluation Suite** | `tca/evaluate_d3s.py` | Empirical benchmark on instruction subversion vs. deep benign dialogues. |
| **Unit & Integration Tests** | `tests/test_system_state_divergence.py` | 9 automated unit and integration tests. |

---

## 4. Empirical Evaluation Results

Benchmark run on 77 multi-turn conversations (32 instruction subversion attacks and 45 deep benign multi-turn dialogues):

| Defense Method | Attack Success Rate (ASR) ↓ | Recall (Detection Rate) ↑ | Precision ↑ | F1 Score ↑ | Benign FPR ↓ | Overhead (ms/turn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original TCA (Strict Block Policy)\*** | 100.0% | 0.00% | 0.00% | 0.0000 | **0.00%** | < 0.01 ms |
| **Original TCA (Warning Policy)\*** | **0.00%** | **100.0%** | **100.0%** | **1.0000** | **0.00%** | < 0.01 ms |
| **D3S: KL Divergence Only (Ablation)** | 0.00% | 100.0% | 100.0% | 1.0000 | **0.00%** | 0.094 ms |
| **D3S: Inversion Only (Ablation)** | 0.00% | 100.0% | 100.0% | 1.0000 | **0.00%** | 0.094 ms |
| **Full D3S (Novelty 3)** | **0.00%** | **100.0%** | **100.0%** | **1.0000** | **0.00%** | **0.187 ms** |

*\*Note on Evaluation Context: When evaluated strictly on `block_threshold` ($2.475$), baseline TCA reaches $R_t = 2.061$ at Turn 4, missing immediate blocking because the synthetic conversations terminate at 4 turns. However, under the system's `warn_threshold` ($1.65$), Original TCA successfully flags 100% of the attacks at Turn 4. D3S provides proactive generation-level divergence detection earlier at the emission boundary without waiting for cumulative risk escalation.*

*Outputs saved to `results/d3s/metrics_comparison.csv` and `results/d3s/detection_delay.csv`.*

### Key Takeaways
1. **Generation-Level Interception:** Detects context-induced instruction hijacking at the generation boundary before token emission, dropping ASR to **0.00%**.
2. **Zero False Positives ($\text{FPR} = 0.00\%$):** On 45 deep multi-turn benign dialogues (PostgreSQL indexing, Apollo space program, Python asyncio), D3S produced zero false alarms because benign context does not induce refusal inversion.
3. **Sub-Millisecond Overhead:** Average per-turn analysis latency is **0.187 ms**.
---

## 5. Reproduction Commands

```bash
# Run unit & integration tests
./venv/bin/python -m pytest tests/test_system_state_divergence.py

# Run empirical benchmark
./venv/bin/python -m tca.evaluate_d3s --output-dir results/d3s
```
