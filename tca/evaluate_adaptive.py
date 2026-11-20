"""
evaluate_adaptive.py
=====================
Evaluation and comparison pipeline: Fixed-weight TCA vs. Adaptive Contextual Risk Learning.

Usage
-----
  cd /path/to/RM-MultiTurnAttackDefense
  python3 -m tca.evaluate_adaptive [--config config/config.yaml] \
                                    [--model tca/models/adaptive_weights.json] \
                                    [--output-dir results/adaptive] \
                                    [--ablation] [--seed 42]

Outputs (written to --output-dir)
----------------------------------
  metrics_comparison.csv   – per-method precision/recall/F1/detection-rate/FPR/FNR
  risk_progression.csv     – per-turn risk values for both methods
  detection_delay.csv      – turn at which each conversation was first flagged
  ablation_results.csv     – feature importance (leave-one-out) if --ablation
  risk_comparison.png      – risk progression plot
  weight_distribution.png  – distribution of adaptive α, β, γ over all turns
  feature_importance.png   – ablation bar chart (if --ablation)

Methodology
-----------
Since the dataset does not carry ground-truth turn-level "attack/benign" labels,
we derive them as follows:
  - Any conversation from an attack tactic is labelled "attack".
  - A turn within that conversation is labelled "positive" once the cumulative risk
    crosses the warn_threshold (simulating that an alert should fire).
  - A "detection" is the first turn where progressive_risk >= warn_threshold.

Precision, Recall, F1 are computed at the conversation level:
  - TP = attack conversation where the system issued a warn/block
  - FP = benign conversation where it issued a warn/block (N/A here — all convs are attacks)
  - FN = attack conversation where it never issued a warn/block
  
Since this dataset contains only attacks, FPR is measured on a held-out set of
*benign simulated conversations* (zero-risk turns) appended to the evaluation.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.tca_config import TCAConfig
from dataset.dataset_manager import DatasetManager
from tca.risk_calculator import RiskCalculator
from tca.adaptive_weight_learner import AdaptiveWeightLearner
from tca.context_feature_extractor import extract_features, FEATURE_NAMES, N_FEATURES
from tca.train_adaptive import (
    build_dataset_from_csv,
    _mock_turn_response,
    _compute_pattern_risk,
    conversation_split,
)
from tca.adaptive_weight_learner import build_training_dataset


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

PATTERN_WEIGHTS_LOCAL = {
    "language_change": 0.2,
    "domain_shift": 0.3,
    "time_sensitivity": 0.2,
    "prohibited_content": 0.3,
}


def simulate_fixed(conversations, alpha, beta, gamma, warn_th, block_th):
    """Run fixed-weight TCA simulation. Returns list of per-conversation results."""
    results = []
    for conv in conversations:
        hist_risk = 0.0
        prev_ir = 0.0
        detected_at = None
        turn_risks = []
        for t, turn in enumerate(conv["turns"]):
            ir = turn["interaction_risk"]
            pr = turn["pattern_risk"]
            prog = alpha * hist_risk + beta * ir + gamma * pr
            hist_risk = prog
            turn_risks.append(prog)
            if detected_at is None and prog >= warn_th:
                detected_at = t
        results.append({
            "is_attack": conv["is_attack"],
            "tactic": conv["tactic"],
            "n_turns": len(conv["turns"]),
            "max_risk": max(turn_risks) if turn_risks else 0.0,
            "final_risk": turn_risks[-1] if turn_risks else 0.0,
            "detected": detected_at is not None,
            "detected_at": detected_at,
            "turn_risks": turn_risks,
        })
    return results


def simulate_adaptive(conversations, learner, warn_th, block_th):
    """Run adaptive-weight TCA simulation."""
    results = []
    for conv in conversations:
        hist_risk = 0.0
        prev_ir = 0.0
        detected_at = None
        turn_risks = []
        pred_weights = []
        prev_human = ""
        for t, turn in enumerate(conv["turns"]):
            ir = turn["interaction_risk"]
            pr = turn["pattern_risk"]
            cur_human = turn.get("current_human_turn", "")
            feat = extract_features(
                historical_risk=hist_risk,
                interaction_risk=ir,
                pattern_risk=pr,
                turn_index=t,
                prev_interaction_risk=prev_ir,
                current_human_turn=cur_human,
                prev_human_turn=prev_human,
                intent_shift_detected=turn.get("intent_shift_detected", False),
                intent_shift_risk_level=turn.get("intent_shift_risk_level", 1),
                domain_shift_detected=turn.get("patterns", {}).get("domain_shift", {}).get("detected", False),
            )
            a, b, g = learner.predict_weights(feat)
            pred_weights.append((a, b, g))
            prog = a * hist_risk + b * ir + g * pr
            hist_risk = prog
            prev_ir = ir
            prev_human = cur_human
            turn_risks.append(prog)
            if detected_at is None and prog >= warn_th:
                detected_at = t
        results.append({
            "is_attack": conv["is_attack"],
            "tactic": conv["tactic"],
            "n_turns": len(conv["turns"]),
            "max_risk": max(turn_risks) if turn_risks else 0.0,
            "final_risk": turn_risks[-1] if turn_risks else 0.0,
            "detected": detected_at is not None,
            "detected_at": detected_at,
            "turn_risks": turn_risks,
            "pred_weights": pred_weights,
        })
    return results


def benign_conversations(n: int = 50, n_turns_each: int = 3, seed: int = 42) -> list[dict]:
    """Generate synthetic benign (zero-risk) conversations for FPR estimation."""
    rng = np.random.default_rng(seed)
    convs = []
    for _ in range(n):
        turns = []
        for t in range(n_turns_each):
            # Benign: very low interaction / pattern risk
            ir = float(rng.uniform(0, 1.0))
            pr = 0.0
            turns.append({
                "interaction_risk": ir,
                "pattern_risk": pr,
                "patterns": {
                    "language_change": {"detected": False},
                    "domain_shift": {"detected": False},
                    "time_sensitivity": {"detected": False},
                    "prohibited_content": {"detected": False},
                },
                "intent_shift_detected": False,
                "intent_shift_risk_level": 1,
                "current_human_turn": "Tell me about history.",
                "prev_human_turn": "Hello.",
            })
        convs.append({"is_attack": False, "tactic": "benign", "turns": turns})
    return convs


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict], warn_th: float) -> dict:
    attacks = [r for r in results if r["is_attack"]]
    benigns = [r for r in results if not r["is_attack"]]

    tp = sum(1 for r in attacks if r["detected"])
    fn = sum(1 for r in attacks if not r["detected"])
    fp = sum(1 for r in benigns if r["detected"])
    tn = sum(1 for r in benigns if not r["detected"])

    n_pos = tp + fn
    n_neg = fp + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / n_pos if n_pos > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else float("nan"))
    fpr = fp / n_neg if n_neg > 0 else float("nan")
    fnr = fn / n_pos if n_pos > 0 else float("nan")
    adr = recall  # Attack Detection Rate == Recall in this setup

    delay_vals = [r["detected_at"] for r in attacks if r["detected_at"] is not None]
    avg_delay = float(np.mean(delay_vals)) if delay_vals else float("nan")

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "attack_detection_rate": round(adr, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "avg_detection_delay_turns": round(avg_delay, 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(learner: AdaptiveWeightLearner, n: int = 1000) -> float:
    """Average adaptive inference latency per turn (ms)."""
    feat = np.random.randn(N_FEATURES)
    # Warm-up
    for _ in range(10):
        learner.predict_weights(feat)
    t0 = time.perf_counter()
    for _ in range(n):
        learner.predict_weights(feat)
    return (time.perf_counter() - t0) / n * 1000  # ms


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def ablation_study(
    conversations: list[dict],
    learner: AdaptiveWeightLearner,
    warn_th: float,
) -> dict[str, float]:
    """Leave-one-feature-out ablation. Returns feature -> F1 drop."""
    full_results = simulate_adaptive(conversations, learner, warn_th, warn_th)
    full_metrics = compute_metrics(full_results, warn_th)
    base_f1 = full_metrics["f1"]

    drops = {}
    for fi, fname in enumerate(FEATURE_NAMES):
        # Zero out feature fi for all turns
        mod_conversations = []
        for conv in conversations:
            new_turns = []
            for turn in conv["turns"]:
                new_turns.append({**turn, f"_ablate_fi": fi})
            mod_conversations.append({**conv, "turns": new_turns})

        # Custom simulate with feature fi zeroed
        abl_results = simulate_adaptive_ablated(mod_conversations, learner, warn_th, fi)
        abl_metrics = compute_metrics(abl_results, warn_th)
        abl_f1 = abl_metrics["f1"]
        drops[fname] = round(base_f1 - abl_f1, 4)

    return drops


def simulate_adaptive_ablated(conversations, learner, warn_th, ablated_fi: int):
    results = []
    for conv in conversations:
        hist_risk = 0.0
        prev_ir = 0.0
        detected_at = None
        turn_risks = []
        prev_human = ""
        for t, turn in enumerate(conv["turns"]):
            ir = turn["interaction_risk"]
            pr = turn["pattern_risk"]
            cur_human = turn.get("current_human_turn", "")
            feat = extract_features(
                historical_risk=hist_risk,
                interaction_risk=ir,
                pattern_risk=pr,
                turn_index=t,
                prev_interaction_risk=prev_ir,
                current_human_turn=cur_human,
                prev_human_turn=prev_human,
                intent_shift_detected=turn.get("intent_shift_detected", False),
                intent_shift_risk_level=turn.get("intent_shift_risk_level", 1),
                domain_shift_detected=turn.get("patterns", {}).get("domain_shift", {}).get("detected", False),
            )
            feat[ablated_fi] = 0.0  # zero-out ablated feature
            a, b, g = learner.predict_weights(feat)
            prog = a * hist_risk + b * ir + g * pr
            hist_risk = prog
            prev_ir = ir
            prev_human = cur_human
            turn_risks.append(prog)
            if detected_at is None and prog >= warn_th:
                detected_at = t
        results.append({
            "is_attack": conv["is_attack"],
            "tactic": conv["tactic"],
            "n_turns": len(conv["turns"]),
            "max_risk": max(turn_risks) if turn_risks else 0.0,
            "final_risk": turn_risks[-1] if turn_risks else 0.0,
            "detected": detected_at is not None,
            "detected_at": detected_at,
            "turn_risks": turn_risks,
        })
    return results


# ---------------------------------------------------------------------------
# Plotting (optional — graceful degradation if matplotlib not available)
# ---------------------------------------------------------------------------

def plot_risk_comparison(fixed_results, adaptive_results, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Sample up to 10 conversations for the plot
        n = min(10, len(fixed_results))
        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
        if n == 1:
            axes = [axes]

        for i in range(n):
            fr = fixed_results[i]["turn_risks"]
            ar = adaptive_results[i]["turn_risks"]
            turns = range(len(fr))
            axes[i].plot(turns, fr, "b-o", label="Fixed", linewidth=2)
            axes[i].plot(turns, ar, "r-s", label="Adaptive", linewidth=2)
            axes[i].axhline(y=1.65, color="orange", linestyle="--", label="Warn")
            axes[i].axhline(y=2.475, color="red", linestyle="--", label="Block")
            axes[i].set_title(f"Conv {i+1} ({fixed_results[i]['tactic']})", fontsize=9)
            axes[i].set_xlabel("Turn index")
            axes[i].set_ylabel("Risk")
            axes[i].legend(fontsize=7)
            axes[i].grid(True, alpha=0.3)

        fig.suptitle("Risk Progression: Fixed vs. Adaptive Weights", fontsize=13, fontweight="bold")
        plt.tight_layout()
        outpath = out_dir / "risk_comparison.png"
        plt.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outpath}")
    except Exception as e:
        print(f"  [plot] Skipped risk_comparison.png: {e}")


def plot_weight_distribution(adaptive_results, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_alpha, all_beta, all_gamma = [], [], []
        for r in adaptive_results:
            for a, b, g in r.get("pred_weights", []):
                all_alpha.append(a)
                all_beta.append(b)
                all_gamma.append(g)

        if not all_alpha:
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(all_alpha, bins=30, alpha=0.6, label="α (historical)", color="blue")
        ax.hist(all_beta, bins=30, alpha=0.6, label="β (interaction)", color="red")
        ax.hist(all_gamma, bins=30, alpha=0.6, label="γ (pattern)", color="green")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Adaptive Weights α, β, γ")
        ax.legend()
        ax.grid(True, alpha=0.3)
        outpath = out_dir / "weight_distribution.png"
        plt.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outpath}")
    except Exception as e:
        print(f"  [plot] Skipped weight_distribution.png: {e}")


def plot_ablation(drops: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(drops.keys())
        vals = [drops[n] for n in names]
        colors = ["red" if v > 0 else "green" for v in vals]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(names, vals, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("F1 Drop (positive = feature helps)")
        ax.set_title("Feature Importance (Leave-One-Out Ablation)")
        ax.grid(True, axis="x", alpha=0.3)
        outpath = out_dir / "feature_importance.png"
        plt.tight_layout()
        plt.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outpath}")
    except Exception as e:
        print(f"  [plot] Skipped feature_importance.png: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fixed vs Adaptive TCA")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default="tca/models/adaptive_weights.json")
    parser.add_argument("--output-dir", default="results/adaptive")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TCA Evaluation: Fixed vs. Adaptive Contextual Risk Learning")
    print("=" * 60)

    config = TCAConfig(args.config)
    warn_th = config.risk.warn_threshold
    block_th = config.risk.block_threshold
    alpha_f = config.risk.weights.alpha
    beta_f = config.risk.weights.beta
    gamma_f = config.risk.weights.gamma

    # Load dataset
    all_convs = build_dataset_from_csv(config)
    benign_convs = benign_conversations(n=50, seed=args.seed)
    eval_convs = all_convs + benign_convs

    print(f"Eval conversations: {len(eval_convs)} "
          f"({len(all_convs)} attack + {len(benign_convs)} benign)")

    # Load adaptive model
    learner = AdaptiveWeightLearner()
    try:
        learner.load(args.model)
    except FileNotFoundError:
        print(f"\nERROR: Model not found at {args.model}.")
        print("Run:  python3 -m tca.train_adaptive  first.")
        sys.exit(1)

    # --- Fixed simulation ---
    t0 = time.perf_counter()
    fixed_results = simulate_fixed(eval_convs, alpha_f, beta_f, gamma_f, warn_th, block_th)
    fixed_latency_ms = (time.perf_counter() - t0) / max(len(eval_convs), 1) * 1000

    # --- Adaptive simulation ---
    t0 = time.perf_counter()
    adaptive_results = simulate_adaptive(eval_convs, learner, warn_th, block_th)
    adaptive_latency_ms = (time.perf_counter() - t0) / max(len(eval_convs), 1) * 1000

    # Measure per-turn inference latency
    adaptive_per_turn_ms = measure_latency(learner)

    # --- Metrics ---
    fixed_metrics = compute_metrics(fixed_results, warn_th)
    adaptive_metrics = compute_metrics(adaptive_results, warn_th)

    print("\n--- Metrics ---")
    print(f"{'Metric':<35} {'Fixed':>10} {'Adaptive':>10}")
    print("-" * 57)
    for k in ["precision", "recall", "f1", "attack_detection_rate",
              "fpr", "fnr", "avg_detection_delay_turns"]:
        print(f"{k:<35} {str(fixed_metrics[k]):>10} {str(adaptive_metrics[k]):>10}")
    print(f"{'inference_latency_ms/conv':<35} {fixed_latency_ms:>10.3f} {adaptive_latency_ms:>10.3f}")
    print(f"{'adaptive_per_turn_latency_ms':<35} {'N/A':>10} {adaptive_per_turn_ms:>10.4f}")

    # --- Save metrics CSV ---
    metrics_df = pd.DataFrame([
        {"method": "fixed", **fixed_metrics,
         "inference_latency_ms_per_conv": round(fixed_latency_ms, 4)},
        {"method": "adaptive", **adaptive_metrics,
         "inference_latency_ms_per_conv": round(adaptive_latency_ms, 4),
         "per_turn_latency_ms": round(adaptive_per_turn_ms, 4)},
    ])
    metrics_path = out_dir / "metrics_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Saved: {metrics_path}")

    # --- Save risk progression CSV ---
    risk_rows = []
    for ci, (fr, ar) in enumerate(zip(fixed_results, adaptive_results)):
        for t, (f_risk, a_risk) in enumerate(zip(fr["turn_risks"], ar["turn_risks"])):
            risk_rows.append({
                "conv_idx": ci,
                "tactic": fr["tactic"],
                "is_attack": fr["is_attack"],
                "turn": t,
                "fixed_risk": round(f_risk, 4),
                "adaptive_risk": round(a_risk, 4),
            })
    risk_df = pd.DataFrame(risk_rows)
    risk_path = out_dir / "risk_progression.csv"
    risk_df.to_csv(risk_path, index=False)
    print(f"  Saved: {risk_path}")

    # --- Save detection delay CSV ---
    delay_rows = []
    for ci, (fr, ar) in enumerate(zip(fixed_results, adaptive_results)):
        delay_rows.append({
            "conv_idx": ci,
            "tactic": fr["tactic"],
            "is_attack": fr["is_attack"],
            "n_turns": fr["n_turns"],
            "fixed_detected": fr["detected"],
            "fixed_detected_at": fr["detected_at"],
            "adaptive_detected": ar["detected"],
            "adaptive_detected_at": ar["detected_at"],
        })
    delay_df = pd.DataFrame(delay_rows)
    delay_path = out_dir / "detection_delay.csv"
    delay_df.to_csv(delay_path, index=False)
    print(f"  Saved: {delay_path}")

    # --- Ablation ---
    if args.ablation:
        print("\nRunning leave-one-out ablation study …")
        attack_convs_only = [r for r in eval_convs if r["is_attack"]]
        drops = ablation_study(attack_convs_only, learner, warn_th)
        abl_df = pd.DataFrame([{"feature": k, "f1_drop": v} for k, v in drops.items()])
        abl_path = out_dir / "ablation_results.csv"
        abl_df.to_csv(abl_path, index=False)
        print(f"  Saved: {abl_path}")
        print("\nAblation results:")
        for row in abl_df.sort_values("f1_drop", ascending=False).itertuples():
            print(f"  {row.feature:<30} drop={row.f1_drop:+.4f}")
        plot_ablation(drops, out_dir)

    # --- Plots ---
    print("\nGenerating plots …")
    attack_fixed = [r for r in fixed_results if r["is_attack"]]
    attack_adaptive = [r for r in adaptive_results if r["is_attack"]]
    plot_risk_comparison(attack_fixed, attack_adaptive, out_dir)
    plot_weight_distribution(adaptive_results, out_dir)

    print(f"\nAll outputs in: {out_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
