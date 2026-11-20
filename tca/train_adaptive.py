"""
train_adaptive.py
==================
Training pipeline for Adaptive Contextual Risk Learning (Novelty 1).

Usage
-----
  cd /path/to/RM-MultiTurnAttackDefense
  python3 -m tca.train_adaptive [--config config/config.yaml] [--output tca/models/adaptive_weights.json]
                                 [--epochs 200] [--lr 0.001] [--val-split 0.15] [--test-split 0.15]
                                 [--seed 42] [--verbose]

What this script does
---------------------
1. Loads the HarmBench dataset.
2. For each conversation row, simulates the TCA turn-by-turn pipeline using the
   MOCK LLM (so no API key is required) to generate realistic interaction_risk,
   pattern_risk, intent_shift, and domain_shift signals.
3. Builds feature vectors via context_feature_extractor.extract_features().
4. Derives heuristic α/β/γ supervision targets via build_training_dataset().
5. Splits by *conversation* (not turn) to prevent label leakage.
6. Trains the AdaptiveWeightLearner MLP.
7. Saves the model to --output.
8. Prints train/val/test loss.

Limitations (documented)
------------------------
- Because real-time LLM labels are expensive, supervision is generated from the
  mock LLM + heuristic weight targets.  The targets are principled proxies, not
  ground-truth optimal weights.
- When you have real LLM runs with known α/β/γ annotations, replace the
  `build_training_dataset` call with your annotated data.
"""

from __future__ import annotations
import argparse
import json
import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Make sure project root is on the path when run as a module
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from config.tca_config import TCAConfig
from dataset.dataset_manager import DatasetManager
from tca.risk_calculator import RiskCalculator
from tca.adaptive_weight_learner import AdaptiveWeightLearner, build_training_dataset
from tca.context_feature_extractor import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Simulate conversation turns using the mock LLM pattern
# ---------------------------------------------------------------------------

def _mock_turn_response(turn_index: int, is_attack: bool):
    """Generate a plausible mock LLM analysis response for one turn."""
    if is_attack:
        final_risk = min(5, 2 + turn_index)
        shift = turn_index >= 1
        domain = turn_index >= 1
        prohibited = turn_index >= 2
    else:
        final_risk = 1
        shift = False
        domain = False
        prohibited = False

    return {
        "intent_shift": {
            "shift_detected": shift,
            "risk_level": final_risk if shift else 1,
        },
        "patterns": {
            "language_change": {"detected": False},
            "domain_shift": {"detected": domain},
            "time_sensitivity": {"detected": False},
            "prohibited_content": {"detected": prohibited},
        },
        "overall_progression_summary": {
            "initial_risk_level": max(1, final_risk - 1),
            "final_risk_level": final_risk,
        },
    }


PATTERN_WEIGHTS = {
    "language_change": 0.2,
    "domain_shift": 0.3,
    "time_sensitivity": 0.2,
    "prohibited_content": 0.3,
}


def _compute_pattern_risk(patterns: dict) -> float:
    score = 0.0
    for name, details in patterns.items():
        if details.get("detected", False):
            score += PATTERN_WEIGHTS.get(name, 0.0)
    return score


def build_dataset_from_csv(config: TCAConfig) -> list[dict]:
    """
    Build conversation records from the HarmBench CSV for training.

    Returns
    -------
    List of ConversationRecord dicts compatible with build_training_dataset().
    """
    dm = DatasetManager(config)
    df = dm.load_data()

    # Attack tactics (all tactics in this dataset represent attack attempts)
    attack_tactics = set(df["tactic"].unique())
    # Heuristic: 'Direct Request' is the most overt attack; others are more covert
    high_risk_tactics = {"Obfuscation", "Injection", "Hidden Intention Streamline"}

    conversations = []
    for _, row in df.iterrows():
        pairs = dm.extract_conversation_pairs(row)
        if len(pairs) < 2:
            continue

        tactic = str(row.get("tactic", ""))
        is_attack = tactic in attack_tactics  # all tactics are attacks here

        turns = []
        prev_human = str(pairs[0][1]) if pairs else ""
        for i in range(1, len(pairs)):
            prev_pair = pairs[i - 1]
            cur_pair = pairs[i]
            cur_human = str(cur_pair[1])

            mock = _mock_turn_response(turn_index=i, is_attack=is_attack)
            ir = float(mock["overall_progression_summary"]["final_risk_level"])
            pr = _compute_pattern_risk(mock["patterns"])

            turns.append({
                "interaction_risk": ir,
                "pattern_risk": pr,
                "patterns": mock["patterns"],
                "intent_shift_detected": mock["intent_shift"]["shift_detected"],
                "intent_shift_risk_level": mock["intent_shift"]["risk_level"],
                "current_human_turn": cur_human,
                "prev_human_turn": prev_human,
            })
            prev_human = cur_human

        if turns:
            conversations.append({
                "is_attack": is_attack,
                "tactic": tactic,
                "turns": turns,
            })

    print(f"[train_adaptive] Built {len(conversations)} conversations, "
          f"{sum(len(c['turns']) for c in conversations)} turns total.")
    return conversations


def conversation_split(
    conversations: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Split conversations (not turns) into train/val/test."""
    rng = np.random.default_rng(seed)
    n = len(conversations)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    test_idx = idx[:n_test]
    val_idx = idx[n_test: n_test + n_val]
    train_idx = idx[n_test + n_val:]
    train = [conversations[i] for i in train_idx]
    val = [conversations[i] for i in val_idx]
    test = [conversations[i] for i in test_idx]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Train Adaptive Contextual Risk Learning model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="tca/models/adaptive_weights.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Adaptive Contextual Risk Learning — Training Pipeline")
    print("=" * 60)
    print(f"Config      : {args.config}")
    print(f"Output      : {args.output}")
    print(f"Epochs      : {args.epochs}")
    print(f"LR          : {args.lr}")
    print(f"Seed        : {args.seed}")
    print()

    config = TCAConfig(args.config)
    conversations = build_dataset_from_csv(config)

    # Split by conversation
    train_convs, val_convs, test_convs = conversation_split(
        conversations,
        val_ratio=args.val_split,
        test_ratio=args.test_split,
        seed=args.seed,
    )
    print(f"Split  — train: {len(train_convs)}, val: {len(val_convs)}, test: {len(test_convs)}")

    # Build feature matrices
    X_train, Y_train, _ = build_training_dataset(train_convs)
    X_val, Y_val, _ = build_training_dataset(val_convs)
    X_test, Y_test, _ = build_training_dataset(test_convs)
    print(f"Turns  — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"Feature names: {FEATURE_NAMES}")
    print()

    # Train
    learner = AdaptiveWeightLearner(
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        verbose=args.verbose or True,
    )
    learner.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val)

    # Evaluate
    pred_test = learner.predict_weights_batch(X_test)
    test_loss = float(np.mean((pred_test - Y_test) ** 2))
    print(f"\nTest MSE  : {test_loss:.6f}")
    print(f"Mean predicted weights — α: {pred_test[:,0].mean():.4f}, "
          f"β: {pred_test[:,1].mean():.4f}, γ: {pred_test[:,2].mean():.4f}")
    print(f"Mean target weights    — α: {Y_test[:,0].mean():.4f}, "
          f"β: {Y_test[:,1].mean():.4f}, γ: {Y_test[:,2].mean():.4f}")

    # Save
    learner.save(args.output)
    print(f"\nModel saved → {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
