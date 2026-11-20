"""
tca/train_zn_classifier.py
==========================
Training Pipeline for Prompt Injection & Jailbreak Classifier on ZN Benchmark.

Usage:
    PYTHONPATH=.:venv/lib/python3.12/site-packages python3 tca/train_zn_classifier.py \
        --data dataset/zn_prompt_injection_bench.csv \
        --output tca/models/zn_prompt_injection_model.json \
        --epochs 20 \
        --lr 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tca.zn_classifier import ZNPromptInjectionClassifier


def train_and_evaluate(
    data_path: str = "dataset/zn_prompt_injection_bench.csv",
    output_model_path: str = "tca/models/zn_prompt_injection_model.json",
    results_dir: str = "results/zn_bench",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 0.05,
    reg: float = 1e-4,
    max_features: int = 5000,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    print("=" * 70)
    print("  TRAINING PIPELINE: ZN PROMPT INJECTION CLASSIFIER")
    print("  Corpus: usezn/zn-prompt-injection-bench (23,699 samples)")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1/4] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    df = pd.read_csv(data_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"  * Train Split:      {len(train_df):,} samples ({train_df['label'].value_counts().to_dict()})")
    print(f"  * Validation Split: {len(val_df):,} samples ({val_df['label'].value_counts().to_dict()})")
    print(f"  * Test Split:       {len(test_df):,} samples ({test_df['label'].value_counts().to_dict()})")

    train_texts = train_df["text"].fillna("").tolist()
    train_labels = (train_df["label"] == "ataque").astype(int).values

    val_texts = val_df["text"].fillna("").tolist()
    val_labels = (val_df["label"] == "ataque").astype(int).values

    test_texts = test_df["text"].fillna("").tolist()
    test_labels = (test_df["label"] == "ataque").astype(int).values

    # 2. Train Model
    print(f"\n[2/4] Training ZNPromptInjectionClassifier ({epochs} epochs, lr={lr}, batch_size={batch_size})...")
    clf = ZNPromptInjectionClassifier(
        max_features=max_features,
        min_df=4,
        reg=reg,
        lr=lr,
        threshold=threshold,
    )

    t_train_start = time.time()
    history = clf.fit(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=5,
        verbose=True,
    )
    train_duration = time.time() - t_train_start
    print(f"  Training finished in {train_duration:.2f} seconds.")

    # 3. Save Model Weights
    print(f"\n[3/4] Saving model weights to: {output_model_path}")
    clf.save(output_model_path)
    print(f"  Model successfully saved ({os.path.getsize(output_model_path) / 1024:.1f} KB).")

    # 4. Comprehensive Evaluation on Test Set (3,510 samples)
    print("\n[4/4] Evaluating on Held-Out Test Set (3,510 samples)...")
    t_test_start = time.time()
    test_probs = clf.predict_proba(test_texts)
    test_preds = (test_probs >= threshold).astype(int)
    test_duration = time.time() - t_test_start

    tp = int(np.sum((test_labels == 1) & (test_preds == 1)))
    fp = int(np.sum((test_labels == 0) & (test_preds == 1)))
    tn = int(np.sum((test_labels == 0) & (test_preds == 0)))
    fn = int(np.sum((test_labels == 1) & (test_preds == 0)))

    total = len(test_labels)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    # Per-family breakdown
    test_df["pred"] = test_preds
    family_metrics = {}
    for fam in test_df["family"].unique():
        sub = test_df[test_df["family"] == fam]
        sub_y = (sub["label"] == "ataque").astype(int).values
        sub_pred = sub["pred"].values
        is_attack_fam = (sub_y[0] == 1)

        if is_attack_fam:
            fam_recall = float(np.mean(sub_pred == 1))
            family_metrics[fam] = {
                "type": "attack",
                "count": len(sub),
                "recall": round(fam_recall * 100, 2),
            }
        else:
            fam_fpr = float(np.mean(sub_pred == 1))
            family_metrics[fam] = {
                "type": "benign",
                "count": len(sub),
                "fpr": round(fam_fpr * 100, 2),
                "utility": round((1.0 - fam_fpr) * 100, 2),
            }

    top_features = clf.get_top_features(top_n=12)

    results = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_duration_seconds": round(train_duration, 2),
        "test_duration_seconds": round(test_duration, 3),
        "inference_ms_per_sample": round(test_duration / total * 1000, 3),
        "throughput_samples_per_sec": round(total / test_duration, 1),
        "test_metrics": {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "specificity": round(spec * 100, 2),
            "fpr": round(fpr * 100, 2),
            "f1_score": round(f1, 4),
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        },
        "family_metrics": family_metrics,
        "history": history,
        "top_features": top_features,
    }

    # Save results to JSON
    os.makedirs(results_dir, exist_ok=True)
    res_path = os.path.join(results_dir, "training_metrics.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("  TEST EVALUATION RESULTS (AFTER TRAINING)")
    print("=" * 70)
    print(f"  * Accuracy:     {acc*100:6.2f}%")
    print(f"  * Precision:    {prec*100:6.2f}%")
    print(f"  * Recall:       {rec*100:6.2f}%")
    print(f"  * Specificity:  {spec*100:6.2f}%")
    print(f"  * FPR:          {fpr*100:6.2f}%")
    print(f"  * F1-Score:     {f1:6.4f}")
    print(f"  * Latency:      {test_duration/total*1000:.3f} ms / sample ({total/test_duration:.1f} samples/sec)")
    print("-" * 70)
    print("  ATTACK FAMILY DETECTION RECALL (AFTER TRAINING):")
    for fam, m in family_metrics.items():
        if m["type"] == "attack":
            print(f"    - {fam:<22} (N={m['count']:<4}): Recall = {m['recall']:6.2f}%")
    print("-" * 70)
    print("  BENIGN FAMILY FALSE POSITIVE RATE (AFTER TRAINING):")
    for fam, m in family_metrics.items():
        if m["type"] == "benign":
            print(f"    - {fam:<22} (N={m['count']:<4}): FPR = {m['fpr']:6.2f}% (Utility = {m['utility']:6.2f}%)")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Prompt Injection Classifier on ZN Benchmark")
    parser.add_argument("--data", default="dataset/zn_prompt_injection_bench.csv", help="Path to CSV dataset")
    parser.add_argument("--output", default="tca/models/zn_prompt_injection_model.json", help="Path to output model JSON")
    parser.add_argument("--results-dir", default="results/zn_bench", help="Results directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--max-features", type=int, default=5000, help="Vocabulary size")
    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data,
        output_model_path=args.output,
        results_dir=args.results_dir,
        epochs=args.epochs,
        lr=args.lr,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
