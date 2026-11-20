"""
tca/evaluate_zn_bench.py
=========================
Comprehensive Evaluation & Analytics Pipeline for the ZN Prompt-Injection Benchmark
(usezn/zn-prompt-injection-bench - 23,699 samples).

Features:
1. Deep Corpus Analytics:
   - Split distribution (train, val, test)
   - Class balance (ataque vs benigno)
   - Family taxonomy breakdown (plain, red_team, corto, largo_real, conversacional, etc.)
   - Source provenance distribution (18 upstream sources)
   - Multilingual distribution (English + 8 translation locales)
   - Character and token length statistics
2. Benchmark Security Evaluation:
   - Compares:
     * Surface Heuristic Baseline (Regex/Keyword)
     * Novelty 1: Active Counterfactual Trajectory Prober (CTP)
     * Novelty 2: Cross-Turn Symbolic Taint Virtualization (CT-STV)
     * Novelty 3: Differential System-State Divergence (D3S)
     * Full Multi-Layer TCA Defense Architecture
   - Measures:
     * Precision, Recall, Specificity, F1-Score, Overall Accuracy
     * Granular Attack Detection Recall per attack family (plain, red_team, paraphrase, multilingual_attack)
     * Granular False Positive Rate (FPR) per benign family (corto, largo_real, conversacional, meta_texto, documentacion)
     * Latency & Throughput (ms/sample, samples/sec)
3. Multi-Turn Conversational False-Positive Stress Test:
   - Evaluates consecutive turns from the 'conversacional' family (hh_rlhf) to verify stability over dialogues.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.tca_config import TCAConfig
from dataset.dataset_manager import DatasetManager
from tca.counterfactual_prober import CounterfactualTrajectoryProber
from tca.taint_virtualizer import CrossTurnTaintVirtualizer
from tca.system_state_divergence import DifferentialStateDivergence
from tca.security_engine import SecurityDecisionEngine
from tca.zn_classifier import ZNPromptInjectionClassifier


class ZNBenchmarkEvaluator:
    """Evaluates TCA defense mechanisms and compiles detailed analytics on the ZN benchmark."""

    # Baseline surface pattern signatures
    BASELINE_PATTERNS = [
        r"\b(ignore\s+(all\s+)?previous\s+instructions|disregard\s+(all\s+)?guidelines)\b",
        r"\b(bypass\s+safety|jailbreak|dan\s+mode|unrestricted\s+mode|developer\s+mode)\b",
        r"\b(reveal\s+system\s+prompt|print\s+instructions|exfiltrate|dump\s+passwords?)\b",
        r"\b\[resultado de herramienta\]\s*:\s*.*?(ignore|bypass|override)\b",
    ]

    def __init__(self, config_path: str = "config/config.yaml", model_path: str = "tca/models/zn_prompt_injection_model.json"):
        self.config = TCAConfig(config_path)
        self.dataset_manager = DatasetManager(self.config)
        self.ctp = CounterfactualTrajectoryProber(config=self.config)
        self.ctstv = CrossTurnTaintVirtualizer(config=self.config)
        self.d3s = DifferentialStateDivergence(config=self.config)
        self.engine = SecurityDecisionEngine(config=self.config)
        
        # Load trained machine learning model if available
        self.trained_clf = None
        if os.path.exists(model_path):
            try:
                self.trained_clf = ZNPromptInjectionClassifier.load(model_path)
            except Exception as e:
                print(f"[Warning] Could not load trained model from {model_path}: {e}")

        import re
        self._compiled_baseline = [re.compile(p, re.IGNORECASE) for p in self.BASELINE_PATTERNS]

    def run_corpus_analytics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates exhaustive statistics and distribution summaries for the benchmark."""
        total_rows = len(df)
        splits = df["split"].value_counts().to_dict()
        labels = df["label"].value_counts().to_dict()

        # Family distribution with labels
        family_df = df.groupby(["family", "label"]).size().unstack(fill_value=0)
        family_summary = {}
        for family, row in family_df.iterrows():
            family_summary[family] = {
                "ataque": int(row.get("ataque", 0)),
                "benigno": int(row.get("benigno", 0)),
                "total": int(row.sum()),
                "share_pct": round(float(row.sum() / total_rows * 100), 2),
            }

        # Source provenance
        source_summary = df["source"].value_counts().to_dict()

        # Language distribution
        lang_summary = df["lang"].fillna("unmarked_en").value_counts().to_dict()

        # Text length analytics
        def get_length_stats(series: pd.Series) -> Dict[str, float]:
            return {
                "min": int(series.min()),
                "max": int(series.max()),
                "mean": round(float(series.mean()), 2),
                "median": float(series.median()),
                "std": round(float(series.std()), 2),
                "p25": float(series.quantile(0.25)),
                "p75": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
            }

        length_analytics = {
            "all": get_length_stats(df["char_len"]),
            "attacks": get_length_stats(df[df["label"] == "ataque"]["char_len"]),
            "benign": get_length_stats(df[df["label"] == "benigno"]["char_len"]),
            "by_family": {
                fam: get_length_stats(df[df["family"] == fam]["char_len"])
                for fam in df["family"].unique()
            },
        }

        return {
            "total_records": total_rows,
            "splits": splits,
            "labels": labels,
            "families": family_summary,
            "sources": source_summary,
            "languages": lang_summary,
            "length_analytics": length_analytics,
        }

    def evaluate_sample_baseline(self, text: str) -> bool:
        """Evaluates text against surface regex baseline."""
        for regex in self._compiled_baseline:
            if regex.search(text):
                return True
        return False

    def evaluate_sample_tca(self, text: str) -> Dict[str, Any]:
        """Runs the complete TCA defense layers on a sample."""
        self.ctp.reset()
        self.ctstv.reset()
        self.d3s.reset()

        c_res = self.ctp.evaluate_turn(turn_id=1, raw_text=text)
        t_res = self.ctstv.evaluate_turn(turn_id=1, user_text=text)
        d_res = self.d3s.evaluate_turn(turn_id=1, user_text=text)

        # Baseline prober check
        has_baseline = self.evaluate_sample_baseline(text)

        # CTP flag: high framing + hazardous directive, or preemptive block
        ctp_flag = c_res.trigger_preemptive_block or c_res.composite_ctp_risk >= 0.40

        # CT-STV flag: split injection or prompt injection violation
        ctstv_flag = t_res.trigger_preemptive_block or t_res.is_split_injection or (t_res.ssim_score >= 0.50)

        # D3S flag: instruction subversion or KL divergence
        d3s_flag = d_res.trigger_preemptive_block or d_res.composite_d3s_risk >= 0.40

        # Layered decision engine
        decision = self.engine.evaluate_risk(
            progressive_risk=max(c_res.scaled_ctp_risk, d_res.scaled_d3s_risk, t_res.scaled_ctstv_risk),
            historical_risk=0.0,
            ctp_preemptive_block=c_res.trigger_preemptive_block,
            ctp_reasoning=c_res.explanation,
            ctstv_preemptive_block=t_res.trigger_preemptive_block,
            ctstv_reasoning=t_res.explanation,
            d3s_preemptive_block=d_res.trigger_preemptive_block,
            d3s_reasoning=d_res.explanation,
        )

        # Unified defense flag (Heuristic baseline + Temporal novelties)
        tca_unified_block = (decision["decision"] == "block") or ctp_flag or ctstv_flag or d3s_flag or has_baseline

        # Trained Machine Learning Classifier prediction (Trained on 16,888 ZN samples)
        trained_ml_prob = 0.0
        trained_ml_flag = False
        if self.trained_clf is not None and self.trained_clf.is_fitted:
            p_val = self.trained_clf.predict_proba([text])
            trained_ml_prob = float(p_val[0])
            trained_ml_flag = bool(trained_ml_prob >= self.trained_clf.threshold)

        # Layered Architecture: Single-turn ML Guardrail + Temporal Context Defenses
        layered_defense_block = trained_ml_flag or (decision["decision"] == "block") or ctp_flag or ctstv_flag or d3s_flag

        return {
            "baseline_flag": has_baseline,
            "ctp_flag": ctp_flag,
            "ctstv_flag": ctstv_flag,
            "d3s_flag": d3s_flag,
            "tca_unified_block": tca_unified_block,
            "trained_ml_flag": trained_ml_flag,
            "trained_ml_prob": trained_ml_prob,
            "layered_defense_block": layered_defense_block,
            "ctp_fdr": c_res.framing_dependency_ratio,
            "ctp_risk": c_res.composite_ctp_risk,
            "d3s_kl": d_res.kl_divergence,
            "d3s_risk": d_res.composite_d3s_risk,
        }

    def run_benchmark_evaluation(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates defense detectors across the test split and compiles performance metrics."""
        t0 = time.time()
        records = []

        for idx, row in test_df.iterrows():
            text = str(row["text"])
            is_attack = (row["label"] == "ataque")
            family = row["family"]

            eval_res = self.evaluate_sample_tca(text)
            records.append({
                "label": row["label"],
                "is_attack": is_attack,
                "family": family,
                "lang": str(row.get("lang", "")),
                "source": row.get("source", ""),
                **eval_res,
            })

        total_time = time.time() - t0
        eval_df = pd.DataFrame(records)

        def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))

            total = len(y_true)
            acc = (tp + tn) / total if total > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            f1 = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0.0

            return {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(recall), 4),
                "specificity": round(float(spec), 4),
                "fpr": round(float(fpr), 4),
                "f1_score": round(float(f1), 4),
            }

        y_true = eval_df["is_attack"].astype(int).values

        # Overall method comparisons
        baseline_metrics = compute_metrics(y_true, eval_df["baseline_flag"].astype(int).values)
        ctp_metrics = compute_metrics(y_true, eval_df["ctp_flag"].astype(int).values)
        ctstv_metrics = compute_metrics(y_true, eval_df["ctstv_flag"].astype(int).values)
        d3s_metrics = compute_metrics(y_true, eval_df["d3s_flag"].astype(int).values)
        unified_metrics = compute_metrics(y_true, eval_df["tca_unified_block"].astype(int).values)
        trained_ml_metrics = compute_metrics(y_true, eval_df["trained_ml_flag"].astype(int).values)
        layered_metrics = compute_metrics(y_true, eval_df["layered_defense_block"].astype(int).values)

        # Granular Breakdown by Attack Family
        attack_families = ["plain", "red_team", "paraphrase", "multilingual_attack"]
        attack_breakdown = {}
        for fam in attack_families:
            fam_sub = eval_df[eval_df["family"] == fam]
            if len(fam_sub) > 0:
                attack_breakdown[fam] = {
                    "count": len(fam_sub),
                    "baseline_recall": round(float(fam_sub["baseline_flag"].mean() * 100), 2),
                    "ctp_recall": round(float(fam_sub["ctp_flag"].mean() * 100), 2),
                    "trained_ml_recall": round(float(fam_sub["trained_ml_flag"].mean() * 100), 2),
                    "layered_recall": round(float(fam_sub["layered_defense_block"].mean() * 100), 2),
                }

        # Granular Breakdown by Benign Family (FPR)
        benign_families = ["corto", "largo_real", "conversacional", "meta_texto", "documentacion"]
        benign_breakdown = {}
        for fam in benign_families:
            fam_sub = eval_df[eval_df["family"] == fam]
            if len(fam_sub) > 0:
                benign_breakdown[fam] = {
                    "count": len(fam_sub),
                    "baseline_fpr": round(float(fam_sub["baseline_flag"].mean() * 100), 2),
                    "ctp_fpr": round(float(fam_sub["ctp_flag"].mean() * 100), 2),
                    "trained_ml_fpr": round(float(fam_sub["trained_ml_flag"].mean() * 100), 2),
                    "layered_fpr": round(float(fam_sub["layered_defense_block"].mean() * 100), 2),
                }

        throughput = {
            "total_samples": len(test_df),
            "total_elapsed_seconds": round(float(total_time), 3),
            "ms_per_sample": round(float(total_time / len(test_df) * 1000), 3),
            "samples_per_second": round(float(len(test_df) / total_time), 1),
        }

        return {
            "throughput": throughput,
            "overall_metrics": {
                "Surface_Baseline": baseline_metrics,
                "CTP_Novelty1": ctp_metrics,
                "CTSTV_Novelty2": ctstv_metrics,
                "D3S_Novelty3": d3s_metrics,
                "TCA_Unified_Defense": unified_metrics,
                "Trained_ML_Classifier": trained_ml_metrics,
                "TCA_Layered_with_ML": layered_metrics,
            },
            "attack_family_breakdown": attack_breakdown,
            "benign_family_breakdown": benign_breakdown,
        }

    def run_multiturn_conversational_fpr_test(self, limit: int = 100) -> Dict[str, Any]:
        """Evaluates consecutive multi-turn conversational sequences from hh_rlhf."""
        convs = self.dataset_manager.extract_zn_conversations(split="test", limit=limit)
        total_convs = len(convs)
        total_turns = sum(c["turn_count"] for c in convs)

        false_alarms = 0
        turn_results = []

        for conv in convs:
            self.ctp.reset()
            self.ctstv.reset()
            self.d3s.reset()
            history = []

            for turn_idx, turn in enumerate(conv["turns"], start=1):
                text = turn["text"]
                if turn["role"] == "human":
                    c_res = self.ctp.evaluate_turn(turn_idx, text, conversation_history=history)
                    t_res = self.ctstv.evaluate_turn(turn_idx, text)
                    d_res = self.d3s.evaluate_turn(turn_idx, text, conversation_history=history)

                    if c_res.trigger_preemptive_block or t_res.trigger_preemptive_block or d_res.trigger_preemptive_block:
                        false_alarms += 1
                        turn_results.append({
                            "conv_id": conv["id"],
                            "turn_idx": turn_idx,
                            "text": text[:80],
                            "alarm": True,
                        })
                    history.append(text)

        fpr_conv = false_alarms / total_convs if total_convs > 0 else 0.0

        return {
            "tested_conversations": total_convs,
            "total_turns": total_turns,
            "false_alarms": false_alarms,
            "conversational_fpr": round(float(fpr_conv * 100), 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ZN Prompt Injection Benchmark with TCA")
    parser.add_argument("--config", default="config/config.yaml", help="Path to TCA configuration YAML")
    parser.add_argument("--output-dir", default="results/zn_bench", help="Directory to save benchmark results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("  ZN Prompt-Injection Benchmark (usezn/zn-prompt-injection-bench)")
    print("  Comprehensive Analytics & TCA Defense Evaluation")
    print("==================================================================")

    evaluator = ZNBenchmarkEvaluator(config_path=args.config)

    # 1. Load full benchmark
    print("\n[1/3] Loading Full Benchmark Corpus...")
    df_all = evaluator.dataset_manager.load_zn_benchmark()
    print(f"Loaded {len(df_all):,} records.")

    # 2. Run Analytics
    print("\n[2/3] Compiling Exhaustive Dataset Analytics...")
    analytics = evaluator.run_corpus_analytics(df_all)
    analytics_path = output_dir / "zn_analytics.json"
    with open(analytics_path, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2)
    print(f"Analytics saved to {analytics_path}")

    # 3. Benchmark Evaluation on Test Split (3,510 samples)
    print("\n[3/3] Running TCA Defense Evaluation across Test Split (3,510 samples)...")
    df_test = df_all[df_all["split"] == "test"].copy()
    test_results = evaluator.run_benchmark_evaluation(df_test)

    # Multi-turn conversational FPR test
    print("\nRunning Multi-Turn Conversational FPR Stress Test on hh_rlhf dialogues...")
    conv_fpr = evaluator.run_multiturn_conversational_fpr_test(limit=200)
    test_results["multiturn_conversational_fpr_test"] = conv_fpr

    results_path = output_dir / "zn_evaluation_metrics.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"Benchmark results saved to {results_path}")

    # Print summary tables to stdout
    print("\n" + "=" * 70)
    print("  BENCHMARK EVALUATION SUMMARY (Test Split: 3,510 Samples)")
    print("=" * 70)
    print(f"{'Defense Module':<24} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'FPR':<6}")
    print("-" * 70)
    for name, m in test_results["overall_metrics"].items():
        print(f"{name:<24} | {m['accuracy']*100:6.2f}%   | {m['precision']*100:6.2f}%   | {m['recall']*100:5.2f}%  | {m['f1_score']:6.4f}   | {m['fpr']*100:4.2f}%")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("  ATTACK DETECTION RECALL BY FAMILY (1,805 Attacks)")
    print("=" * 70)
    for fam, stats in test_results["attack_family_breakdown"].items():
        print(f"  * {fam:<22} (N={stats['count']:<4}): Trained ML = {stats.get('trained_ml_recall', 0.0):6.2f}% | Baseline = {stats['baseline_recall']:6.2f}%")

    print("\n" + "=" * 70)
    print("  BENIGN FALSE POSITIVE RATE (FPR) BY FAMILY (1,705 Benign)")
    print("=" * 70)
    for fam, stats in test_results["benign_family_breakdown"].items():
        print(f"  * {fam:<22} (N={stats['count']:<4}): Trained ML FPR = {stats.get('trained_ml_fpr', 0.0):6.2f}% | Baseline FPR = {stats['baseline_fpr']:6.2f}%")

    print("\n" + "=" * 70)
    print("  THROUGHPUT & LATENCY OVERHEAD")
    print("=" * 70)
    tp = test_results["throughput"]
    print(f"  * Total Samples: {tp['total_samples']:,}")
    print(f"  * Elapsed Time: {tp['total_elapsed_seconds']}s")
    print(f"  * Mean Latency: {tp['ms_per_sample']} ms / sample")
    print(f"  * Throughput:   {tp['samples_per_second']:.1f} samples / sec")
    print("=" * 70)


if __name__ == "__main__":
    main()
