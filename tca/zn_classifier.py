"""
tca/zn_classifier.py
====================
Trained Machine Learning Classifier for Prompt Injection and Jailbreak Detection.
Trained on the ZN Prompt-Injection Evaluation Corpus (usezn/zn-prompt-injection-bench).

Features:
- Pure NumPy implementation to ensure zero C-extension / SciPy ABI dependency issues.
- Sublinear TF-IDF + n-gram vectorizer with document-frequency filtering.
- Structural & security heuristic feature augmentations (delimiters, brackets, tool output markers).
- Logistic / sigmoid regression with Adam optimizer, L2 regularization, and threshold calibration.
- Full serialization (to/from JSON) for instant deployment and reproducibility.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ZNPromptInjectionClassifier:
    """Pure-NumPy trained machine learning classifier for prompt injection detection."""

    # Key structural patterns indicative of prompt injection/jailbreaking attempts
    STRUCTURAL_PATTERNS = [
        re.compile(r"\[resultado de herramienta\]", re.IGNORECASE),
        re.compile(r"\b(ignore|disregard)\s+(all\s+)?(previous|prior)\s+(instructions|directives|prompts)\b", re.IGNORECASE),
        re.compile(r"\b(system\s+prompt|developer\s+mode|dan\s+mode|jailbreak|unrestricted)\b", re.IGNORECASE),
        re.compile(r"\b(bypass\s+filters?|override\s+safety|without\s+limitations?)\b", re.IGNORECASE),
        re.compile(r"```[a-z]*\n.*?\n```", re.DOTALL),
        re.compile(r"(\bformat\b|\bjson\b|\bschema\b)\s*:\s*\{", re.IGNORECASE),
    ]

    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 4,
        reg: float = 1e-4,
        lr: float = 0.05,
        threshold: float = 0.50,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.reg = reg
        self.lr = lr
        self.threshold = threshold

        # Fitted state
        self.vocab: List[str] = []
        self.word2idx: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([], dtype=np.float32)
        self.weights: np.ndarray = np.array([], dtype=np.float32)
        self.bias: float = 0.0
        self.is_fitted: bool = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercases and extracts clean word tokens."""
        text = str(text).lower()
        return re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text)

    def _extract_structural_features(self, text: str) -> np.ndarray:
        """Extracts dense structural and statistical features for a single text."""
        t = str(text)
        n_chars = max(1, len(t))
        uppercase_ratio = sum(1 for c in t if c.isupper()) / float(n_chars)
        punct_count = sum(1 for c in t if c in "!?:;{}[]`'\"")
        punct_ratio = punct_count / float(n_chars)

        pattern_matches = [
            1.0 if pat.search(t) else 0.0 for pat in self.STRUCTURAL_PATTERNS
        ]

        feats = [
            math.log1p(n_chars) / 10.0,
            uppercase_ratio,
            punct_ratio,
            *pattern_matches,
        ]
        return np.array(feats, dtype=np.float32)

    def _build_features(self, texts: List[str], fit_vocab: bool = False) -> np.ndarray:
        """Vectorizes a list of texts into a dense feature matrix."""
        n_samples = len(texts)

        if fit_vocab:
            df_counter = Counter()
            for text in texts:
                tokens = set(self.tokenize(text))
                df_counter.update(tokens)

            # Select top vocabulary words meeting min_df
            filtered_vocab = [w for w, c in df_counter.most_common(self.max_features) if c >= self.min_df]
            self.vocab = filtered_vocab
            self.word2idx = {w: i for i, w in enumerate(self.vocab)}

            # Smooth IDF
            n_docs = float(n_samples)
            doc_freqs = np.array([df_counter[w] for w in self.vocab], dtype=np.float32)
            self.idf = np.log((1.0 + n_docs) / (1.0 + doc_freqs)) + 1.0

        n_vocab = len(self.vocab)
        n_struct = 3 + len(self.STRUCTURAL_PATTERNS)
        total_dim = n_vocab + n_struct

        X = np.zeros((n_samples, total_dim), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = self.tokenize(text)
            counts = Counter(tokens)
            
            # TF-IDF sparse embedding
            for w, cnt in counts.items():
                if w in self.word2idx:
                    j = self.word2idx[w]
                    X[i, j] = np.log1p(cnt) * self.idf[j]

            # Normalize text component to unit norm
            text_norm = np.linalg.norm(X[i, :n_vocab])
            if text_norm > 0:
                X[i, :n_vocab] /= text_norm

            # Append structural features
            X[i, n_vocab:] = self._extract_structural_features(text)

        return X

    def fit(
        self,
        train_texts: List[str],
        train_labels: Union[List[int], np.ndarray],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[Union[List[int], np.ndarray]] = None,
        epochs: int = 20,
        batch_size: int = 256,
        early_stopping_patience: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Trains the classifier on labeled texts using the Adam optimizer with binary cross-entropy.
        """
        if verbose:
            print(f"[ZNClassifier] Building vocabulary and feature matrix ({len(train_texts):,} samples)...")

        X_train = self._build_features(train_texts, fit_vocab=True)
        y_train = np.array(train_labels, dtype=np.float32)

        has_val = val_texts is not None and val_labels is not None
        if has_val:
            X_val = self._build_features(val_texts, fit_vocab=False)
            y_val = np.array(val_labels, dtype=np.float32)

        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0

        # Adam state
        m_W = np.zeros_like(self.weights)
        v_W = np.zeros_like(self.weights)
        m_b = 0.0
        v_b = 0.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        n_batches = int(np.ceil(n_samples / float(batch_size)))
        history = {"train_loss": [], "val_acc": [], "val_f1": []}

        best_val_f1 = -1.0
        best_W = self.weights.copy()
        best_b = self.bias
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(n_samples)
            epoch_loss = 0.0

            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                xb, yb = X_train[idx], y_train[idx]

                # Forward pass
                z = xb @ self.weights + self.bias
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

                # Binary cross-entropy loss
                p_safe = np.clip(p, 1e-7, 1.0 - 1e-7)
                bce = -np.mean(yb * np.log(p_safe) + (1.0 - yb) * np.log(1.0 - p_safe))
                loss = bce + 0.5 * self.reg * float(np.sum(self.weights ** 2))
                epoch_loss += loss * len(xb)

                # Backprop
                dz = (p - yb) / float(len(xb))
                dW = xb.T @ dz + self.reg * self.weights
                db = float(np.sum(dz))

                # Adam parameter update
                t = (epoch - 1) * n_batches + b + 1
                m_W = beta1 * m_W + (1.0 - beta1) * dW
                v_W = beta2 * v_W + (1.0 - beta2) * (dW ** 2)
                m_W_hat = m_W / (1.0 - beta1 ** t)
                v_W_hat = v_W / (1.0 - beta2 ** t)
                self.weights -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + eps)

                m_b = beta1 * m_b + (1.0 - beta1) * db
                v_b = beta2 * v_b + (1.0 - beta2) * (db ** 2)
                m_b_hat = m_b / (1.0 - beta1 ** t)
                v_b_hat = v_b / (1.0 - beta2 ** t)
                self.bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

            mean_train_loss = epoch_loss / float(n_samples)
            history["train_loss"].append(round(mean_train_loss, 4))

            if has_val:
                val_probs = self.predict_proba_features(X_val)
                val_preds = (val_probs >= self.threshold).astype(int)
                val_acc = float(np.mean(val_preds == y_val))

                tp = int(np.sum((y_val == 1) & (val_preds == 1)))
                fp = int(np.sum((y_val == 0) & (val_preds == 1)))
                fn = int(np.sum((y_val == 1) & (val_preds == 0)))
                val_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                val_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                val_f1 = (2 * val_prec * val_rec) / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0.0

                history["val_acc"].append(round(val_acc, 4))
                history["val_f1"].append(round(val_f1, 4))

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_W = self.weights.copy()
                    best_b = self.bias
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                    print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {mean_train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"  Early stopping triggered at epoch {epoch} (best Val F1: {best_val_f1:.4f})")
                    self.weights = best_W
                    self.bias = best_b
                    break

        self.is_fitted = True
        return history

    def predict_proba_features(self, X: np.ndarray) -> np.ndarray:
        """Computes probabilities given a precomputed feature matrix."""
        z = X @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Computes posterior attack probability P(Attack | Text) in [0.0, 1.0]."""
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted or loaded.")
        X = self._build_features(texts, fit_vocab=False)
        return self.predict_proba_features(X)

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> np.ndarray:
        """Predicts binary labels (1 = attack, 0 = benign)."""
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(texts)
        return (probs >= th).astype(int)

    def get_top_features(self, top_n: int = 15) -> Dict[str, List[Tuple[str, float]]]:
        """Returns the most discriminative attack and benign vocabulary tokens."""
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted.")

        n_vocab = len(self.vocab)
        text_weights = self.weights[:n_vocab]
        sorted_indices = np.argsort(text_weights)

        top_benign_indices = sorted_indices[:top_n]
        top_attack_indices = sorted_indices[-top_n:][::-1]

        return {
            "attack_tokens": [(self.vocab[i], round(float(text_weights[i]), 4)) for i in top_attack_indices],
            "benign_tokens": [(self.vocab[i], round(float(text_weights[i]), 4)) for i in top_benign_indices],
        }

    def save(self, filepath: str):
        """Serializes model parameters and vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "reg": self.reg,
            "lr": self.lr,
            "threshold": self.threshold,
            "vocab": self.vocab,
            "idf": self.idf.tolist(),
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "ZNPromptInjectionClassifier":
        """Deserializes and reconstructs a trained model from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        clf = cls(
            max_features=data["max_features"],
            min_df=data["min_df"],
            reg=data["reg"],
            lr=data["lr"],
            threshold=data["threshold"],
        )
        clf.vocab = data["vocab"]
        clf.word2idx = {w: i for i, w in enumerate(clf.vocab)}
        clf.idf = np.array(data["idf"], dtype=np.float32)
        clf.weights = np.array(data["weights"], dtype=np.float32)
        clf.bias = float(data["bias"])
        clf.is_fitted = True
        return clf
