"""
adaptive_weight_learner.py
===========================
Implements Adaptive Contextual Risk Learning (Novelty 1).

Architecture
------------
A lightweight 2-hidden-layer MLP is trained to predict three logits (z_α, z_β, z_γ).
Softmax converts them to non-negative weights that sum to 1:

    α_t, β_t, γ_t = softmax(z_α, z_β, z_γ)

Progressive risk is then:

    R_t = α_t * R_(t-1) + β_t * I_t + γ_t * P_t

Training strategy
-----------------
Because we lack real-time LLM labels for α/β/γ targets directly, we derive a
**heuristic supervision signal** from the data:

For each labeled turn we know:
  - Whether the conversation ultimately led to a blocked/warned/allowed outcome.
  - The pattern of risk over the conversation (escalating, flat, sudden spike).

We use a closed-form optimal-weight heuristic:
  "Given the final outcome label, what weighting would have minimised prediction
   error relative to a threshold boundary?"

This is documented clearly so users know it is a heuristic, not ground truth.

If a user later has real α/β/γ annotations, they can substitute them in
`build_training_dataset()`.

The MLP is implemented in pure NumPy to avoid scipy/sklearn ABI conflicts with the
venv's NumPy 1.26.4.

Public API
----------
AdaptiveWeightLearner
    .fit(X, y_alpha, y_beta, y_gamma)
    .predict_weights(feature_vec)  -> (alpha, beta, gamma)
    .save(path) / .load(path)

build_training_dataset(conversations)  -> (X, y_alpha, y_beta, y_gamma, meta)

Limitations
-----------
- Heuristic labels are proxies; they may not reflect ground-truth optimal weights.
- No GPU acceleration; training is CPU-only.
- Bag-of-words semantic similarity is a lightweight approximation.
- The model assumes stationarity across conversation topics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from tca.context_feature_extractor import extract_features, N_FEATURES, FEATURE_NAMES


# ---------------------------------------------------------------------------
# MLP (pure NumPy, no sklearn dependency)
# ---------------------------------------------------------------------------

class _MLP:
    """Two-hidden-layer MLP with ReLU activations and softmax output (3 units).

    Parameters
    ----------
    n_in : int
        Input dimension (number of features).
    hidden : tuple[int, int]
        Number of units in each hidden layer.
    lr : float
        Learning rate for Adam optimiser.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_in: int = N_FEATURES,
        hidden: Tuple[int, int] = (32, 16),
        lr: float = 1e-3,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden

        # He initialisation for ReLU layers
        self.W1 = rng.standard_normal((n_in, h1)) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(h1)
        self.W2 = rng.standard_normal((h1, h2)) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = rng.standard_normal((h2, 3)) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(3)

        self.lr = lr
        # Adam state
        self._t = 0
        self._params = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self._m: Dict[str, np.ndarray] = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self._v: Dict[str, np.ndarray] = {p: np.zeros_like(getattr(self, p)) for p in self._params}

    # --- forward / backward ---

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Forward pass. Returns (probs, cache) where probs shape is (N, 3)."""
        a1 = self._relu(X @ self.W1 + self.b1)
        a2 = self._relu(a1 @ self.W2 + self.b2)
        z3 = a2 @ self.W3 + self.b3
        probs = self._softmax(z3)
        cache = {"X": X, "a1": a1, "a2": a2, "z3": z3, "probs": probs}
        return probs, cache

    def loss(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """MSE loss between predicted weights and target weights."""
        return float(np.mean((probs - targets) ** 2))

    def backward_and_update(self, cache: dict, targets: np.ndarray) -> float:
        """Backprop + Adam update. Returns batch loss."""
        probs = cache["probs"]
        N = probs.shape[0]

        # MSE gradient at output
        d_loss = 2.0 * (probs - targets) / N  # (N, 3)

        # Softmax Jacobian (diagonal - off-diagonal)
        # dL/dz3_j = sum_k dL/dprob_k * prob_k*(delta_jk - prob_j)
        s = probs  # (N, 3)
        # Vectorised: dL/dz = dL/dprob * (probs*(1-probs)) - cross terms
        # = s*(d_loss - (d_loss*s).sum(-1, keepdims=True))
        dz3 = s * (d_loss - (d_loss * s).sum(axis=-1, keepdims=True))  # (N, 3)

        dW3 = cache["a2"].T @ dz3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ self.W3.T
        da2[cache["a2"] <= 0] = 0  # ReLU grad
        dW2 = cache["a1"].T @ da2
        db2 = da2.sum(axis=0)

        da1 = da2 @ self.W2.T
        da1[cache["a1"] <= 0] = 0
        dW1 = cache["X"].T @ da1
        db1 = da1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        self._adam_step(grads)
        return self.loss(probs, targets)

    def _adam_step(self, grads: dict, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self._t += 1
        for p in self._params:
            g = grads[p]
            self._m[p] = beta1 * self._m[p] + (1 - beta1) * g
            self._v[p] = beta2 * self._v[p] + (1 - beta2) * g ** 2
            m_hat = self._m[p] / (1 - beta1 ** self._t)
            v_hat = self._v[p] / (1 - beta2 ** self._t)
            setattr(self, p, getattr(self, p) - self.lr * m_hat / (np.sqrt(v_hat) + eps))

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X)
        return probs

    def to_dict(self) -> dict:
        return {
            p: getattr(self, p).tolist()
            for p in self._params
        }

    def from_dict(self, d: dict) -> "_MLP":
        for p in self._params:
            setattr(self, p, np.array(d[p], dtype=np.float64))
        return self


# ---------------------------------------------------------------------------
# Feature normaliser
# ---------------------------------------------------------------------------

class _Normaliser:
    """Standard-score normalisation (zero mean, unit variance) per feature."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> "_Normaliser":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # avoid division by zero
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before transform().")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
        }

    def from_dict(self, d: dict) -> "_Normaliser":
        if d.get("mean") is not None:
            self.mean_ = np.array(d["mean"], dtype=np.float64)
            self.std_ = np.array(d["std"], dtype=np.float64)
            self.fitted = True
        return self


# ---------------------------------------------------------------------------
# Heuristic label generation
# ---------------------------------------------------------------------------

def _heuristic_weights(
    interaction_risk: float,
    pattern_risk: float,
    historical_risk: float,
    turn_index: int,
    is_attack: bool,
) -> Tuple[float, float, float]:
    """Generate heuristic (alpha, beta, gamma) supervision targets.

    Design rationale (documented heuristic, not ground truth):
    -----------------------------------------------------------
    - Early turns (low turn_index): weight history less (α low), weight current
      interaction more (β high) since there is little history.
    - Attack conversations with high pattern_risk: upweight pattern (γ).
    - Attack conversations with high interaction_risk and low pattern: upweight β.
    - Benign conversations: weight history more (α high) for smooth accumulation.

    Returns
    -------
    (alpha, beta, gamma) : three non-negative values summing to 1.
    """
    # Base logits tuned by domain reasoning
    if is_attack:
        if pattern_risk > 0.5:
            z = np.array([0.2, 0.5, 1.0])  # dominant gamma
        elif interaction_risk >= 4.0:
            z = np.array([0.2, 1.0, 0.3])  # dominant beta
        else:
            z = np.array([0.3, 0.7, 0.4])
    else:
        # Benign: more weight on history once we have turns
        z = np.array([0.6 + 0.1 * min(turn_index, 3), 0.5, 0.2])

    # Softmax
    e = np.exp(z - z.max())
    weights = e / e.sum()
    return float(weights[0]), float(weights[1]), float(weights[2])


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

ConversationRecord = Dict[str, Any]


def build_training_dataset(
    conversations: List[ConversationRecord],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build feature matrix X and target matrix Y from conversation records.

    Parameters
    ----------
    conversations : list of ConversationRecord
        Each record is a dict with keys:
            "is_attack"      : bool
            "tactic"         : str
            "turns"          : list of turn dicts, each containing:
                "interaction_risk"       : float (0-5)
                "pattern_risk"           : float (0-1)
                "patterns"               : dict  (pattern_name -> {detected: bool})
                "intent_shift_detected"  : bool
                "intent_shift_risk_level": int
                "current_human_turn"     : str
                "prev_human_turn"        : str

    Returns
    -------
    X : np.ndarray, shape (N_turns_total, 9)
    Y : np.ndarray, shape (N_turns_total, 3) – target [alpha, beta, gamma]
    meta : dict with "conversation_ids" per sample (for split safety)
    """
    rows_X, rows_Y, conv_ids = [], [], []

    for conv_idx, conv in enumerate(conversations):
        turns = conv.get("turns", [])
        is_attack = bool(conv.get("is_attack", False))
        historical_risk = 0.0
        prev_interaction_risk = 0.0

        for turn_idx, turn in enumerate(turns):
            ir = float(turn.get("interaction_risk", 0.0))
            pr = float(turn.get("pattern_risk", 0.0))
            patterns = turn.get("patterns", {})
            domain_shift = bool(patterns.get("domain_shift", {}).get("detected", False))
            shift_det = bool(turn.get("intent_shift_detected", False))
            shift_rl = int(turn.get("intent_shift_risk_level", 1))
            cur_text = str(turn.get("current_human_turn", ""))
            prev_text = str(turn.get("prev_human_turn", ""))

            feat = extract_features(
                historical_risk=historical_risk,
                interaction_risk=ir,
                pattern_risk=pr,
                turn_index=turn_idx,
                prev_interaction_risk=prev_interaction_risk,
                current_human_turn=cur_text,
                prev_human_turn=prev_text,
                intent_shift_detected=shift_det,
                intent_shift_risk_level=shift_rl,
                domain_shift_detected=domain_shift,
            )

            alpha, beta, gamma = _heuristic_weights(
                interaction_risk=ir,
                pattern_risk=pr,
                historical_risk=historical_risk,
                turn_index=turn_idx,
                is_attack=is_attack,
            )

            # Compute R_t with heuristic weights to advance historical risk
            historical_risk = alpha * historical_risk + beta * ir + gamma * pr
            prev_interaction_risk = ir

            rows_X.append(feat)
            rows_Y.append([alpha, beta, gamma])
            conv_ids.append(conv_idx)

    X = np.array(rows_X, dtype=np.float64)
    Y = np.array(rows_Y, dtype=np.float64)
    meta = {"conversation_ids": np.array(conv_ids, dtype=np.int64)}
    return X, Y, meta


# ---------------------------------------------------------------------------
# Main learner
# ---------------------------------------------------------------------------

class AdaptiveWeightLearner:
    """Trains and serves context-adaptive weights α_t, β_t, γ_t.

    Usage
    -----
    >>> learner = AdaptiveWeightLearner()
    >>> learner.fit(X, Y)                # X shape (N,9), Y shape (N,3)
    >>> alpha, beta, gamma = learner.predict_weights(feature_vec)  # shape (9,)
    >>> learner.save("tca/models/adaptive_weights.json")
    >>> learner2 = AdaptiveWeightLearner().load("tca/models/adaptive_weights.json")
    """

    def __init__(
        self,
        hidden: Tuple[int, int] = (32, 16),
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

        self._mlp = _MLP(n_in=N_FEATURES, hidden=hidden, lr=lr, seed=seed)
        self._norm = _Normaliser()
        self._fitted = False
        self.train_history: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> "AdaptiveWeightLearner":
        """Train the MLP on (X, Y) pairs.

        Parameters
        ----------
        X : np.ndarray, shape (N, 9)
        Y : np.ndarray, shape (N, 3)  – target [alpha, beta, gamma] rows
        X_val, Y_val : optional validation set (not used for early stopping yet)
        """
        X_norm = self._norm.fit_transform(X)
        rng = np.random.default_rng(self.seed)
        N = X_norm.shape[0]

        for epoch in range(self.epochs):
            idx = rng.permutation(N)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, N, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                Xb = X_norm[batch_idx]
                Yb = Y[batch_idx]
                _, cache = self._mlp.forward(Xb)
                loss = self._mlp.backward_and_update(cache, Yb)
                epoch_loss += loss
                steps += 1

            avg = epoch_loss / max(steps, 1)
            self.train_history.append(avg)

            if self.verbose and (epoch % 20 == 0 or epoch == self.epochs - 1):
                val_msg = ""
                if X_val is not None and Y_val is not None:
                    pv = self._mlp.predict(self._norm.transform(X_val))
                    val_loss = float(np.mean((pv - Y_val) ** 2))
                    val_msg = f"  val_loss={val_loss:.5f}"
                print(f"[Adaptive] epoch {epoch:3d}/{self.epochs}  train_loss={avg:.5f}{val_msg}")

        self._fitted = True
        return self

    def predict_weights(self, feature_vec: np.ndarray) -> Tuple[float, float, float]:
        """Predict (alpha_t, beta_t, gamma_t) for a single turn.

        Parameters
        ----------
        feature_vec : np.ndarray, shape (9,)

        Returns
        -------
        (alpha, beta, gamma) : floats summing to 1.
        """
        if not self._fitted:
            raise RuntimeError(
                "AdaptiveWeightLearner is not fitted. Call fit() or load() first."
            )
        x = self._norm.transform(feature_vec.reshape(1, -1))
        probs = self._mlp.predict(x)[0]
        return float(probs[0]), float(probs[1]), float(probs[2])

    def predict_weights_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict weights for a batch. Returns shape (N, 3)."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        X_norm = self._norm.transform(X)
        return self._mlp.predict(X_norm)

    # --- persistence ---

    def save(self, path: str) -> None:
        """Save model weights and normaliser to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mlp": self._mlp.to_dict(),
            "normaliser": self._norm.to_dict(),
            "config": {
                "hidden": list(self.hidden),
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "seed": self.seed,
            },
            "train_history": self.train_history,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[AdaptiveWeightLearner] Model saved to {path}")

    def load(self, path: str) -> "AdaptiveWeightLearner":
        """Load model from JSON file."""
        with open(path) as f:
            payload = json.load(f)
        cfg = payload.get("config", {})
        self.hidden = tuple(cfg.get("hidden", (32, 16)))
        self.lr = cfg.get("lr", 1e-3)
        self.epochs = cfg.get("epochs", 200)
        self.batch_size = cfg.get("batch_size", 32)
        self.seed = cfg.get("seed", 42)
        self.train_history = payload.get("train_history", [])
        self._mlp = _MLP(n_in=N_FEATURES, hidden=self.hidden, lr=self.lr, seed=self.seed)
        self._mlp.from_dict(payload["mlp"])
        self._norm = _Normaliser().from_dict(payload["normaliser"])
        self._fitted = True
        print(f"[AdaptiveWeightLearner] Model loaded from {path}")
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted
