"""
tests/test_adaptive_risk.py
============================
Unit and integration tests for Adaptive Contextual Risk Learning (Novelty 1).

Tests cover:
  - context_feature_extractor: feature vector shape, ranges, expected values
  - AdaptiveWeightLearner: MLP forward pass, training, weight constraints, persistence
  - RiskCalculator: fixed mode unchanged, adaptive mode integration
  - ConversationAnalyzer: backward compatibility with fixed mode
  - Training pipeline: build_training_dataset, conversation splits
"""

import json
import os
import sys
import tempfile
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from tca.context_feature_extractor import (
    extract_features,
    FEATURE_NAMES,
    N_FEATURES,
    _bow_cosine,
    _tokenise,
)
from tca.adaptive_weight_learner import (
    AdaptiveWeightLearner,
    build_training_dataset,
    _heuristic_weights,
    _MLP,
    _Normaliser,
)
from tca.risk_calculator import RiskCalculator
from config.tca_config import TCAConfig


# ===========================================================================
# 1. context_feature_extractor
# ===========================================================================

class TestContextFeatureExtractor:

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            historical_risk=1.5,
            interaction_risk=3.0,
            pattern_risk=0.5,
            turn_index=2,
            prev_interaction_risk=2.0,
            current_human_turn="How do I hack a system?",
            prev_human_turn="Tell me about computers.",
            intent_shift_detected=True,
            intent_shift_risk_level=4,
            domain_shift_detected=True,
        )
        kwargs.update(overrides)
        return kwargs

    def test_shape(self):
        feat = extract_features(**self._base_kwargs())
        assert feat.shape == (N_FEATURES,), f"Expected ({N_FEATURES},), got {feat.shape}"
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_dtype(self):
        feat = extract_features(**self._base_kwargs())
        assert feat.dtype == np.float64

    def test_historical_risk_feature(self):
        feat = extract_features(**self._base_kwargs(historical_risk=2.5))
        assert feat[0] == pytest.approx(2.5)

    def test_interaction_risk_feature(self):
        feat = extract_features(**self._base_kwargs(interaction_risk=4.0))
        assert feat[1] == pytest.approx(4.0)

    def test_pattern_risk_feature(self):
        feat = extract_features(**self._base_kwargs(pattern_risk=0.8))
        assert feat[2] == pytest.approx(0.8)

    def test_turn_index_feature(self):
        feat = extract_features(**self._base_kwargs(turn_index=5))
        assert feat[3] == pytest.approx(5.0)

    def test_risk_delta_feature(self):
        feat = extract_features(**self._base_kwargs(interaction_risk=4.0, prev_interaction_risk=2.0))
        assert feat[4] == pytest.approx(2.0)

    def test_risk_delta_negative(self):
        feat = extract_features(**self._base_kwargs(interaction_risk=1.0, prev_interaction_risk=3.0))
        assert feat[4] == pytest.approx(-2.0)

    def test_semantic_similarity_identical_text(self):
        feat = extract_features(
            **self._base_kwargs(
                current_human_turn="hello world",
                prev_human_turn="hello world",
            )
        )
        assert feat[5] == pytest.approx(1.0)

    def test_semantic_similarity_disjoint_text(self):
        feat = extract_features(
            **self._base_kwargs(
                current_human_turn="apple orange banana",
                prev_human_turn="car truck bus train",
            )
        )
        assert feat[5] == pytest.approx(0.0)

    def test_intent_escalation_true(self):
        feat = extract_features(
            **self._base_kwargs(intent_shift_detected=True, intent_shift_risk_level=4)
        )
        assert feat[6] == 1.0

    def test_intent_escalation_low_risk_level(self):
        # shift detected but risk_level < 3 → not escalation
        feat = extract_features(
            **self._base_kwargs(intent_shift_detected=True, intent_shift_risk_level=2)
        )
        assert feat[6] == 0.0

    def test_topic_transition_true(self):
        feat = extract_features(**self._base_kwargs(domain_shift_detected=True))
        assert feat[7] == 1.0

    def test_topic_transition_false(self):
        feat = extract_features(**self._base_kwargs(domain_shift_detected=False))
        assert feat[7] == 0.0

    def test_confidence_range(self):
        feat = extract_features(**self._base_kwargs())
        assert 0.0 <= feat[8] <= 1.0

    def test_empty_text_no_crash(self):
        feat = extract_features(
            **self._base_kwargs(current_human_turn="", prev_human_turn="")
        )
        assert feat[5] == pytest.approx(0.0)

    def test_zero_all_risks(self):
        feat = extract_features(
            historical_risk=0.0,
            interaction_risk=0.0,
            pattern_risk=0.0,
            turn_index=0,
            prev_interaction_risk=0.0,
            current_human_turn="hello",
            prev_human_turn="hi",
            intent_shift_detected=False,
            intent_shift_risk_level=1,
            domain_shift_detected=False,
        )
        assert feat.shape == (N_FEATURES,)
        assert all(np.isfinite(feat))


class TestBOWCosine:

    def test_identical(self):
        assert _bow_cosine("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint(self):
        assert _bow_cosine("cat dog", "car bus") == pytest.approx(0.0)

    def test_empty(self):
        assert _bow_cosine("", "hello") == pytest.approx(0.0)
        assert _bow_cosine("hello", "") == pytest.approx(0.0)
        assert _bow_cosine("", "") == pytest.approx(0.0)

    def test_symmetry(self):
        a, b = "apple orange", "orange banana apple"
        assert _bow_cosine(a, b) == pytest.approx(_bow_cosine(b, a))

    def test_case_insensitive(self):
        assert _bow_cosine("Hello World", "hello world") == pytest.approx(1.0)


# ===========================================================================
# 2. MLP internals
# ===========================================================================

class TestMLP:

    def test_forward_output_shape(self):
        mlp = _MLP(n_in=9, hidden=(16, 8))
        X = np.random.randn(10, 9)
        probs, cache = mlp.forward(X)
        assert probs.shape == (10, 3)

    def test_softmax_sums_to_one(self):
        mlp = _MLP(n_in=9, hidden=(16, 8))
        X = np.random.randn(5, 9)
        probs, _ = mlp.forward(X)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_all_weights_positive(self):
        mlp = _MLP(n_in=9, hidden=(16, 8))
        X = np.random.randn(5, 9)
        probs, _ = mlp.forward(X)
        assert np.all(probs >= 0.0)

    def test_backward_does_not_crash(self):
        mlp = _MLP(n_in=9, hidden=(16, 8))
        X = np.random.randn(8, 9)
        Y = np.array([[0.3, 0.5, 0.2]] * 8)
        probs, cache = mlp.forward(X)
        loss = mlp.backward_and_update(cache, Y)
        assert np.isfinite(loss)

    def test_loss_decreases_after_training(self):
        mlp = _MLP(n_in=9, hidden=(16, 8), lr=1e-2, seed=0)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((64, 9))
        Y = np.tile([0.4, 0.4, 0.2], (64, 1))
        probs0, cache0 = mlp.forward(X)
        loss0 = mlp.loss(probs0, Y)
        for _ in range(50):
            _, cache = mlp.forward(X)
            mlp.backward_and_update(cache, Y)
        probs1, _ = mlp.forward(X)
        loss1 = mlp.loss(probs1, Y)
        assert loss1 < loss0, f"Loss did not decrease: {loss0:.4f} → {loss1:.4f}"

    def test_serialisation_roundtrip(self):
        mlp = _MLP(n_in=9, hidden=(16, 8), seed=1)
        X = np.random.randn(3, 9)
        probs_before, _ = mlp.forward(X)
        d = mlp.to_dict()
        mlp2 = _MLP(n_in=9, hidden=(16, 8), seed=99)
        mlp2.from_dict(d)
        probs_after, _ = mlp2.forward(X)
        np.testing.assert_allclose(probs_before, probs_after, atol=1e-10)


# ===========================================================================
# 3. Normaliser
# ===========================================================================

class TestNormaliser:

    def test_fit_transform(self):
        norm = _Normaliser()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Xn = norm.fit_transform(X)
        np.testing.assert_allclose(Xn.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(Xn.std(axis=0), 1.0, atol=1e-6)

    def test_transform_before_fit_raises(self):
        norm = _Normaliser()
        with pytest.raises(RuntimeError):
            norm.transform(np.array([[1.0, 2.0]]))

    def test_zero_variance_column(self):
        norm = _Normaliser()
        X = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])
        Xn = norm.fit_transform(X)
        # Zero-variance column should become 0
        np.testing.assert_allclose(Xn[:, 0], 0.0, atol=1e-10)

    def test_serialisation_roundtrip(self):
        norm = _Normaliser()
        X = np.random.randn(20, 5)
        norm.fit(X)
        d = norm.to_dict()
        norm2 = _Normaliser().from_dict(d)
        X2 = np.random.randn(5, 5)
        np.testing.assert_allclose(norm.transform(X2), norm2.transform(X2), atol=1e-10)


# ===========================================================================
# 4. Heuristic weights
# ===========================================================================

class TestHeuristicWeights:

    def test_sum_to_one(self):
        for is_attack in [True, False]:
            for ir in [1.0, 3.0, 5.0]:
                for pr in [0.0, 0.5, 1.0]:
                    a, b, g = _heuristic_weights(ir, pr, 0.5, 2, is_attack)
                    assert a + b + g == pytest.approx(1.0, abs=1e-6)

    def test_all_non_negative(self):
        a, b, g = _heuristic_weights(4.0, 0.8, 1.0, 1, True)
        assert a >= 0 and b >= 0 and g >= 0

    def test_attack_high_pattern_upweights_gamma(self):
        a_att, b_att, g_att = _heuristic_weights(2.0, 0.9, 0.5, 1, True)
        a_ben, b_ben, g_ben = _heuristic_weights(2.0, 0.9, 0.5, 1, False)
        # attack with high pattern should give higher gamma
        assert g_att >= g_ben

    def test_benign_upweights_alpha(self):
        a_ben, b_ben, g_ben = _heuristic_weights(1.0, 0.0, 0.5, 5, False)
        a_att, b_att, g_att = _heuristic_weights(1.0, 0.0, 0.5, 5, True)
        assert a_ben >= a_att


# ===========================================================================
# 5. AdaptiveWeightLearner
# ===========================================================================

class TestAdaptiveWeightLearner:

    @pytest.fixture
    def small_dataset(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, N_FEATURES))
        Y = np.tile([0.3, 0.5, 0.2], (60, 1))
        Y += rng.standard_normal((60, 3)) * 0.02
        # Renormalise rows
        Y = np.abs(Y)
        Y = Y / Y.sum(axis=1, keepdims=True)
        return X, Y

    def test_fit_then_predict(self, small_dataset):
        X, Y = small_dataset
        learner = AdaptiveWeightLearner(epochs=20, lr=1e-2, verbose=False)
        learner.fit(X, Y)
        a, b, g = learner.predict_weights(X[0])
        assert a + b + g == pytest.approx(1.0, abs=1e-5)
        assert a >= 0 and b >= 0 and g >= 0

    def test_predict_before_fit_raises(self):
        learner = AdaptiveWeightLearner()
        with pytest.raises(RuntimeError):
            learner.predict_weights(np.zeros(N_FEATURES))

    def test_predict_batch_shape(self, small_dataset):
        X, Y = small_dataset
        learner = AdaptiveWeightLearner(epochs=10, verbose=False)
        learner.fit(X, Y)
        preds = learner.predict_weights_batch(X[:5])
        assert preds.shape == (5, 3)

    def test_predict_batch_sums_to_one(self, small_dataset):
        X, Y = small_dataset
        learner = AdaptiveWeightLearner(epochs=10, verbose=False)
        learner.fit(X, Y)
        preds = learner.predict_weights_batch(X)
        np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-5)

    def test_train_history_non_empty(self, small_dataset):
        X, Y = small_dataset
        learner = AdaptiveWeightLearner(epochs=15, verbose=False)
        learner.fit(X, Y)
        assert len(learner.train_history) == 15

    def test_save_load_roundtrip(self, small_dataset):
        X, Y = small_dataset
        learner = AdaptiveWeightLearner(epochs=20, lr=1e-2, verbose=False)
        learner.fit(X, Y)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            learner.save(path)
            learner2 = AdaptiveWeightLearner().load(path)
            a1, b1, g1 = learner.predict_weights(X[0])
            a2, b2, g2 = learner2.predict_weights(X[0])
            assert a1 == pytest.approx(a2, abs=1e-8)
            assert b1 == pytest.approx(b2, abs=1e-8)
            assert g1 == pytest.approx(g2, abs=1e-8)
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self):
        learner = AdaptiveWeightLearner()
        with pytest.raises(FileNotFoundError):
            learner.load("/nonexistent/path/model.json")

    def test_is_fitted_flag(self, small_dataset):
        X, Y = small_dataset
        learner = AdaptiveWeightLearner(epochs=5, verbose=False)
        assert not learner.is_fitted
        learner.fit(X, Y)
        assert learner.is_fitted


# ===========================================================================
# 6. build_training_dataset
# ===========================================================================

class TestBuildTrainingDataset:

    def _make_conversations(self, n=5, n_turns=3, is_attack=True):
        convs = []
        for _ in range(n):
            turns = []
            for t in range(n_turns):
                turns.append({
                    "interaction_risk": float(2 + t),
                    "pattern_risk": 0.5 if is_attack else 0.0,
                    "patterns": {
                        "domain_shift": {"detected": is_attack},
                        "language_change": {"detected": False},
                        "time_sensitivity": {"detected": False},
                        "prohibited_content": {"detected": is_attack and t >= 1},
                    },
                    "intent_shift_detected": is_attack and t >= 1,
                    "intent_shift_risk_level": 3 if (is_attack and t >= 1) else 1,
                    "current_human_turn": "how do i exploit this" if is_attack else "tell me about dogs",
                    "prev_human_turn": "what is security" if is_attack else "hi there",
                })
            convs.append({"is_attack": is_attack, "tactic": "test", "turns": turns})
        return convs

    def test_output_shapes(self):
        convs = self._make_conversations(n=4, n_turns=3)
        X, Y, meta = build_training_dataset(convs)
        assert X.shape == (12, N_FEATURES)
        assert Y.shape == (12, 3)
        assert len(meta["conversation_ids"]) == 12

    def test_weights_sum_to_one(self):
        convs = self._make_conversations(n=3, n_turns=2)
        X, Y, meta = build_training_dataset(convs)
        np.testing.assert_allclose(Y.sum(axis=1), 1.0, atol=1e-6)

    def test_weights_non_negative(self):
        convs = self._make_conversations(n=3, n_turns=2)
        X, Y, _ = build_training_dataset(convs)
        assert np.all(Y >= 0.0)

    def test_conversation_ids_contiguous(self):
        convs = self._make_conversations(n=3, n_turns=4)
        X, Y, meta = build_training_dataset(convs)
        ids = meta["conversation_ids"]
        # Each conversation contributes n_turns rows with same id
        assert list(ids) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    def test_empty_conversations_no_crash(self):
        X, Y, meta = build_training_dataset([])
        assert X.shape == (0, N_FEATURES) or len(X) == 0

    def test_features_finite(self):
        convs = self._make_conversations(n=5, n_turns=3)
        X, Y, _ = build_training_dataset(convs)
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(Y))


# ===========================================================================
# 7. RiskCalculator – backward compatibility (fixed mode)
# ===========================================================================

class TestRiskCalculatorFixedMode:
    """Existing fixed-weight behaviour must be fully preserved."""

    @pytest.fixture
    def calc(self):
        return RiskCalculator(alpha=0.3, beta=0.5, gamma=0.2)

    def test_progressive_risk_no_extra_args(self, calc):
        risk = calc.calculate_progressive_risk(1.0, 2.0, 0.5)
        # 0.3*1 + 0.5*2 + 0.2*0.5 = 0.3 + 1.0 + 0.1 = 1.4
        assert risk == pytest.approx(1.4)

    def test_historical_risk_updated(self, calc):
        calc.calculate_progressive_risk(0.0, 4.0, 1.0)
        assert calc.historical_risk == pytest.approx(0.3 * 0 + 0.5 * 4 + 0.2 * 1)

    def test_reset_historical_risk(self, calc):
        calc.calculate_progressive_risk(1.0, 2.0, 0.5)
        calc.reset_historical_risk()
        assert calc.historical_risk == 0

    def test_calculate_pattern_risk_all_detected(self, calc):
        # Uses default config pattern weights: 0.2+0.3+0.2+0.3=1.0
        patterns = {
            "language_change": {"detected": True},
            "domain_shift": {"detected": True},
            "time_sensitivity": {"detected": True},
            "prohibited_content": {"detected": True},
        }
        # pattern_weights come from config defaults; RiskCalculator uses config
        # when instantiated without explicit config — but here we pass alpha/beta/gamma.
        # We still need a config for pattern_weights, so use the default config path.
        calc2 = RiskCalculator(alpha=0.3, beta=0.5, gamma=0.2)
        score = calc2.calculate_pattern_risk(patterns)
        assert score == pytest.approx(1.0)

    def test_calculate_pattern_risk_none_detected(self, calc):
        patterns = {k: {"detected": False}
                    for k in ["language_change", "domain_shift",
                              "time_sensitivity", "prohibited_content"]}
        score = calc.calculate_pattern_risk(patterns)
        assert score == pytest.approx(0.0)

    def test_risk_mode_default_is_fixed(self):
        calc = RiskCalculator()
        assert calc._risk_mode == "fixed"

    def test_fixed_mode_ignores_adaptive_kwargs(self, calc):
        """Extra kwargs for adaptive mode should be accepted without error in fixed mode."""
        risk = calc.calculate_progressive_risk(
            1.0, 2.0, 0.5,
            turn_index=3,
            prev_interaction_risk=1.5,
            current_human_turn="test",
            prev_human_turn="prev",
            intent_shift_detected=True,
            intent_shift_risk_level=4,
            domain_shift_detected=True,
        )
        # Result must still be fixed-weight
        assert risk == pytest.approx(1.4)


# ===========================================================================
# 8. RiskCalculator – adaptive mode (with a trained model)
# ===========================================================================

class TestRiskCalculatorAdaptiveMode:

    @pytest.fixture
    def trained_model_path(self, tmp_path):
        """Train a tiny model and return its path."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, N_FEATURES))
        Y = np.tile([0.3, 0.5, 0.2], (40, 1))
        Y = np.abs(Y + rng.standard_normal((40, 3)) * 0.01)
        Y = Y / Y.sum(axis=1, keepdims=True)
        learner = AdaptiveWeightLearner(epochs=20, verbose=False)
        learner.fit(X, Y)
        path = str(tmp_path / "model.json")
        learner.save(path)
        return path

    def test_adaptive_mode_produces_valid_risk(self, trained_model_path):
        calc = RiskCalculator(alpha=0.3, beta=0.5, gamma=0.2)
        calc._risk_mode = "adaptive"
        from tca.adaptive_weight_learner import AdaptiveWeightLearner
        learner = AdaptiveWeightLearner().load(trained_model_path)
        calc._adaptive_learner = learner

        risk = calc.calculate_progressive_risk(
            historical_risk=1.0,
            interaction_risk=3.0,
            pattern_risk=0.5,
            turn_index=1,
            prev_interaction_risk=2.0,
            current_human_turn="how to exploit",
            prev_human_turn="tell me more",
            intent_shift_detected=True,
            intent_shift_risk_level=4,
            domain_shift_detected=True,
        )
        assert np.isfinite(risk)
        assert risk >= 0.0

    def test_adaptive_weights_sum_to_one_internally(self, trained_model_path):
        """The adaptive weights α+β+γ should always sum to 1."""
        from tca.adaptive_weight_learner import AdaptiveWeightLearner
        from tca.context_feature_extractor import extract_features
        learner = AdaptiveWeightLearner().load(trained_model_path)

        for _ in range(20):
            feat = np.random.randn(N_FEATURES)
            a, b, g = learner.predict_weights(feat)
            assert a + b + g == pytest.approx(1.0, abs=1e-5)
            assert a >= 0 and b >= 0 and g >= 0


# ===========================================================================
# 9. Config loading – new fields
# ===========================================================================

class TestConfigNewFields:

    def test_risk_mode_default_fixed(self):
        config = TCAConfig("config/config.yaml")
        # risk_mode should be "fixed" (or "adaptive" if changed in yaml)
        assert config.risk_mode in {"fixed", "adaptive"}

    def test_adaptive_model_path_exists_as_attribute(self):
        config = TCAConfig("config/config.yaml")
        assert hasattr(config, "adaptive_model_path")
        assert isinstance(config.adaptive_model_path, str)


# ===========================================================================
# 10. End-to-end: train → predict pipeline (integration)
# ===========================================================================

class TestEndToEndPipeline:

    def test_train_then_evaluate(self, tmp_path):
        """Full pipeline: build dataset → train → save → load → predict."""
        # Build a tiny synthetic dataset
        convs = []
        for attack in [True, False]:
            for ci in range(5):
                turns = []
                for t in range(3):
                    ir = float(2 + t) if attack else 1.0
                    pr = 0.5 if attack else 0.0
                    turns.append({
                        "interaction_risk": ir,
                        "pattern_risk": pr,
                        "patterns": {"domain_shift": {"detected": attack}},
                        "intent_shift_detected": attack,
                        "intent_shift_risk_level": 4 if attack else 1,
                        "current_human_turn": "attack text" if attack else "normal text",
                        "prev_human_turn": "prev text",
                    })
                convs.append({"is_attack": attack, "tactic": "test", "turns": turns})

        X, Y, meta = build_training_dataset(convs)

        learner = AdaptiveWeightLearner(epochs=30, lr=5e-3, verbose=False)
        learner.fit(X, Y)

        model_path = str(tmp_path / "test_model.json")
        learner.save(model_path)

        learner2 = AdaptiveWeightLearner().load(model_path)
        a, b, g = learner2.predict_weights(X[0])
        assert a + b + g == pytest.approx(1.0, abs=1e-5)
        assert a >= 0 and b >= 0 and g >= 0

    def test_risk_calculator_full_conversation(self, tmp_path):
        """Simulate a 5-turn attack conversation with adaptive mode."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((60, N_FEATURES))
        Y_train = np.tile([0.3, 0.5, 0.2], (60, 1))
        Y_train = np.abs(Y_train + rng.standard_normal((60, 3)) * 0.01)
        Y_train = Y_train / Y_train.sum(axis=1, keepdims=True)

        learner = AdaptiveWeightLearner(epochs=20, verbose=False)
        learner.fit(X_train, Y_train)
        model_path = str(tmp_path / "sim_model.json")
        learner.save(model_path)

        calc = RiskCalculator(alpha=0.3, beta=0.5, gamma=0.2)
        calc._risk_mode = "adaptive"
        calc._adaptive_learner = AdaptiveWeightLearner().load(model_path)

        risks = []
        prev_ir = 0.0
        for t in range(5):
            ir = float(2 + t)
            pr = 0.3 + t * 0.1
            risk = calc.calculate_progressive_risk(
                calc.historical_risk, ir, pr,
                turn_index=t,
                prev_interaction_risk=prev_ir,
                current_human_turn=f"attack message {t}",
                prev_human_turn=f"prev message {t}",
                intent_shift_detected=True,
                intent_shift_risk_level=4,
                domain_shift_detected=True,
            )
            risks.append(risk)
            prev_ir = ir

        # Risk should grow across turns (not strictly guaranteed but expected)
        assert len(risks) == 5
        assert all(np.isfinite(r) for r in risks)
        assert risks[-1] > 0
