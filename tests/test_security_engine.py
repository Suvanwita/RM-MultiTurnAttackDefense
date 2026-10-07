import pytest
from tca.security_engine import SecurityDecisionEngine


@pytest.fixture
def engine():
    # warn = 1.65, block = 2.475
    return SecurityDecisionEngine(warn_threshold=1.65, block_threshold=2.475)


def test_allow_decision(engine):
    res = engine.evaluate_risk(progressive_risk=1.0, historical_risk=0.5)
    assert res["decision"] == "allow"
    assert "acceptable limits" in res["message"]


def test_warn_decision(engine):
    res = engine.evaluate_risk(progressive_risk=1.8, historical_risk=1.2)
    assert res["decision"] == "warn"
    assert "warning" in res["message"]


def test_block_decision(engine):
    res = engine.evaluate_risk(progressive_risk=2.5, historical_risk=2.0)
    assert res["decision"] == "block"
    assert "blocked" in res["message"]


def test_threshold_exact_boundaries(engine):
    warn_edge = engine.evaluate_risk(progressive_risk=1.65, historical_risk=1.0)
    assert warn_edge["decision"] == "warn"

    block_edge = engine.evaluate_risk(progressive_risk=2.475, historical_risk=2.0)
    assert block_edge["decision"] == "block"
