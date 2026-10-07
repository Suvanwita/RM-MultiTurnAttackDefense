import pytest
from tca.risk_calculator import RiskCalculator


@pytest.fixture
def risk_calc():
    return RiskCalculator(alpha=0.3, beta=0.5, gamma=0.2)


def test_calculate_pattern_risk_none_detected(risk_calc):
    patterns = {
        "language_change": {"detected": False},
        "domain_shift": {"detected": False},
        "time_sensitivity": {"detected": False},
        "prohibited_content": {"detected": False},
    }
    score = risk_calc.calculate_pattern_risk(patterns)
    assert score == pytest.approx(0.0)


def test_calculate_pattern_risk_all_detected(risk_calc):
    patterns = {
        "language_change": {"detected": True},
        "domain_shift": {"detected": True},
        "time_sensitivity": {"detected": True},
        "prohibited_content": {"detected": True},
    }
    score = risk_calc.calculate_pattern_risk(patterns)
    # Sum of pattern weights in config: 0.2 + 0.3 + 0.2 + 0.3 = 1.0
    assert score == pytest.approx(1.0)


def test_progressive_risk_calculation(risk_calc):
    # progressive_risk = 0.3 * 1.0 + 0.5 * 2.0 + 0.2 * 0.5 = 0.3 + 1.0 + 0.1 = 1.4
    prog_risk = risk_calc.calculate_progressive_risk(
        historical_risk=1.0,
        interaction_risk=2.0,
        pattern_risk=0.5,
    )
    assert prog_risk == pytest.approx(1.4)
    assert risk_calc.historical_risk == pytest.approx(1.4)


def test_reset_historical_risk(risk_calc):
    risk_calc.calculate_progressive_risk(1.0, 2.0, 0.5)
    assert risk_calc.historical_risk > 0
    risk_calc.reset_historical_risk()
    assert risk_calc.historical_risk == 0
