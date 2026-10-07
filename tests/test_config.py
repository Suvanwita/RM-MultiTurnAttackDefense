import pytest
import os
from config.tca_config import TCAConfig, WeightConfig, PatternWeights, RiskConfig


def test_load_valid_config():
    config = TCAConfig("config/config.yaml")
    assert config.risk.warn_threshold == 1.65
    assert config.risk.block_threshold == 2.475
    assert config.file_path == "dataset/harmbench_behaviors.csv"
    assert config.risk.weights.alpha == 0.3
    assert config.risk.weights.beta == 0.5
    assert config.risk.weights.gamma == 0.2


def test_weight_validation_success():
    wc = WeightConfig().load_from_dict({"alpha": 0.4, "beta": 0.4, "gamma": 0.2})
    wc.validate_weights()
    assert wc.alpha == 0.4


def test_weight_validation_failure():
    wc = WeightConfig().load_from_dict({"alpha": 0.8, "beta": 0.8, "gamma": 0.2})
    with pytest.raises(ValueError, match="Invalid weights"):
        wc.validate_weights()


def test_pattern_weight_validation_failure():
    pw = PatternWeights().load_from_dict({
        "language_change": 0.5,
        "domain_shift": 0.5,
        "time_sensitivity": 0.5,
        "prohibited_content": 0.5,
    })
    with pytest.raises(ValueError, match="Invalid pattern weights"):
        pw.validate_weights()


def test_log_config_runs(capsys):
    config = TCAConfig("config/config.yaml")
    config.risk.log_config()
    captured = capsys.readouterr()
    assert "Risk Configuration:" in captured.out
