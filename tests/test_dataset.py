import pytest
import pandas as pd
from config.tca_config import TCAConfig
from dataset.dataset_manager import DatasetManager


@pytest.fixture
def dataset_manager():
    config = TCAConfig("config/config.yaml")
    return DatasetManager(config)


def test_dataset_loading(dataset_manager):
    df = dataset_manager.load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "tactic" in df.columns
    assert "Source" in df.columns


def test_extract_conversation_pairs(dataset_manager):
    df = dataset_manager.load_data()
    row = df.iloc[0]
    pairs = dataset_manager.extract_conversation_pairs(row)
    assert isinstance(pairs, list)
    assert len(pairs) > 0
    # Each pair is a tuple of (assistant, human)
    for pair in pairs:
        assert isinstance(pair, tuple)
        assert len(pair) == 2


def test_dataset_metadata(dataset_manager, capsys):
    df = dataset_manager.load_data()
    metadata = dataset_manager.get_metadata(df)
    assert "total_records" in metadata
    assert metadata["total_records"] == len(df)
    assert "unique_tactics" in metadata
