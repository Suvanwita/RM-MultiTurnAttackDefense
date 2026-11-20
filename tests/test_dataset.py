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


def test_zn_benchmark_loading(dataset_manager):
    df = dataset_manager.load_zn_benchmark(split="test", limit=50)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "text" in df.columns
    assert "label" in df.columns
    assert "family" in df.columns
    assert "source" in df.columns
    assert set(df["label"].unique()).issubset({"ataque", "benigno"})


def test_zn_benchmark_metadata(dataset_manager):
    df = dataset_manager.load_zn_benchmark(limit=100)
    meta = dataset_manager.get_zn_metadata(df)
    assert "total_records" in meta
    assert meta["total_records"] == 100
    assert "label_distribution" in meta
    assert "family_distribution" in meta
    assert "char_length_stats" in meta


def test_zn_conversations_extraction(dataset_manager):
    convs = dataset_manager.extract_zn_conversations(split="test", limit=5)
    assert isinstance(convs, list)
    assert len(convs) == 5
    for conv in convs:
        assert "turns" in conv
        assert len(conv["turns"]) >= 2
        assert conv["family"] == "conversacional"
