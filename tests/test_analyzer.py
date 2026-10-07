import pytest
import asyncio
from intent.conversation_analyzer import ConversationAnalyzer
from llms.llm_manager import LLMManager
from prompts.prompt_templates import PromptManager


@pytest.mark.asyncio
async def test_mock_llm_analysis():
    pm = PromptManager()
    llm = LLMManager(prompt_manager=pm)
    prev_pair = ("Hello, how can I help?", "Tell me about cars.")
    current_pair = ("Cars are motor vehicles.", "How to hotwire a car?")

    res = await llm.analyze_conversation("mock", prev_pair, current_pair)
    assert "overall_progression_summary" in res
    assert "patterns" in res
    assert res["patterns"]["domain_shift"]["detected"] is True
    assert res["patterns"]["prohibited_content"]["detected"] is True


@pytest.mark.asyncio
async def test_conversation_analyzer_single_row():
    analyzer = ConversationAnalyzer(config_path="config/config.yaml", llm_type="mock")
    df = analyzer.dataset_manager.load_data()
    first_row = df.iloc[0]

    # Should run through pairs and calculate progressive risk and decisions
    await analyzer.analyze_row(0, first_row)
    assert analyzer.historical_risk > 0


@pytest.mark.asyncio
async def test_conversation_analyzer_batch():
    analyzer = ConversationAnalyzer(config_path="config/config.yaml", llm_type="mock")
    # Run 1 row of 'Direct Request'
    await analyzer.analyze_conversations(tactic_name="Direct Request", max_rows=1)
    assert analyzer.historical_risk > 0


def test_missing_api_key_raises_informative_error():
    pm = PromptManager()
    llm = LLMManager(prompt_manager=pm)
    # Ensure client is None when no key is set
    llm.openai_client = None

    with pytest.raises(ValueError, match="Missing OPENAI_API_KEY"):
        asyncio.run(llm._call_gpt("test prompt"))
