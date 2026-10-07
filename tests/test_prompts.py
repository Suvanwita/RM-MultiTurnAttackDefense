import pytest
from prompts.prompt_templates import PromptManager


@pytest.fixture
def prompt_manager():
    return PromptManager()


def test_get_templates(prompt_manager):
    for llm in ["gpt", "claude", "gemini", "llama"]:
        template = prompt_manager.get_template(llm)
        assert isinstance(template, str)
        assert "{prev_pair[0]}" in template
        assert "{current_pair[1]}" in template


def test_format_template(prompt_manager):
    template = prompt_manager.get_template("gpt")
    prev_pair = ("Hi, how can I help you?", "Can you tell me how locks work?")
    current_pair = ("Locks have pins and tumblers.", "How can I pick one without keys?")
    formatted = template.format(prev_pair=prev_pair, current_pair=current_pair)
    assert "Can you tell me how locks work?" in formatted
    assert "How can I pick one without keys?" in formatted
