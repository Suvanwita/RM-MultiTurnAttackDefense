# Temporal Context Awareness (TCA) Framework for Securing LLMs

## Description

This repository contains the implementation of the **Temporal Context Awareness (TCA) framework** as described in our paper: [Full Paper on arXiv](https://arxiv.org/pdf/2503.15560). The TCA framework is a novel defense mechanism designed to protect Large Language Models (LLMs) from multi-turn manipulation attacks. These attacks exploit the conversational nature of LLMs by gradually building context across multiple turns to elicit harmful or unauthorized responses.

## Key Components

The TCA framework combines several robust components to detect and mitigate adversarial tactics:

- **Semantic Drift Analysis:** Tracks shifts in conversation topics and intents.
- **Cross-Turn Consistency Verification:** Ensures alignment of user intent across conversational turns.
- **Progressive Risk Scoring:** Dynamically evaluates risks based on interaction patterns and historical data.

## Features

- **Risk Evaluation Metrics and Security Decision Engine:** Implements advanced risk assessment and mitigation strategies.
- **Adversarial Tactic Analysis:** Detects tactics including:
  - Direct Requests
  - Obfuscation
  - Hidden Intentions
  - Request Framing
  - Output Format Exploitation
  - Injection Attacks
  - Echoing Tactics
- **Multi-LLM Support:** Compatible with popular LLMs such as GPT, Claude, and Gemini.
- **Dataset Integration:** Automated testing on scenarios from the MHJ dataset ([Scale AI’s work](https://scale.com/research/mhj)).
- **Modular Python Codebase:** Built for extensibility and experimentation.

## Getting Started

### Prerequisites

- Python 3.8 to 3.12
- Virtual environment tool (`venv` or `uv`)

### Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install the required dependencies and package in editable mode:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Configure environment variables:
   Copy `.env.example` to `.env` and supply any API keys you wish to use:
   ```bash
   cp .env.example .env
   ```
   *(Note: You can also use the built-in `mock` LLM option to test without API keys.)*

## Usage

### 1. Run the Conversation Analyzer (Interactive Mode)
```bash
python intent/conversation_analyzer.py
```
You will be prompted to select an LLM (`gpt`, `claude`, `gemini`, or `mock`), select an attack tactic, and specify how many rows to analyze.

### 2. Run with Command-Line Arguments
```bash
python intent/conversation_analyzer.py --llm mock --tactic "Direct Request" --rows 2
```

### 3. Run the Automated Test Suite
```bash
pytest
```

## Features

- **Custom Risk Scoring:** Adjust weights and thresholds in `config/config.yaml` for fine-grained risk evaluation.
- **Security Decision Engine:** Tailor decision logic (`allow`, `warn`, `block`) to align with your security protocols.
- **Multi-LLM & Mock Support:** Seamlessly switch between OpenAI GPT, Anthropic Claude, Google Gemini, or offline Mock modes.
- **Extendability:** Easily incorporate additional components or custom datasets.

## Citation

```bibtex
@inproceedings{TCA2025,
  author         = {Kulkarni, P. and Namer, A.},
  title          = {Temporal Context Awareness (TCA) Framework for Securing LLMs},
  booktitle      = {Proceedings of the IEEE Conference on Artificial Intelligence (CAI)},
  year           = {2025},
  eprint         = {2503.15560},
  archivePrefix  = {arXiv},
  primaryClass   = {cs.LG},
  url            = {https://arxiv.org/pdf/2503.15560}
}

  

