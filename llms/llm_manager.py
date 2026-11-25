import os
import re
from dotenv import load_dotenv, find_dotenv

import json
from typing import Dict, Any, Tuple
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai

# import replicate
from prompts.prompt_templates import PromptManager
from config.tca_config import TCAConfig
from prompts.prompt_response import Response

class LLMManager:
    def __init__(self, prompt_manager: PromptManager, config_path: str = "config/config.yaml"):
        """
        Initializes the LLMManager with the necessary API clients and keys.

        Args:
            prompt_manager (PromptManager): The PromptManager instance for managing prompts.
            config_path (str): Path to configuration YAML file.
        """
        self.prompt_manager = prompt_manager
        self.config = TCAConfig(config_path)
        # Load environment variables from .env file
        load_dotenv(find_dotenv())

        # Access environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_GENAI_API_KEY")
        self.replicate_api_key = os.getenv("LLAMA_API_KEY")

        gcp_project_id = self.config.gcp_project_id
        if gcp_project_id:
            print(f"Loaded GCP project ID: {gcp_project_id}")

        # Lazy / optional client initializations
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        self.anthropic_client = AsyncAnthropic(api_key=self.anthropic_api_key) if self.anthropic_api_key else None

    async def analyze_conversation(
        self, llm_type: str, prev_pair: Tuple[str, str], current_pair: Tuple[str, str]
    ) -> Dict[str, Any]:
        """
        Analyzes a conversation using the specified LLM type.

        Args:
            llm_type (str): The type of LLM (e.g., "gpt", "claude", "gemini", "mock").
            prev_pair (Tuple[str, str]): The previous assistant-human conversation pair.
            current_pair (Tuple[str, str]): The current assistant-human conversation pair.

        Returns:
            Dict[str, Any]: The analysis result from the LLM.
        """
        template = self.prompt_manager.get_template(llm_type)
        print("*" * 50)
        print("Formatting prompt now")
        prompt = template.format(prev_pair=prev_pair, current_pair=current_pair)

        handlers = {
            "gpt": self._call_gpt,
            "claude": self._call_claude,
            "gemini": self._call_gemini,
            "mock": self._call_mock,
        }

        handler = handlers.get(llm_type)
        if not handler:
            raise ValueError(f"Unsupported LLM type: {llm_type}. Supported types: {list(handlers.keys())}")

        if llm_type == "mock":
            user_text = current_pair[1] if current_pair and len(current_pair) > 1 else ""
            return await self._call_mock(prompt, user_text=user_text)

        return await handler(prompt)

    async def _call_gpt(self, prompt: str) -> Dict[str, Any]:
        """
        Calls OpenAI GPT model with the provided prompt.

        Args:
            prompt (str): The input prompt for GPT.

        Returns:
            Dict[str, Any]: The response as a JSON dictionary.
        """
        if not self.openai_client:
            raise ValueError("Missing OPENAI_API_KEY environment variable. Please set it in .env or your environment.")

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        # Extract content from OpenAI response
        response_content = response.choices[0].message.content
        parsed_data = self.parse_content(response_content)

        return {
            "content": response_content,
            **parsed_data,
        }

    def parse_content(self, content: str) -> dict:
        """
        Parse the LLM response content to extract structured information.
        Cleans markdown formatting and parses JSON.
        """
        try:
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            return json.loads(cleaned)
        except Exception:
            return {
                "risk_level": None,
                "recommendations": [],
                "intent_shift": {},
                "prompt_attack": {},
                "patterns": {},
                "overall_progression_summary": {},
            }

    async def _call_claude(self, prompt: str) -> Dict[str, Any]:
        """
        Calls Anthropic Claude model with the provided prompt.

        Args:
            prompt (str): The input prompt for Claude.

        Returns:
            Dict[str, Any]: The response as a JSON dictionary.
        """
        if not self.anthropic_client:
            raise ValueError("Missing ANTHROPIC_API_KEY environment variable. Please set it in .env or your environment.")

        response = await self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        content_text = response.content[0].text
        parsed_data = self.parse_content(content_text)
        return {
            "content": content_text,
            **parsed_data,
        }

    async def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        """
        Calls Google Generative AI Gemini model with the provided prompt.

        Args:
            prompt (str): The input prompt for Gemini.

        Returns:
            Dict[str, Any]: The response as a JSON dictionary.
        """
        if not self.google_api_key:
            raise ValueError("Missing GOOGLE_GENAI_API_KEY environment variable. Please set it in .env or your environment.")

        genai.configure(api_key=self.google_api_key)
        preferred_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        candidate_models = list(dict.fromkeys([preferred_model, "gemini-3.1-flash-lite", "gemini-flash-latest"]))

        last_error = None
        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = await model.generate_content_async(prompt)
                content_text = response.text
                parsed_data = self.parse_content(content_text)
                return {
                    "content": content_text,
                    **parsed_data,
                }
            except Exception as e:
                last_error = e
                continue

        raise last_error

    async def _call_mock(self, prompt: str, user_text: str = "") -> Dict[str, Any]:
        """
        Simulated LLM response for testing, live interactive demos, and offline execution.
        Dynamically detects hazardous directives or returns benign status.
        """
        eval_text = (user_text or prompt).lower()
        has_hazard = bool(re.search(
            r"\b(hotwire|bomb|handgun|firearm|exploit|bypass|password|kill|poison|malware|drop database|"
            r"override|illegal|fake id|minor|purchase|weapon|steal|exfiltrate|credentials|inject|"
            r"jailbreak|ransomware|keylogger|rootkit|payload|shell|synthesize|detonate|skimmer|opioid)\b",
            eval_text
        ))
        risk_level = 4 if has_hazard else 1
        mock_data = {
            "intent_shift": {
                "previous_intent": "General inquiry / baseline topic",
                "current_intent": "Manipulative or sensitive topic escalation" if has_hazard else "General conversational inquiry",
                "shift_detected": has_hazard,
                "shift_description": "Detected progression towards restricted domain" if has_hazard else "Coherent standard domain",
                "risk_level": risk_level,
            },
            "prompt_attack": {
                "detected": has_hazard,
                "evidence": "Observed pattern attempting to elicit sensitive output" if has_hazard else "No adversarial pattern detected",
            },
            "patterns": {
                "language_change": {
                    "detected": False,
                    "evidence": "Same language across turns",
                },
                "domain_shift": {
                    "detected": has_hazard,
                    "evidence": "Topic transitioned from informational to sensitive domain" if has_hazard else "Consistent conversation domain",
                },
                "time_sensitivity": {
                    "detected": False,
                    "evidence": "No urgency detected",
                },
                "prohibited_content": {
                    "detected": has_hazard,
                    "evidence": "Referenced prohibited actions/materials" if has_hazard else "Standard safe request",
                },
            },
            "risk_level": risk_level,
            "overall_progression_summary": {
                "initial_risk_level": 1,
                "final_risk_level": risk_level,
                "risk_progression": f"1 -> {risk_level}",
            },
            "recommendations": [
                "Decline to generate prohibited guidance" if has_hazard else "Respond helpfully and accurately",
                "Guide the user to safe educational or defensive alternatives" if has_hazard else "Maintain helpful tone",
            ],
        }
        content_json = json.dumps(mock_data)
        return {
            "content": content_json,
            **mock_data,
        }

    # async def _call_llama(self, prompt: str) -> Dict[str, Any]:
    #     """
    #     Calls LLaMA model hosted on Replicate with the provided prompt.

    #     Args:
    #         prompt (str): The input prompt for LLaMA.

    #     Returns:
    #         Dict[str, Any]: The response as a JSON dictionary.
    #     """
    #     output = await self.replicate_client.async_run(
    #         "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    #         input={"prompt": prompt},
    #     )
    #     return json.loads(output)
