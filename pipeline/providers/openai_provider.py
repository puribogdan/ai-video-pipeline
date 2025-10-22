# pipeline/providers/openai_provider.py
import json
from typing import List, Dict, Any

# Modern SDK (>=2.x)
from openai import OpenAI

# Optional: legacy import for fallback only (lazy use)
import importlib

class OpenAIProvider:
    """OpenAI API provider implementation (Responses API first, Chat Completions fallback)"""

    def __init__(self, api_key: str, model: str = "gpt-5"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _ensure_json_system(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            return messages
        return [
            {
                "role": "system",
                "content": (
                    "You are a JSON API. Reply with a single valid JSON object only, "
                    "with no extra text or code fences."
                ),
            },
            *messages,
        ]

    def _parse_output_text(self, result: Any) -> str:
        # Preferred accessor on modern SDK
        text = getattr(result, "output_text", None)
        if text:
            return text

        # Conservative fallback: walk the first text-bearing path if present
        try:
            # Many builds expose: result.output[0].content[0].text
            return result.output[0].content[0].text  # type: ignore[attr-defined,index]
        except Exception:
            return str(result)

    def chat_json(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Send chat messages and return a parsed JSON dict.
        `messages`: list of dicts with 'role' in {'system','user','assistant'} and 'content': str
        """
        full_messages = self._ensure_json_system(messages)

        # Prefer the modern Responses API if available
        if hasattr(self.client, "responses"):
            # Convert messages to input format for Responses API
            input_text = "\n".join([msg["content"] for msg in full_messages if msg["role"] != "system"])
            result = self.client.responses.create(
                model=self.model,
                input=input_text,
                reasoning={"effort": "low"},
                text={"verbosity": "low"},
            )
            text = self._parse_output_text(result)
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                snippet = text[:600].replace("\n", "\\n")
                raise ValueError(
                    f"OpenAI returned non-JSON text (first 600 chars): {snippet}"
                ) from e

        # No fallback - raise error if modern Responses API is not available
        raise RuntimeError("Modern OpenAI Responses API is not available. Please update your OpenAI SDK.")
