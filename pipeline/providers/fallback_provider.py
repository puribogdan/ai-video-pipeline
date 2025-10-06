# pipeline/providers/fallback_provider.py
import time
import json
from typing import List, Dict, Any, Optional
from .base import LLMProvider
from .claude_provider import ClaudeProvider
from .openai_provider import OpenAIProvider

class FallbackLLMProvider(LLMProvider):
    """Fallback provider that tries OpenAI first, then Claude"""

    def __init__(self, claude_key: str, openai_key: str):
        self.primary = OpenAIProvider(openai_key)
        self.fallback = ClaudeProvider(claude_key)
        self.last_used_provider = None
        self.fallback_delay = 1.0  # seconds
    
    def chat_json(self, messages: List[Dict], **kwargs) -> Dict:
        """
        Try OpenAI first, fallback to Claude if it fails

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response as dictionary

        Raises:
            Exception: If both providers fail
        """
        # Try primary provider (OpenAI) first
        try:
            print("[DEBUG] Attempting OpenAI (primary)...")
            result = self.primary.chat_json(messages, **kwargs)
            self.last_used_provider = "openai"
            print("[DEBUG] Successfully used OpenAI")
            return result

        except Exception as primary_error:
            print(f"[WARNING] OpenAI failed: {primary_error}")
            print("[DEBUG] Falling back to Claude...")

            # Add small delay before fallback attempt
            time.sleep(self.fallback_delay)

            try:
                result = self.fallback.chat_json(messages, **kwargs)
                self.last_used_provider = "claude"
                print("[DEBUG] Successfully used Claude as fallback")
                return result

            except Exception as fallback_error:
                print("[ERROR] Both OpenAI and Claude failed")
                print(f"[ERROR] OpenAI error: {primary_error}")
                print(f"[ERROR] Claude error: {fallback_error}")
                # Raise the fallback error since it's the most recent
                raise fallback_error
    
    def get_last_used_provider(self) -> Optional[str]:
        """Get the name of the last provider that was successfully used"""
        return self.last_used_provider