# pipeline/providers/claude_provider.py
import json
import anthropic
from typing import List, Dict, Any
from .base import LLMProvider

class ClaudeProvider(LLMProvider):
    """Claude API provider implementation"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
    
    def chat_json(self, messages: List[Dict], **kwargs) -> Dict:
        """
        Send chat messages to Claude and return JSON response
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Parsed JSON response as dictionary
        """
        # Convert OpenAI-style messages to Claude format
        claude_messages = []
        system_message = ""
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": claude_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        
        # Add system message if present
        if system_message:
            request_params["system"] = system_message
            
        # Add temperature if specified (Claude uses 0-1 range)
        if "temperature" in kwargs:
            request_params["temperature"] = kwargs["temperature"]
            
        # Make the API call
        response = self.client.messages.create(**request_params)
        
        # Return the parsed JSON content
        return json.loads(response.content[0].text)