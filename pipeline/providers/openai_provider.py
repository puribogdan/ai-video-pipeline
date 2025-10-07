# pipeline/providers/openai_provider.py
import json
from openai import OpenAI
from typing import List, Dict, Any
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"  # Using gpt-4o as it's reliable and cost-effective
    
    def chat_json(self, messages: List[Dict], **kwargs) -> Dict:
        """
        Send chat messages to OpenAI and return JSON response
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Parsed JSON response as dictionary
        """
        # Prepare request parameters
        request_params = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "response_format": {"type": "json_object"}
        }
        
        # Add temperature if specified (OpenAI uses 0-2 range)
        if "temperature" in kwargs:
            request_params["temperature"] = kwargs["temperature"]
            
        # Add max_tokens if specified
        if "max_tokens" in kwargs:
            request_params["max_tokens"] = kwargs["max_tokens"]
        
        # Make the API call
        response = self.client.chat.completions.create(**request_params)
        
        # Return the parsed JSON content
        return json.loads(response.choices[0].message.content)