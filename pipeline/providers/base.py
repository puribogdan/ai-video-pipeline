# pipeline/providers/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMProvider(ABC):
    """Base interface for LLM providers"""
    
    @abstractmethod
    def chat_json(self, messages: List[Dict], **kwargs) -> Dict:
        """
        Send chat messages and return JSON response
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Parsed JSON response as dictionary
        """
        pass