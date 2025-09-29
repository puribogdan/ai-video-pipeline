# pipeline/providers/claude_provider.py
import json
import anthropic
from typing import List, Dict, Any
from .base import LLMProvider

class ClaudeProvider(LLMProvider):
    """Claude API provider implementation"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-opus-4-1-20250805"
    
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
        try:
            response = self.client.messages.create(**request_params)
        except Exception as api_error:
            print(f"[ERROR] Claude API call failed: {type(api_error).__name__}: {api_error}")
            raise api_error

        # Add detailed logging to debug the response
        print(f"[DEBUG] Claude API Response type: {type(response)}")
        print(f"[DEBUG] Response content type: {type(response.content)}")
        print(f"[DEBUG] Response content length: {len(response.content) if hasattr(response.content, '__len__') else 'N/A'}")

        # Validate response structure
        if not response.content:
            print("[ERROR] Claude API returned empty content array!")
            print(f"[ERROR] Full response object: {response}")
            raise ValueError("Claude API returned empty content array")

        if not hasattr(response.content[0], 'text') or response.content[0].text is None:
            print("[ERROR] Claude API returned invalid content block!")
            print(f"[ERROR] Content block: {response.content[0]}")
            print(f"[ERROR] Full response object: {response}")
            raise ValueError("Claude API returned invalid content block")

        content_text = response.content[0].text.strip()

        # Check if text is empty after stripping whitespace
        if not content_text:
            print("[ERROR] Claude API returned empty response content!")
            print(f"[ERROR] Full response object: {response}")
            raise ValueError("Claude API returned empty response content")

        print(f"[DEBUG] Content text length: {len(content_text)}")
        print(f"[DEBUG] Raw content: {repr(content_text)}")

        # Handle markdown code blocks (common with newer models)
        if content_text.startswith('```json'):
            # Extract JSON from markdown code block
            lines = content_text.split('\n')
            json_lines = []
            in_code_block = False

            for line in lines:
                if line.strip() == '```json':
                    in_code_block = True
                    continue
                elif line.strip() == '```' and in_code_block:
                    break
                elif in_code_block:
                    json_lines.append(line)

            content_text = '\n'.join(json_lines).strip()
            print(f"[DEBUG] Extracted JSON from markdown: {repr(content_text)}")

        print(f"[DEBUG] Attempting to parse JSON: {content_text[:200]}...")

        # Try to extract JSON if there's text before it
        json_text = content_text.strip()

        # Look for JSON object start
        json_start = json_text.find('{')
        if json_start == -1:
            print(f"[ERROR] No JSON object found in response")
            print(f"[ERROR] Raw content: {repr(content_text)}")
            raise ValueError("No JSON object found in Claude response")

        # Extract from first { to last }
        json_end = json_text.rfind('}')
        if json_end == -1:
            print(f"[ERROR] No JSON object end found in response")
            print(f"[ERROR] Raw content: {repr(content_text)}")
            raise ValueError("No JSON object end found in Claude response")

        json_text = json_text[json_start:json_end + 1]
        print(f"[DEBUG] Extracted JSON: {json_text[:200]}...")

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as json_error:
            print(f"[ERROR] Failed to parse JSON response from Claude: {json_error}")
            print(f"[ERROR] Extracted JSON: {repr(json_text)}")
            print(f"[ERROR] Raw content: {repr(content_text)}")
            print(f"[ERROR] Full response object: {response}")
            raise ValueError(f"Claude API returned invalid JSON: {json_error}")