# pipeline/providers/claude_provider.py
import json
import anthropic
import time
import base64
from typing import List, Dict, Any
from .base import LLMProvider

PORTRAIT_DESCRIPTION_PROMPT = """Describe the main subject in this image in one short paragraph. Include:
- What it is (person, animal, character, etc.)
- Key visual features (age/size, colors, distinctive characteristics)
- What they're wearing or how they look.
Use this format: A [subject] with [key features] wearing/looking [appearance details]
Keep it under 30 words and focus only on visual details needed to recognize them in an illustration."""

class ClaudeProvider(LLMProvider):
    """Claude API provider implementation"""

    def __init__(self, api_key: str, max_retries: int = 3, base_delay: float = 1.0):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5-20250929"
        self.max_retries = max_retries
        self.base_delay = base_delay
    
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
            "model": kwargs.get("model", self.model),
            "messages": claude_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        
        # Add system message if present
        if system_message:
            request_params["system"] = system_message
            
        # Add temperature if specified (Claude uses 0-1 range)
        if "temperature" in kwargs:
            request_params["temperature"] = kwargs["temperature"]
            
        # Make the API call with retry mechanism
        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.messages.create(**request_params)
                break  # Success, exit retry loop
            except Exception as api_error:
                error_type = type(api_error).__name__
                error_msg = str(api_error)

                # Check if it's an overloaded error (529)
                if "529" in error_msg or "overloaded" in error_msg.lower():
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"[WARNING] Claude API overloaded (attempt {attempt + 1}/{self.max_retries + 1}), retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"[ERROR] Claude API overloaded after {self.max_retries + 1} attempts")
                        raise api_error
                else:
                    # For other errors, don't retry
                    print(f"[ERROR] Claude API call failed: {error_type}: {error_msg}")
                    raise api_error

        # If we get here without response, it means all retries failed
        if response is None:
            raise RuntimeError("Failed to get response from Claude API after all retries")

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
    
    def describe_portrait(self, image_path: str) -> str:
        """
        Analyze a portrait image and return a description using Claude.
        
        Args:
            image_path: Path to the portrait image file
            
        Returns:
            Portrait description string
        """
        try:
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare the request
            request_params = {
                "model": self.model,
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": PORTRAIT_DESCRIPTION_PROMPT
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",  # Will be adjusted based on actual file type
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            }
            
            # Detect image type for proper media_type
            if image_path.lower().endswith('.png'):
                request_params["messages"][0]["content"][1]["source"]["media_type"] = "image/png"
            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                request_params["messages"][0]["content"][1]["source"]["media_type"] = "image/jpeg"
            elif image_path.lower().endswith('.webp'):
                request_params["messages"][0]["content"][1]["source"]["media_type"] = "image/webp"
            
            # Make the API call
            response = self.client.messages.create(**request_params)
            
            if not response.content or not hasattr(response.content[0], 'text'):
                raise ValueError("Invalid response from Claude API")
                
            description = response.content[0].text.strip()
            return description
            
        except Exception as e:
            print(f"[ERROR] Failed to generate portrait description: {e}")
            raise