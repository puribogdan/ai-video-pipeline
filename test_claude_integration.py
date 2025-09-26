#!/usr/bin/env python3
"""
Test script to verify Claude integration is working properly.
"""
import json
import sys
from pathlib import Path

# Add the pipeline directory to Python path
sys.path.append(str(Path(__file__).parent / "pipeline"))

from providers.factory import get_llm_provider
from config import settings

def test_claude_integration():
    """Test Claude integration with a simple prompt."""

    print("Testing Claude Integration...")
    print("=" * 50)

    # Check if API keys are configured
    has_claude = bool(settings.CLAUDE_API_KEY)
    has_openai = bool(settings.OPENAI_API_KEY)

    print(f"CLAUDE_API_KEY configured: {has_claude}")
    print(f"OPENAI_API_KEY configured: {has_openai}")
    print()

    if not has_claude and not has_openai:
        print("No API keys found in environment!")
        print("Please add CLAUDE_API_KEY and/or OPENAI_API_KEY to your .env file")
        return False

    try:
        # Get the provider
        print("Getting LLM provider...")
        provider = get_llm_provider()
        print(f"Using provider: {type(provider).__name__}")
        print()

        # Test with a simple prompt
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON."},
            {"role": "user", "content": "What is 2+2? Respond with JSON format: {\"answer\": number}"}
        ]

        print("Sending test request to LLM...")
        response = provider.chat_json(
            model="claude-opus-4-1-20250805",
            messages=test_messages,
            max_tokens=100
        )

        print("Received response:")
        print(json.dumps(response, indent=2))
        print()

        # Validate response
        if isinstance(response, dict) and "answer" in response:
            answer = response["answer"]
            if answer == 4:
                print("Test PASSED! Claude integration is working correctly.")
                return True
            else:
                print(f"Unexpected answer: {answer} (expected 4)")
                return False
        else:
            print("Response format unexpected")
            return False

    except Exception as e:
        print(f"Test FAILED with error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def test_provider_selection():
    """Test that provider selection logic works correctly."""

    print("\nTesting Provider Selection Logic...")
    print("=" * 50)

    try:
        provider = get_llm_provider()
        provider_name = type(provider).__name__

        if "Fallback" in provider_name:
            print("Fallback provider selected (both keys available)")
            print("Will try Claude first, then OpenAI if needed")
        elif "Claude" in provider_name:
            print("Claude provider selected (only Claude key available)")
        elif "OpenAI" in provider_name:
            print("OpenAI provider selected (only OpenAI key available)")
        else:
            print(f"Unknown provider: {provider_name}")

        return True

    except Exception as e:
        print(f"Provider selection failed: {e}")
        return False

if __name__ == "__main__":
    print("Claude Integration Test")
    print("=" * 50)

    # Test provider selection
    selection_ok = test_provider_selection()

    # Test actual API call
    integration_ok = test_claude_integration()

    print("\n" + "=" * 50)
    if selection_ok and integration_ok:
        print("ALL TESTS PASSED! Claude integration is working correctly.")
        sys.exit(0)
    else:
        print("Some tests failed. Check your configuration and API keys.")
        sys.exit(1)