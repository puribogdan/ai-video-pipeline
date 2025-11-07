# pipeline/providers/factory.py
from .fallback_provider import FallbackLLMProvider
from .claude_provider import ClaudeProvider
from .openai_provider import OpenAIProvider
from ..config import settings

def get_llm_provider():
    """
    Returns the best available LLM provider:
    1. Fallback provider if both Claude and OpenAI keys are available
    2. Single provider if only one key is available
    3. Raises error if no keys are configured
    """

    has_claude = bool(settings.CLAUDE_API_KEY)
    has_openai = bool(settings.OPENAI_API_KEY)

    if has_claude and has_openai:
        print("[DEBUG] Using Claude (primary) + OpenAI (fallback)")
        return FallbackLLMProvider(
            claude_key=settings.CLAUDE_API_KEY or "",
            openai_key=settings.OPENAI_API_KEY or ""
        )
    elif has_claude:
        print("[DEBUG] Using Claude only")
        return ClaudeProvider(settings.CLAUDE_API_KEY or "")
    elif has_openai:
        print("[DEBUG] Using OpenAI only")
        return OpenAIProvider(settings.OPENAI_API_KEY or "")
    else:
        available_keys = []
        if not has_claude:
            available_keys.append("CLAUDE_API_KEY")
        if not has_openai:
            available_keys.append("OPENAI_API_KEY")
        raise ValueError(f"Missing API keys: {', '.join(available_keys)}. Add them to your .env file.")