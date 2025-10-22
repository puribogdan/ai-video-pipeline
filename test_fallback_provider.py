from pipeline.providers.fallback_provider import FallbackLLMProvider
from pipeline.config import settings

if not settings.CLAUDE_API_KEY or not settings.OPENAI_API_KEY:
    raise ValueError("Both CLAUDE_API_KEY and OPENAI_API_KEY must be set for fallback test")

provider = FallbackLLMProvider(
    claude_key=settings.CLAUDE_API_KEY,
    openai_key=settings.OPENAI_API_KEY
)

messages = [{"role": "user", "content": "Return a JSON object with a 'haiku' field containing a haiku about code."}]

result = provider.chat_json(messages)

print("Fallback provider result:")
print(result)
print(f"Last used provider: {provider.last_used_provider}")