from pipeline.providers.openai_provider import OpenAIProvider
from pipeline.config import settings

if not settings.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set for GPT-5 test")

provider = OpenAIProvider(api_key=settings.OPENAI_API_KEY, model="gpt-5")

messages = [{"role": "user", "content": "Return a JSON object with a 'response' field containing 'Hello from GPT-5'."}]

result = provider.chat_json(messages)

print("GPT-5 provider result:")
print(result)