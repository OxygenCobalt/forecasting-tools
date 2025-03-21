import typing_extensions

from forecasting_tools.ai_models.model_interfaces.combined_llm_archetype import (
    CombinedLlmArchetype,
)


@typing_extensions.deprecated(
    "LLM calls will slowly be moved to the GeneralLlm class", category=None
)
class GptO1Preview(CombinedLlmArchetype):
    # See OpenAI Limit on the account dashboard for most up-to-date limit
    MODEL_NAME: str = "openai/o1-preview"
    REASONING_EFFORT: str = "high"
    REQUESTS_PER_PERIOD_LIMIT: int = 8_000
    REQUEST_PERIOD_IN_SECONDS: int = 60
    TIMEOUT_TIME: int = 120
    TOKENS_PER_PERIOD_LIMIT: int = 2_000_000
    TOKEN_PERIOD_IN_SECONDS: int = 60
