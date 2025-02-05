from forecasting_tools.ai_models.model_archetypes.general_llm import (
    GeneralTextToTextLlm,
)


class GptO1(GeneralTextToTextLlm):
    # See OpenAI Limit on the account dashboard for most up-to-date limit
    MODEL_NAME: str = "openai/o1"
    REASONING_EFFORT: str = "high"
    REQUESTS_PER_PERIOD_LIMIT: int = 8_000
    REQUEST_PERIOD_IN_SECONDS: int = 60
    TIMEOUT_TIME: int = 120
    TOKENS_PER_PERIOD_LIMIT: int = 2_000_000
    TOKEN_PERIOD_IN_SECONDS: int = 60
