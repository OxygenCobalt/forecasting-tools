from forecasting_tools.ai_models.model_archetypes.general_llm import (
    GeneralTextToTextLlm,
)


class GptO3Mini(GeneralTextToTextLlm):
    # See OpenAI Limit on the account dashboard for most up-to-date limit
    MODEL_NAME: str = "openai/o3-mini"
    REASONING_EFFORT: str = "high"
    REQUESTS_PER_PERIOD_LIMIT: int = 30_000
    REQUEST_PERIOD_IN_SECONDS: int = 60
    TIMEOUT_TIME: int = 300
    TOKENS_PER_PERIOD_LIMIT: int = 150_000_000
    TOKEN_PERIOD_IN_SECONDS: int = 60
