from typing import Final

from forecasting_tools.ai_models.model_interfaces.combined_llm_archetype import (
    CombinedLlmArchetype,
)


class Perplexity(CombinedLlmArchetype):
    MODEL_NAME: Final[str] = "perplexity/sonar-pro"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = (
        40  # Technically 50, but giving wiggle room
    )
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 120
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 2000000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
